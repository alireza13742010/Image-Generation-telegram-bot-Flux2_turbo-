"""
FLUX.2 [dev] 4-bit + Turbo LoRA - queued image server for n8n / Telegram.

One GPU = one image at a time. Extra requests wait in a FIFO queue, and n8n
polls each job to show a progress bar and an estimated waiting time.

Install:  pip install fastapi uvicorn
Run:      uvicorn flux_queue_server:app --host 0.0.0.0 --port 8000
          (keep a single worker: do NOT use --workers or --reload)

Endpoints
  POST /jobs             {"prompt": "...", "user": "123", "seed": 42}  -> accepted? + job info
  GET  /jobs/{id}        status, progress %, ETA, queue position
  GET  /jobs/{id}/image  the finished picture
  GET  /queue            how many images run at once / how many are waiting
  GET  /health
"""
import gc
import inspect
import io
import random
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from diffusers import Flux2Pipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
#  CONFIG (model settings are the same as your script)
# ─────────────────────────────────────────────────────────────

LOCAL_BASE_DIR      = "./models/flux2-dev-bnb-4bit"
LOCAL_TURBO_DIR     = "./models/flux2-turbo-lora"
TURBO_LORA_FILENAME = "flux.2-turbo-lora.safetensors"

DEVICE = "cuda:0"
DTYPE  = torch.bfloat16

TURBO_SIGMAS        = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]
HEIGHT              = 1024
WIDTH               = 1024
GUIDANCE_SCALE      = 2.5
NUM_INFERENCE_STEPS = 8

MAX_CONCURRENT = 1      # one GPU -> one image at a time (informational; the worker is single)
MAX_QUEUE      = 10     # how many requests may wait behind the one running
KEEP_SECONDS   = 3600   # finished jobs (and their images) are kept this long

OUTPUT_FORMAT = "JPEG"  # "JPEG" (small, fast to upload) or "PNG"
JPEG_QUALITY  = 95

# Rough seconds per stage. They are refined automatically after every finished job,
# so the progress bar and the "time left" estimate get more accurate over time.
STAGES = ["encode", "load", "denoise", "finish"]
STAGE_EST = {"encode": 45.0, "load": 40.0, "denoise": 40.0, "finish": 3.0}
STAGE_LABEL = {
    "queued":  "Waiting in the queue",
    "encode":  "Reading your prompt",
    "load":    "Loading the image model",
    "denoise": "Generating the image",
    "finish":  "Finishing up",
    "done":    "Done",
}

# ─────────────────────────────────────────────────────────────
#  JOB STATE
# ─────────────────────────────────────────────────────────────


@dataclass
class Job:
    id: str
    user: str
    prompt: str
    seed: int
    created: float = field(default_factory=time.time)
    status: str = "queued"          # queued | running | done | error
    stage: str = "queued"
    stage_started: float = 0.0
    step: int = 0
    total_steps: int = NUM_INFERENCE_STEPS
    started: float = 0.0
    finished: float = 0.0
    error: str = ""
    image: Optional[bytes] = None
    stage_times: Dict[str, float] = field(default_factory=dict)


cond = threading.Condition()     # re-entrant lock + wake-up signal for the worker
jobs: Dict[str, Job] = {}
waiting: List[str] = []          # FIFO of job ids that have not started yet
current: Optional[str] = None    # id of the job on the GPU right now


def free_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def set_stage(job: Job, stage: str) -> None:
    with cond:
        now = time.time()
        if job.stage in STAGE_EST and job.stage_started:
            job.stage_times[job.stage] = now - job.stage_started
        job.stage = stage
        job.stage_started = now
        job.step = 0


def total_estimate() -> float:
    return sum(STAGE_EST.values())


def update_estimates(job: Job) -> None:
    for stage in STAGES:
        measured = job.stage_times.get(stage)
        if measured:
            STAGE_EST[stage] = 0.7 * STAGE_EST[stage] + 0.3 * measured


def job_progress(job: Job):
    """(fraction done 0..1, estimated seconds left) for a job that is running."""
    now = time.time()
    done = left = 0.0
    reached = False
    for stage in STAGES:
        est = STAGE_EST[stage]
        if stage == job.stage:
            reached = True
            if stage == "denoise" and job.step > 0:
                frac = job.step / job.total_steps
            else:
                frac = min((now - job.stage_started) / est, 0.95)
            done += est * frac
            left += est * (1.0 - frac)
        elif reached:
            left += est
        else:
            done += est
    total = done + left
    return (done / total if total else 0.0), left


def queue_info() -> dict:
    """Caller must hold `cond`."""
    return {
        "max_concurrent": MAX_CONCURRENT,
        "running": 1 if current else 0,
        "queued": len(waiting),
        "queue_length": len(waiting),
        "max_queue": MAX_QUEUE,
        "avg_seconds": round(total_estimate(), 1),
    }


def snapshot(job: Job) -> dict:
    """Everything n8n needs to draw the progress message. Caller must hold `cond`."""
    now = time.time()
    ahead = 0
    if job.status == "done":
        pct, eta = 100, 0.0
    elif job.status == "error":
        pct, eta = 0, 0.0
    elif job.status == "queued":
        idx = waiting.index(job.id) if job.id in waiting else 0
        running_left = 0.0
        if current and current in jobs:
            running_left = job_progress(jobs[current])[1]
        ahead = (1 if current else 0) + idx
        pct = 0
        eta = running_left + (idx + 1) * total_estimate()
    else:
        frac, eta = job_progress(job)
        pct = int(frac * 100)
    end = job.finished or now
    return {
        "job_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "stage_label": STAGE_LABEL.get(job.stage, ""),
        "progress_percent": pct,
        "eta_seconds": int(eta),
        "ahead": ahead,
        "elapsed_seconds": int(end - job.created),
        "total_seconds": int(end - job.created),
        "prompt": job.prompt,
        "seed": job.seed,
        "error": job.error,
        **queue_info(),
    }


def purge_old() -> None:
    """Caller must hold `cond`."""
    cutoff = time.time() - KEEP_SECONDS
    for jid in [j.id for j in jobs.values() if j.finished and j.finished < cutoff]:
        del jobs[jid]


# ─────────────────────────────────────────────────────────────
#  THE TWO-STAGE PIPELINE (same approach as your script)
# ─────────────────────────────────────────────────────────────


def encode_prompt(prompt: str):
    """Stage 1: load only text_encoder + tokenizer, encode, free VRAM."""
    text_pipe = Flux2Pipeline.from_pretrained(
        LOCAL_BASE_DIR,
        transformer=None,
        vae=None,
        torch_dtype=DTYPE,
        local_files_only=True,
    ).to(DEVICE)
    try:
        with torch.no_grad():
            prompt_embeds, _text_ids = text_pipe.encode_prompt(
                prompt=prompt,
                device=DEVICE,
            )
        prompt_embeds = prompt_embeds.to("cpu")
    finally:
        del text_pipe
        free_vram()
    return prompt_embeds


def render(job: Job, prompt_embeds):
    """Stage 2: load transformer + VAE + Turbo LoRA, generate, free VRAM."""
    pipe = Flux2Pipeline.from_pretrained(
        LOCAL_BASE_DIR,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=DTYPE,
        local_files_only=True,
    ).to(DEVICE)
    try:
        pipe.load_lora_weights(LOCAL_TURBO_DIR, weight_name=TURBO_LORA_FILENAME)
        set_stage(job, "denoise")

        kwargs = dict(
            height=HEIGHT,
            width=WIDTH,
            sigmas=TURBO_SIGMAS,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator(DEVICE).manual_seed(job.seed),
            num_images_per_prompt=1,
            prompt_embeds=prompt_embeds.to(DEVICE),
        )

        # Real step-by-step progress if this diffusers version supports the callback;
        # otherwise the progress bar falls back to time-based estimates.
        if "callback_on_step_end" in inspect.signature(pipe.__call__).parameters:
            def on_step_end(_pipe, step, _timestep, callback_kwargs):
                job.step = step + 1
                return callback_kwargs
            kwargs["callback_on_step_end"] = on_step_end

        image = pipe(**kwargs).images[0]
    finally:
        # No pipe.to("cpu") here: bitsandbytes 4-bit models can't always be moved.
        # Deleting the pipeline is enough to release the VRAM.
        del pipe
        free_vram()
    return image


def run_job(job: Job) -> None:
    try:
        embeds = encode_prompt(job.prompt)
        set_stage(job, "load")
        image = render(job, embeds)
        set_stage(job, "finish")

        buf = io.BytesIO()
        if OUTPUT_FORMAT.upper() == "PNG":
            image.save(buf, format="PNG")
        else:
            image.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)

        with cond:
            job.image = buf.getvalue()
            set_stage(job, "done")
            job.status = "done"
            job.finished = time.time()
            update_estimates(job)
    except Exception as exc:
        traceback.print_exc()
        free_vram()
        with cond:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished = time.time()


def worker() -> None:
    """The only thread that touches the GPU, so jobs run strictly one at a time."""
    global current
    while True:
        with cond:
            while not waiting:
                cond.wait()
            job = jobs[waiting.pop(0)]
            current = job.id
            job.status = "running"
            job.started = time.time()
            set_stage(job, "encode")
        try:
            run_job(job)
        finally:
            with cond:
                current = None


# ─────────────────────────────────────────────────────────────
#  HTTP API
# ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=worker, daemon=True, name="gpu-worker").start()
    yield


app = FastAPI(title="FLUX.2 turbo queue server", lifespan=lifespan)


class JobRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    user: str = Field("anonymous", max_length=100)   # e.g. the Telegram user id
    seed: Optional[int] = None                       # random if omitted


@app.get("/health")
def health():
    with cond:
        return {"status": "ok", "cuda": torch.cuda.is_available(), **queue_info()}


@app.get("/queue")
def get_queue():
    with cond:
        return queue_info()


@app.post("/jobs")
def create_job(req: JobRequest):
    """Always answers 200; check the `accepted` field."""
    with cond:
        purge_old()

        active = [j for j in jobs.values() if j.user == req.user and j.status in ("queued", "running")]
        if active:
            return {"accepted": False, "reason": "user_busy", "job_id": active[0].id, **queue_info()}
        if len(waiting) >= MAX_QUEUE:
            return {"accepted": False, "reason": "queue_full", **queue_info()}

        job = Job(
            id=uuid.uuid4().hex[:12],
            user=req.user,
            prompt=req.prompt,
            seed=req.seed if req.seed is not None else random.randint(0, 2**31 - 1),
        )
        jobs[job.id] = job
        waiting.append(job.id)
        cond.notify()
        return {"accepted": True, **snapshot(job)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with cond:
        job = jobs.get(job_id)
        if job is None:
            return {"status": "not_found", "job_id": job_id}
        return snapshot(job)


@app.get("/jobs/{job_id}/image")
def get_image(job_id: str):
    with cond:
        job = jobs.get(job_id)
        data = job.image if job and job.status == "done" else None
    if data is None:
        raise HTTPException(status_code=404, detail="image not ready or expired")
    png = OUTPUT_FORMAT.upper() == "PNG"
    return Response(
        content=data,
        media_type="image/png" if png else "image/jpeg",
        headers={"Content-Disposition": f'inline; filename="flux_{job_id}.{"png" if png else "jpg"}"'},
    )
