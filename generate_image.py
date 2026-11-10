"""
Generate an image with FLUX.2 [dev] + Turbo LoRA — loads from LOCAL disk
============================================================================
Run download_flux2_weights.py FIRST. This script only reads from your
local ./models folder — the only network call it makes is the remote
text-encoder API request (prompt text only, not model weights).

Prerequisites:
    pip install -U diffusers transformers accelerate bitsandbytes torch requests
    huggingface-cli login      # needed for the remote text-encoder API auth
"""

import io
import torch
import requests
from huggingface_hub import get_token
from diffusers import Flux2Pipeline


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

LOCAL_BASE_DIR       = "./models/flux2-dev-bnb-4bit"     # from download script
LOCAL_TURBO_DIR      = "./models/flux2-turbo-lora"        # from download script
TURBO_LORA_FILENAME  = "flux.2-turbo-lora.safetensors"

DEVICE = "cuda:0"
DTYPE  = torch.bfloat16          # VAE runs in bf16 even though the rest is 4-bit

# Must match how the weights were downloaded — if you downloaded WITHOUT
# skipping the text_encoder folder, set this to False instead.
USE_REMOTE_TEXT_ENCODER = True

OUTPUT_PATH = "dwarf_lord_by_AI.png"

# Pre-shifted sigma schedule for 8-step turbo inference (required
# alongside the LoRA weights — this is NOT the pipeline's default).
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

HEIGHT = 1024
WIDTH  = 1024
GUIDANCE_SCALE       = 2.5     # Turbo LoRA's recommended value — don't push much higher
NUM_INFERENCE_STEPS  = 8       # Turbo is trained specifically for this step count
SEED = 42


# ─────────────────────────────────────────────────────────────
#  REMOTE TEXT ENCODER
# ─────────────────────────────────────────────────────────────

def remote_text_encoder(prompts):
    """Encodes prompt(s) via HF's hosted FLUX.2 text-encoder endpoint —
    the only network call this script makes besides this."""
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    response.raise_for_status()
    prompt_embeds = torch.load(io.BytesIO(response.content))
    return prompt_embeds.to(DEVICE)


# ─────────────────────────────────────────────────────────────
#  LOAD PIPELINE — from local disk, no download
# ─────────────────────────────────────────────────────────────

print(f"Loading FLUX.2 [dev] from local folder: {LOCAL_BASE_DIR}")

pipe_kwargs = dict(torch_dtype=DTYPE, local_files_only=True)
if USE_REMOTE_TEXT_ENCODER:
    pipe_kwargs["text_encoder"] = None   # not present locally — use remote instead

pipe = Flux2Pipeline.from_pretrained(LOCAL_BASE_DIR, **pipe_kwargs).to(DEVICE)

print(f"Loading Turbo LoRA from: {LOCAL_TURBO_DIR}/{TURBO_LORA_FILENAME}")
pipe.load_lora_weights(LOCAL_TURBO_DIR, weight_name=TURBO_LORA_FILENAME)

print("✅  Pipeline ready.\n")


# ─────────────────────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────────────────────

prompt = (
    "A close-up richly detailed oil-painted fantasy character portrait of a dwarf lord, "
    "in the style of classic epic fantasy book cover illustration, painted with visible "
    "brushwork, deep glazes, and masterful realistic rendering; dark warm background fading "
    "from soft tan light into deep umber shadow; stocky and powerfully built with a broad, "
    "muscular upper body and short bowed legs; mismatched eyes — one deep obsidian black, "
    "the other a vivid piercing green — painted with subtle wet reflective highlights; his "
    "nose is reduced to a rough stub, most of it sliced away in battle, a deep jagged scar "
    "running from cheekbone to jaw, rendered with realistic skin texture and healed tissue "
    "detail; his jaw is uneven and asymmetrical, short bristly mismatched-colored hair falls "
    "unevenly, catching warm rim light against the dark background; he looks directly at the "
    "viewer with a shrewd, knowing half-smile, head tilted slightly, radiating cunning and "
    "quiet authority; he wears fine dark burnished plate armor with intricate gold filigree "
    "engraving across the chestplate and pauldrons, a heavy crimson cape fastened by a gold "
    "lion-head clasp draped over one shoulder, the fabric rendered with deep realistic folds, "
    "soft velvet sheen, and directional highlights; one hand rests near the pommel of a sword "
    "at his hip, fine detail in the rings and knuckles; dramatic Rembrandt-style side lighting "
    "falls across his face and armor from camera-left, carving strong contrast between warm "
    "lit highlights and deep soft shadow; softly blurred moody background with subtle vignette, "
    "in the tradition of classic fantasy portrait painting, realistic skin texture with fine "
    "pores and natural asymmetry, muted rich color palette of crimson, aged gold, and dark "
    "steel, elegant and painterly, gallery-quality digital oil painting, sharp focus on the "
    "face, noble and commanding presence despite his stature"
)

negative_prompt = (
    "flat cel-shaded animation, cartoon, anime style, vector art, chibi proportions, "
    "smooth plastic skin, glossy toon shading, thick black outlines, intact undamaged nose, "
    "symmetrical unscarred face, matching eye colors, blank vacant stare, looking away from "
    "camera, plain or drab clothing, timid expression, low detail, blurry, extra limbs, "
    "deformed hands, malformed face, text, watermark, signature, oversaturated flat colors, "
    "harsh overexposure, lens flare artifacts, unrealistic proportions, exaggerated scars "
    "beyond described, bright flat even lighting, washed-out colors, child-like features, "
    "tall average human proportions, non-dwarf body type, 3d render, video game render, "
    "airbrushed digital art"
)


# ─────────────────────────────────────────────────────────────
#  GENERATE
# ─────────────────────────────────────────────────────────────

generate_kwargs = dict(
    height=HEIGHT,
    width=WIDTH,
    sigmas=TURBO_SIGMAS,               # required — this is what makes 8 steps work well
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=torch.Generator(DEVICE).manual_seed(SEED),
    num_images_per_prompt=1,
)

if USE_REMOTE_TEXT_ENCODER:
    print("Encoding prompt + negative prompt via remote text encoder ...")
    generate_kwargs["prompt_embeds"] = remote_text_encoder([prompt])
    generate_kwargs["negative_prompt_embeds"] = remote_text_encoder([negative_prompt])
else:
    generate_kwargs["prompt"] = prompt
    generate_kwargs["negative_prompt"] = negative_prompt

print("Generating image (8 turbo steps) ...")
image = pipe(**generate_kwargs).images[0]

image.save(OUTPUT_PATH)
print(f"\n✅  Saved: {OUTPUT_PATH}")


# ─────────────────────────────────────────────────────────────
#  CLEANUP
# ─────────────────────────────────────────────────────────────

pipe.to("cpu")
del pipe
torch.cuda.empty_cache()
