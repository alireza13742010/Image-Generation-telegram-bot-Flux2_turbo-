"""
Download FLUX.2 [dev] (4-bit) + Turbo LoRA weights — RUN THIS ONCE
======================================================================
Downloads everything needed for generation into local folders, so the
generation script never has to touch the network again (aside from the
remote text-encoder API call at inference time).

IMPORTANT — this intentionally SKIPS the text_encoder subfolder.
diffusers/FLUX.2-dev-bnb-4bit's text encoder is a 4-bit quantized
Mistral-Small-3.2-24B (~12-14GB on its own). Since generate_image.py
uses the remote HF-hosted text encoder instead of a local one, we never
need those files — skipping them saves roughly half the total download.

Prerequisites:
    pip install -U huggingface_hub

    # FLUX.2 is a GATED model — request access first:
    # https://huggingface.co/black-forest-labs/FLUX.2-dev
    huggingface-cli login
"""

import os
from huggingface_hub import snapshot_download


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

BASE_REPO   = "diffusers/FLUX.2-dev-bnb-4bit"
TURBO_REPO  = "fal/FLUX.2-dev-Turbo"

LOCAL_BASE_DIR  = "./models/flux2-dev-bnb-4bit"
LOCAL_TURBO_DIR = "./models/flux2-turbo-lora"

TURBO_LORA_FILENAME = "flux.2-turbo-lora.safetensors"

# Skip the text_encoder subfolder entirely (~12-14GB saved) since
# generate_image.py uses the remote text encoder API instead.
SKIP_LOCAL_TEXT_ENCODER = False


def folder_size_gb(path: str) -> float:
    total_bytes = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 ** 3)


def download_base_model():
    print("=" * 60)
    print(f"  Downloading base model: {BASE_REPO}")
    print("=" * 60)

    ignore_patterns = ["text_encoder/*"] if SKIP_LOCAL_TEXT_ENCODER else None
    if SKIP_LOCAL_TEXT_ENCODER:
        print("  (skipping text_encoder/ subfolder — using remote text encoder instead)")

    snapshot_download(
        repo_id=BASE_REPO,
        local_dir=LOCAL_BASE_DIR,
        ignore_patterns=ignore_patterns,
    )

    size_gb = folder_size_gb(LOCAL_BASE_DIR)
    print(f"\n✅  Base model downloaded to: {LOCAL_BASE_DIR}")
    print(f"    Size on disk: {size_gb:.1f} GB")


def download_turbo_lora():
    print("\n" + "=" * 60)
    print(f"  Downloading Turbo LoRA: {TURBO_REPO}")
    print("=" * 60)

    # allow_patterns restricts to just the main LoRA file, skipping the
    # duplicate ComfyUI-format weights (comfy/*.safetensors) and example
    # images, saving an extra ~2.76GB of redundant download.
    snapshot_download(
        repo_id=TURBO_REPO,
        local_dir=LOCAL_TURBO_DIR,
        allow_patterns=[TURBO_LORA_FILENAME, "README.md"],
    )

    size_gb = folder_size_gb(LOCAL_TURBO_DIR)
    print(f"\n✅  Turbo LoRA downloaded to: {LOCAL_TURBO_DIR}")
    print(f"    Size on disk: {size_gb:.1f} GB")


def main():
    download_base_model()
    download_turbo_lora()

    total_gb = folder_size_gb(LOCAL_BASE_DIR) + folder_size_gb(LOCAL_TURBO_DIR)

    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Total disk usage: {total_gb:.1f} GB")
    print(f"  Base model : {LOCAL_BASE_DIR}")
    print(f"  Turbo LoRA : {LOCAL_TURBO_DIR}/{TURBO_LORA_FILENAME}")
    print("\n  Next step: run generate_image.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
