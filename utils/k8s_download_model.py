#!/usr/bin/env python3
"""
Script to run on K8s pod to download HuggingFace models
Usage: python3 k8s_download_model.py <model_name> <save_path> [hf_token]
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_huggingface_hub():
    try:
        import huggingface_hub

        return huggingface_hub
    except ImportError:
        logging.error("huggingface_hub not installed. Please install huggingface_hub")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 k8s_download_model.py <model_name> <save_path> [hf_token]")
        sys.exit(1)

    model_name = sys.argv[1]
    save_path = sys.argv[2]
    hf_token = sys.argv[3]

    # Install huggingface_hub if needed
    hf_hub = get_huggingface_hub()

    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading model: {model_name}")
    print(f"Save path: {save_path}")

    try:
        path = hf_hub.snapshot_download(
            repo_id=model_name,
            repo_type="model",
            revision=None,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print(f"Successfully downloaded model to: {path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
