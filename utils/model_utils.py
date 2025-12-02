import logging
import os
import subprocess

from utils.k8s_download_model import get_huggingface_hub

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Downloading models supported only for external team recipes and only from HuggingFace
def download_model(cfg):
    if cfg.get("training_config", None) != None and cfg.cluster_type in ["k8s", "slurm"]:
        model_name_or_path = cfg.get("local_model_name_or_path", None)
        if model_name_or_path == None:
            return
        model_name_or_path_li = model_name_or_path.split("/")
        model_name = model_name_or_path_li[-2] + "/" + model_name_or_path_li[-1]  # Ex: meta-llama/llama3.1

        # Try different possible paths for hf_access_token
        hf_token = cfg.recipes.run.get("hf_access_token", None)
        if hf_token is None:
            logging.error("HF Access token not found in configuration")
            return
        download_model_from_hf(cfg, hf_token, model_name, model_name_or_path)

    else:
        logging.info("Skipping model download as this is hyperpod native recipe")


def download_model_from_hf(cfg, hf_access_token, model_name, save_path):
    if hf_access_token == None:
        logging.error("HF Access token not found. Please set the value in common_config.yaml")

    if cfg.cluster_type == "k8s":
        download_model_on_k8s(cfg.cluster.general_pod, model_name, hf_access_token, save_path)
        return
    elif cfg.cluster_type == "slurm":
        logging.info(f"Initiating model download from HF for {model_name}")
        hf_hub = get_huggingface_hub()
        path = hf_hub.snapshot_download(
            repo_id=model_name,
            repo_type="model",
            revision=None,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            allow_patterns=None,
            ignore_patterns=None,
            token=hf_access_token,
        )
        return
    else:
        logging.error(
            f"Invalid cluster type {cfg.cluster_type}. Model downloads are only supported types are k8s and slurm"
        )
        return


def download_model_on_k8s(
    pod_name, model_name, hf_token, save_path, script_path="/data/hp-recipe-validator/k8s_download_model.py"
):
    """Copy script to pod and execute model download"""
    try:
        # Copy script to pod
        copy_script_to_pod(pod_name, script_path)

        # Execute download script on pod
        logging.info(f"Downloading {model_name} on pod {pod_name}...")
        cmd = ["kubectl", "exec", pod_name, "--", "python3", script_path, model_name, save_path, hf_token]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logging.info(f"Successfully downloaded {model_name}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading model {model_name}: {e}")
        if e.stderr:
            logging.error(f"Error output: {e.stderr}")
        return False


def copy_script_to_pod(pod_name, remote_path):
    """Copy k8s_download_model.py to pod"""
    script_path = os.path.join(os.path.dirname(__file__), "k8s_download_model.py")
    subprocess.run(["kubectl", "cp", script_path, f"{pod_name}:{remote_path}"], check=True)
    print(f"Copied script to {pod_name}:{remote_path}")
