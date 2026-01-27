import json
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_model_id_map():
    """Load the JumpStart model ID mapping from JSON file"""
    with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
        return json.load(f)


def get_jumpstart_model_id(model_name):
    """
    Extract base model name from path and look up JumpStart model ID.

    Args:
        model_name: name in the yaml

    Returns:
        JumpStart model ID or None if not found
    """
    model_id_map = load_model_id_map()

    # Look up in the map
    jumpstart_id = model_id_map.get(model_name)

    if jumpstart_id:
        logging.info(f"Mapped '{model_name}' to JumpStart model ID: '{jumpstart_id}'")
        return jumpstart_id
    else:
        logging.warning(f"Model '{model_name}' not found in JumpStart model ID map")
        return None


# Downloading models from S3 JumpStart cache for k8 and from hf for slurm
# Does not work with Verl for now due to fsx in different location
def download_model(cfg):
    if cfg.get("training_config", None) != None and cfg.cluster_type in ["k8s", "slurm"]:
        model_name_or_path = cfg.get("local_model_name_or_path", None)
        if model_name_or_path == None:
            return

        if cfg.cluster_type == "k8s":
            # Get JumpStart model ID from the map
            jumpstart_model_id = get_jumpstart_model_id(cfg["recipes"]["run"]["name"])
            if jumpstart_model_id is None:
                logging.error(f"Cannot download model: '{model_name_or_path}' not found in JumpStart model ID map")
                return
            download_model_on_k8s(cfg.cluster.general_pod, jumpstart_model_id, model_name_or_path)
        else:
            # Try different possible paths for hf_access_token
            model_name_or_path_li = model_name_or_path.split("/")
            model_name = model_name_or_path_li[-2] + "/" + model_name_or_path_li[-1]  # Ex: meta-llama/llama3.1

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

    if cfg.cluster_type == "slurm":
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
    pod_name, jumpstart_model_id, save_path, script_path="/data/hp-recipe-validator/k8s_download_model.py"
):
    """Copy script to pod and execute model download from S3"""
    try:
        # Copy script to pod
        copy_script_to_pod(pod_name, script_path)

        # Execute download script on pod
        logging.info(f"Downloading {jumpstart_model_id} on pod {pod_name}...")
        cmd = ["kubectl", "exec", pod_name, "--", "python3", script_path, jumpstart_model_id, save_path]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logging.info(f"Successfully downloaded {jumpstart_model_id}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading model {jumpstart_model_id}: {e}")
        if e.stderr:
            logging.error(f"Error output: {e.stderr}")
        return False


def copy_script_to_pod(pod_name, remote_path):
    """Copy k8s_download_model.py to pod"""
    script_path = os.path.join(os.path.dirname(__file__), "k8s_download_model.py")
    subprocess.run(["kubectl", "cp", script_path, f"{pod_name}:{remote_path}"], check=True)
    print(f"Copied script to {pod_name}:{remote_path}")
