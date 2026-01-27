#!/usr/bin/env python3
"""
Script to run on K8s pod to download models from S3 JumpStart cache
Usage: python3 k8s_download_model.py <jumpstart_model_id> <save_path>
"""

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

S3_COMMAND_TEMPLATE = (
    "aws sagemaker describe-hub-content "
    "--hub-name SageMakerPublicHub "
    "--hub-content-name {model_id} "
    "--hub-content-type Model "
    "--query HubContentDocument "
    "--output text | jq -r .{field}"
)


# S3 bucket configuration
def get_s3_bucket(jumpstart_model_id):
    try:
        cmd = S3_COMMAND_TEMPLATE.format(model_id=jumpstart_model_id, field="TrainingArtifactUri")
        result = subprocess.check_output(cmd, shell=True, text=True).strip()
        print("Result of training artifact: " + result)

        # if trainingArtifactUri is null or ends in .tar.gz fall back to hostingArtifactUri
        if "s3" not in result or result.endswith(".tar.gz"):
            cmd = S3_COMMAND_TEMPLATE.format(model_id=jumpstart_model_id, field="HostingArtifactUri")
            result = subprocess.check_output(cmd, shell=True, text=True).strip()
            print("Result of hosting artifact: " + result)

        return result

    except subprocess.CalledProcessError as e:
        print("Model id {jumpstart_model_id} s3 path was unable to be found")
        print("Error: " + e)
        return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 k8s_download_model.py <jumpstart_model_id> <save_path>")
        sys.exit(1)

    jumpstart_model_id = sys.argv[1]
    save_path = sys.argv[2]

    s3_source = get_s3_bucket(jumpstart_model_id)
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading model from S3: {s3_source}")
    print(f"Save path: {save_path}")

    try:
        cmd = ["aws", "s3", "sync", s3_source, save_path, "--no-progress"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully downloaded model to: {save_path}")

    except Exception as e:
        print(f"Error downloading model from S3: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
