# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from pathlib import Path

SM_ADAPTER_REPO = "https://github.com/aws/sagemaker-hyperpod-training-adapter-for-nemo.git"
NEMO_REPO = "https://github.com/NVIDIA/NeMo.git"
NEMO_REPO_TAG = "v2.0.0rc0"  # [TODO] move to v2.0.0 once it is released

SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH = {
    "llama": "examples/llama/llama_pretrain.py",
    "mistral": "examples/mistral/mistral_pretrain.py",
    "mixtral": "examples/mixtral/mixtral_pretrain.py",
}

NEURONX_REPO_URI = "https://github.com/aws-neuron/neuronx-distributed-training.git"
NEURONX_REPO_TAG = "main"
NEURONX_CONF_PATH = "examples/conf"

# utility directory to more easily navigate to other parts of the package
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # package root
