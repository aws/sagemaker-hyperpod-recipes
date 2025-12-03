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

# Evaluation container constants
EVAL_CONTAINER_IMAGE = "{account_id}.dkr.ecr.{region}.amazonaws.com/smdistributed-modelparallel:2.4.1-gpu-py311-cu121"

# Region to account mapping for evaluation containers
EVAL_REGION_ACCOUNT_MAP = {
    "us-east-1": "658645717510",
    "us-west-2": "658645717510",
    "eu-west-1": "658645717510",
    "ap-southeast-1": "658645717510",
    "ap-northeast-1": "658645717510",
}
