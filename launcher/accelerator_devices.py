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

acceleratorDevices = {
    "p4d.24xlarge": 8,
    "p4de.24xlarge": 8,
    "p5.48xlarge": 8,
    "trn1.2xlarge": 1,
    "trn1.32xlarge": 16,
    "trn1n.32xlarge": 16,
    "g5.xlarge": 1,
    "g5.2xlarge": 1,
    "g5.4xlarge": 1,
    "g5.8xlarge": 1,
    "g5.12xlarge": 4,
    "g5.16xlarge": 1,
    "g5.24xlarge": 4,
    "g5.48xlarge": 8,
}

coresPerAcceleratorDevice = {
    "p4d.24xlarge": 1,
    "p4de.24xlarge": 1,
    "p5.48xlarge": 1,
    "trn1.2xlarge": 2,
    "trn1.32xlarge": 2,
    "trn1n.32xlarge": 2,
    "g5.xlarge": 1,
    "g5.2xlarge": 1,
    "g5.4xlarge": 1,
    "g5.8xlarge": 1,
    "g5.12xlarge": 1,
    "g5.16xlarge": 1,
    "g5.24xlarge": 1,
    "g5.48xlarge": 1,
}


def get_num_accelerator_devices(instance_type: str):
    """
    Get the number of accelerator devices on an instance type.
    Accelerator device could be GPU or Trainium chips
    :param instance_type: AWS EC2 instance type
    :return: number of accelerator devices for the instance type or None if instance
    type not in the accelerator devices map
    """
    if instance_type not in acceleratorDevices:
        return None

    return acceleratorDevices[instance_type]


def get_num_cores_per_accelerator(instance_type: str):
    """
    Get the number of cores per accelerator device on an instance type.
    Currently, Trainium has 2 cores per device while Nvida has 1 core per device.
    :param instance_type: AWS EC2 instance type
    :return: number of cores for the accelerator device or None if instance type
    not in the map
    """
    if instance_type not in coresPerAcceleratorDevice:
        return None

    return coresPerAcceleratorDevice[instance_type]
