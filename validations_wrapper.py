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

from functools import wraps
from typing import Any, Callable, TypeVar, cast

from omegaconf import DictConfig

from launcher.config_validator.type_validator import TypeValidator
from launcher.config_validator.value_validator import ValueValidator

_T = TypeVar("_T", bound=Callable[..., Any])


def validate_config(fn: _T) -> _T:
    @wraps(fn)
    def validations_wrapper(config: DictConfig, *args, **kwargs) -> DictConfig:
        """
        Execute all validations in this function
        """
        type_validator = TypeValidator(config)
        type_validator.validate()
        schema_validator = ValueValidator(config)
        schema_validator.validate()

        return fn(config, *args, **kwargs)

    return cast(_T, validations_wrapper)
