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

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from launcher.config_validator.schema.nova_schema_validation import NovaRecipeValidator
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

        # Add Pydantic schema validation here
        if "recipes" in config:
            try:
                recipes_dict = OmegaConf.to_container(config.recipes, resolve=True)

                # Detect framework type based on model_type
                model_type = recipes_dict.get("run", {}).get("model_type", "")

                # Apply framework-specific validation
                if model_type and model_type.startswith("amazon.nova"):
                    # Check if this is a distillation job
                    training_config = recipes_dict.get("training_config", {})
                    is_distillation = training_config.get("distillation_data") == "true"

                    if not is_distillation:
                        # Validate Nova config only for non-distillation jobs
                        NovaRecipeValidator(**recipes_dict)
                else:
                    # Default to skip validation
                    pass
            except ValidationError as e:
                raise e

        return fn(config, *args, **kwargs)

    return cast(_T, validations_wrapper)
