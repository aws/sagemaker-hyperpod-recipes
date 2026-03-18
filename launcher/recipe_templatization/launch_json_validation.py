from __future__ import annotations

import re as _re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Launch type discriminator ─────────────────────────────────────────────────


class LaunchType(str, Enum):
    K8S = "k8s"
    K8S_NOVA = "k8s_nova"
    SMTJ = "smtj"


# ── Helper: derive LaunchType from processor + platform ───────────────────────
# Import processor classes here to avoid circular imports at the call site.
# If circular imports are a concern, move this function to a separate
# launch_type_utils.py module.


def get_launch_type(recipe_template_processor, platform: str) -> LaunchType:
    """
    Derive the LaunchType from the recipe template processor class and platform.

    Args:
        recipe_template_processor: An instance of a BaseRecipeTemplateProcessor subclass.
        platform: The deployment platform string (e.g. "k8s", "sm_jobs").

    Returns:
        The appropriate LaunchType enum value.
    """
    # Import here to avoid top-level circular dependency
    from launcher.recipe_templatization.nova.nova_recipe_template_processor import (
        NovaRecipeTemplateProcessor,
    )

    if platform == "sm_jobs":
        return LaunchType.SMTJ

    if isinstance(recipe_template_processor, NovaRecipeTemplateProcessor):
        return LaunchType.K8S_NOVA

    return LaunchType.K8S


# ── Sub-models ────────────────────────────────────────────────────────────────


class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    FLOAT = "float"


class RecipeOverrideParameter(BaseModel):
    type: ParameterType
    required: bool
    enum: Optional[List[Union[str, int, float, bool]]] = None
    default: Optional[Union[str, int, float, bool]] = None

    model_config = {"extra": "allow"}

    @field_validator("enum", mode="before")
    @classmethod
    def validate_enum_matches_type(cls, v: Optional[List[Any]], info) -> Optional[List[Any]]:
        if v is None:
            return v
        param_type = info.data.get("type")
        type_checks: Dict[str, Any] = {
            ParameterType.INTEGER: lambda x: isinstance(x, int) and not isinstance(x, bool),
            ParameterType.STRING: lambda x: isinstance(x, str),
            ParameterType.FLOAT: lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            ParameterType.BOOLEAN: lambda x: isinstance(x, bool),
        }
        if param_type and param_type in type_checks:
            check = type_checks[param_type]
            for item in v:
                if not check(item):
                    raise ValueError(f"Enum value '{item}' is not valid for parameter type '{param_type.value}'")
        return v


class LaunchMetadata(BaseModel):
    Name: str = Field(..., min_length=1, description="Recipe name")
    # No min_length on InstanceTypes — custom validator owns the error message
    InstanceTypes: List[str] = Field(..., description="Supported instance types")

    model_config = {"extra": "allow"}

    @field_validator("InstanceTypes")
    @classmethod
    def validate_instance_types(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("InstanceTypes must contain at least one instance type")
        return v


class RegionalParameters(BaseModel):
    model_config = {"extra": "allow"}


# ── Key sets ──────────────────────────────────────────────────────────────────

_COMMON_FIXED_KEYS = {"metadata", "recipe_override_parameters", "regional_parameters", "launch_type"}

# K8s: all non-fixed keys must be Helm-rendered *.yaml files
_K8S_FIXED_KEYS = _COMMON_FIXED_KEYS

# K8s Nova: adds training_recipe.json as a structured field
_K8S_NOVA_FIXED_KEYS = _COMMON_FIXED_KEYS | {"training_recipe.json"}

# SMTJ: known structured keys — no extra keys allowed beyond these
_SMTJ_FIXED_KEYS = _COMMON_FIXED_KEYS | {
    "training_recipe.yaml",
    "output_path",
    "launch_overrides",
    "tensorboard_config",
}

# Required keys per launch type
_REQUIRED_KEYS: Dict[LaunchType, set] = {
    LaunchType.K8S: {"training-config.yaml"},
    LaunchType.K8S_NOVA: {"training-config.yaml"},
    LaunchType.SMTJ: {"training_recipe.yaml", "output_path", "launch_overrides"},
}

# Fixed key sets per launch type (used to identify extra/template keys)
_FIXED_KEYS: Dict[LaunchType, set] = {
    LaunchType.K8S: _K8S_FIXED_KEYS,
    LaunchType.K8S_NOVA: _K8S_NOVA_FIXED_KEYS,
    LaunchType.SMTJ: _SMTJ_FIXED_KEYS,
}

# Pattern for valid Helm-rendered YAML template filenames
_YAML_FILENAME_RE = _re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*\.yaml$")


# ── Schema ────────────────────────────────────────────────────────────────────


class LaunchJsonSchema(BaseModel):
    """
    Unified schema for launch.json files produced by:
      - _create_launch_json       (K8s / Helm-rendered templates)
      - _create_launch_json       (K8s Nova / Helm-rendered templates + training_recipe.json)
      - make_smtj_launch_json     (SMTJ, flat SageMaker estimator args)

    The `launch_type` field is required and must be set by the caller using
    `get_launch_type(recipe_template_processor, platform)` before constructing
    this schema. This makes the validation branch explicit and removes heuristic
    key detection.
    """

    # ── Discriminator field ───────────────────────────────────────────────────
    launch_type: LaunchType = Field(..., description="Explicit launch type — set by the caller", exclude=True)

    # ── Shared structured fields ──────────────────────────────────────────────
    metadata: Optional[LaunchMetadata] = Field(None, description="Recipe metadata")
    recipe_override_parameters: Optional[Dict[str, RecipeOverrideParameter]] = Field(
        None, description="Parameters that can be overridden"
    )
    regional_parameters: Optional[RegionalParameters] = Field(None, description="Region-specific parameters")

    # ── SMTJ only ─────────────────────────────────────────────────────────────
    training_recipe_yaml: Optional[str] = Field(
        None,
        alias="training_recipe.yaml",
        description="Templated recipe YAML content (SMTJ only)",
    )
    output_path: Optional[str] = Field(None, description="S3 output path for the training job (SMTJ only)")
    launch_overrides: Optional[Dict[str, Any]] = Field(
        None, description="Recipe overrides from sm_jobs_config (SMTJ only)"
    )
    tensorboard_config: Optional[Dict[str, Any]] = Field(
        None, description="Optional TensorBoard configuration (SMTJ only)"
    )

    # ── Nova K8s only ─────────────────────────────────────────────────────────
    training_recipe_json: Optional[Dict[str, Any]] = Field(
        None,
        alias="training_recipe.json",
        description="Untemplated recipe as a JSON object (Nova K8s only)",
    )

    model_config = {
        "extra": "allow",  # absorbs Helm *.yaml template files and SMTJ estimator kwargs
        "populate_by_name": True,
    }

    # ── Validators (defined in reverse execution order — last defined runs first) ──
    #
    # Pydantic v2 runs mode="before" model validators in REVERSE definition order.
    # Desired execution order:
    #   1. validate_required_keys          (runs first  — defined last)
    #   2. validate_template_files         (runs second — defined second)
    #   3. validate_smtj_metadata_present  (runs third  — defined first)

    @model_validator(mode="before")
    @classmethod
    def validate_smtj_metadata_present(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs third. For SMTJ, structured fields must be present."""
        if data.get("launch_type") == LaunchType.SMTJ:
            for key in ("metadata", "recipe_override_parameters", "regional_parameters"):
                if key not in data:
                    raise ValueError(f"SMTJ launch.json is missing required structured field: '{key}'")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_template_files(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs second. Validates content of files/fields — only when the key is present.
        Guards with `is not None` so missing keys are left to validate_required_keys.
        """
        launch_type = data.get("launch_type")

        if launch_type == LaunchType.SMTJ:
            # Validate recipe YAML content only if the key is present
            recipe_yaml = data.get("training_recipe.yaml")
            if recipe_yaml is not None and (not isinstance(recipe_yaml, str) or not recipe_yaml.strip()):
                raise ValueError("'training_recipe.yaml' must be a non-empty string")

            # Validate output_path content only if the key is present
            output_path = data.get("output_path")
            if output_path is not None and (not isinstance(output_path, str) or not output_path.strip()):
                raise ValueError("'output_path' must be a non-empty string")

        elif launch_type in (LaunchType.K8S, LaunchType.K8S_NOVA):
            fixed_keys = _FIXED_KEYS.get(launch_type, _K8S_FIXED_KEYS)
            extra = {k: v for k, v in data.items() if k not in fixed_keys}

            if not extra:
                raise ValueError("K8s launch.json must contain at least one Helm-rendered YAML template file")

            for filename, content in extra.items():
                # All extra keys in K8s must be valid *.yaml filenames
                if not _YAML_FILENAME_RE.match(filename):
                    raise ValueError(
                        f"Unexpected key '{filename}' in {launch_type.value} launch.json. "
                        f"All non-structured keys must be Helm-rendered *.yaml template files."
                    )
                if not isinstance(content, str):
                    raise ValueError(f"Template file '{filename}' must be a string, " f"got {type(content).__name__}")
                if not content.strip():
                    raise ValueError(f"Template file '{filename}' must not be empty")

            # For Nova K8s, validate training_recipe.json if present
            if launch_type == LaunchType.K8S_NOVA:
                recipe_json = data.get("training_recipe.json")
                if recipe_json is not None and not recipe_json:
                    raise ValueError("'training_recipe.json' must be a non-empty dict for Nova recipes")

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_required_keys(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs first. Enforces required keys for the given launch type."""
        launch_type = data.get("launch_type")

        if launch_type is None:
            raise ValueError(
                "'launch_type' is required. Use get_launch_type(recipe_template_processor, platform) "
                "to derive the correct value before constructing LaunchJsonSchema."
            )

        # Coerce string to enum if needed (e.g. when deserializing from JSON)
        if isinstance(launch_type, str):
            try:
                launch_type = LaunchType(launch_type)
                data["launch_type"] = launch_type
            except ValueError:
                valid = [e.value for e in LaunchType]
                raise ValueError(f"Invalid launch_type '{launch_type}'. Must be one of: {valid}")

        required = _REQUIRED_KEYS.get(launch_type, set())
        missing = required - data.keys()
        if missing:
            raise ValueError(f"{launch_type.value.upper()} launch.json is missing required key(s): {sorted(missing)}")
        return data

    @model_validator(mode="after")
    def validate_extra_fields_by_launch_type(self) -> "LaunchJsonSchema":
        """
        Runs after all declared fields are validated.
        Validates the content of extra fields (model_extra) based on launch type.
        """
        launch_type = self.launch_type

        if launch_type in (LaunchType.K8S, LaunchType.K8S_NOVA):
            # All extra fields must be non-empty strings (Helm-rendered YAML content)
            # The key format is already validated in validate_template_files (mode="before")
            for filename, content in (self.model_extra or {}).items():
                if not isinstance(content, str):
                    raise ValueError(f"Template file '{filename}' must be a string, got {type(content).__name__}")
                if not content.strip():
                    raise ValueError(f"Template file '{filename}' must not be empty")

        elif launch_type == LaunchType.SMTJ:
            # SMTJ estimator kwargs: validate known ones if present
            extra = self.model_extra or {}

            instance_count = extra.get("instance_count")
            if instance_count is not None and (not isinstance(instance_count, int) or instance_count < 1):
                raise ValueError(f"'instance_count' must be a positive integer, got: {instance_count!r}")

            volume_size = extra.get("volume_size")
            if volume_size is not None and (not isinstance(volume_size, int) or volume_size < 1):
                raise ValueError(f"'volume_size' must be a positive integer (GB), got: {volume_size!r}")

            max_run = extra.get("max_run")
            if max_run is not None and (not isinstance(max_run, int) or max_run < 1):
                raise ValueError(f"'max_run' must be a positive integer (seconds), got: {max_run!r}")

        return self
