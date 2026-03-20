from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from launcher.recipe_templatization.launch_json_validation import (
    LaunchJsonSchema,
    LaunchMetadata,
    LaunchType,
    ParameterType,
    RecipeOverrideParameter,
    get_launch_type,
)
from launcher.recipe_templatization.nova.nova_recipe_template_processor import (
    NovaRecipeTemplateProcessor,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def valid_k8s_launch_json():
    """Minimal valid K8s launch.json dict."""
    return {
        "launch_type": LaunchType.K8S,
        "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{name}}
""",
        "metadata": {
            "Name": "llmft_llama3_8b",
            "InstanceTypes": ["ml.p5.48xlarge"],
        },
        "recipe_override_parameters": {
            "global_batch_size": {
                "type": "integer",
                "required": False,
                "enum": [16, 32, 64],
                "default": 16,
            }
        },
        "regional_parameters": {},
    }


@pytest.fixture
def valid_smtj_launch_json():
    """Minimal valid SMTJ launch.json dict."""
    return {
        "launch_type": LaunchType.SMTJ,
        "training_recipe.yaml": """\
recipes:
  run:
    name: {{name}}
""",
        "output_path": "s3://my-bucket/output",
        "launch_overrides": {"learning_rate": 0.001},
        "metadata": {
            "Name": "llmft_llama3_8b",
            "InstanceTypes": ["ml.p5.48xlarge"],
        },
        "recipe_override_parameters": {
            "global_batch_size": {
                "type": "integer",
                "required": False,
                "enum": [16, 32, 64],
                "default": 16,
            }
        },
        "regional_parameters": {},
        "instance_count": 1,
        "volume_size": 100,
        "max_run": 86400,
    }


@pytest.fixture
def valid_nova_ppo_launch_json():
    """Minimal valid Nova PPO K8s launch.json dict."""
    return {
        "launch_type": LaunchType.K8S_NOVA,
        "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config-{{name}}-rm
""",
        "training-ag.yaml": """\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {{name}}-ag
""",
        "training.yaml": """\
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {{name}}-rm
""",
        "training_recipe.json": {
            "recipes": {
                "run": {"name": "my-ppo-run", "model_name_or_path": "nova-pro/prod"},
                "ppo_actor_train": {"max_length": 8192},
            }
        },
        "metadata": {
            "Name": "nova_pro_1_0_p5_gpu_ppo",
            "InstanceTypes": ["ml.p5.48xlarge"],
        },
        "recipe_override_parameters": {
            "global_batch_size": {
                "type": "integer",
                "required": True,
                "enum": [160],
                "default": 160,
            }
        },
        "regional_parameters": {},
    }


@pytest.fixture
def valid_nova_rft_launch_json():
    """Minimal valid Nova RFT K8s launch.json dict (multiple YAML templates + training_recipe.json)."""
    return {
        "launch_type": LaunchType.K8S_NOVA,
        "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config-{{name}}
""",
        "training.yaml": """\
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {{name}}
""",
        "training_recipe.json": {
            "recipes": {
                "run": {
                    "name": "my-rft-run",
                    "model_name_or_path": "nova-lite/prod",
                    "replicas": 4,
                    "generation_replicas": 2,
                    "rollout_worker_replicas": 2,
                }
            }
        },
        "metadata": {
            "Name": "nova_lite_1_0_p5_gpu_rft",
            "InstanceTypes": ["ml.p5.48xlarge"],
        },
        "recipe_override_parameters": {
            "global_batch_size": {
                "type": "integer",
                "required": True,
                "enum": [64],
                "default": 64,
            }
        },
        "regional_parameters": {},
    }


# ── get_launch_type ───────────────────────────────────────────────────────────


class TestGetLaunchType:
    def test_nova_processor_k8s_platform_returns_k8s_nova(self):
        processor = MagicMock(spec=NovaRecipeTemplateProcessor)
        assert get_launch_type(processor, "k8s") == LaunchType.K8S_NOVA

    def test_nova_processor_sm_jobs_platform_returns_smtj(self):
        # sm_jobs always returns SMTJ regardless of processor type
        processor = MagicMock(spec=NovaRecipeTemplateProcessor)
        assert get_launch_type(processor, "sm_jobs") == LaunchType.SMTJ

    def test_non_nova_processor_k8s_platform_returns_k8s(self):
        # Any non-Nova processor on k8s → K8S
        processor = MagicMock()  # generic mock, not a NovaRecipeTemplateProcessor
        assert get_launch_type(processor, "k8s") == LaunchType.K8S

    def test_non_nova_processor_sm_jobs_platform_returns_smtj(self):
        processor = MagicMock()
        assert get_launch_type(processor, "sm_jobs") == LaunchType.SMTJ

    def test_none_processor_k8s_returns_k8s(self):
        # None processor is not a NovaRecipeTemplateProcessor → K8S
        assert get_launch_type(None, "k8s") == LaunchType.K8S

    def test_none_processor_sm_jobs_returns_smtj(self):
        assert get_launch_type(None, "sm_jobs") == LaunchType.SMTJ

    def test_unknown_platform_non_nova_returns_k8s(self):
        # Any platform that isn't "sm_jobs" falls through to K8S for non-Nova
        processor = MagicMock()
        assert get_launch_type(processor, "slurm") == LaunchType.K8S

    def test_unknown_platform_nova_returns_k8s_nova(self):
        # Any platform that isn't "sm_jobs" falls through to K8S_NOVA for Nova
        processor = MagicMock(spec=NovaRecipeTemplateProcessor)
        assert get_launch_type(processor, "slurm") == LaunchType.K8S_NOVA


# ── ParameterType ─────────────────────────────────────────────────────────────


class TestParameterType:
    def test_valid_types(self):
        for t in ("string", "integer", "boolean", "float"):
            assert ParameterType(t).value == t

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            ParameterType("list")


# ── RecipeOverrideParameter ───────────────────────────────────────────────────


class TestRecipeOverrideParameter:
    def test_valid_string_enum(self):
        p = RecipeOverrideParameter(type="string", required=False, enum=["a", "b"], default="a")
        assert p.enum == ["a", "b"]

    def test_valid_integer_enum(self):
        p = RecipeOverrideParameter(type="integer", required=False, enum=[16, 32, 64], default=16)
        assert p.enum == [16, 32, 64]

    def test_valid_float_enum(self):
        p = RecipeOverrideParameter(type="float", required=False, enum=[0.1, 0.01], default=0.1)
        assert p.enum == [0.1, 0.01]

    def test_valid_boolean_enum(self):
        p = RecipeOverrideParameter(type="boolean", required=False, enum=[True, False], default=True)
        assert p.enum == [True, False]

    def test_none_enum_is_valid(self):
        p = RecipeOverrideParameter(type="string", required=True)
        assert p.enum is None

    def test_integer_enum_rejects_strings(self):
        with pytest.raises(ValidationError, match="not valid for parameter type 'integer'"):
            RecipeOverrideParameter(type="integer", required=False, enum=["16", "32"])

    def test_string_enum_rejects_integers(self):
        with pytest.raises(ValidationError, match="not valid for parameter type 'string'"):
            RecipeOverrideParameter(type="string", required=False, enum=[1, 2])

    def test_boolean_rejects_integers(self):
        with pytest.raises(ValidationError, match="not valid for parameter type 'boolean'"):
            RecipeOverrideParameter(type="boolean", required=False, enum=[0, 1])

    def test_extra_fields_allowed(self):
        p = RecipeOverrideParameter(type="string", required=False, description="my param")
        assert p.model_extra["description"] == "my param"


# ── LaunchMetadata ────────────────────────────────────────────────────────────


class TestLaunchMetadata:
    def test_valid_metadata(self):
        m = LaunchMetadata(Name="my_recipe", InstanceTypes=["ml.p5.48xlarge"])
        assert m.Name == "my_recipe"
        assert m.InstanceTypes == ["ml.p5.48xlarge"]

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            LaunchMetadata(Name="", InstanceTypes=["ml.p5.48xlarge"])

    def test_empty_instance_types_raises(self):
        with pytest.raises(ValidationError, match="InstanceTypes must contain at least one"):
            LaunchMetadata(Name="my_recipe", InstanceTypes=[])

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            LaunchMetadata(InstanceTypes=["ml.p5.48xlarge"])

    def test_missing_instance_types_raises(self):
        with pytest.raises(ValidationError):
            LaunchMetadata(Name="my_recipe")

    def test_extra_fields_allowed(self):
        m = LaunchMetadata(
            Name="my_recipe",
            InstanceTypes=["ml.p5.48xlarge"],
            DisplayName="My Recipe",
        )
        assert m.model_extra["DisplayName"] == "My Recipe"


# ── LaunchJsonSchema — K8s ────────────────────────────────────────────────────


class TestLaunchJsonSchemaK8s:
    def test_valid_k8s(self, valid_k8s_launch_json):
        schema = LaunchJsonSchema(**valid_k8s_launch_json)
        assert schema.metadata.Name == "llmft_llama3_8b"

    def test_missing_training_config_raises(self, valid_k8s_launch_json):
        del valid_k8s_launch_json["training-config.yaml"]
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(**valid_k8s_launch_json)

    def test_no_yaml_template_files_raises(self):
        # Only fixed keys, no Helm-rendered YAML files
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(
                launch_type=LaunchType.K8S,
                metadata={"Name": "r", "InstanceTypes": ["ml.p5.48xlarge"]},
                recipe_override_parameters={},
                regional_parameters={},
            )

    def test_non_string_template_file_raises(self, valid_k8s_launch_json):
        valid_k8s_launch_json["extra-template.yaml"] = {"key": "value"}  # dict, not string
        with pytest.raises(ValidationError, match="must be a string"):
            LaunchJsonSchema(**valid_k8s_launch_json)

    def test_empty_template_file_raises(self, valid_k8s_launch_json):
        valid_k8s_launch_json["training-config.yaml"] = "   "
        with pytest.raises(ValidationError, match="must not be empty"):
            LaunchJsonSchema(**valid_k8s_launch_json)

    def test_unexpected_non_yaml_key_raises(self, valid_k8s_launch_json):
        # A key without a .yaml extension is rejected in K8s mode
        valid_k8s_launch_json["unexpected_config"] = "some value"
        with pytest.raises(ValidationError, match="Unexpected key"):
            LaunchJsonSchema(**valid_k8s_launch_json)

    def test_json_extension_key_raises(self, valid_k8s_launch_json):
        # .json keys are not valid Helm template filenames in plain K8s mode
        valid_k8s_launch_json["config.json"] = "{}"
        with pytest.raises(ValidationError, match="Unexpected key"):
            LaunchJsonSchema(**valid_k8s_launch_json)

    def test_multiple_yaml_templates(self, valid_k8s_launch_json):
        valid_k8s_launch_json["worker.yaml"] = "kind: Pod"
        valid_k8s_launch_json["service.yaml"] = "kind: Service"
        schema = LaunchJsonSchema(**valid_k8s_launch_json)
        assert "worker.yaml" in schema.model_extra
        assert "service.yaml" in schema.model_extra

    def test_optional_metadata_for_k8s(self, valid_k8s_launch_json):
        # metadata is optional in the schema (K8s path doesn't enforce it at schema level)
        del valid_k8s_launch_json["metadata"]
        del valid_k8s_launch_json["recipe_override_parameters"]
        del valid_k8s_launch_json["regional_parameters"]
        schema = LaunchJsonSchema(**valid_k8s_launch_json)
        assert schema.metadata is None

    def test_instance_type_override_parameter(self, valid_k8s_launch_json):
        valid_k8s_launch_json["recipe_override_parameters"]["instance_type"] = {
            "type": "string",
            "required": False,
            "enum": ["ml.p5.48xlarge"],
            "default": "ml.p5.48xlarge",
        }
        schema = LaunchJsonSchema(**valid_k8s_launch_json)
        assert "instance_type" in schema.recipe_override_parameters

    def test_model_dump_preserves_yaml_keys(self, valid_k8s_launch_json):
        schema = LaunchJsonSchema(**valid_k8s_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        assert "training-config.yaml" in dumped

    def test_model_dump_excludes_launch_type(self, valid_k8s_launch_json):
        # launch_type is a validation-only field and must not appear in the output artifact
        schema = LaunchJsonSchema(**valid_k8s_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        assert "launch_type" not in dumped

    def test_verl_recipe_no_instance_type_override(self, valid_k8s_launch_json):
        valid_k8s_launch_json["metadata"]["Name"] = "verl-llama3_8b"
        schema = LaunchJsonSchema(**valid_k8s_launch_json)
        assert "instance_type" not in schema.recipe_override_parameters


# ── LaunchJsonSchema — SMTJ ───────────────────────────────────────────────────


class TestLaunchJsonSchemaSmtjV2:
    def test_valid_smtj(self, valid_smtj_launch_json):
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.output_path == "s3://my-bucket/output"
        assert schema.training_recipe_yaml.startswith("recipes:")

    def test_missing_training_recipe_yaml_raises(self, valid_smtj_launch_json):
        del valid_smtj_launch_json["training_recipe.yaml"]
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_missing_output_path_raises(self, valid_smtj_launch_json):
        del valid_smtj_launch_json["output_path"]
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_missing_launch_overrides_raises(self, valid_smtj_launch_json):
        del valid_smtj_launch_json["launch_overrides"]
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_empty_training_recipe_yaml_raises(self, valid_smtj_launch_json):
        valid_smtj_launch_json["training_recipe.yaml"] = "   "
        with pytest.raises(ValidationError, match="non-empty string"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_empty_output_path_raises(self, valid_smtj_launch_json):
        valid_smtj_launch_json["output_path"] = ""
        with pytest.raises(ValidationError, match="non-empty string"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_missing_metadata_raises(self, valid_smtj_launch_json):
        del valid_smtj_launch_json["metadata"]
        with pytest.raises(ValidationError, match="missing required structured field"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_missing_recipe_override_parameters_raises(self, valid_smtj_launch_json):
        del valid_smtj_launch_json["recipe_override_parameters"]
        with pytest.raises(ValidationError, match="missing required structured field"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_missing_regional_parameters_raises(self, valid_smtj_launch_json):
        del valid_smtj_launch_json["regional_parameters"]
        with pytest.raises(ValidationError, match="missing required structured field"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_optional_tensorboard_config(self, valid_smtj_launch_json):
        valid_smtj_launch_json["tensorboard_config"] = {"s3_output_path": "s3://bucket/tb"}
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.tensorboard_config["s3_output_path"] == "s3://bucket/tb"

    def test_null_launch_overrides_is_valid(self, valid_smtj_launch_json):
        valid_smtj_launch_json["launch_overrides"] = None
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.launch_overrides is None

    def test_additional_estimator_kwargs_in_extra(self, valid_smtj_launch_json):
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.model_extra["instance_count"] == 1
        assert schema.model_extra["volume_size"] == 100
        assert schema.model_extra["max_run"] == 86400

    def test_model_dump_preserves_dotted_key(self, valid_smtj_launch_json):
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        assert "training_recipe.yaml" in dumped
        assert "training_recipe_yaml" not in dumped

    def test_model_dump_excludes_launch_type(self, valid_smtj_launch_json):
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        assert "launch_type" not in dumped

    def test_deleted_override_params_not_in_schema(self, valid_smtj_launch_json):
        for key in ["instance_type", "namespace", "instance_count", "replicas"]:
            valid_smtj_launch_json["recipe_override_parameters"].pop(key, None)
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        for key in ["instance_type", "namespace", "replicas"]:
            assert key not in schema.recipe_override_parameters

    def test_integer_enum_in_override_params(self, valid_smtj_launch_json):
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        param = schema.recipe_override_parameters["global_batch_size"]
        assert param.enum == [16, 32, 64]
        assert param.type == ParameterType.INTEGER


# ── LaunchJsonSchema — Nova K8s ───────────────────────────────────────────────


class TestLaunchJsonSchemaNovaK8s:
    def test_valid_nova_ppo(self, valid_nova_ppo_launch_json):
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        assert schema.metadata.Name == "nova_pro_1_0_p5_gpu_ppo"
        assert schema.training_recipe_json["recipes"]["run"]["name"] == "my-ppo-run"
        # Additional Helm YAML templates land in model_extra
        assert "training-ag.yaml" in schema.model_extra
        assert "training.yaml" in schema.model_extra

    def test_valid_nova_rft(self, valid_nova_rft_launch_json):
        schema = LaunchJsonSchema(**valid_nova_rft_launch_json)
        assert schema.metadata.Name == "nova_lite_1_0_p5_gpu_rft"
        assert schema.training_recipe_json["recipes"]["run"]["replicas"] == 4

    def test_missing_training_config_raises(self, valid_nova_ppo_launch_json):
        # training-config.yaml is required for Nova K8s just as for plain K8s
        del valid_nova_ppo_launch_json["training-config.yaml"]
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(**valid_nova_ppo_launch_json)

    def test_missing_training_recipe_json_is_valid(self, valid_nova_ppo_launch_json):
        # training_recipe.json is optional — Nova recipes may omit it
        del valid_nova_ppo_launch_json["training_recipe.json"]
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        assert schema.training_recipe_json is None

    def test_empty_training_recipe_json_raises(self, valid_nova_ppo_launch_json):
        # An explicitly empty dict is rejected
        valid_nova_ppo_launch_json["training_recipe.json"] = {}
        with pytest.raises(ValidationError, match="non-empty dict"):
            LaunchJsonSchema(**valid_nova_ppo_launch_json)

    def test_training_recipe_json_must_be_dict(self, valid_nova_ppo_launch_json):
        # A string value for training_recipe.json is rejected (wrong type)
        valid_nova_ppo_launch_json["training_recipe.json"] = "not a dict"
        with pytest.raises(ValidationError):
            LaunchJsonSchema(**valid_nova_ppo_launch_json)

    def test_non_string_template_file_raises(self, valid_nova_ppo_launch_json):
        valid_nova_ppo_launch_json["extra-template.yaml"] = {"key": "value"}
        with pytest.raises(ValidationError, match="must be a string"):
            LaunchJsonSchema(**valid_nova_ppo_launch_json)

    def test_empty_template_file_raises(self, valid_nova_ppo_launch_json):
        valid_nova_ppo_launch_json["training-config.yaml"] = "   "
        with pytest.raises(ValidationError, match="must not be empty"):
            LaunchJsonSchema(**valid_nova_ppo_launch_json)

    def test_unexpected_non_yaml_key_raises(self, valid_nova_ppo_launch_json):
        # Non-.yaml extra keys are rejected in Nova K8s mode too
        valid_nova_ppo_launch_json["unexpected_config"] = "some value"
        with pytest.raises(ValidationError, match="Unexpected key"):
            LaunchJsonSchema(**valid_nova_ppo_launch_json)

    def test_training_recipe_json_not_treated_as_extra_yaml(self, valid_nova_ppo_launch_json):
        # training_recipe.json is a fixed structured key — it must NOT be validated
        # as a Helm YAML template file (i.e. the .json extension check must not fire)
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        assert schema.training_recipe_json is not None
        assert "training_recipe.json" not in schema.model_extra

    def test_model_dump_preserves_training_recipe_json_alias(self, valid_nova_ppo_launch_json):
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        assert "training_recipe.json" in dumped
        assert "training_recipe_json" not in dumped

    def test_model_dump_excludes_launch_type(self, valid_nova_ppo_launch_json):
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        assert "launch_type" not in dumped

    def test_model_dump_preserves_yaml_template_keys(self, valid_nova_ppo_launch_json):
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        assert "training-config.yaml" in dumped
        assert "training-ag.yaml" in dumped
        assert "training.yaml" in dumped

    def test_nova_metadata_extra_fields_allowed(self, valid_nova_ppo_launch_json):
        # Nova metadata carries extra fields like DisplayName, CustomizationTechnique, etc.
        valid_nova_ppo_launch_json["metadata"]["DisplayName"] = "Nova Pro PPO"
        valid_nova_ppo_launch_json["metadata"]["CustomizationTechnique"] = "ppo"
        valid_nova_ppo_launch_json["metadata"]["InstanceCount"] = 10
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        assert schema.metadata.model_extra["DisplayName"] == "Nova Pro PPO"
        assert schema.metadata.model_extra["CustomizationTechnique"] == "ppo"
        assert schema.metadata.model_extra["InstanceCount"] == 10

    def test_nova_integer_enum_in_override_params(self, valid_nova_ppo_launch_json):
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        param = schema.recipe_override_parameters["global_batch_size"]
        assert param.enum == [160]
        assert param.type == ParameterType.INTEGER
        assert param.required is True

    def test_nova_ppo_plain_k8s_launch_type_rejects_training_recipe_json_as_extra(self):
        """
        If launch_type=K8S (not K8S_NOVA) is passed but training_recipe.json is present,
        the .json key is treated as an unexpected extra key and rejected.
        """
        data = {
            "launch_type": LaunchType.K8S,  # plain K8s, not Nova
            "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
""",
            "training_recipe.json": {"recipes": {"run": {"name": "test"}}},
            "metadata": {"Name": "test", "InstanceTypes": ["ml.p5.48xlarge"]},
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        with pytest.raises(ValidationError, match="Unexpected key"):
            LaunchJsonSchema(**data)


# ── Round-trip serialization ──────────────────────────────────────────────────


class TestRoundTrip:
    def test_k8s_round_trip(self, valid_k8s_launch_json):
        schema = LaunchJsonSchema(**valid_k8s_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        # Re-inject launch_type — excluded from serialization (validation-only field)
        dumped["launch_type"] = LaunchType.K8S
        schema2 = LaunchJsonSchema(**dumped)
        assert schema2.metadata.Name == schema.metadata.Name

    def test_smtj_round_trip(self, valid_smtj_launch_json):
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        # Re-inject launch_type — excluded from serialization (validation-only field)
        dumped["launch_type"] = LaunchType.SMTJ
        schema2 = LaunchJsonSchema(**dumped)
        assert schema2.output_path == schema.output_path
        assert schema2.training_recipe_yaml == schema.training_recipe_yaml

    def test_nova_k8s_round_trip(self, valid_nova_ppo_launch_json):
        schema = LaunchJsonSchema(**valid_nova_ppo_launch_json)
        dumped = schema.model_dump(mode="json", exclude_none=True, by_alias=True)
        dumped["launch_type"] = LaunchType.K8S_NOVA
        schema2 = LaunchJsonSchema(**dumped)
        assert schema2.metadata.Name == schema.metadata.Name
        assert schema2.training_recipe_json == schema.training_recipe_json


# ── Edge cases: conflicting / ambiguous launch types ─────────────────────────


class TestConflictingLaunchTypeKeys:
    def test_smtj_keys_with_k8s_template_file(self):
        """
        SMTJ dict that also contains a K8s-style training-config.yaml.
        training-config.yaml lands in model_extra as an extra field — no error.
        """
        data = {
            "launch_type": LaunchType.SMTJ,
            "training_recipe.yaml": """\
recipes:
  run:
    name: {{name}}
""",
            "output_path": "s3://my-bucket/output",
            "launch_overrides": {"learning_rate": 0.001},
            "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
""",
            "metadata": {
                "Name": "llmft_llama3_8b",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        schema = LaunchJsonSchema(**data)
        assert schema.output_path == "s3://my-bucket/output"
        assert "training-config.yaml" in schema.model_extra

    def test_smtj_missing_required_keys_raises(self):
        """
        Explicit LaunchType.SMTJ but missing training_recipe.yaml and
        launch_overrides — fails the SMTJ required-key check.
        """
        data = {
            "launch_type": LaunchType.SMTJ,
            "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
""",
            "output_path": "s3://my-bucket/output",
            "metadata": {
                "Name": "llmft_llama3_8b",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(**data)

    def test_both_training_recipe_yaml_and_training_config_yaml_with_smtj_keys(self):
        """
        Dict has both training_recipe.yaml (SMTJ) and training-config.yaml (K8s),
        plus SMTJ sentinel keys. With launch_type=SMTJ, training-config.yaml is
        an extra field. Validates successfully.
        """
        data = {
            "launch_type": LaunchType.SMTJ,
            "training_recipe.yaml": """\
recipes:
  run:
    name: {{name}}
""",
            "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
""",
            "output_path": "s3://my-bucket/output",
            "launch_overrides": {},
            "metadata": {
                "Name": "llmft_llama3_8b",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        schema = LaunchJsonSchema(**data)
        assert schema.training_recipe_yaml.startswith("recipes:")
        assert "training-config.yaml" in schema.model_extra

    def test_no_sentinel_keys_and_no_template_files_raises(self):
        """
        K8s dict with no Helm YAML files — fails the K8s required-key check.
        """
        data = {
            "launch_type": LaunchType.K8S,
            "metadata": {
                "Name": "llmft_llama3_8b",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(**data)

    def test_nova_k8s_with_smtj_output_path_key(self):
        """
        Nova K8s dict that accidentally includes output_path (an SMTJ-only key).
        With launch_type=K8S_NOVA, output_path lands in model_extra as an extra
        field — it is not a valid *.yaml filename, so it is rejected.
        """
        data = {
            "launch_type": LaunchType.K8S_NOVA,
            "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config-{{name}}
""",
            "training_recipe.json": {"recipes": {"run": {"name": "my-ppo-run"}}},
            "output_path": "s3://my-bucket/output",  # SMTJ-only key — not a *.yaml filename
            "metadata": {
                "Name": "nova_pro_1_0_p5_gpu_ppo",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        with pytest.raises(ValidationError, match="Unexpected key"):
            LaunchJsonSchema(**data)

    def test_nova_k8s_with_plain_k8s_launch_type_rejects_training_recipe_json(self):
        """
        A Nova-style dict (has training_recipe.json) passed with launch_type=K8S
        (not K8S_NOVA). training_recipe.json is not in the plain K8s fixed-key set,
        so it is treated as an extra key — the .json extension is rejected.
        """
        data = {
            "launch_type": LaunchType.K8S,  # wrong type for a Nova recipe
            "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config-{{name}}
""",
            "training_recipe.json": {"recipes": {"run": {"name": "my-ppo-run"}}},
            "metadata": {
                "Name": "nova_pro_1_0_p5_gpu_ppo",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        with pytest.raises(ValidationError, match="Unexpected key"):
            LaunchJsonSchema(**data)

    def test_nova_k8s_missing_training_config_but_has_other_yaml_templates_raises(self):
        """
        Nova K8s dict that has other YAML templates but is missing the required
        training-config.yaml — fails the K8S_NOVA required-key check.
        """
        data = {
            "launch_type": LaunchType.K8S_NOVA,
            # training-config.yaml intentionally omitted
            "training-ag.yaml": """\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {{name}}-ag
""",
            "training.yaml": """\
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {{name}}
""",
            "training_recipe.json": {"recipes": {"run": {"name": "my-ppo-run"}}},
            "metadata": {
                "Name": "nova_pro_1_0_p5_gpu_ppo",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        with pytest.raises(ValidationError, match="missing required key"):
            LaunchJsonSchema(**data)

    def test_missing_launch_type_raises(self):
        """
        Omitting launch_type entirely raises a clear error pointing to
        get_launch_type() — no silent fallback to a default type.
        """
        data = {
            "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
""",
            "metadata": {
                "Name": "llmft_llama3_8b",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        with pytest.raises(ValidationError, match="'launch_type' is required"):
            LaunchJsonSchema(**data)

    def test_invalid_launch_type_string_raises(self):
        """
        An unrecognized launch_type string raises a clear error listing valid values.
        """
        data = {
            "launch_type": "slurm",  # not a valid LaunchType value
            "training-config.yaml": """\
apiVersion: v1
kind: ConfigMap
""",
            "metadata": {
                "Name": "llmft_llama3_8b",
                "InstanceTypes": ["ml.p5.48xlarge"],
            },
            "recipe_override_parameters": {},
            "regional_parameters": {},
        }
        with pytest.raises(ValidationError, match="Invalid launch_type"):
            LaunchJsonSchema(**data)


# ── Invalid S3 path validation ────────────────────────────────────────────────
#
# The current schema accepts any non-empty string for output_path.
# These tests document the current behavior (no S3 format enforcement)
# and serve as a baseline if stricter validation is added later.
# To enforce S3 URI format, add a field_validator for output_path in LaunchJsonSchema.


class TestInvalidS3Paths:
    def test_valid_s3_path(self, valid_smtj_launch_json):
        """Standard s3:// URI passes validation."""
        valid_smtj_launch_json["output_path"] = "s3://my-bucket/output/path"
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.output_path == "s3://my-bucket/output/path"

    def test_valid_s3_path_no_prefix(self, valid_smtj_launch_json):
        """S3 URI with bucket only (no key prefix) passes validation."""
        valid_smtj_launch_json["output_path"] = "s3://my-bucket"
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.output_path == "s3://my-bucket"

    def test_non_s3_uri_currently_accepted(self, valid_smtj_launch_json):
        """
        Non-S3 URIs (e.g. https://) are currently accepted since output_path
        is validated only as a non-empty string. This test documents that behavior.
        Update match to pytest.raises(ValidationError) if S3 format is enforced.
        """
        valid_smtj_launch_json["output_path"] = "https://not-an-s3-path.com/output"
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.output_path == "https://not-an-s3-path.com/output"

    def test_local_path_currently_accepted(self, valid_smtj_launch_json):
        """
        Local filesystem paths are currently accepted. Documents current behavior.
        """
        valid_smtj_launch_json["output_path"] = "/local/path/to/output"
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.output_path == "/local/path/to/output"

    def test_empty_s3_path_raises(self, valid_smtj_launch_json):
        """Empty string is rejected by the non-empty string validator."""
        valid_smtj_launch_json["output_path"] = ""
        with pytest.raises(ValidationError, match="non-empty string"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_whitespace_only_s3_path_raises(self, valid_smtj_launch_json):
        """Whitespace-only string is rejected by the non-empty string validator."""
        valid_smtj_launch_json["output_path"] = "   "
        with pytest.raises(ValidationError, match="non-empty string"):
            LaunchJsonSchema(**valid_smtj_launch_json)

    def test_s3_path_missing_bucket_currently_accepted(self, valid_smtj_launch_json):
        """
        Malformed S3 URI with no bucket name (s3://) is currently accepted.
        Documents current behavior — add a field_validator to reject this.
        """
        valid_smtj_launch_json["output_path"] = "s3://"
        schema = LaunchJsonSchema(**valid_smtj_launch_json)
        assert schema.output_path == "s3://"
