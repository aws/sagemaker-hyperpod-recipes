"""
Launch.json content validators for different recipe types and platforms.

This module provides comprehensive validation functions for launch.json files
generated from different recipe types (LLMFT, Nova, VERL, Evaluation) on
different platforms (K8s, SM Jobs).
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Supported open source evaluation model names (from evaluation_regional_parameters.json)
SUPPORTED_OPEN_SOURCE_EVAL_RECIPES = {
    "deepseek-llm-r1-distill-llama-70b",
    "deepseek-llm-r1-distill-llama-8b",
    "deepseek-llm-r1-distill-qwen-1-5b",
    "deepseek-llm-r1-distill-qwen-7b",
    "deepseek-llm-r1-distill-qwen-14b",
    "deepseek-llm-r1-distill-qwen-32b",
    "meta-textgeneration-llama-3-1-8b-instruct",
    "meta-textgeneration-llama-3-2-1b-instruct",
    "meta-textgeneration-llama-3-2-3b-instruct",
    "meta-textgeneration-llama-3-3-70b-instruct",
    "meta-vlm-llama-4-scout-17b-16e-instruct",
    "huggingface-llm-qwen2-5-14b-instruct",
    "huggingface-llm-qwen2-5-32b-instruct",
    "huggingface-llm-qwen2-5-72b-instruct",
    "huggingface-llm-qwen2-5-7b-instruct",
    "huggingface-reasoning-qwen3-06b",
    "huggingface-reasoning-qwen3-32b",
    "huggingface-reasoning-qwen3-4b",
    "huggingface-reasoning-qwen3-8b",
    "huggingface-reasoning-qwen3-1-7b",
    "huggingface-reasoning-qwen3-14b",
    "openai-reasoning-gpt-oss-20b",
    "openai-reasoning-gpt-oss-120b",
}


def identify_recipe_type(recipe_name: str) -> str:
    """
    Identify recipe type from recipe name.

    Returns:
        str: One of "llmft", "nova_training", "nova_eval", "verl", "open_source_eval"

    Raises:
        ValueError: If recipe type cannot be identified (unknown recipe)
    """
    recipe_name_lower = recipe_name.lower()

    # Check if recipe name matches supported open source evaluation models
    # These use model names with underscores instead of hyphens
    recipe_name_with_hyphens = recipe_name_lower.replace("_", "-")
    if recipe_name_with_hyphens in SUPPORTED_OPEN_SOURCE_EVAL_RECIPES:
        return "open_source_eval"

    # Check for evaluation recipes first (more specific)
    if "eval" in recipe_name_lower:
        if "nova" in recipe_name_lower:
            return "nova_eval"
        else:
            return "open_source_eval"

    # Check for training recipes
    if "llmft" in recipe_name_lower or "llm_finetuning" in recipe_name_lower:
        return "llmft"
    elif "verl" in recipe_name_lower:
        return "verl"
    elif "nova" in recipe_name_lower:
        return "nova_training"

    # Unknown recipe type - this should fail the test
    raise ValueError(
        f"Unknown recipe type for: {recipe_name}. Recipe does not match any known pattern (llmft, nova, verl, evaluation, or supported open source models)."
    )


def validate_k8s_template_variables(template_content: str, template_name: str, recipe_name: str) -> List[str]:
    """
    Validate that K8s template has proper variable replacements and no hardcoded ECR URIs.

    Uses a dictionary mapping template name patterns to their required variables.

    Args:
        template_content: The YAML template content as string
        template_name: Name of the template file
        recipe_name: Name of the recipe (for PPO/RFT detection)

    Returns:
        List of validation errors
    """
    errors = []
    import re

    # ALWAYS check for hardcoded ECR URIs in ALL templates
    ecr_pattern = r'\d+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[^\s"\'}]+'
    ecr_matches = re.findall(ecr_pattern, template_content)
    if ecr_matches:
        unique_ecrs = list(set(ecr_matches))[:2]
        errors.append(
            f"{template_name}: Found hardcoded ECR URI(s) that should be template variables: {unique_ecrs}..."
        )

    # Check for test_container placeholder (should never appear in templates)
    if "test_container" in template_content:
        errors.append(
            f"{template_name}: Found 'test_container' placeholder that should be replaced with template variable"
        )

    # Check if this is a Nova recipe (init_container_image is Nova-specific)
    is_nova = "nova" in recipe_name.lower()
    is_rft = is_nova and "rft" in recipe_name.lower() and "smtj" not in recipe_name.lower()

    # Dictionary mapping template file patterns to required variables
    template_variable_requirements = {}

    if is_nova:
        # Nova-specific templates with init_container_image
        template_variable_requirements = {
            "training-config.yaml": ["{{name}}"],  # Config files have no containers
            "evaluation.yaml": ["{{name}}", "{{container_image}}"],
            "evaluation-config.yaml": ["{{name}}"],
            # Nova PPO
            "training-ag.yaml": ["{{name}}", "{{actor_generation_container_image}}", "{{init_container_image}}"],
            # Nova RFT templates
            "vllm-generation": ["{{name}}", "{{rft_generation_container_image}}"],
            "hub": ["{{name}}", "{{rft_storm_container_image}}"],
            "prompt-rbs": ["{{name}}", "{{rft_storm_container_image}}", "{{init_container_image}}"],
            "nats-server": ["{{name}}", "{{rft_nats_server_container_image}}", "{{rft_nats_reloader_container_image}}"],
            "redis": ["{{name}}", "{{rft_redis_container_image}}"],
        }

        # training.yaml for Nova
        if is_rft:
            template_variable_requirements["training.yaml"] = ["{{name}}", "{{container_image}}"]  # Nova RFT: no init
        else:
            template_variable_requirements["training.yaml"] = [
                "{{name}}",
                "{{container_image}}",
                "{{init_container_image}}",
            ]  # Nova non-RFT: has init
    else:
        # Non-Nova recipes (LLMFT, VERL, open source eval) - simpler requirements
        template_variable_requirements = {
            "training.yaml": ["{{name}}", "{{container_image}}", "{{namespace}}"],
            "training-config.yaml": ["{{name}}"],
            "evaluation.yaml": ["{{name}}", "{{container_image}}", "{{namespace}}"],
            "evaluation-config.yaml": ["{{name}}"],
        }

    # Find matching requirements for this template
    required_vars = []
    for pattern, variables in template_variable_requirements.items():
        if pattern in template_name:
            required_vars = variables
            break

    # If no specific pattern matched, use default
    if not required_vars:
        required_vars = ["{{name}}", "{{container_image}}"]

    # Check for missing required variables
    for var in required_vars:
        if var not in template_content:
            errors.append(f"{template_name}: Missing required template variable: {var}")

    return errors


def validate_k8s_templates(launch_json: Dict, recipe_type: str) -> List[str]:
    """
    Validate that K8s platform launch.json has required template files and proper templatization.

    Args:
        launch_json: The launch.json content
        recipe_type: Type of recipe (llmft, nova_training, etc.)

    Returns:
        List of validation errors
    """
    errors = []

    # Define valid recipe types
    valid_recipe_types = ["llmft", "nova_training", "nova_eval", "verl", "open_source_eval"]

    # Fail if recipe_type is not in the preset list
    if recipe_type not in valid_recipe_types:
        errors.append(f"Unrecognized recipe type: {recipe_type}. Must be one of: {', '.join(valid_recipe_types)}")
        return errors

    # Get recipe name for PPO/RFT detection
    recipe_name = launch_json.get("metadata", {}).get("Name", "")

    # Common templates for all K8s recipes
    required_templates = []

    if recipe_type in ["llmft", "nova_training", "verl", "nova_eval"]:
        required_templates = ["training.yaml", "training-config.yaml"]
    elif recipe_type == "open_source_eval":
        # Open source evaluation uses evaluation-specific templates
        required_templates = ["evaluation.yaml", "evaluation-config.yaml"]

    # Validate each required template
    for template in required_templates:
        if template not in launch_json:
            errors.append(f"Missing required K8s template: {template}")
        elif not isinstance(launch_json[template], str):
            errors.append(f"K8s template {template} must be string, got {type(launch_json[template]).__name__}")
        elif len(launch_json[template].strip()) == 0:
            errors.append(f"K8s template {template} is empty")
        else:
            # Validate template variables
            errors.extend(validate_k8s_template_variables(launch_json[template], template, recipe_name))

    # Check for additional templates in complex jobs
    if "ppo" in recipe_name.lower():
        if "training-ag.yaml" not in launch_json:
            errors.append("Nova PPO recipe missing training-ag.yaml template")
        elif isinstance(launch_json.get("training-ag.yaml"), str):
            errors.extend(
                validate_k8s_template_variables(launch_json["training-ag.yaml"], "training-ag.yaml", recipe_name)
            )

    if "rft" in recipe_name.lower() and "smtj" not in recipe_name.lower():
        rft_templates = ["vllm-generation.yaml", "hub.yaml", "prompt-rbs.yaml", "nats-server.yaml"]
        for template in rft_templates:
            if template not in launch_json:
                errors.append(f"Nova RFT recipe missing {template} template")
            elif isinstance(launch_json.get(template), str):
                errors.extend(validate_k8s_template_variables(launch_json[template], template, recipe_name))

    return errors


def validate_sm_jobs_content(launch_json: Dict, recipe_type: str) -> List[str]:
    """
    Validate that SM Jobs platform launch.json has required content.

    Args:
        launch_json: The launch.json content
        recipe_type: Type of recipe

    Returns:
        List of validation errors
    """
    errors = []

    # SM Jobs requires training_recipe.yaml
    if "training_recipe.yaml" not in launch_json:
        errors.append("SM Jobs launch.json missing training_recipe.yaml")
    elif not isinstance(launch_json["training_recipe.yaml"], str):
        errors.append(f"training_recipe.yaml must be string, got {type(launch_json['training_recipe.yaml']).__name__}")
    elif len(launch_json["training_recipe.yaml"].strip()) == 0:
        errors.append("training_recipe.yaml is empty")

    # SM Jobs may have tensorboard_config for training recipes
    if recipe_type in ["llmft", "nova_training", "verl"]:
        if "tensorboard_config" in launch_json:
            tb_config = launch_json["tensorboard_config"]
            if not isinstance(tb_config, dict):
                errors.append(f"tensorboard_config must be dict, got {type(tb_config).__name__}")

    # SM Jobs should NOT have K8s-specific templates
    k8s_templates = ["training.yaml", "training-config.yaml", "evaluation.yaml", "evaluation-config.yaml"]
    for template in k8s_templates:
        if template in launch_json:
            errors.append(f"SM Jobs launch.json should not contain K8s template: {template}")

    return errors


def validate_metadata_schema(metadata: Dict) -> List[str]:
    """
    Comprehensive metadata validation against js_schema.json with type and constraint checking.
    Combines schema validation with recipe-type-specific validation.

    Args:
        metadata: The metadata section

    Returns:
        List of validation errors
    """
    errors = []

    # Check required fields
    required_fields = ["Name", "DisplayName", "Type", "Versions"]
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")

    # Validate Name (string, maxLength 255)
    if "Name" in metadata:
        if not isinstance(metadata["Name"], str):
            errors.append(f"Name must be string, got {type(metadata['Name']).__name__}")
        elif len(metadata["Name"]) > 255:
            errors.append(f"Name exceeds maxLength of 255 characters (length: {len(metadata['Name'])})")

    # Validate DisplayName (string, maxLength 255)
    if "DisplayName" in metadata:
        if not isinstance(metadata["DisplayName"], str):
            errors.append(f"DisplayName must be string, got {type(metadata['DisplayName']).__name__}")
        elif len(metadata["DisplayName"]) > 255:
            errors.append(f"DisplayName exceeds maxLength of 255 characters (length: {len(metadata['DisplayName'])})")

    # Validate Type (string, enum)
    if "Type" in metadata:
        if not isinstance(metadata["Type"], str):
            errors.append(f"Type must be string, got {type(metadata['Type']).__name__}")
        elif metadata["Type"] not in ["FineTuning", "Evaluation"]:
            errors.append(f"Invalid Type: {metadata['Type']}. Must be 'FineTuning' or 'Evaluation'")

    # Get recipe type and check if it's open source evaluation
    recipe_type = metadata.get("Type")
    display_name = metadata.get("DisplayName", "")
    is_open_source_eval = "Open Source Evaluation" in display_name

    # Validate CustomizationTechnique (string, enum, case-insensitive)
    if "CustomizationTechnique" in metadata:
        if not isinstance(metadata["CustomizationTechnique"], str):
            errors.append(
                f"CustomizationTechnique must be string, got {type(metadata['CustomizationTechnique']).__name__}"
            )
        else:
            valid_techniques = ["SFT", "DPO", "RLAIF", "RLVR", "PPO", "CPT", "DISTILL"]
            technique_upper = metadata["CustomizationTechnique"].upper()
            if technique_upper not in valid_techniques:
                errors.append(
                    f"Invalid CustomizationTechnique: {metadata['CustomizationTechnique']}. Must be one of: {', '.join(valid_techniques)}"
                )

        # CustomizationTechnique should only be in FineTuning recipes
        if recipe_type == "Evaluation":
            errors.append("Evaluation recipe should not have CustomizationTechnique field")
    elif recipe_type == "FineTuning":
        # FineTuning recipes MUST have CustomizationTechnique
        errors.append("Training recipe missing CustomizationTechnique")

    # Validate EvaluationType (string, enum)
    # Note: EvaluationType is required for Nova evaluation but NOT generated for open source evaluation
    if "EvaluationType" in metadata:
        if not isinstance(metadata["EvaluationType"], str):
            errors.append(f"EvaluationType must be string, got {type(metadata['EvaluationType']).__name__}")
        else:
            valid_eval_types = [
                "DeterministicTextBenchmark",
                "DeterministicMultiModalBenchmark",
                "DeterministicEvaluation",
                "LLMAJEvaluation",
            ]
            if metadata["EvaluationType"] not in valid_eval_types:
                errors.append(
                    f"Invalid EvaluationType: {metadata['EvaluationType']}. Must be one of: {', '.join(valid_eval_types)}"
                )

        # EvaluationType should only be in Evaluation recipes
        if recipe_type == "FineTuning":
            errors.append("Training recipe should not have EvaluationType field")
    elif recipe_type == "Evaluation" and not is_open_source_eval:
        # Non-open-source evaluation recipes (e.g., Nova evaluation) MUST have EvaluationType
        errors.append("Evaluation recipe missing EvaluationType (required for non open-source evaluation)")

    # Validate IsSubscriptionModel (boolean)
    if "IsSubscriptionModel" in metadata:
        if not isinstance(metadata["IsSubscriptionModel"], bool):
            errors.append(f"IsSubscriptionModel must be boolean, got {type(metadata['IsSubscriptionModel']).__name__}")

    # Validate Peft (string, enum)
    if "Peft" in metadata:
        if not isinstance(metadata["Peft"], str):
            errors.append(f"Peft must be string, got {type(metadata['Peft']).__name__}")
        elif metadata["Peft"].upper() not in ["LORA", "QLORA"]:
            errors.append(f"Invalid Peft: {metadata['Peft']}. Must be 'LORA' or 'QLORA'")

        # Peft should only be in FineTuning recipes
        if recipe_type == "Evaluation":
            errors.append("Evaluation recipe should not have Peft field")

    # Validate SequenceLength (string, enum)
    if "SequenceLength" in metadata:
        if not isinstance(metadata["SequenceLength"], str):
            errors.append(f"SequenceLength must be string, got {type(metadata['SequenceLength']).__name__}")
        else:
            valid_seq_lengths = ["1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K"]
            if metadata["SequenceLength"] not in valid_seq_lengths:
                errors.append(
                    f"Invalid SequenceLength: {metadata['SequenceLength']}. Must be one of: {', '.join(valid_seq_lengths)}"
                )

    # Validate Hardware (string, enum)
    if "Hardware" in metadata:
        if not isinstance(metadata["Hardware"], str):
            errors.append(f"Hardware must be string, got {type(metadata['Hardware']).__name__}")
        elif metadata["Hardware"] not in ["GPU", "TRAINIUM", "CPU"]:
            errors.append(f"Invalid Hardware: {metadata['Hardware']}. Must be 'GPU', 'TRAINIUM', or 'CPU'")

    # Validate Versions (array of strings)
    if "Versions" in metadata:
        if not isinstance(metadata["Versions"], list):
            errors.append(f"Versions must be array, got {type(metadata['Versions']).__name__}")
        else:
            for idx, version in enumerate(metadata["Versions"]):
                if not isinstance(version, str):
                    errors.append(f"Versions[{idx}] must be string, got {type(version).__name__}")

    # Validate InstanceCount (integer)
    if "InstanceCount" in metadata:
        if not isinstance(metadata["InstanceCount"], int) or isinstance(metadata["InstanceCount"], bool):
            errors.append(f"InstanceCount must be integer, got {type(metadata['InstanceCount']).__name__}")
        elif metadata["InstanceCount"] < 1:
            errors.append(f"InstanceCount must be >= 1, got {metadata['InstanceCount']}")

        # InstanceCount should only be in FineTuning recipes
        if recipe_type == "Evaluation":
            errors.append("Evaluation recipe should not have InstanceCount field")

    # Validate ServerlessMeteringType (string, enum)
    if "ServerlessMeteringType" in metadata:
        if not isinstance(metadata["ServerlessMeteringType"], str):
            errors.append(
                f"ServerlessMeteringType must be string, got {type(metadata['ServerlessMeteringType']).__name__}"
            )
        elif metadata["ServerlessMeteringType"] not in ["Token-based", "Hourly"]:
            errors.append(
                f"Invalid ServerlessMeteringType: {metadata['ServerlessMeteringType']}. Must be 'Token-based' or 'Hourly'"
            )

    # Validate string fields
    string_fields = [
        "SmtjImageUri",
        "Description",
        "RecipeFilePath",
        "HpEksPayloadTemplateS3Uri",
        "HpEksOverrideParamsS3Uri",
        "SmtjRecipeTemplateS3Uri",
        "SmtjOverrideParamsS3Uri",
    ]
    for field in string_fields:
        if field in metadata and not isinstance(metadata[field], str):
            errors.append(f"{field} must be string, got {type(metadata[field]).__name__}")

    # Validate array fields
    array_fields = ["UseCases", "BestFor", "SupportedInstanceTypes"]
    for field in array_fields:
        if field in metadata:
            if not isinstance(metadata[field], list):
                errors.append(f"{field} must be array, got {type(metadata[field]).__name__}")
            else:
                # Validate array items are strings
                for idx, item in enumerate(metadata[field]):
                    if not isinstance(item, str):
                        errors.append(f"{field}[{idx}] must be string, got {type(item).__name__}")

    # Validate HostingConfigs (array of objects)
    if "HostingConfigs" in metadata:
        if not isinstance(metadata["HostingConfigs"], list):
            errors.append(f"HostingConfigs must be array, got {type(metadata['HostingConfigs']).__name__}")
        else:
            for idx, config in enumerate(metadata["HostingConfigs"]):
                if not isinstance(config, dict):
                    errors.append(f"HostingConfigs[{idx}] must be object, got {type(config).__name__}")
                else:
                    # Validate required InstanceType
                    if "InstanceType" not in config:
                        errors.append(f"HostingConfigs[{idx}] missing required field 'InstanceType'")
                    elif not isinstance(config["InstanceType"], str):
                        errors.append(
                            f"HostingConfigs[{idx}].InstanceType must be string, got {type(config['InstanceType']).__name__}"
                        )

                    # Validate optional fields
                    if "Profile" in config and not isinstance(config["Profile"], str):
                        errors.append(
                            f"HostingConfigs[{idx}].Profile must be string, got {type(config['Profile']).__name__}"
                        )

                    if "EcrAddress" in config and not isinstance(config["EcrAddress"], str):
                        errors.append(
                            f"HostingConfigs[{idx}].EcrAddress must be string, got {type(config['EcrAddress']).__name__}"
                        )

                    if "Environment" in config:
                        if not isinstance(config["Environment"], dict):
                            errors.append(
                                f"HostingConfigs[{idx}].Environment must be object, got {type(config['Environment']).__name__}"
                            )

                    # Validate ComputeResourceRequirements
                    if "ComputeResourceRequirements" in config:
                        crr = config["ComputeResourceRequirements"]
                        if not isinstance(crr, dict):
                            errors.append(
                                f"HostingConfigs[{idx}].ComputeResourceRequirements must be object, got {type(crr).__name__}"
                            )
                        else:
                            # Check required fields
                            required_crr = [
                                "MinMemoryRequiredInMb",
                                "NumberOfCpuCoresRequired",
                                "NumberOfAcceleratorDevicesRequired",
                            ]
                            for req_field in required_crr:
                                if req_field not in crr:
                                    errors.append(
                                        f"HostingConfigs[{idx}].ComputeResourceRequirements missing required field '{req_field}'"
                                    )
                                elif not isinstance(crr[req_field], int) or isinstance(crr[req_field], bool):
                                    errors.append(
                                        f"HostingConfigs[{idx}].ComputeResourceRequirements.{req_field} must be integer, got {type(crr[req_field]).__name__}"
                                    )
                                elif (
                                    req_field in ["MinMemoryRequiredInMb", "NumberOfCpuCoresRequired"]
                                    and crr[req_field] < 1
                                ):
                                    errors.append(
                                        f"HostingConfigs[{idx}].ComputeResourceRequirements.{req_field} must be >= 1, got {crr[req_field]}"
                                    )
                                elif req_field == "NumberOfAcceleratorDevicesRequired" and crr[req_field] < 0:
                                    errors.append(
                                        f"HostingConfigs[{idx}].ComputeResourceRequirements.{req_field} must be >= 0, got {crr[req_field]}"
                                    )

    return errors


def validate_regional_parameters(regional_params: Dict, platform: str, recipe_name: str = "") -> List[str]:
    """
    Validate regional_parameters section.

    Args:
        regional_params: The regional_parameters section
        platform: Platform type (k8s or sm_jobs)
        recipe_name: Recipe name for type-specific validation

    Returns:
        List of validation errors
    """
    errors = []

    if not isinstance(regional_params, dict):
        errors.append(f"regional_parameters must be dict, got {type(regional_params).__name__}")
        return errors

    # Check for expected container image parameters
    expected_params = []
    if platform == "k8s":
        # K8s can have hp_eks_regional_ecr_uri and/or container_image
        expected_params = ["hp_eks_regional_ecr_uri", "container_image"]
    elif platform == "sm_jobs":
        # SM Jobs should have smtj_regional_ecr_uri or container_image
        expected_params = ["smtj_regional_ecr_uri", "container_image"]

    # At least one expected parameter should be present
    has_container_param = any(param in regional_params for param in expected_params)
    if not has_container_param and len(regional_params) > 0:
        errors.append(f"regional_parameters missing expected container image parameter for {platform} platform")

    # Validate structure of each parameter
    for param_name, param_value in regional_params.items():
        if not isinstance(param_value, dict):
            errors.append(f"regional_parameters.{param_name} must be dict, got {type(param_value).__name__}")
            continue

        # Each regional parameter should have stage keys (prod, gamma, etc.)
        valid_stages = {"prod", "gamma", "beta"}
        for stage_name in param_value.keys():
            if stage_name not in valid_stages:
                errors.append(f"Invalid stage in regional_parameters.{param_name}: {stage_name}. Valid: {valid_stages}")

    # Nova-specific container validation for K8s platform
    if platform == "k8s" and recipe_name:
        is_nova = "nova" in recipe_name.lower()

        if is_nova:
            is_rft = "rft" in recipe_name.lower() and "smtj" not in recipe_name.lower()
            is_ppo = "ppo" in recipe_name.lower()

            # Non-RFT Nova recipes should have init_container_image
            if not is_rft and "init_container_image" not in regional_params:
                errors.append(f"Nova recipe missing init_container_image in regional_parameters")

            # PPO recipes should have actor_generation_container_image
            if is_ppo and "actor_generation_container_image" not in regional_params:
                errors.append(f"Nova PPO recipe missing actor_generation_container_image in regional_parameters")

            # RFT recipes should have all RFT container images
            if is_rft:
                required_rft = {
                    "rft_generation_container_image",
                    "rft_storm_container_image",
                    "rft_nats_server_container_image",
                    "rft_nats_reloader_container_image",
                    "rft_redis_container_image",
                }
                available = set(regional_params.keys())
                missing_rft = required_rft - available
                if missing_rft:
                    errors.append(f"Nova RFT recipe missing container images in regional_parameters: {missing_rft}")

    return errors


def validate_recipe_override_parameters(override_params: Dict) -> List[str]:
    """
    Validate recipe_override_parameters section.

    Args:
        override_params: The recipe_override_parameters section

    Returns:
        List of validation errors
    """
    errors = []

    if not isinstance(override_params, dict):
        errors.append(f"recipe_override_parameters must be dict, got {type(override_params).__name__}")
        return errors

    # Each parameter should have specific fields
    required_fields = ["type", "required"]
    optional_fields = ["default", "enum", "description"]

    for param_name, param_config in override_params.items():
        if not isinstance(param_config, dict):
            errors.append(f"recipe_override_parameters.{param_name} must be dict, got {type(param_config).__name__}")
            continue

        # Check required fields
        for field in required_fields:
            if field not in param_config:
                errors.append(f"recipe_override_parameters.{param_name} missing required field: {field}")

        # Validate type field
        if "type" in param_config:
            valid_types = ["string", "integer", "number", "float", "boolean", "array", "object"]
            if param_config["type"] not in valid_types:
                errors.append(f"recipe_override_parameters.{param_name}.type must be one of: {', '.join(valid_types)}")

        # Validate required field
        if "required" in param_config:
            if not isinstance(param_config["required"], bool):
                errors.append(f"recipe_override_parameters.{param_name}.required must be boolean")

    return errors


def validate_training_recipe_json(training_recipe: Dict) -> List[str]:
    """
    Validate training_recipe.json section.

    Args:
        training_recipe: The training_recipe.json section

    Returns:
        List of validation errors
    """
    errors = []

    if not isinstance(training_recipe, dict):
        errors.append(f"training_recipe.json must be dict, got {type(training_recipe).__name__}")
        return errors

    # Should have a "recipes" key containing the actual recipe
    if "recipes" not in training_recipe:
        errors.append("training_recipe.json missing 'recipes' key")
        return errors

    recipes_content = training_recipe["recipes"]
    if not isinstance(recipes_content, dict):
        errors.append(f"training_recipe.json.recipes must be dict, got {type(recipes_content).__name__}")

    return errors


def validate_launch_json_content(launch_json: Dict, platform: str, recipe_name: Optional[str] = None) -> List[str]:
    """
    Main validation function for launch.json content.

    Args:
        launch_json: The complete launch.json content
        platform: Platform type ("k8s" or "sm_jobs")
        recipe_name: Optional recipe name (will use metadata.Name if not provided)

    Returns:
        List of validation errors
    """
    errors = []

    # Get recipe name
    if recipe_name is None:
        recipe_name = launch_json.get("metadata", {}).get("Name", "")

    if not recipe_name:
        errors.append("Cannot validate launch.json: recipe name not found")
        return errors

    # Identify recipe type
    recipe_type = identify_recipe_type(recipe_name)

    # Validate based on platform
    if platform == "k8s":
        errors.extend(validate_k8s_templates(launch_json, recipe_type))
    elif platform == "sm_jobs":
        errors.extend(validate_sm_jobs_content(launch_json, recipe_type))
    else:
        errors.append(f"Unknown platform: {platform}")
        return errors

    # Validate metadata using comprehensive schema validator
    if "metadata" in launch_json:
        errors.extend(validate_metadata_schema(launch_json["metadata"]))
    else:
        errors.append("launch.json missing metadata section")

    # Validate regional_parameters with recipe name for type-specific checks
    if "regional_parameters" in launch_json:
        errors.extend(validate_regional_parameters(launch_json["regional_parameters"], platform, recipe_name))
    else:
        errors.append("launch.json missing regional_parameters section")

    # Validate recipe_override_parameters
    if "recipe_override_parameters" in launch_json:
        errors.extend(validate_recipe_override_parameters(launch_json["recipe_override_parameters"]))
    else:
        errors.append("launch.json missing recipe_override_parameters section")

    # Validate training_recipe.json (if present)
    if "training_recipe.json" in launch_json:
        errors.extend(validate_training_recipe_json(launch_json["training_recipe.json"]))

    return errors
