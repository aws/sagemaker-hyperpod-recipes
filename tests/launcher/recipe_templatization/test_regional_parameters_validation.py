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

"""
Unit tests for validating regional parameters configuration.

Tests AWS account ID consistency across prod and gamma environments.
Handles various URI formats including @sha extensions.
"""

import json
import re

import pytest

RECIPE_TYPES = ["nova", "llmft", "verl", "checkpointless"]

# Expected account-to-region mapping (each region should consistently use same account per environment)
REGION_ACCOUNT_MAPPING = {
    "nova": {
        "prod": {
            "us-east-1": "708977205387",
            "eu-west-2": "470633809225",
            "us-west-2": "176779409107",
        },
        "gamma": {
            "us-east-1": "900867814919",
        },
    },
    "llmft": {
        "prod": {
            "ap-south-1": "423350936952",
            "us-east-2": "556809692997",
            "eu-central-1": "391061375763",
            "eu-south-2": "330290781619",
            "eu-north-1": "963403601044",
            "eu-west-1": "942446708630",
            "eu-west-2": "016839105697",
            "us-west-1": "827510180725",
            "sa-east-1": "311136344257",
            "us-east-1": "327873000638",
            "ap-northeast-1": "356859066553",
            "ap-southeast-1": "885852567298",
            "ap-southeast-2": "304708117039",
            "us-west-2": "920498770698",
        },
        "gamma": {
            "us-west-2": "839249767557",
            "us-east-1": "190594010507",
        },
        "beta": {
            "us-west-2": "300869608763",
        },
    },
    "verl": {
        "prod": {
            "ap-south-1": "423350936952",
            "us-east-2": "556809692997",
            "eu-central-1": "391061375763",
            "eu-south-2": "330290781619",
            "eu-north-1": "963403601044",
            "eu-west-1": "942446708630",
            "eu-west-2": "016839105697",
            "us-west-1": "827510180725",
            "sa-east-1": "311136344257",
            "us-east-1": "327873000638",
            "ap-northeast-1": "356859066553",
            "ap-southeast-1": "885852567298",
            "ap-southeast-2": "304708117039",
            "us-west-2": "920498770698",
        },
        "gamma": {
            "us-west-2": "839249767557",
            "us-east-1": "190594010507",
        },
        "beta": {
            "us-west-2": "300869608763",
        },
    },
}

# Expected account IDs per processor
ACCOUNT_IDS = {
    "nova": {
        "gamma": {"900867814919"},
        "prod": {"708977205387", "470633809225", "176779409107"},
    },
    "llmft": {
        "prod": {
            "963403601044",
            "920498770698",
            "016839105697",
            "330290781619",
            "556809692997",
            "423350936952",
            "311136344257",
            "356859066553",
            "827510180725",
            "304708117039",
            "942446708630",
            "327873000638",
            "391061375763",
            "885852567298",
        },
        "gamma": {"839249767557", "190594010507"},
        "beta": {"300869608763"},
    },
    "verl": {
        "prod": {
            "963403601044",
            "920498770698",
            "016839105697",
            "330290781619",
            "556809692997",
            "423350936952",
            "311136344257",
            "356859066553",
            "827510180725",
            "304708117039",
            "942446708630",
            "327873000638",
            "391061375763",
            "885852567298",
        },
        "gamma": {"839249767557", "190594010507"},
        "beta": {"300869608763"},
    },
    "checkpointless": {
        "prod": "839249767557",
        "gamma": "839249767557",
    },  # Checkpointless uses same account for both
}

# Regional parameters file paths
REGIONAL_PARAMS_FILES = {
    "nova": "launcher/recipe_templatization/nova/nova_regional_parameters.json",
    "llmft": "launcher/recipe_templatization/llmft/llmft_regional_parameters.json",
    "verl": "launcher/recipe_templatization/verl/verl_regional_parameters.json",
    "checkpointless": "launcher/recipe_templatization/checkpointless/checkpointless_regional_parameters.json",
}

# Valid platforms - only these are allowed
VALID_PLATFORMS = {"k8s", "sm_jobs"}

# Valid regional parameter keys - only these are allowed in platform configs
VALID_REGIONAL_PARAM_KEYS = {
    "container_image",
    "init_container_image",
    "actor_generation_container_image",
    "rft_generation_container_image",
    "rft_nats_server_container_image",
    "rft_nats_reloader_container_image",
    "rft_storm_container_image",
    "rft_redis_container_image",
}

# Valid stages/environments - only these are allowed
VALID_STAGES = {"prod", "gamma", "beta"}


@pytest.fixture(scope="module")
def regional_params_nova():
    """Load Nova regional parameters once for all tests."""
    with open(REGIONAL_PARAMS_FILES["nova"]) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def regional_params_llmft():
    """Load LLMFT regional parameters once for all tests."""
    with open(REGIONAL_PARAMS_FILES["llmft"]) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def regional_params_verl():
    """Load VERL regional parameters once for all tests."""
    with open(REGIONAL_PARAMS_FILES["verl"]) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def regional_params_checkpointless():
    """Load Checkpointless regional parameters once for all tests."""
    with open(REGIONAL_PARAMS_FILES["checkpointless"]) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_regional_params(
    regional_params_nova,
    regional_params_llmft,
    regional_params_verl,
    regional_params_checkpointless,
):
    """Combined fixture providing all regional parameters."""
    return {
        "nova": regional_params_nova,
        "llmft": regional_params_llmft,
        "verl": regional_params_verl,
        "checkpointless": regional_params_checkpointless,
    }


def extract_account_ids_from_uris(data, environment):
    """
    Extract all AWS account IDs from ECR URIs in a specific environment.

    Handles various URI formats:
    - Standard: 123456789012.dkr.ecr.us-east-1.amazonaws.com/repo:tag
    - With SHA: 123456789012.dkr.ecr.us-east-1.amazonaws.com/repo:tag@sha256:hash

    Also handles "all" section which applies to all recipes.

    Args:
        data: The regional parameters data
        environment: Either "prod", "gamma", or "beta"

    Returns:
        set: Set of account IDs found
    """
    account_ids = set()
    # Regex matches account ID before .dkr.ecr (handles all URI formats)
    account_pattern = re.compile(r"(\d{12})\.dkr\.ecr")

    for recipe_name, recipe_config in data.items():
        if recipe_name.startswith("_"):  # Skip comment fields
            continue

        for platform in VALID_PLATFORMS:
            if platform not in recipe_config:
                continue

            platform_config = recipe_config[platform]

            # Check ALL regional parameter keys, not just container_image
            for param_key in VALID_REGIONAL_PARAM_KEYS:
                if param_key not in platform_config:
                    continue

                param_config = platform_config[param_key]

                if not isinstance(param_config, dict):
                    continue

                # Handle both nested (prod/gamma/beta) and direct structure
                if environment in param_config:
                    env_config = param_config[environment]

                    if not isinstance(env_config, dict):
                        continue

                    # Extract from all regions in this environment
                    for region, uri in env_config.items():
                        if isinstance(uri, str) and uri:  # Non-empty string
                            match = account_pattern.search(uri)
                            if match:
                                account_ids.add(match.group(1))

    return account_ids


class TestRegionalParametersAccountIDs:
    """Test AWS account ID validation across all regional parameters files."""

    @pytest.mark.parametrize("processor", RECIPE_TYPES)
    def test_prod_account_id_consistency(self, processor, all_regional_params):
        """Test that only the expected prod account ID is used in prod sections."""
        expected_prod_accounts = ACCOUNT_IDS[processor]["prod"]
        regional_params = all_regional_params[processor]

        prod_account_ids = extract_account_ids_from_uris(regional_params, "prod")

        # Handle both single account (string) and multiple accounts (set)
        if isinstance(expected_prod_accounts, str):
            expected_prod_accounts = {expected_prod_accounts}

        # Assert only expected prod account IDs are present
        assert prod_account_ids == expected_prod_accounts, (
            f"[{processor}] Expected only prod accounts {expected_prod_accounts} in prod sections, "
            f"but found: {prod_account_ids}"
        )

    @pytest.mark.parametrize("processor", RECIPE_TYPES)
    def test_gamma_account_id_consistency(self, processor, all_regional_params):
        """Test that only the expected gamma account ID is used in gamma sections."""
        expected_gamma_accounts = ACCOUNT_IDS[processor]["gamma"]
        regional_params = all_regional_params[processor]

        gamma_account_ids = extract_account_ids_from_uris(regional_params, "gamma")

        # Handle both single account (string) and multiple accounts (set)
        if isinstance(expected_gamma_accounts, str):
            expected_gamma_accounts = {expected_gamma_accounts}

        # Assert only expected gamma account IDs are present (or empty if no gamma URIs)
        if gamma_account_ids:
            assert gamma_account_ids == expected_gamma_accounts, (
                f"[{processor}] Expected only gamma accounts {expected_gamma_accounts} in gamma sections, "
                f"but found: {gamma_account_ids}"
            )

    @pytest.mark.parametrize("processor", RECIPE_TYPES)
    def test_beta_account_id_consistency(self, processor, all_regional_params):
        """Test that only the expected beta account ID is used in beta sections."""
        if "beta" not in ACCOUNT_IDS[processor]:
            return  # Skip if processor doesn't have beta accounts

        expected_beta_accounts = ACCOUNT_IDS[processor]["beta"]
        regional_params = all_regional_params[processor]

        beta_account_ids = extract_account_ids_from_uris(regional_params, "beta")

        # Handle both single account (string) and multiple accounts (set)
        if isinstance(expected_beta_accounts, str):
            expected_beta_accounts = {expected_beta_accounts}

        # Assert only expected beta account IDs are present (or empty if no beta URIs)
        if beta_account_ids:
            assert beta_account_ids == expected_beta_accounts, (
                f"[{processor}] Expected only beta accounts {expected_beta_accounts} in beta sections, "
                f"but found: {beta_account_ids}"
            )

    @pytest.mark.parametrize("processor", ["nova", "llmft", "verl"])
    def test_region_account_mapping_consistency(self, processor, all_regional_params):
        """Test that each region consistently uses the correct account ID per environment."""
        if processor not in REGION_ACCOUNT_MAPPING:
            pytest.skip(f"No region mapping defined for {processor}")

        regional_params = all_regional_params[processor]
        violations = []
        account_pattern = re.compile(r"(\d{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws")

        for recipe_name, recipe_config in regional_params.items():
            if recipe_name.startswith("_"):
                continue

            for platform_name, platform_config in recipe_config.items():
                if not isinstance(platform_config, dict):
                    continue

                # Check all regional parameters
                for param_name, param_config in platform_config.items():
                    if param_name not in VALID_REGIONAL_PARAM_KEYS:
                        continue

                    if not isinstance(param_config, dict):
                        continue

                    # Check each environment
                    for env_name, env_config in param_config.items():
                        if env_name not in REGION_ACCOUNT_MAPPING[processor]:
                            continue

                        expected_mapping = REGION_ACCOUNT_MAPPING[processor][env_name]

                        if not isinstance(env_config, dict):
                            continue

                        # Validate each region uses the correct account
                        for region_name, uri in env_config.items():
                            if not isinstance(uri, str) or not uri:
                                continue

                            match = account_pattern.search(uri)
                            if not match:
                                continue

                            actual_account = match.group(1)
                            actual_region = match.group(2)

                            # Check if region is in expected mapping
                            if region_name in expected_mapping:
                                expected_account = expected_mapping[region_name]

                                # Verify account ID matches
                                if actual_account != expected_account:
                                    violations.append(
                                        {
                                            "recipe": recipe_name,
                                            "platform": platform_name,
                                            "param": param_name,
                                            "environment": env_name,
                                            "region": region_name,
                                            "expected_account": expected_account,
                                            "actual_account": actual_account,
                                            "uri": uri,
                                        }
                                    )

                                # Verify URI region matches config region
                                if actual_region != region_name:
                                    violations.append(
                                        {
                                            "recipe": recipe_name,
                                            "platform": platform_name,
                                            "param": param_name,
                                            "environment": env_name,
                                            "config_region": region_name,
                                            "uri_region": actual_region,
                                            "issue": "Region in URI doesn't match config key",
                                            "uri": uri,
                                        }
                                    )

        assert not violations, (
            f"[{processor}] Found region-to-account mapping violations. "
            f"Each region should consistently use the same account per environment. "
            f"Expected mapping: {json.dumps(REGION_ACCOUNT_MAPPING[processor], indent=2)}\n"
            f"Violations:\n{json.dumps(violations, indent=2)}"
        )

    @pytest.mark.parametrize("processor", RECIPE_TYPES)
    def test_no_unauthorized_account_ids(self, processor, all_regional_params):
        """Test that only authorized account IDs are used (no mixing with other processors' accounts)."""
        expected_prod_accounts = ACCOUNT_IDS[processor]["prod"]
        expected_gamma_accounts = ACCOUNT_IDS[processor]["gamma"]
        regional_params = all_regional_params[processor]

        prod_account_ids = extract_account_ids_from_uris(regional_params, "prod")
        gamma_account_ids = extract_account_ids_from_uris(regional_params, "gamma")
        beta_account_ids = extract_account_ids_from_uris(regional_params, "beta")
        all_found_accounts = prod_account_ids | gamma_account_ids | beta_account_ids

        # Handle both single account (string) and multiple accounts (set)
        if isinstance(expected_prod_accounts, str):
            expected_prod_accounts = {expected_prod_accounts}
        if isinstance(expected_gamma_accounts, str):
            expected_gamma_accounts = {expected_gamma_accounts}

        expected_accounts = expected_prod_accounts | expected_gamma_accounts

        # Add beta accounts if they exist
        if "beta" in ACCOUNT_IDS[processor]:
            expected_beta_accounts = ACCOUNT_IDS[processor]["beta"]
            if isinstance(expected_beta_accounts, str):
                expected_beta_accounts = {expected_beta_accounts}
            expected_accounts |= expected_beta_accounts

        unexpected_accounts = all_found_accounts - expected_accounts

        assert not unexpected_accounts, (
            f"[{processor}] Found unauthorized account IDs: {unexpected_accounts}. "
            f"Expected only: {expected_accounts}"
        )

    @pytest.mark.parametrize("processor", RECIPE_TYPES)
    def test_uri_format_validation(self, processor, all_regional_params):
        """Test that all container URIs follow the expected ECR URI format."""
        regional_params = all_regional_params[processor]

        # ECR URI pattern (handles with/without @sha256)
        # Format: account.dkr.ecr.region.amazonaws.com/repo:tag[@sha256:hash]
        ecr_pattern = re.compile(
            r"^\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-zA-Z0-9/_-]+:[a-zA-Z0-9._-]+(@sha256:[a-f0-9]+)?$"
        )

        invalid_format_uris = []

        for recipe_name, recipe_config in regional_params.items():
            if recipe_name.startswith("_"):
                continue

            for platform in VALID_PLATFORMS:
                if platform not in recipe_config:
                    continue

                platform_config = recipe_config[platform]

                if "container_image" not in platform_config:
                    continue

                container_config = platform_config["container_image"]

                for env in VALID_STAGES:
                    if env not in container_config:
                        continue

                    env_config = container_config[env]

                    for region, uri in env_config.items():
                        if isinstance(uri, str) and uri:  # Non-empty string
                            if not ecr_pattern.match(uri):
                                invalid_format_uris.append(
                                    {
                                        "recipe": recipe_name,
                                        "platform": platform,
                                        "environment": env,
                                        "region": region,
                                        "uri": uri,
                                        "issue": "URI doesn't match expected ECR format",
                                    }
                                )

        assert not invalid_format_uris, (
            f"[{processor}] Found {len(invalid_format_uris)} URIs with invalid format. "
            f"Expected format: account.dkr.ecr.region.amazonaws.com/repo:tag[@sha256:hash]. "
            f"Invalid entries:\n{json.dumps(invalid_format_uris, indent=2)}"
        )

    @pytest.mark.parametrize("processor", RECIPE_TYPES)
    def test_only_valid_platforms(self, processor, all_regional_params):
        """Test that only valid platforms are used (k8s, sm_jobs)."""
        regional_params = all_regional_params[processor]

        invalid_platforms = []

        for recipe_name, recipe_config in regional_params.items():
            if recipe_name.startswith("_"):
                continue

            # Get all platform keys
            recipe_platforms = set(recipe_config.keys())

            # Find any invalid platforms
            unexpected_platforms = recipe_platforms - VALID_PLATFORMS

            if unexpected_platforms:
                invalid_platforms.append(
                    {
                        "recipe": recipe_name,
                        "invalid_platforms": list(unexpected_platforms),
                        "expected_platforms": list(VALID_PLATFORMS),
                    }
                )

        assert not invalid_platforms, (
            f"[{processor}] Found recipes with invalid platform keys. "
            f"Only {VALID_PLATFORMS} are allowed. "
            f"Invalid entries:\n{json.dumps(invalid_platforms, indent=2)}"
        )

    @pytest.mark.parametrize("processor", RECIPE_TYPES)
    def test_only_valid_regional_param_keys(self, processor, all_regional_params):
        """Test that only valid regional parameter keys are used (currently only 'container_image')."""
        regional_params = all_regional_params[processor]

        invalid_param_keys = []

        for recipe_name, recipe_config in regional_params.items():
            if recipe_name.startswith("_"):
                continue

            for platform_name, platform_config in recipe_config.items():
                if not isinstance(platform_config, dict):
                    continue

                # Validate platform first
                if platform_name not in VALID_PLATFORMS:
                    invalid_param_keys.append(
                        {
                            "recipe": recipe_name,
                            "issue": f"Invalid platform '{platform_name}'",
                            "valid_platforms": list(VALID_PLATFORMS),
                        }
                    )
                    continue

                # Get all keys in platform config
                platform_param_keys = set(platform_config.keys())

                # Find any invalid param keys
                unexpected_keys = platform_param_keys - VALID_REGIONAL_PARAM_KEYS

                if unexpected_keys:
                    invalid_param_keys.append(
                        {
                            "recipe": recipe_name,
                            "platform": platform_name,
                            "invalid_keys": list(unexpected_keys),
                            "expected_keys": list(VALID_REGIONAL_PARAM_KEYS),
                        }
                    )

        assert not invalid_param_keys, (
            f"[{processor}] Found platform configs with invalid parameter keys. "
            f"Only {VALID_REGIONAL_PARAM_KEYS} are allowed. "
            f"If adding new parameter types, update VALID_REGIONAL_PARAM_KEYS. "
            f"Invalid entries:\n{json.dumps(invalid_param_keys, indent=2)}"
        )

    @pytest.mark.parametrize("processor", RECIPE_TYPES)
    def test_structure_pattern(self, processor, all_regional_params):
        """Test structure follows: recipe -> platform -> regional_param -> env (prod/gamma) -> region -> value."""
        regional_params = all_regional_params[processor]

        violations = []

        for recipe_name, recipe_config in regional_params.items():
            if recipe_name.startswith("_"):
                continue

            # Level 1: Check recipe_config is dict
            if not isinstance(recipe_config, dict):
                violations.append({"recipe": recipe_name, "issue": "Recipe value must be dict"})
                continue

            for platform_name, platform_config in recipe_config.items():
                # Validate platform
                if platform_name not in VALID_PLATFORMS:
                    violations.append(
                        {
                            "recipe": recipe_name,
                            "issue": f"Invalid platform '{platform_name}', expected {VALID_PLATFORMS}",
                        }
                    )
                    continue

                if not isinstance(platform_config, dict):
                    violations.append(
                        {
                            "recipe": recipe_name,
                            "platform": platform_name,
                            "issue": "Platform value must be dict",
                        }
                    )
                    continue

                for param_name, param_config in platform_config.items():
                    # Validate regional param
                    if param_name not in VALID_REGIONAL_PARAM_KEYS:
                        violations.append(
                            {
                                "recipe": recipe_name,
                                "platform": platform_name,
                                "issue": f"Invalid regional param '{param_name}', expected {VALID_REGIONAL_PARAM_KEYS}",
                            }
                        )
                        continue

                    if not isinstance(param_config, dict):
                        violations.append(
                            {
                                "recipe": recipe_name,
                                "platform": platform_name,
                                "regional_param": param_name,
                                "issue": "Regional param value must be dict",
                            }
                        )
                        continue

                    for env_name, env_config in param_config.items():
                        # Validate environment against VALID_STAGES
                        if env_name not in VALID_STAGES:
                            violations.append(
                                {
                                    "recipe": recipe_name,
                                    "platform": platform_name,
                                    "regional_param": param_name,
                                    "issue": f"Invalid environment '{env_name}', expected {VALID_STAGES}",
                                }
                            )
                            continue

                        if not isinstance(env_config, dict):
                            violations.append(
                                {
                                    "recipe": recipe_name,
                                    "platform": platform_name,
                                    "regional_param": param_name,
                                    "environment": env_name,
                                    "issue": "Environment value must be dict",
                                }
                            )
                            continue

                        # Validate regions have string values
                        for region_name, region_value in env_config.items():
                            if not isinstance(region_value, str):
                                violations.append(
                                    {
                                        "recipe": recipe_name,
                                        "platform": platform_name,
                                        "regional_param": param_name,
                                        "environment": env_name,
                                        "region": region_name,
                                        "issue": f"Region value must be string, got {type(region_value).__name__}",
                                    }
                                )

        assert not violations, (
            f"[{processor}] Structure violations found. "
            f"Expected: recipe -> platform -> regional_param -> env -> region -> value. "
            f"Violations:\n{json.dumps(violations, indent=2)}"
        )
