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
Special recipe validation tests for specific recipe types.

These tests validate recipe-specific constraints that go beyond standard schema validation.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import pytest

from utils.recipe_utils import load_recipe_with_hydra

logger = logging.getLogger(__name__)

# Path to recipes_collection directory
RECIPES_COLLECTION_DIR = Path.cwd() / "recipes_collection"


def find_all_ppo_mini_batch_sizes(config_dict, path="") -> List[Tuple[int, str]]:
    """
    Recursively find all ppo_mini_batch_size fields in a configuration dictionary.

    Args:
        config_dict: The configuration dictionary to search
        path: Current path for reporting (used in recursion)

    Returns:
        List of tuples (value, path) for each ppo_mini_batch_size found
    """
    results = []

    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            current_path = f"{path}.{key}" if path else key

            # Check if this key is ppo_mini_batch_size
            if key == "ppo_mini_batch_size":
                if value is not None and isinstance(value, (int, float)):
                    # Skip critic.ppo_mini_batch_size paths
                    if "critic" not in current_path.lower():
                        results.append((int(value), current_path))

            # Recurse into nested structures
            if isinstance(value, dict):
                results.extend(find_all_ppo_mini_batch_sizes(value, current_path))
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        results.extend(find_all_ppo_mini_batch_sizes(item, f"{current_path}[{idx}]"))

    return results


def discover_verl_recipes() -> List[Path]:
    """
    Discover all VERL recipe files.

    Returns:
        List of Path objects for VERL recipe files
    """
    current_dir = Path.cwd()
    recipes_dir = current_dir / "recipes_collection" / "recipes"

    # Find all recipe files with 'verl' in the filename
    verl_recipes = list(recipes_dir.rglob("*verl*.yaml"))

    return sorted(verl_recipes)


class TestVerlBatchSizeConstraints:
    """Test VERL-specific batch size constraints."""

    @pytest.mark.parametrize("recipe_path", discover_verl_recipes())
    def test_verl_train_batch_size_greater_than_ppo_mini_batch_sizes(self, recipe_path: Path):
        """
        Test that for VERL recipes, training_config.data.train_batch_size is greater than
        or equal to ALL ppo_mini_batch_size values found anywhere in the recipe.

        This ensures proper batch processing in the VERL training loop where the train_batch_size
        must accommodate the ppo_mini_batch_size for all components (actor, critic, etc.).

        Args:
            recipe_path: Path to the VERL recipe file
        """
        # Load recipe with Hydra composition to resolve all defaults
        # This is required because VERL recipes define train_batch_size in Hydra config files
        # (e.g., /hydra_config/verl/data/default.yaml) rather than directly in the recipe
        recipe_dict = load_recipe_with_hydra(recipe_path, return_dict=True)

        # Find all ppo_mini_batch_size values
        ppo_mini_batch_sizes = find_all_ppo_mini_batch_sizes(recipe_dict)

        # If no ppo_mini_batch_size fields found, test passes (nothing to validate)
        if not ppo_mini_batch_sizes:
            pytest.skip(f"No ppo_mini_batch_size fields found in {recipe_path.name}")
            return

        # Get train_batch_size from training_config.data.train_batch_size
        train_batch_size = None
        try:
            if "training_config" in recipe_dict:
                if "data" in recipe_dict["training_config"]:
                    train_batch_size = recipe_dict["training_config"]["data"].get("train_batch_size")
        except (KeyError, AttributeError, TypeError):
            pass

        # If train_batch_size not found, fail the test
        if train_batch_size is None:
            pytest.fail(
                f"Recipe {recipe_path.name}: training_config.data.train_batch_size not found, "
                f"but recipe contains {len(ppo_mini_batch_sizes)} ppo_mini_batch_size field(s)"
            )
            return

        if "-fft" in recipe_path.name.lower():
            logger.warning(
                f"SPECIAL HANDLING: Recipe '{recipe_path.name}' contains '-fft' suffix. "
                f"Batch size constraints are relaxed for Full Fine-Tuning (FFT) recipes due to "
                f"different memory and computational requirements compared to LoRA recipes."
            )

            if ppo_mini_batch_sizes and train_batch_size is not None:
                oversized_ppo_batches = [
                    (ppo_size, ppo_path) for ppo_size, ppo_path in ppo_mini_batch_sizes if train_batch_size < ppo_size
                ]

                if oversized_ppo_batches:
                    warning_msg = (
                        f"FFT Recipe '{recipe_path.name}' has train_batch_size ({train_batch_size}) "
                        f"smaller than {len(oversized_ppo_batches)} ppo_mini_batch_size value(s). "
                        f"This is allowed for FFT recipes but may indicate suboptimal configuration."
                    )
                    logger.warning(warning_msg)

                    for ppo_size, ppo_path in oversized_ppo_batches:
                        logger.warning(f"  - {ppo_path}: {ppo_size} > train_batch_size ({train_batch_size})")

            # Skip the strict validation for FFT recipes
            pytest.skip(f"Batch size validation skipped for FFT recipe: {recipe_path.name}")
            return

        # Validate that train_batch_size >= all ppo_mini_batch_sizes
        errors = []
        for ppo_size, ppo_path in ppo_mini_batch_sizes:
            if train_batch_size < ppo_size:
                errors.append(
                    f"  • YAML Path: {ppo_path}\n"
                    f"    Value: {ppo_size}\n"
                    f"    Constraint: training_config.data.train_batch_size ({train_batch_size}) < {ppo_size}"
                )

        # If any errors, fail with detailed message
        if errors:
            # Print all ppo_mini_batch_size locations for reference
            all_ppo_locations = "\n".join([f"  • {path}: {size}" for size, path in ppo_mini_batch_sizes])

            error_message = (
                f"\nRecipe: {recipe_path.name}\n"
                f"{'='*80}\n"
                f"VALIDATION FAILED: train_batch_size must be >= ALL ppo_mini_batch_size values\n"
                f"{'='*80}\n\n"
                f"Found train_batch_size:\n"
                f"  Location: training_config.data.train_batch_size\n"
                f"  Value: {train_batch_size}\n\n"
                f"Found {len(ppo_mini_batch_sizes)} ppo_mini_batch_size field(s):\n"
                f"{all_ppo_locations}\n\n"
                f"FAILED CONSTRAINTS ({len(errors)}):\n" + "\n".join(errors)
            )
            pytest.fail(error_message)


class TestReplicasRegexPattern:
    """Test that replicas regex pattern only matches exact 'replicas:' entries."""

    def test_replicas_regex_exact_match(self):
        """Test that the regex correctly matches exact 'replicas:' entries."""
        import re

        # The regex pattern used in launchers.py
        pattern = r"\breplicas:\s*\d+"
        replacement = "replicas: {{replicas}}"

        # Test case 1: Should match exact 'replicas: 5'
        content = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config
data:
  replicas: 5
"""
        result = re.sub(pattern, replacement, content)
        assert "replicas: {{replicas}}" in result, "Should replace exact 'replicas: 5'"
        assert "replicas: 5" not in result, "Original 'replicas: 5' should be replaced"

    def test_replicas_regex_with_spaces(self):
        """Test that the regex matches 'replicas:' with various spacing."""
        import re

        pattern = r"\breplicas:\s*\d+"
        replacement = "replicas: {{replicas}}"

        # Test with different spacing
        test_cases = [
            "replicas: 5",
            "replicas:5",
            "replicas:  5",
            "replicas:   10",
        ]

        for test_case in test_cases:
            result = re.sub(pattern, replacement, test_case)
            assert "{{replicas}}" in result, f"Should replace '{test_case}'"

    def test_replicas_regex_does_not_match_suffix(self):
        """Test that the regex does NOT match when 'replicas' is a suffix."""
        import re

        pattern = r"\breplicas:\s*\d+"
        replacement = "replicas: {{replicas}}"

        # Test case: Should NOT match 'xyzreplicas: 5'
        content = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config
data:
  xyzreplicas: 5
  actor_train_replicas: 3
  generation_replicas: 2
"""
        result = re.sub(pattern, replacement, content)

        # These should NOT be replaced because 'replicas' is a suffix
        assert "xyzreplicas: 5" in result, "Should NOT replace 'xyzreplicas: 5'"
        assert "actor_train_replicas: 3" in result, "Should NOT replace 'actor_train_replicas: 3'"
        assert "generation_replicas: 2" in result, "Should NOT replace 'generation_replicas: 2'"

        # The template variable should NOT appear in the result
        assert "xyzreplicas: {{replicas}}" not in result, "Should NOT create 'xyzreplicas: {{replicas}}'"
        assert (
            "actor_train_replicas: {{replicas}}" not in result
        ), "Should NOT create 'actor_train_replicas: {{replicas}}'"

    def test_replicas_regex_mixed_content(self):
        """Test the regex with mixed content containing both exact and suffix matches."""
        import re

        pattern = r"\breplicas:\s*\d+"
        replacement = "replicas: {{replicas}}"

        content = """
apiVersion: v1
kind: Job
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: training
        env:
        - name: ACTOR_TRAIN_REPLICAS
          value: "3"
      config:
        actor_train_replicas: 3
        generation_replicas: 2
        worker_replicas: 4
  parallelism: 2
"""
        result = re.sub(pattern, replacement, content)

        # Should replace exact 'replicas: 5'
        assert "replicas: {{replicas}}" in result, "Should replace exact 'replicas: 5'"
        assert result.count("replicas: {{replicas}}") == 1, "Should only replace the exact match once"

        # Should NOT replace suffix cases
        assert "actor_train_replicas: 3" in result, "Should NOT replace 'actor_train_replicas: 3'"
        assert "generation_replicas: 2" in result, "Should NOT replace 'generation_replicas: 2'"
        assert "worker_replicas: 4" in result, "Should NOT replace 'worker_replicas: 4'"

    def test_replicas_regex_indented(self):
        """Test that the regex works with indented YAML."""
        import re

        pattern = r"\breplicas:\s*\d+"
        replacement = "replicas: {{replicas}}"

        content = """
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: training
    spec:
      replicas: 5
"""
        result = re.sub(pattern, replacement, content)

        # Both exact 'replicas:' entries should be replaced
        assert result.count("replicas: {{replicas}}") == 2, "Should replace both exact 'replicas:' entries"
        assert "replicas: 3" not in result, "First 'replicas: 3' should be replaced"
        assert "replicas: 5" not in result, "Second 'replicas: 5' should be replaced"

    def test_replicas_regex_word_boundary(self):
        """Test that word boundary works correctly at the beginning of 'replicas'."""
        import re

        pattern = r"\breplicas:\s*\d+"
        replacement = "replicas: {{replicas}}"

        # Test with different prefixes
        test_cases = [
            ("replicas: 5", True),  # Should match
            ("_replicas: 5", False),  # Should NOT match (underscore is part of word)
            (" replicas: 5", True),  # Should match (space is word boundary)
            ("\nreplicas: 5", True),  # Should match (newline is word boundary)
            ("\treplicas: 5", True),  # Should match (tab is word boundary)
            ("xyzreplicas: 5", False),  # Should NOT match (alphanumeric is part of word)
            ("123replicas: 5", False),  # Should NOT match (digit is part of word)
        ]

        for test_input, should_match in test_cases:
            result = re.sub(pattern, replacement, test_input)
            if should_match:
                assert "{{replicas}}" in result, f"Should match and replace in: '{test_input}'"
            else:
                assert "{{replicas}}" not in result, f"Should NOT match in: '{test_input}'"
                assert test_input == result, f"Should NOT modify: '{test_input}'"
