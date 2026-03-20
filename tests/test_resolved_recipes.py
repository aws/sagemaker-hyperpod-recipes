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
Tests for resolved recipe generator.

Key test:
- test_resolved_recipes_match_disk - authoritative validation that resolved YAMLs match generator output

Environment variables:
- GOLDEN_TEST_WRITE=true: Regenerate resolved recipes before validation
"""

import os

from scripts.generate_resolved_recipes import (
    ResolvedRecipeError,
    generate_resolved_recipes,
)


class TestResolvedRecipeValidation:
    """
    Authoritative validation that resolved recipe YAMLs match generator output.

    This ensures that the self-contained YAMLs in recipes_collection/recipes/
    exactly match what the Hydra composition generator produces from
    hyperpod_recipes/recipes_src/.
    """

    def test_resolved_recipes_match_disk(self):
        """
        All resolved recipe YAMLs on disk exactly match what the generator produces.

        When GOLDEN_TEST_WRITE=true, regenerates YAMLs first then validates.
        Otherwise, only validates existing YAMLs against generator output.

        Failures indicate:
        1. Recipes manually edited and drifted from generator
        2. Generator updated but recipes not regenerated
        3. New recipes added without running generator
        4. Source recipes in recipes_src/ modified without regenerating

        Fix: python scripts/generate_resolved_recipes.py
        Or:  GOLDEN_TEST_WRITE=true pytest tests/test_resolved_recipes.py
        """

        # Regenerate resolved recipes if GOLDEN_TEST_WRITE is set
        write = os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes")
        try:
            generate_resolved_recipes(write=write)
        except ResolvedRecipeError as e:
            msg = [
                "Mismatch found in resolved recipes",
                "To fix: python scripts/generate_resolved_recipes.py",
                "Or:     GOLDEN_TEST_WRITE=true pytest tests/test_resolved_recipes.py",
            ]
            raise RuntimeError("\n".join(msg)) from e
