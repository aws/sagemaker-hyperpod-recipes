#!/usr/bin/env python3
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
Launcher Script Generator

Config-driven script generator. All customization is in launcher_scripts_config.yaml.
Template selection is based on the `run.model_type` field in recipes.

Usage:
    # Generate all launcher scripts
    python -m scripts.launcher_scripts_generator.generate_launcher_scripts

    # Check mode (like formatters: exits non-zero if files would change)
    python -m scripts.launcher_scripts_generator.generate_launcher_scripts --check

    # Check with diffs shown
    python -m scripts.launcher_scripts_generator.generate_launcher_scripts --check --diff
"""

import argparse
import difflib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from omegaconf import OmegaConf

from hyperpod_recipes import list_recipes as _list_recipes
from hyperpod_recipes.recipe import Recipe as HpRecipe

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


class LauncherConfig:
    """Loads configuration from YAML file."""

    _instance: Optional["LauncherConfig"] = None
    _data: Dict[str, Any] = {}

    def __new__(cls) -> "LauncherConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            config_path = Path(__file__).parent / "launcher_scripts_config.yaml"
            with open(config_path) as f:
                cls._data = yaml.safe_load(f) or {}
        return cls._instance

    @property
    def settings(self) -> Dict[str, Any]:
        return self._data.get("settings", {})

    @property
    def manually_maintained(self) -> Dict[str, str]:
        return self.settings.get("manually_maintained", {})

    @property
    def preserved_dirs(self) -> List[str]:
        return self.settings.get("preserved_dirs", [])

    @property
    def excluded_recipe_dirs(self) -> List[str]:
        return self.settings.get("excluded_recipe_dirs", [])

    @property
    def excluded_recipe_patterns(self) -> List[str]:
        return self.settings.get("excluded_recipe_patterns", [])

    @property
    def recipes_dir(self) -> str:
        return self.settings.get("recipes_dir", "recipes_collection/recipes")

    @property
    def output_dir(self) -> str:
        return self.settings.get("output_dir", "launcher_scripts")

    @property
    def header(self) -> str:
        return self._data.get("header", "#!/bin/bash\n")

    @property
    def templates(self) -> Dict[str, str]:
        return self._data.get("templates", {})

    @property
    def default_template(self) -> str:
        return self._data.get("default_template", "llm_finetuning_aws")

    @classmethod
    def reset(cls):
        """Reset for testing."""
        cls._instance = None
        cls._data = {}


# =============================================================================
# Script Generator
# =============================================================================


class LauncherScriptGenerator:
    """Generates launcher scripts for recipes."""

    def __init__(self, hp_recipe: HpRecipe):
        self.recipe_name = Path(hp_recipe.path).stem
        # recipe_path is the recipes_collection-relative path (no extension), used in the
        # generated shell script body (e.g. recipes=fine-tuning/llama/recipe_name)
        self.recipe_path = hp_recipe.name
        self.config = LauncherConfig()
        self._recipe_data: Optional[Dict] = None
        self._hp_recipe = hp_recipe

    @property
    def recipe_data(self) -> Dict:
        if self._recipe_data is None:
            try:
                self._recipe_data = (
                    OmegaConf.to_container(self._hp_recipe.config, resolve=False, throw_on_missing=False) or {}
                )
            except Exception:
                self._recipe_data = {}
        return self._recipe_data

    def get_model_type(self) -> str:
        run = self.recipe_data.get("run", {})
        if isinstance(run, dict) and run.get("model_type"):
            return run.get("model_type")
        return ""

    def get_template_key(self) -> str:
        """Select template based on model_type."""
        model_type = self.get_model_type()

        # Exact match
        if model_type in self.config.templates:
            if model_type == "verl" and "sft" in self.recipe_path:
                return "verl-sft"

            return model_type

        # Prefix match for Nova models
        if model_type.startswith("amazon.nova"):
            return "nova"

        return self.config.default_template

    def generate(self, output_dir: Path, script_name: Optional[str] = None, dry_run: bool = False) -> Tuple[Path, str]:
        """Generate launcher script.

        Args:
            output_dir: Directory to write script to
            script_name: Override script filename
            dry_run: If True, return content without writing to disk

        Returns:
            Tuple of (script_path, content)
        """
        script_name = script_name or f"run_{self.recipe_name.replace('-', '_')}.sh"
        script_path = output_dir / script_name
        content = self._render()

        if not dry_run:
            script_path.write_text(content)
            script_path.chmod(0o755)

        return script_path, content

    def _render(self) -> str:
        """Render script from header + template."""
        placeholders = {
            "recipe_path": self.recipe_path,
            "recipe_name": self.recipe_name,
            "run_name": self._get_run_name(),
            "model_save_name": self._get_model_save_name(),
        }
        header = self.config.header.format(**placeholders)
        template = self.config.templates.get(self.get_template_key(), "")
        return header + "\n" + template.format(**placeholders)

    def _get_run_name(self) -> str:
        run = self.recipe_data.get("run", {})
        return (
            run.get("name", self.recipe_name.replace("_", "-"))
            if isinstance(run, dict)
            else self.recipe_name.replace("_", "-")
        )

    def _get_model_save_name(self) -> str:
        model = self.recipe_data.get("training_config", {}).get("model_config", {})
        name = model.get("model_save_name") or model.get("model_name_or_path") or self.recipe_name
        return name.rstrip("/").split("/")[-1].replace("_", "-") if "/" in name else name.replace("_", "-")


# =============================================================================
# Orchestrator
# =============================================================================


class LauncherScriptGenerationOrchestrator:
    """Discovers recipes and generates all launcher scripts."""

    def __init__(self):
        self.config = LauncherConfig()
        self.root = Path(__file__).parent.parent.parent
        self.output_dir = self.root / self.config.output_dir
        self.generated: set = set()

    def discover_recipes(self) -> List[HpRecipe]:
        """Find all recipes via list_recipes(), excluding configured directories."""
        excluded_dirs = self.config.excluded_recipe_dirs
        excluded_patterns = self.config.excluded_recipe_patterns
        recipes = _list_recipes()
        recipes = [r for r in recipes if not any(r.name.startswith(excl) for excl in excluded_dirs)]
        recipes = [r for r in recipes if not any(excl in Path(r.path).stem.lower() for excl in excluded_patterns)]
        return sorted(recipes, key=lambda r: r.name)

    def infer_model_family(self, hp_recipe: HpRecipe) -> str:
        """Extract model family from recipe path (2nd directory level)."""
        parts = hp_recipe.name.split("/")
        return parts[1].lower() if len(parts) >= 2 else "misc"

    def run(self):
        """Generate all launcher scripts."""
        recipes = self.discover_recipes()
        print(f"Found {len(recipes)} recipes\n")

        stats = {"generated": 0, "skipped": 0, "errors": 0}

        for hp_recipe in recipes:
            # Handle manually maintained scripts
            if hp_recipe.name in self.config.manually_maintained:
                script_path = self.output_dir / self.config.manually_maintained[hp_recipe.name]
                self.generated.add(script_path)
                stats["skipped"] += 1
                print(f"⚠ {Path(hp_recipe.path).stem} → {script_path.relative_to(self.root)} (manually maintained)")
                continue

            # Generate script
            try:
                out_dir = self.output_dir / self.infer_model_family(hp_recipe)
                out_dir.mkdir(parents=True, exist_ok=True)

                gen = LauncherScriptGenerator(hp_recipe)
                script_path, _ = gen.generate(out_dir)
                self.generated.add(script_path)
                stats["generated"] += 1
                print(
                    f"✓ {Path(hp_recipe.path).stem} → {script_path.relative_to(self.root)} [model_type={gen.get_model_type() or '(none)'} → {gen.get_template_key()}]"
                )
            except Exception as e:
                stats["errors"] += 1
                print(f"✗ {Path(hp_recipe.path).stem}: {e}")

        self._cleanup()
        self._print_summary(stats)

    def _cleanup(self):
        """Remove scripts no longer backed by recipes."""
        preserved = set(self.config.preserved_dirs)
        maintained = {self.output_dir / p for p in self.config.manually_maintained.values()}

        for subdir in self.output_dir.iterdir():
            if not subdir.is_dir() or subdir.name in preserved:
                continue
            for script in subdir.glob("*.sh"):
                if script not in self.generated and script not in maintained:
                    print(f"🗑️  Removing: {script.relative_to(self.root)}")
                    script.unlink()

    def _print_summary(self, stats: Dict):
        print(f"\n{'='*60}")
        print(f"Generated: {stats['generated']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}")
        print(f"Output: {self.output_dir}")
        print("=" * 60)

    def check(self, show_diff: bool = False) -> List[Tuple[Path, Optional[str], str]]:
        """Check if generated scripts match what's on disk.

        Args:
            show_diff: If True, print unified diffs for mismatches

        Returns:
            List of (path, actual_content_or_None, expected_content) for mismatches.
            Empty list means all scripts match.
        """
        mismatches: List[Tuple[Path, Optional[str], str]] = []

        for hp_recipe in self.discover_recipes():
            # Skip manually maintained scripts
            if hp_recipe.name in self.config.manually_maintained:
                continue

            out_dir = self.output_dir / self.infer_model_family(hp_recipe)
            gen = LauncherScriptGenerator(hp_recipe)
            expected_path, expected_content = gen.generate(out_dir, dry_run=True)

            if not expected_path.exists():
                mismatches.append((expected_path, None, expected_content))
                if show_diff:
                    print(f"\n--- MISSING: {expected_path.relative_to(self.root)}")
            else:
                actual_content = expected_path.read_text()
                if actual_content != expected_content:
                    mismatches.append((expected_path, actual_content, expected_content))
                    if show_diff:
                        diff = difflib.unified_diff(
                            actual_content.splitlines(keepends=True),
                            expected_content.splitlines(keepends=True),
                            fromfile=f"a/{expected_path.relative_to(self.root)}",
                            tofile=f"b/{expected_path.relative_to(self.root)}",
                        )
                        print("".join(diff))

        return mismatches


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate launcher scripts for all recipes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  Generate all launcher scripts
  %(prog)s --check          Check if scripts are up-to-date (exits non-zero if not)
  %(prog)s --check --diff   Check and show diffs for out-of-date scripts
""",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if generated scripts match disk (like formatter --check mode)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show unified diffs for mismatched scripts (requires --check)",
    )
    args = parser.parse_args()

    orchestrator = LauncherScriptGenerationOrchestrator()

    if args.check:
        mismatches = orchestrator.check(show_diff=args.diff)
        if mismatches:
            print(f"\n{'='*60}")
            print(f"ERROR: {len(mismatches)} script(s) out of sync with generator")
            print("=" * 60)
            for path, actual, _ in mismatches:
                status = "MISSING" if actual is None else "DIFFERS"
                print(f"  [{status}] {path.relative_to(orchestrator.root)}")
            print(f"\nRun without --check to regenerate:\n  python {sys.argv[0]}")
            sys.exit(1)
        else:
            print("All launcher scripts are up-to-date.")
            sys.exit(0)
    else:
        orchestrator.run()


if __name__ == "__main__":
    main()
