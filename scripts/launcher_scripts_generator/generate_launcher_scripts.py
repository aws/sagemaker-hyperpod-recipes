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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

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


@contextmanager
def hydra_compose_context(config_dir: Path):
    """Context manager for Hydra compose that handles GlobalHydra cleanup."""
    GlobalHydra.instance().clear()
    try:
        initialize_config_dir(config_dir=str(config_dir), version_base=None)
        yield
    finally:
        GlobalHydra.instance().clear()


class LauncherScriptGenerator:
    """Generates launcher scripts for recipes."""

    # Class-level cache for Hydra config directory
    _hydra_config_dir: Optional[Path] = None

    def __init__(self, recipe_name: str, recipe_path: str, **kwargs):
        self.recipe_name = recipe_name
        self.recipe_path = recipe_path
        self.config = LauncherConfig()
        self._recipe_data: Optional[Dict] = None
        self._recipes_root = Path(__file__).parent.parent.parent / "recipes_collection/recipes"

    @property
    def recipe_data(self) -> Dict:
        """Lazy-load recipe YAML with Hydra defaults resolved via OmegaConf."""
        if self._recipe_data is None:
            recipe_file = self._recipes_root / f"{self.recipe_path}.yaml"
            if recipe_file.exists():
                self._recipe_data = self._load_with_hydra(recipe_file)
            else:
                self._recipe_data = {}
        return self._recipe_data

    def _load_with_hydra(self, recipe_file: Path) -> Dict:
        """Load recipe YAML, resolving Hydra defaults if present."""
        raw = yaml.safe_load(recipe_file.read_text()) or {}
        if "defaults" not in raw:
            return raw

        # Derive paths from recipe_file: .../recipes/{category}/{model}/{recipe}.yaml
        # searchpath = category dir (for hydra_config), config_dir = recipes_collection
        searchpath = recipe_file.parent.parent
        config_dir = self._recipes_root.parent

        with hydra_compose_context(config_dir):
            cfg = compose(
                config_name="config",
                overrides=[f"hydra.searchpath=[file://{searchpath}]", f"recipes={self.recipe_path}"],
            )
            return OmegaConf.to_container(cfg.get("recipes", cfg), resolve=True)

    def get_model_type(self) -> str:
        """Get run.model_type from recipe, or infer from Hydra defaults."""
        run = self.recipe_data.get("run", {})
        if isinstance(run, dict) and run.get("model_type"):
            return run.get("model_type")

        # Infer from Hydra defaults
        defaults = self.recipe_data.get("defaults", [])
        for default in defaults:
            if isinstance(default, str) and "/hydra_config/verl/" in default:
                return "verl"
            if isinstance(default, str) and "/hydra_config/llmft/" in default:
                return "llm_finetuning_aws"

        return ""

    def get_template_key(self) -> str:
        """Select template based on model_type."""
        model_type = self.get_model_type()

        # Exact match
        if model_type in self.config.templates:
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
        self.recipes_dir = self.root / self.config.recipes_dir
        self.output_dir = self.root / self.config.output_dir
        self.generated: set = set()

    def discover_recipes(self) -> List[Path]:
        """Find all recipe YAML files, excluding configured directories."""
        excluded = self.config.excluded_recipe_dirs
        recipes = []
        for recipe_path in self.recipes_dir.rglob("*.yaml"):
            rel_path = str(recipe_path.relative_to(self.recipes_dir))
            # Skip if recipe is in an excluded directory
            if not any(rel_path.startswith(excl) for excl in excluded):
                recipes.append(recipe_path)
        return sorted(recipes)

    def infer_model_family(self, recipe_path: Path) -> str:
        """Extract model family from recipe path (2nd directory level)."""
        try:
            parts = recipe_path.relative_to(self.recipes_dir).parts
            return parts[1].lower() if len(parts) >= 2 else "misc"
        except ValueError:
            return "misc"

    def run(self):
        """Generate all launcher scripts."""
        recipes = self.discover_recipes()
        print(f"Found {len(recipes)} recipes\n")

        stats = {"generated": 0, "skipped": 0, "errors": 0}

        for recipe_path in recipes:
            rel_path = str(recipe_path.relative_to(self.recipes_dir).with_suffix(""))

            # Handle manually maintained scripts
            if rel_path in self.config.manually_maintained:
                script_path = self.output_dir / self.config.manually_maintained[rel_path]
                self.generated.add(script_path)
                stats["skipped"] += 1
                print(f"⚠ {recipe_path.stem} → {script_path.relative_to(self.root)} (manually maintained)")
                continue

            # Generate script
            try:
                out_dir = self.output_dir / self.infer_model_family(recipe_path)
                out_dir.mkdir(parents=True, exist_ok=True)

                gen = LauncherScriptGenerator(recipe_path.stem, rel_path)
                script_path, _ = gen.generate(out_dir)
                self.generated.add(script_path)
                stats["generated"] += 1
                print(
                    f"✓ {recipe_path.stem} → {script_path.relative_to(self.root)} [model_type={gen.get_model_type() or '(none)'} → {gen.get_template_key()}]"
                )
            except Exception as e:
                stats["errors"] += 1
                print(f"✗ {recipe_path.stem}: {e}")

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

        for recipe_path in self.discover_recipes():
            rel_path = str(recipe_path.relative_to(self.recipes_dir).with_suffix(""))

            # Skip manually maintained scripts
            if rel_path in self.config.manually_maintained:
                continue

            out_dir = self.output_dir / self.infer_model_family(recipe_path)
            gen = LauncherScriptGenerator(recipe_path.stem, rel_path)
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
