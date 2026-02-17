"""
Launch.json generation utilities.

Generates launch.json files for recipes across multiple types:
- nova, llmft, verl, evaluation, checkpointless

Supports both k8s and sm_jobs job types.
"""

import argparse
import json
import logging
import os
import random
import shutil
import string
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Default recipe prefixes to search for
DEFAULT_RECIPE_PREFIXES = ["llmft", "nova", "verl", "evaluation", "checkpointless"]

# Default job types
DEFAULT_JOB_TYPES = ["k8s", "sm_jobs"]


class LaunchJsonGenerator:
    """Generate launch.json files for all recipes."""

    def __init__(self, working_dir: Optional[str] = None):
        """
        Initialize the LaunchJsonGenerator.

        Args:
            working_dir: Working directory for subprocess calls. Defaults to current directory.
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.recipes_dir = self.working_dir / "recipes_collection" / "recipes"
        self.unsupported_recipes_path = (
            self.working_dir / "launcher" / "recipe_templatization" / "unsupported_recipes_list.json"
        )
        self.eval_regional_params_path = (
            self.working_dir
            / "launcher"
            / "recipe_templatization"
            / "evaluation"
            / "evaluation_regional_parameters.json"
        )

        # Load unsupported recipes list
        self.unsupported_recipes = set()
        if self.unsupported_recipes_path.exists():
            with open(self.unsupported_recipes_path) as f:
                unsupported_data = json.load(f)
                for category_recipes in unsupported_data.values():
                    for recipe_path in category_recipes:
                        self.unsupported_recipes.add(Path(recipe_path))

        # Load evaluation regional parameters for model mappings
        self.eval_model_mapping = {}
        if self.eval_regional_params_path.exists():
            with open(self.eval_regional_params_path) as f:
                eval_data = json.load(f)
                self.eval_model_mapping = eval_data.get("js_model_name_instance_mapping", {})

    def discover_recipes(
        self,
        prefixes: Optional[List[str]] = None,
        recipe_filter: Optional[str] = None,
    ) -> List[Tuple[Path, str, Optional[str]]]:
        """
        Discover all recipes matching the prefixes.

        Args:
            prefixes: List of recipe prefixes to search for. Defaults to all supported prefixes.
            recipe_filter: Optional filter string to match against recipe paths.

        Returns:
            List of tuples: (recipe_path, prefix, model_name)
            - For most recipes: model_name is None
            - For open_source evaluation recipes: expanded into one tuple per model
        """
        if prefixes is None:
            prefixes = DEFAULT_RECIPE_PREFIXES

        recipes = []
        excluded_count = 0

        for prefix in prefixes:
            # Handle special case for evaluation - search only in open-source directory
            if prefix == "evaluation":
                search_dir = self.recipes_dir / "evaluation" / "open-source"
                recipe_files = search_dir.glob("*.yaml") if search_dir.exists() else []
            else:
                # For other prefixes, recursively search for files matching the prefix
                search_dir = self.recipes_dir
                recipe_files = search_dir.rglob(f"*{prefix}*.yaml") if search_dir.exists() else []

            for recipe_file in recipe_files:
                # Apply filter if set
                if recipe_filter is not None:
                    if recipe_filter not in str(recipe_file):
                        continue

                # Check if recipe is in unsupported list
                recipe_relative = recipe_file.relative_to(self.working_dir)
                if recipe_relative in self.unsupported_recipes:
                    excluded_count += 1
                    logger.info(f"Excluding unsupported recipe: {recipe_relative}")
                    continue

                # For evaluation prefix (open-source), expand into one test per model
                if prefix == "evaluation":
                    for model_name in self.eval_model_mapping.keys():
                        recipes.append((recipe_file, prefix, model_name))
                else:
                    recipes.append((recipe_file, prefix, None))

        unique_recipe_files = len(set(r[0] for r in recipes))
        logger.info(f"Discovered {unique_recipe_files} recipe file(s), {len(recipes)} total tests")
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} unsupported recipes")

        return sorted(recipes, key=lambda x: (x[0].name, x[2] or ""))

    @staticmethod
    def generate_valid_k8s_name(recipe_name: str) -> str:
        """
        Generate a K8s-compliant run name from recipe filename.

        Ensures:
        - DNS-1123 compliant (lowercase alphanumeric + hyphens only)
        - Max 53 characters (Helm limitation)
        - Adds random 5-char suffix to prevent collisions
        """
        random_hash = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        k8s_name = recipe_name.lower().replace("_", "-")
        k8s_name = "".join(c if c.isalnum() or c == "-" else "-" for c in k8s_name)

        while "--" in k8s_name:
            k8s_name = k8s_name.replace("--", "-")

        k8s_name = k8s_name.strip("-")

        if len(k8s_name) > 47:
            k8s_name = k8s_name[:47].rstrip("-")

        return f"{k8s_name}-{random_hash}"

    def get_nova_params(
        self,
        recipe_path: Path,
        recipe_rel_path: Path,
        job_type: str,
        run_name: str,
        output_dir: str,
        instance_type: str,
    ) -> List[str]:
        """Generate Nova-specific parameters."""
        is_evaluation = "/evaluation/" in str(recipe_path)

        if is_evaluation:
            if job_type == "k8s":
                return [
                    f"recipes={recipe_rel_path}",
                    f"instance_type={instance_type}",
                    f"base_results_dir={output_dir}",
                    "cluster=k8s",
                    "cluster_type=k8s",
                    "launch_json=true",
                    "container=test_container",
                ]
            else:  # sm_jobs
                sm_instance_type = (
                    instance_type.replace("ml.", "") if instance_type.startswith("ml.") else instance_type
                )
                return [
                    f"recipes={recipe_rel_path}",
                    f"instance_type={sm_instance_type}",
                    f"base_results_dir={output_dir}",
                    "cluster=sm_jobs",
                    "cluster_type=sm_jobs",
                    "launch_json=true",
                    "container=test_container",
                ]

        if job_type == "k8s":
            return [
                f"recipes={recipe_rel_path}",
                f"instance_type={instance_type}",
                f"recipes.run.name={run_name}",
                f"base_results_dir={output_dir}",
                "container=test_container",
                "cluster=k8s",
                "cluster_type=k8s",
                "launch_json=true",
            ]
        else:  # sm_jobs
            sm_instance_type = instance_type.replace("ml.", "") if instance_type.startswith("ml.") else instance_type
            return [
                f"recipes={recipe_rel_path}",
                "cluster=sm_jobs",
                "cluster_type=sm_jobs",
                f"instance_type={sm_instance_type}",
                f"recipes.run.name={run_name}",
                f"base_results_dir={output_dir}",
                "+cluster.sm_jobs_config.output_path=s3://test_path",
                "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
                "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
                "container=test_container",
                "+env_vars.NEMO_LAUNCHER_DEBUG=1",
                "git.use_default=false",
                "git.entry_script=/app/src/train_hp.py",
                "launch_json=true",
            ]

    def get_llmft_params(
        self,
        recipe_rel_path: Path,
        job_type: str,
        run_name: str,
        output_dir: str,
        instance_type: str,
    ) -> List[str]:
        """Generate LLMFT-specific parameters."""
        if job_type == "k8s":
            return [
                f"recipes={recipe_rel_path}",
                f"instance_type={instance_type}",
                f"recipes.run.name={run_name}",
                f"base_results_dir={output_dir}",
                "container=test_container",
                "git.use_default=false",
                "+cluster.persistent_volume_claims.0.claimName=fsx-claim",
                "+cluster.persistent_volume_claims.0.mountPath=/data",
                "cluster=k8s",
                "cluster_type=k8s",
                "launch_json=true",
            ]
        else:  # sm_jobs
            sm_instance_type = instance_type.replace("ml.", "") if instance_type.startswith("ml.") else instance_type
            return [
                f"recipes={recipe_rel_path}",
                "cluster=sm_jobs",
                "cluster_type=sm_jobs",
                f"instance_type={sm_instance_type}",
                f"recipes.run.name={run_name}",
                f"base_results_dir={output_dir}",
                "+cluster.sm_jobs_config.output_path=s3://test_path",
                "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
                "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
                "container=test_container",
                "+env_vars.NEMO_LAUNCHER_DEBUG=1",
                "git.use_default=false",
                "git.entry_script=/app/src/train_hp.py",
                "launch_json=true",
            ]

    def get_checkpointless_params(
        self,
        recipe_rel_path: Path,
        job_type: str,
        run_name: str,
        output_dir: str,
        instance_type: str,
    ) -> List[str]:
        """Generate Checkpointless-specific parameters."""
        # Checkpointless recipes follow similar pattern to LLMFT
        return self.get_llmft_params(recipe_rel_path, job_type, run_name, output_dir, instance_type)

    def get_verl_params(
        self,
        recipe_rel_path: Path,
        job_type: str,
        run_name: str,
        output_dir: str,
        instance_type: str,
    ) -> List[str]:
        """Generate Verl-specific parameters."""
        # Verl recipes follow similar pattern to LLMFT
        return self.get_llmft_params(recipe_rel_path, job_type, run_name, output_dir, instance_type)

    def get_open_source_eval_params(
        self,
        recipe_path: Path,
        recipe_rel_path: Path,
        job_type: str,
        output_dir: str,
        model_name: str,
    ) -> List[str]:
        """Generate open source evaluation-specific parameters."""
        instance_types = self.eval_model_mapping.get(model_name, ["ml.p5.48xlarge"])
        instance_type = instance_types[0]
        eval_job_name = f"eval-{model_name}-job"

        base_params = [
            f"recipes={recipe_rel_path}",
            f"cluster_type={job_type}",
            f"cluster={job_type}",
            f"base_results_dir={output_dir}",
            f"recipes.run.base_model_name={model_name}",
            f"recipes.run.name={eval_job_name}",
            "recipes.output.eval_results_dir=''",
            "container=test_container",
            "git.use_default=false",
            "launch_json=true",
        ]

        recipe_name = recipe_path.stem.lower()
        if "llmaj" in recipe_name:
            base_params.append("recipes.run.inference_data_s3_path=''")
        elif "deterministic" in recipe_name:
            base_params.extend(
                [
                    "recipes.run.model_name_or_path=''",
                    "recipes.run.data_s3_path=''",
                ]
            )

        if job_type == "k8s":
            base_params.extend(
                [
                    f"instance_type={instance_type}",
                    "+cluster.persistent_volume_claims.0.claimName=fsx-claim",
                    "+cluster.persistent_volume_claims.0.mountPath=/data",
                ]
            )
        else:  # sm_jobs
            sm_instance_type = instance_type.replace("ml.", "") if instance_type.startswith("ml.") else instance_type
            base_params.extend(
                [
                    f"instance_type={sm_instance_type}",
                    "+cluster.sm_jobs_config.output_path=s3://test_path",
                    "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
                    "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
                ]
            )

        return base_params

    def get_job_params(
        self,
        recipe_path: Path,
        job_type: str,
        run_name: str,
        output_dir: str,
        model_name: Optional[str] = None,
    ) -> List[str]:
        """Generate job parameters for launch.json generation."""
        recipe_rel_path = recipe_path.relative_to(self.recipes_dir).with_suffix("")
        recipe_name = recipe_path.stem.lower()

        # Route to recipe-specific parameter generation
        if model_name is not None:
            return self.get_open_source_eval_params(recipe_path, recipe_rel_path, job_type, output_dir, model_name)

        # Read recipe file to extract instance_type
        with open(recipe_path, "r") as f:
            recipe_data = yaml.safe_load(f)

        instance_types = recipe_data.get("instance_types", ["ml.p5.48xlarge"])
        instance_type = instance_types[0] if instance_types else "ml.p5.48xlarge"

        if "nova" in recipe_name:
            return self.get_nova_params(recipe_path, recipe_rel_path, job_type, run_name, output_dir, instance_type)
        elif "checkpointless" in recipe_name:
            return self.get_checkpointless_params(recipe_rel_path, job_type, run_name, output_dir, instance_type)
        elif "verl" in recipe_name:
            return self.get_verl_params(recipe_rel_path, job_type, run_name, output_dir, instance_type)
        elif "llmft" in recipe_name:
            return self.get_llmft_params(recipe_rel_path, job_type, run_name, output_dir, instance_type)
        else:
            # Default to LLMFT pattern
            return self.get_llmft_params(recipe_rel_path, job_type, run_name, output_dir, instance_type)

    def generate_launch_json(
        self,
        recipe_path: Path,
        job_type: str,
        output_dir: str,
        model_name: Optional[str] = None,
        timeout: int = 180,
    ) -> Optional[Path]:
        """
        Generate launch.json for a single recipe.

        Returns:
        Tuple of (launch_json_path, status) where status is one of:
        - "success": launch.json generated successfully
        - "skipped": recipe not supported on this platform
        - "failed": generation failed (error message in status)
        """
        recipe_name = recipe_path.stem
        display_name = f"{recipe_name}_{model_name}" if model_name else recipe_name

        # Generate K8s-compliant name
        base_name = display_name.lower().replace("_", "-")
        if len(base_name) > 53:
            base_name = base_name[:53].rstrip("-")
        run_name = base_name

        # Create unique temp directory for this recipe
        recipe_output_dir = tempfile.mkdtemp(prefix=f"{recipe_name[:20]}_{job_type}_", dir=output_dir)

        job_params = self.get_job_params(recipe_path, job_type, run_name, recipe_output_dir, model_name)
        cmd = ["python3", "main.py"] + job_params

        env = os.environ.copy()
        env["HYDRA_FULL_ERROR"] = "1"
        env["AWS_REGION"] = "us-east-1"

        proc_result = subprocess.run(
            cmd,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )

        if proc_result.stdout:
            logger.debug(f"STDOUT from {display_name}:\n{proc_result.stdout}")
        if proc_result.stderr:
            logger.debug(f"STDERR from {display_name}:\n{proc_result.stderr}")

        # Check if recipe is not supported on this platform
        combined_output = proc_result.stdout + proc_result.stderr
        skip_indicators = [
            "No regional parameters found",
            "No supported regional parameters for platform",
            "No container image found for platform",
            "has not been implemented yet",
        ]
        if any(indicator in combined_output for indicator in skip_indicators):
            logger.info(f"Skipped: {display_name} not supported on {job_type}")
            return None, "skipped"

        if proc_result.returncode != 0:
            logger.error(f"Generation failed for {display_name}: {proc_result.stderr[:500]}")
            return None, f"failed: {proc_result.stderr[:200]}"

        # Find generated launch.json
        output_path = None
        for line in proc_result.stdout.split("\n"):
            if "Outputs are in:" in line:
                output_path = Path(line.split("Outputs are in:")[1].strip())
                break

        if output_path and output_path.exists():
            run_dir = output_path
        else:
            recipe_temp_path = Path(recipe_output_dir)
            created_dirs = [d for d in recipe_temp_path.iterdir() if d.is_dir()]
            if not created_dirs:
                logger.error(f"No directory created for {display_name}")
                return None
            run_dir = created_dirs[0]

        # Find launch.json in appropriate location
        if job_type == "k8s":
            for subdir in ["k8s_templates", "k8s_template", ""]:
                launch_json_path = run_dir / subdir / "launch.json" if subdir else run_dir / "launch.json"
                if launch_json_path.exists():
                    return launch_json_path, "success"
        else:
            launch_json_path = run_dir / "launch.json"
            if launch_json_path.exists():
                return launch_json_path, "success"

    def generate_all_launch_jsons(
        self,
        recipes: List[Tuple[Path, str, Optional[str]]],
        output_dir: str,
        job_types: Optional[List[str]] = None,
        subfolder: str = "",
    ) -> Dict[str, Dict[str, Optional[Path]]]:
        """
        Generate launch.jsons for all recipes.

        Args:
            recipes: List of (recipe_path, prefix, model_name) tuples
            output_dir: Base output directory
            job_types: List of job types to generate. Defaults to ["k8s", "sm_jobs"]
            subfolder: Subfolder within output_dir

        Returns:
            Dict mapping recipe_file -> {job_type: launch_json_path}
        """
        if job_types is None:
            job_types = DEFAULT_JOB_TYPES

        full_output_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
        os.makedirs(full_output_dir, exist_ok=True)

        results = {}
        total = len(recipes) * len(job_types)
        current = 0
        success_count = 0
        skip_count = 0
        fail_count = 0
        failed_recipes = []

        for recipe_path, prefix, model_name in recipes:
            recipe_key = str(recipe_path)
            if model_name:
                recipe_key = f"{recipe_path}:{model_name}"

            if recipe_key not in results:
                results[recipe_key] = {}

            for job_type in job_types:
                current += 1
                recipe_name = recipe_path.stem
                display_name = f"{recipe_name}_{model_name}" if model_name else recipe_name

                print(f"  [{current}/{total}] Generating: {display_name} ({job_type})")

                launch_json_path, status = self.generate_launch_json(recipe_path, job_type, full_output_dir, model_name)

                results[recipe_key][job_type] = launch_json_path

                if status == "success":
                    # Copy to standardized location
                    dest_dir = os.path.join(full_output_dir, display_name, job_type)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, "launch.json")
                    if str(launch_json_path) != dest_path:
                        shutil.copy2(launch_json_path, dest_path)
                        results[recipe_key][job_type] = Path(dest_path)

                    success_count += 1
                    print(f"           ✓ Generated: {dest_path}")
                elif status == "skipped":
                    skip_count += 1
                    print(f"           ⊘ Skipped (not supported on {job_type})")
                else:  # status == "failed"
                    fail_count += 1
                    failed_recipes.append((display_name, job_type, status))
                    print(f"           ✗ FAILED")

                    raise RuntimeError(f"Failed to generate launch.json for {display_name} ({job_type})")

        # Print summary
        print(f"\n  Summary: {success_count} generated, {skip_count} skipped, {fail_count} failed")

        if failed_recipes:
            failed_list = "\n".join(f"  - {name} ({jt})" for name, jt, _ in failed_recipes)
            raise RuntimeError(f"Failed to generate {fail_count} launch.json(s):\n{failed_list}")

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate launch.json files for recipes")
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=DEFAULT_RECIPE_PREFIXES,
        help=f"Recipe prefixes to process (default: {DEFAULT_RECIPE_PREFIXES})",
    )
    parser.add_argument(
        "--job-types",
        nargs="+",
        default=DEFAULT_JOB_TYPES,
        help=f"Job types to generate (default: {DEFAULT_JOB_TYPES})",
    )
    parser.add_argument(
        "--output-dir",
        default="results/launch_jsons",
        help="Output directory (default: results/launch_jsons)",
    )
    parser.add_argument(
        "--filter",
        help="Filter recipes by path substring",
    )
    parser.add_argument(
        "--working-dir",
        help="Working directory (default: current directory)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    generator = LaunchJsonGenerator(working_dir=args.working_dir)

    logger.info("Discovering recipes...")
    recipes = generator.discover_recipes(
        prefixes=args.prefixes,
        recipe_filter=args.filter,
    )

    if not recipes:
        logger.warning("No recipes found")
        return

    logger.info(f"Found {len(recipes)} recipes to process")

    results = generator.generate_all_launch_jsons(
        recipes=recipes,
        output_dir=args.output_dir,
        job_types=args.job_types,
    )

    # Summary
    success_count = sum(1 for r in results.values() for p in r.values() if p is not None)
    total_count = sum(len(r) for r in results.values())
    logger.info(f"Generated {success_count}/{total_count} launch.json files")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
