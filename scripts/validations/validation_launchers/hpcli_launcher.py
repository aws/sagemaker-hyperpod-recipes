"""HyperPod CLI (hyp) recipe job launcher for integration testing.

Uses a Builder pattern (HpCliRecipeJobBuilder) to construct and execute recipe jobs
through the full CLI lifecycle: hyp init → configure → validate → create → monitor → delete.

Overrides the private hub's SmtjImageUri before launching to ensure the correct
container image is always used (same pattern as ServerlessValidationLauncher).
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .base_launcher import BaseLauncher
from .hub_override_utils import override_hub_recipe_images

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI execution helper
# ---------------------------------------------------------------------------


@dataclass
class CLIResult:
    """Result of a HyperPod CLI command execution."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


EXPECTED_INIT_FILES = ["config.yaml", "k8s.jinja", ".override_spec.json"]


def run_hyp_command(
    args: list[str],
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout: int = 120,
    endpoint_url: str = "",
) -> CLIResult:
    """Execute a 'hyp' CLI command via subprocess.

    Injects AWS_ENDPOINT_URL_SAGEMAKER so the CLI talks to the correct
    SageMaker endpoint where the private hub lives.
    """
    full_cmd = ["hyp"] + args
    run_env = {**os.environ.copy()}
    if endpoint_url:
        run_env["AWS_ENDPOINT_URL_SAGEMAKER"] = endpoint_url
    if env:
        run_env.update(env)

    logger.info(f"Running: {' '.join(full_cmd)}")

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=run_env,
        )
        cli_result = CLIResult(
            command=full_cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if not cli_result.success:
            logger.warning(f"  {cli_result.command} failed (rc={cli_result.returncode})")
            if result.stderr:
                logger.warning(f"  stderr: {result.stderr[:500]}")
        return cli_result

    except subprocess.TimeoutExpired:
        return CLIResult(
            command=full_cmd,
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
        )


# ---------------------------------------------------------------------------
# Builder — constructs and executes HP CLI recipe jobs step-by-step
# ---------------------------------------------------------------------------


class HpCliRecipeJobBuilder:
    """Builder for constructing and executing HP CLI recipe jobs.

    Separates job *construction* (fluent `with_*` setters) from *execution*
    (individual ``init`` / ``configure`` / ``validate`` / ``create`` methods).

    Usage::

        builder = (HpCliRecipeJobBuilder(endpoint_url=url, aws_env=env)
                   .with_model(model_arn)
                   .with_technique("SFT")
                   .with_instance_type("ml.p4d.24xlarge")
                   .with_namespace("default")
                   .with_job_name("integ-test-12345")
                   .with_data_config(
                       data_path="/fsx/data",
                       output_path="/fsx/output",
                       results_dir="/fsx/results",
                       training_data_name="train.jsonl",
                       validation_data_name="val.jsonl",
                       validation_data_path="/fsx/val",
                   ))

        # Execute steps individually (for granular testing)
        builder.init()
        builder.configure()
        builder.validate()
        builder.create()

        # Or iterate (see HpCliValidationLauncher.launch_job)
        builder.cleanup()
    """

    def __init__(self, endpoint_url: str = "", aws_env: Optional[dict] = None):
        self._endpoint_url = endpoint_url
        self._aws_env = aws_env or {}

        # Configurable fields — set via fluent setters
        self._model_arn: Optional[str] = None
        self._technique: Optional[str] = None
        self._instance_type: Optional[str] = None
        self._namespace: str = "default"
        self._job_name: Optional[str] = None

        # Data configuration
        self._data_path: Optional[str] = None
        self._output_path: Optional[str] = None
        self._results_directory: Optional[str] = None
        self._training_data_name: Optional[str] = None
        self._validation_data_name: Optional[str] = None
        self._validation_data_path: Optional[str] = None
        self._max_epochs: Optional[str] = None

        # Internal state
        self._parent_dir: Optional[str] = None
        self._job_dir: Optional[str] = None
        self._submitted: bool = False

    # ---- Fluent setters (each returns self for chaining) -----------------

    def with_model(self, model_arn: str) -> "HpCliRecipeJobBuilder":
        """Set the model ARN (private hub content ARN or JumpStart / HuggingFace ID)."""
        self._model_arn = model_arn
        return self

    def with_technique(self, technique: str) -> "HpCliRecipeJobBuilder":
        """Set the training technique (SFT, DPO, RLAIF, RLVR, etc.)."""
        self._technique = technique
        return self

    def with_instance_type(self, instance_type: str) -> "HpCliRecipeJobBuilder":
        """Set the instance type (e.g. ml.p4d.24xlarge)."""
        self._instance_type = instance_type
        return self

    def with_namespace(self, namespace: str) -> "HpCliRecipeJobBuilder":
        """Set the Kubernetes namespace for the job."""
        self._namespace = namespace
        return self

    def with_job_name(self, name: str) -> "HpCliRecipeJobBuilder":
        """Set a custom job name."""
        self._job_name = name
        return self

    def with_data_config(
        self,
        data_path: str,
        output_path: str,
        results_dir: str,
        training_data_name: str,
        validation_data_name: str,
        validation_data_path: str,
        model_name_or_path: Optional[str] = None,
        max_epochs: Optional[str] = None,
    ) -> "HpCliRecipeJobBuilder":
        """Set all data-related paths in one call.

        ``model_name_or_path`` overrides the recipe default so the training
        container loads weights from a local/FSx path instead of attempting
        a HuggingFace download (which fails for gated models without a token).
        """
        self._data_path = data_path
        self._output_path = output_path
        self._results_directory = results_dir
        self._training_data_name = training_data_name
        self._validation_data_name = validation_data_name
        self._validation_data_path = validation_data_path
        self._model_name_or_path = model_name_or_path
        self._max_epochs = max_epochs
        return self

    # ---- CLI step methods ------------------------------------------------

    def init(self) -> CLIResult:
        """Run ``hyp init hyp-recipe-job`` — creates job directory with
        config.yaml, k8s.jinja, .override_spec.json.
        """
        self._require_fields("init", ["_model_arn", "_technique", "_instance_type"])
        self._parent_dir = tempfile.mkdtemp(prefix="hp-cli-")
        self._job_dir = os.path.join(self._parent_dir, "job")
        result = run_hyp_command(
            [
                "init",
                "hyp-recipe-job",
                self._job_dir,
                "--model-id",
                self._model_arn,
                "--technique",
                self._technique,
                "--instance-type",
                self._instance_type,
            ],
            endpoint_url=self._endpoint_url,
            env=self._aws_env,
        )
        if result.success:
            self._patch_k8s_jinja()
        return result

    def _patch_k8s_jinja(self):
        """Fix SCRIPT_PATH in k8s.jinja to use the correct container entrypoint."""
        jinja_path = os.path.join(self._job_dir, "k8s.jinja")
        if not os.path.exists(jinja_path):
            return
        with open(jinja_path, "r") as f:
            content = f.read()
        content = re.sub(
            r'(- name: SCRIPT_PATH\s*\n\s*value: )"[^"]*"',
            r'\1"/app/src/train_hp.py"',
            content,
        )
        with open(jinja_path, "w") as f:
            f.write(content)

    def configure(self) -> CLIResult:
        """Run ``hyp configure`` — fills in config.yaml fields.

        Only passes options that exist in the recipe's ``.override_spec.json``.
        This ensures compatibility with any recipe regardless of which
        parameters it exposes.
        """
        self._require_fields("configure", ["_job_dir", "_job_name", "_data_path"])

        # Verify init artifacts exist
        config_path = os.path.join(self._job_dir, "config.yaml")
        logger.info(f"Job dir: {self._job_dir}, config.yaml exists: {os.path.exists(config_path)}")

        # Read available options from the override spec
        available_options = self._read_override_spec_options()

        params = {
            "name": self._job_name,
            "namespace": self._namespace,
            "data-path": self._data_path,
            "output-path": self._output_path,
            "results-directory": self._results_directory,
            "training-data-name": self._training_data_name,
            "validation-data-name": self._validation_data_name,
            "validation-data-path": self._validation_data_path,
            "model-name-or-path": getattr(self, "_model_name_or_path", None),
            "max-epochs": getattr(self, "_max_epochs", None),
        }
        args = ["configure"]
        for key, value in params.items():
            if value is None:
                continue
            # If we have a spec, skip options not supported by this recipe
            if available_options and key not in available_options:
                logger.info(f"Skipping --{key}: not in recipe override spec")
                continue
            args.extend([f"--{key}", str(value)])
        return run_hyp_command(
            args,
            cwd=self._job_dir,
            endpoint_url=self._endpoint_url,
            env=self._aws_env,
        )

    def _read_override_spec_options(self) -> set:
        """Read parameter names from .override_spec.json and return as CLI option names.

        Returns an empty set if the spec file doesn't exist (in which case
        configure() will pass all non-None params through without filtering).
        """
        import json

        assert self._job_dir is not None  # guaranteed by _require_fields in configure()
        spec_path = os.path.join(self._job_dir, ".override_spec.json")
        if not os.path.exists(spec_path):
            logger.warning(f".override_spec.json not found at {spec_path}, skipping option filtering")
            return set()

        try:
            with open(spec_path) as f:
                spec = json.load(f)
            # Spec is a flat dict: {"param_name": {"type": ..., "required": ...}, ...}
            # Convert parameter names (snake_case) to CLI option names (kebab-case)
            options = {name.replace("_", "-") for name in spec.keys()}
            logger.info(f"Override spec has {len(options)} configurable options: {sorted(options)}")
            return options
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read override spec: {e}, skipping option filtering")
            return set()

    def validate(self) -> CLIResult:
        """Run ``hyp validate`` — checks config.yaml against the recipe's parameter schema."""
        self._require_fields("validate", ["_job_dir"])
        return run_hyp_command(
            ["validate"],
            cwd=self._job_dir,
            endpoint_url=self._endpoint_url,
            env=self._aws_env,
        )

    def create(self) -> CLIResult:
        """Run ``hyp create`` — renders K8s manifest and submits to the cluster."""
        self._require_fields("create", ["_job_dir"])
        result = run_hyp_command(
            ["create"],
            cwd=self._job_dir,
            timeout=180,
            endpoint_url=self._endpoint_url,
            env=self._aws_env,
        )
        if result.success:
            self._submitted = True
        return result

    def describe(self) -> CLIResult:
        """Run ``hyp describe hyp-recipe-job`` — inspect job status and conditions."""
        self._require_fields("describe", ["_job_name"])
        assert self._job_name is not None  # guaranteed by _require_fields
        return run_hyp_command(
            ["describe", "hyp-recipe-job", "--job-name", self._job_name, "-n", self._namespace],
            endpoint_url=self._endpoint_url,
            env=self._aws_env,
        )

    def list_jobs(self) -> CLIResult:
        """Run ``hyp list hyp-recipe-job`` — list all recipe jobs in the namespace."""
        return run_hyp_command(
            ["list", "hyp-recipe-job", "-n", self._namespace],
            endpoint_url=self._endpoint_url,
            env=self._aws_env,
        )

    def delete(self) -> CLIResult:
        """Run ``hyp delete hyp-recipe-job`` — delete the submitted job."""
        self._require_fields("delete", ["_job_name"])
        assert self._job_name is not None  # guaranteed by _require_fields
        return run_hyp_command(
            ["delete", "hyp-recipe-job", "--job-name", self._job_name, "-n", self._namespace],
            endpoint_url=self._endpoint_url,
            env=self._aws_env,
        )

    def patch_template_image(self, expected_image: str) -> bool:
        """Replace ``{{container_image}}`` (or stale ECR URIs) in the local ``k8s.jinja``.

        Call this **after** ``init()`` and **before** ``create()`` so the
        rendered K8s manifest always uses the correct container image — even
        when the S3 template still contains an unresolved Jinja placeholder
        or an outdated ECR URI.

        Returns True if the file was patched, False if no patching was needed
        or the file was not found.
        """
        import re

        if not self._job_dir:
            logger.warning("patch_template_image called before init(), skipping")
            return False

        k8s_path = os.path.join(self._job_dir, "k8s.jinja")
        if not os.path.exists(k8s_path):
            logger.warning(f"k8s.jinja not found at {k8s_path}, skipping image patch")
            return False

        with open(k8s_path) as f:
            content = f.read()

        original = content

        # 1. Replace Jinja placeholder
        if "{{container_image}}" in content:
            content = content.replace("{{container_image}}", expected_image)
            logger.info(f"Patched {{{{container_image}}}} → {expected_image} in k8s.jinja")

        # 2. Replace any stale ECR image URIs that don't match expected
        ecr_pattern = re.compile(
            r"\d+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[^\s\"']+:[^\s\"':]+",
        )
        for match in set(ecr_pattern.findall(content)):
            if match != expected_image:
                content = content.replace(match, expected_image)
                logger.info(f"Patched stale ECR image {match} → {expected_image} in k8s.jinja")

        if content != original:
            with open(k8s_path, "w") as f:
                f.write(content)
            return True

        logger.info("k8s.jinja already has the correct image, no patch needed")
        return False

    def patch_entry_script(self, entry_script: str = "/app/src/train_hp.py") -> bool:
        """Replace the ``SCRIPT_PATH`` env var value in the local ``k8s.jinja``.

        The K8s template from the hub has ``SCRIPT_PATH`` set to the NeMo
        adapter path (e.g. ``examples/llama/llama_pretrain.py``), but the
        LLMFT container uses ``/app/src/train_hp.py``.  This mirrors what
        ``generate_launch_jsons.py`` does with ``git.entry_script``.

        Call this **after** ``init()`` and **before** ``create()``.

        Returns True if the file was patched, False otherwise.
        """
        import re

        if not self._job_dir:
            logger.warning("patch_entry_script called before init(), skipping")
            return False

        k8s_path = os.path.join(self._job_dir, "k8s.jinja")
        if not os.path.exists(k8s_path):
            logger.warning(f"k8s.jinja not found at {k8s_path}, skipping entry script patch")
            return False

        with open(k8s_path) as f:
            content = f.read()

        original = content

        # Match the SCRIPT_PATH env var value in the K8s manifest.
        # Pattern: `value: "examples/llama/llama_pretrain.py"` or without quotes.
        # The template embeds it as:
        #   - name: SCRIPT_PATH
        #     value: "examples/llama/llama_pretrain.py"
        content = re.sub(
            r'(name:\s*SCRIPT_PATH\s*\n\s*value:\s*)"?([^"\n]+)"?',
            rf'\1"{entry_script}"',
            content,
        )

        if content != original:
            with open(k8s_path, "w") as f:
                f.write(content)
            logger.info(f"Patched SCRIPT_PATH → {entry_script} in k8s.jinja")
            return True

        logger.info("k8s.jinja SCRIPT_PATH already correct, no patch needed")
        return False

    def cleanup(self):
        """Delete the K8s job (if submitted) and remove the temp directory."""
        if self._submitted:
            self.delete()
        if self._parent_dir:
            shutil.rmtree(self._parent_dir, ignore_errors=True)

    # ---- Properties ------------------------------------------------------

    @property
    def job_dir(self) -> Optional[str]:
        """Path to the job directory created by ``init``."""
        return self._job_dir

    @property
    def job_name(self) -> Optional[str]:
        return self._job_name

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def is_submitted(self) -> bool:
        return self._submitted

    # ---- Internal helpers ------------------------------------------------

    def _require_fields(self, step_name: str, fields: list[str]):
        """Raise ValueError if any required field is None for the given step.

        After this call returns, all listed fields are guaranteed to be non-None.
        """
        missing = [f.lstrip("_") for f in fields if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"Cannot run '{step_name}': missing required fields: {', '.join(missing)}. "
                f"Use the corresponding .with_*() setter first."
            )


# ---------------------------------------------------------------------------
# Launcher — thin adapter that wires the builder into the validation framework
# ---------------------------------------------------------------------------


class HpCliValidationLauncher(BaseLauncher):
    """HyperPod CLI launcher — uses HpCliRecipeJobBuilder for recipe jobs on K8s.

    Plugs into the validation framework via ``select_validation_launcher("HPCLI")``.
    """

    # Steps to execute in order — makes it easy to add/skip steps
    _PIPELINE_STEPS = ["init", "configure", "validate", "create"]

    def __init__(self, job_recorder, config):
        super().__init__(job_recorder, config)

        self.hpcli_config = config.hpcli_config
        self.endpoint_url = getattr(self.hpcli_config, "endpoint", "")
        self.model_arn = self.hpcli_config.model_arn
        self._namespace = getattr(self.hpcli_config, "namespace", "default")

        self._hub_overridden = False
        self._hub_override_lock = threading.Lock()

    # ---- Hub override (same as before) -----------------------------------

    def _ensure_hub_overridden(self):
        """Override private hub container image once (thread-safe)."""
        with self._hub_override_lock:
            if not self._hub_overridden:
                self._override_hub_container_image()
                self._hub_overridden = True

    def _override_hub_container_image(self):
        """Override SmtjImageUri and K8s template image in the private hub.

        Delegates to the shared ``override_hub_recipe_images`` utility with
        ``update_k8s_templates=True`` (HPCLI launches K8s jobs).
        """
        expected_image = getattr(self.hpcli_config, "expected_container_image", "")
        if not expected_image:
            logger.info("No expected_container_image configured, skipping hub override")
            return

        arn_parts = self.model_arn.split("/")
        hub_name, model_name = arn_parts[1], arn_parts[3]

        client_kwargs = {"region_name": self.hpcli_config.region}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        sm = self.boto_session.client("sagemaker", **client_kwargs)
        s3 = self.boto_session.client("s3", region_name=self.hpcli_config.region)

        override_hub_recipe_images(
            sagemaker_client=sm,
            hub_name=hub_name,
            model_name=model_name,
            expected_image=expected_image,
            s3_client=s3,
            update_k8s_templates=True,
        )

    # ---- Job lifecycle (builder pattern) ---------------------------------

    def _build_job(self, recipe: str) -> HpCliRecipeJobBuilder:
        """Construct a fully-configured job builder from config + recipe.

        Passes assumed-role credentials so the kubeconfig exec plugin
        (aws eks get-token --role-arn) can re-assume the role for K8s auth.
        """
        technique = self._extract_technique(recipe)
        instance_type = getattr(self.hpcli_config, "instance_type", "ml.g5.48xlarge")
        job_name = f"integ-cli-{os.environ.get('USER', 'test')[:8]}-{int(time.time()) % 100000}"

        return (
            HpCliRecipeJobBuilder(endpoint_url=self.endpoint_url, aws_env=self.aws_env)
            .with_model(self.model_arn)
            .with_technique(technique)
            .with_instance_type(instance_type)
            .with_namespace(self._namespace)
            .with_job_name(job_name)
            .with_data_config(
                data_path=self.hpcli_config.data_path,
                output_path=self.hpcli_config.output_path,
                results_dir=self.hpcli_config.results_directory,
                training_data_name=self.hpcli_config.training_data_name,
                validation_data_name=self.hpcli_config.validation_data_name,
                validation_data_path=self.hpcli_config.validation_data_path,
                model_name_or_path=getattr(self.hpcli_config, "model_name_or_path", None),
                max_epochs=getattr(self.hpcli_config, "max_epochs", None),
            )
        )

    def launch_job(self, recipe: str) -> bool:
        """Full CLI flow using the builder: override hub → init → configure → validate → create → monitor → delete."""
        self._ensure_hub_overridden()

        builder = self._build_job(recipe)

        try:
            # Run each pipeline step in order
            for step_name in self._PIPELINE_STEPS:
                step_fn = getattr(builder, step_name)
                result = step_fn()
                if not result.success:
                    self._record_failure(
                        recipe,
                        f"hyp {step_name} failed:\nstderr: {result.stderr}\nstdout: {result.stdout}",
                    )
                    return False
                logger.info(f"hyp {step_name} succeeded for job '{builder.job_name}'")

            # Monitor until completion
            return self._monitor_job(recipe, builder)

        except Exception as e:
            self._record_failure(recipe, str(e))
            return False
        finally:
            builder.cleanup()

    def _monitor_job(
        self, recipe: str, builder: HpCliRecipeJobBuilder, timeout: int = 3600, poll_interval: int = 30
    ) -> bool:
        """Poll ``builder.describe()`` until job reaches a terminal status."""
        start = time.time()
        terminal_statuses = ["succeeded", "completed", "failed"]

        while time.time() - start < timeout:
            result = builder.describe()
            if result.success:
                stdout_lower = result.stdout.lower()
                for status in terminal_statuses:
                    if status in stdout_lower:
                        if "failed" in stdout_lower:
                            self._record_failure(recipe, f"Job {builder.job_name} failed:\n{result.stdout}")
                            return False
                        self.job_recorder.update_job(
                            input_filename=recipe,
                            status="Complete",
                            output_log=f"Job {builder.job_name} completed successfully",
                        )
                        return True

            elapsed = int(time.time() - start)
            logger.info(f"Job {builder.job_name}: waiting... ({elapsed}s / {timeout}s)")
            time.sleep(poll_interval)

        self._record_failure(recipe, f"Job {builder.job_name} timed out after {timeout}s")
        return False

    # ---- Helpers ---------------------------------------------------------

    def _record_failure(self, recipe: str, message: str):
        logger.error(message)
        self.job_recorder.update_job(input_filename=recipe, status="Failed", output_log=message)

    def _extract_technique(self, recipe: str) -> str:
        """Extract training technique from recipe filename or config."""
        recipe_lower = recipe.lower()
        if "dpo" in recipe_lower:
            return "DPO"
        if "rlaif" in recipe_lower:
            return "RLAIF"
        if "rlvr" in recipe_lower:
            return "RLVR"
        if "ppo" in recipe_lower:
            return "PPO"
        if "cpt" in recipe_lower:
            return "CPT"
        return getattr(self.hpcli_config, "default_technique", "SFT")
