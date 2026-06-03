"""Unit tests for HpCliValidationLauncher and HpCliRecipeJobBuilder."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.validations.validation_launchers.hpcli_launcher import (
    CLIResult,
    HpCliRecipeJobBuilder,
    HpCliValidationLauncher,
    run_hyp_command,
)

# =============================================================================
# CLIResult
# =============================================================================


class TestCLIResult:
    def test_success_when_returncode_zero(self):
        r = CLIResult(command=["hyp", "init"], returncode=0, stdout="ok", stderr="")
        assert r.success is True

    def test_failure_when_returncode_nonzero(self):
        r = CLIResult(command=["hyp", "init"], returncode=1, stdout="", stderr="err")
        assert r.success is False


# =============================================================================
# run_hyp_command
# =============================================================================


class TestRunHypCommand:
    @patch("scripts.validations.validation_launchers.hpcli_launcher.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        result = run_hyp_command(["validate"], cwd="/tmp")

        assert result.success
        assert result.stdout == "output"
        assert mock_run.call_args[0][0] == ["hyp", "validate"]

    @patch("scripts.validations.validation_launchers.hpcli_launcher.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="bad")
        result = run_hyp_command(["bad-cmd"])

        assert not result.success

    @patch("scripts.validations.validation_launchers.hpcli_launcher.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="hyp", timeout=5)
        result = run_hyp_command(["slow"], timeout=5)

        assert result.returncode == -1
        assert "timed out" in result.stderr.lower()

    @patch("scripts.validations.validation_launchers.hpcli_launcher.subprocess.run")
    def test_injects_endpoint_url(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        run_hyp_command(["validate"], endpoint_url="https://beta.example.com")

        env = mock_run.call_args[1]["env"]
        assert env["AWS_ENDPOINT_URL_SAGEMAKER"] == "https://beta.example.com"

    @patch("scripts.validations.validation_launchers.hpcli_launcher.subprocess.run")
    def test_no_endpoint_url_when_empty(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        run_hyp_command(["validate"], endpoint_url="")

        env = mock_run.call_args[1]["env"]
        assert "AWS_ENDPOINT_URL_SAGEMAKER" not in env


# =============================================================================
# HpCliRecipeJobBuilder — fluent construction
# =============================================================================


class TestBuilderConstruction:
    """Test that the builder's fluent API correctly stores configuration."""

    def test_with_model(self):
        b = HpCliRecipeJobBuilder().with_model("arn:aws:sagemaker:us-west-2:123:hub-content/Hub/Model/m")
        assert b._model_arn == "arn:aws:sagemaker:us-west-2:123:hub-content/Hub/Model/m"

    def test_with_technique(self):
        b = HpCliRecipeJobBuilder().with_technique("DPO")
        assert b._technique == "DPO"

    def test_with_instance_type(self):
        b = HpCliRecipeJobBuilder().with_instance_type("ml.p4d.24xlarge")
        assert b._instance_type == "ml.p4d.24xlarge"

    def test_with_namespace(self):
        b = HpCliRecipeJobBuilder().with_namespace("training")
        assert b.namespace == "training"

    def test_with_job_name(self):
        b = HpCliRecipeJobBuilder().with_job_name("test-job-42")
        assert b.job_name == "test-job-42"

    def test_chaining(self):
        """All setters return self for fluent chaining."""
        b = (
            HpCliRecipeJobBuilder()
            .with_model("m")
            .with_technique("SFT")
            .with_instance_type("ml.g5.48xlarge")
            .with_namespace("ns")
            .with_job_name("j")
            .with_data_config(
                data_path="/d",
                output_path="/o",
                results_dir="/r",
                training_data_name="t",
                validation_data_name="v",
                validation_data_path="/vp",
            )
        )
        assert b._model_arn == "m"
        assert b._technique == "SFT"
        assert b._instance_type == "ml.g5.48xlarge"
        assert b.namespace == "ns"
        assert b.job_name == "j"
        assert b._data_path == "/d"
        assert b._output_path == "/o"

    def test_initial_state(self):
        b = HpCliRecipeJobBuilder()
        assert b.job_dir is None
        assert b.job_name is None
        assert b.is_submitted is False
        assert b.namespace == "default"


# =============================================================================
# HpCliRecipeJobBuilder — require_fields validation
# =============================================================================


class TestBuilderRequireFields:
    """Test that steps fail fast when required fields are missing."""

    def test_init_fails_without_model(self):
        b = HpCliRecipeJobBuilder().with_technique("SFT").with_instance_type("ml.g5.48xlarge")
        try:
            b.init()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "model_arn" in str(e)

    def test_init_fails_without_technique(self):
        b = HpCliRecipeJobBuilder().with_model("m").with_instance_type("ml.g5.48xlarge")
        try:
            b.init()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "technique" in str(e)

    def test_configure_fails_without_job_name(self):
        b = HpCliRecipeJobBuilder()
        b._job_dir = "/tmp/fake"  # simulate init completed
        b._data_path = "/data"
        try:
            b.configure()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "job_name" in str(e)

    def test_describe_fails_without_job_name(self):
        b = HpCliRecipeJobBuilder()
        try:
            b.describe()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "job_name" in str(e)


# =============================================================================
# HpCliRecipeJobBuilder — CLI step execution
# =============================================================================


class TestBuilderInit:
    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_init_builds_correct_args(self, mock_cmd):
        mock_cmd.return_value = CLIResult([], 0, "", "")
        b = (
            HpCliRecipeJobBuilder()
            .with_model("arn:aws:sagemaker:us-west-2:123:hub-content/Hub/Model/m")
            .with_technique("SFT")
            .with_instance_type("ml.g5.48xlarge")
        )
        b.init()

        args = mock_cmd.call_args[0][0]
        assert args[0:2] == ["init", "hyp-recipe-job"]
        assert "--model-id" in args
        assert "--technique" in args
        assert "--instance-type" in args
        assert b.job_dir is not None

    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_init_creates_job_dir_path(self, mock_cmd):
        mock_cmd.return_value = CLIResult([], 0, "", "")
        b = HpCliRecipeJobBuilder().with_model("m").with_technique("SFT").with_instance_type("t")
        b.init()
        assert b.job_dir is not None
        assert b.job_dir.endswith("/job")


class TestBuilderConfigure:
    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_configure_builds_correct_args(self, mock_cmd):
        mock_cmd.return_value = CLIResult([], 0, "", "")
        b = (
            HpCliRecipeJobBuilder()
            .with_job_name("test-job")
            .with_data_config(
                data_path="/d",
                output_path="/o",
                results_dir="/r",
                training_data_name="t",
                validation_data_name="v",
                validation_data_path="/vp",
            )
        )
        b._job_dir = "/tmp/fake"  # simulate init
        b.configure()

        args = mock_cmd.call_args[0][0]
        assert args[0] == "configure"
        assert "--name" in args
        assert "--data-path" in args

    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_configure_skips_none_values(self, mock_cmd):
        mock_cmd.return_value = CLIResult([], 0, "", "")
        b = HpCliRecipeJobBuilder().with_job_name("j")
        b._job_dir = "/tmp/fake"
        b._data_path = "/d"
        # Other data fields are None — should be skipped
        b.configure()

        args = mock_cmd.call_args[0][0]
        assert "--output-path" not in args  # was None, should be skipped


class TestBuilderCreate:
    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_create_sets_submitted_on_success(self, mock_cmd):
        mock_cmd.return_value = CLIResult([], 0, "", "")
        b = HpCliRecipeJobBuilder()
        b._job_dir = "/tmp/fake"
        b.create()
        assert b.is_submitted is True

    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_create_not_submitted_on_failure(self, mock_cmd):
        mock_cmd.return_value = CLIResult([], 1, "", "error")
        b = HpCliRecipeJobBuilder()
        b._job_dir = "/tmp/fake"
        b.create()
        assert b.is_submitted is False


# =============================================================================
# HpCliRecipeJobBuilder — patch_template_image
# =============================================================================


class TestBuilderPatchTemplateImage:
    """Test that patch_template_image replaces placeholders and stale ECR URIs."""

    def test_replaces_jinja_placeholder(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        k8s = job_dir / "k8s.jinja"
        k8s.write_text("image: {{container_image}}\nother: stuff\n")

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        expected = "111222333444.dkr.ecr.us-west-2.amazonaws.com/repo:tag"
        assert b.patch_template_image(expected) is True

        content = k8s.read_text()
        assert expected in content
        assert "{{container_image}}" not in content

    def test_replaces_stale_ecr_uri(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        old_image = "999888777666.dkr.ecr.us-east-1.amazonaws.com/old-repo:old-tag"
        k8s = job_dir / "k8s.jinja"
        k8s.write_text(f"image: {old_image}\n")

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        expected = "111222333444.dkr.ecr.us-west-2.amazonaws.com/new-repo:new-tag"
        assert b.patch_template_image(expected) is True

        content = k8s.read_text()
        assert expected in content
        assert old_image not in content

    def test_no_patch_when_image_already_correct(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        expected = "111222333444.dkr.ecr.us-west-2.amazonaws.com/repo:tag"
        k8s = job_dir / "k8s.jinja"
        k8s.write_text(f"image: {expected}\n")

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        assert b.patch_template_image(expected) is False

    def test_returns_false_when_no_job_dir(self):
        b = HpCliRecipeJobBuilder()
        assert b.patch_template_image("some-image") is False

    def test_returns_false_when_k8s_jinja_missing(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        # No k8s.jinja file created

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        assert b.patch_template_image("some-image") is False

    def test_replaces_both_placeholder_and_stale_ecr(self, tmp_path):
        """Template may have a mix — placeholder on one line, stale ECR on another."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        old_ecr = "555666777888.dkr.ecr.eu-west-1.amazonaws.com/stale:v1"
        k8s = job_dir / "k8s.jinja"
        k8s.write_text(f"main: {{{{container_image}}}}\ninit: {old_ecr}\n")

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        expected = "111222333444.dkr.ecr.us-west-2.amazonaws.com/repo:tag"
        assert b.patch_template_image(expected) is True

        content = k8s.read_text()
        assert content.count(expected) == 2
        assert "{{container_image}}" not in content
        assert old_ecr not in content


# =============================================================================
# HpCliRecipeJobBuilder — patch_entry_script
# =============================================================================


class TestBuilderPatchEntryScript:
    """Test that patch_entry_script replaces SCRIPT_PATH in the K8s manifest."""

    def test_replaces_nemo_adapter_script_path(self, tmp_path):
        """Standard case: hub template has examples/llama/llama_pretrain.py."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        k8s = job_dir / "k8s.jinja"
        k8s.write_text(
            "              - name: SCRIPT_PATH\n"
            '                value: "examples/llama/llama_pretrain.py"\n'
            "              - name: SCRIPT_ARGS\n"
        )

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        assert b.patch_entry_script("/app/src/train_hp.py") is True

        content = k8s.read_text()
        assert '"/app/src/train_hp.py"' in content
        assert "llama_pretrain.py" not in content
        # SCRIPT_ARGS should be untouched
        assert "SCRIPT_ARGS" in content

    def test_replaces_unquoted_script_path(self, tmp_path):
        """Some templates may not quote the value."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        k8s = job_dir / "k8s.jinja"
        k8s.write_text(
            "              - name: SCRIPT_PATH\n"
            "                value: examples/mistral/mistral_pretrain.py\n"
            "              - name: SCRIPT_ARGS\n"
        )

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        assert b.patch_entry_script("/app/src/train_hp.py") is True
        content = k8s.read_text()
        assert '"/app/src/train_hp.py"' in content
        assert "mistral_pretrain.py" not in content

    def test_no_patch_when_already_correct(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        k8s = job_dir / "k8s.jinja"
        k8s.write_text("              - name: SCRIPT_PATH\n" '                value: "/app/src/train_hp.py"\n')

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        assert b.patch_entry_script("/app/src/train_hp.py") is False

    def test_returns_false_when_no_job_dir(self):
        b = HpCliRecipeJobBuilder()
        assert b.patch_entry_script() is False

    def test_returns_false_when_k8s_jinja_missing(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        assert b.patch_entry_script() is False

    def test_default_entry_script(self, tmp_path):
        """Default argument should be /app/src/train_hp.py."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        k8s = job_dir / "k8s.jinja"
        k8s.write_text(
            "              - name: SCRIPT_PATH\n" '                value: "examples/llama/llama_pretrain.py"\n'
        )

        b = HpCliRecipeJobBuilder()
        b._job_dir = str(job_dir)

        # Call without argument — uses default
        assert b.patch_entry_script() is True
        content = k8s.read_text()
        assert '"/app/src/train_hp.py"' in content


class TestBuilderCleanup:
    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    @patch("scripts.validations.validation_launchers.hpcli_launcher.shutil.rmtree")
    def test_cleanup_deletes_job_and_dir(self, mock_rmtree, mock_cmd):
        mock_cmd.return_value = CLIResult([], 0, "", "")
        b = HpCliRecipeJobBuilder().with_job_name("j")
        b._parent_dir = "/tmp/hp-cli-xyz"
        b._submitted = True
        b.cleanup()

        mock_cmd.assert_called_once()  # delete was called
        mock_rmtree.assert_called_once_with("/tmp/hp-cli-xyz", ignore_errors=True)

    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    @patch("scripts.validations.validation_launchers.hpcli_launcher.shutil.rmtree")
    def test_cleanup_skips_delete_when_not_submitted(self, mock_rmtree, mock_cmd):
        b = HpCliRecipeJobBuilder()
        b._parent_dir = "/tmp/hp-cli-xyz"
        b._submitted = False
        b.cleanup()

        mock_cmd.assert_not_called()  # delete was NOT called
        mock_rmtree.assert_called_once()


# =============================================================================
# HpCliValidationLauncher
# =============================================================================


def _make_launcher():
    """Create a launcher with mocked boto session."""
    config = OmegaConf.create(
        {
            "platform": "HPCLI",
            "hpcli_config": {
                "endpoint": "https://api.sagemaker.beta.us-west-2.ml-platform.aws.a2z.com",
                "region": "us-west-2",
                "namespace": "default",
                "model_arn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/TestHub/Model/test-model",
                "expected_container_image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
                "instance_type": "ml.g5.48xlarge",
                "default_technique": "SFT",
                "data_path": "/data/train.jsonl",
                "output_path": "/opt/ml/model",
                "results_directory": "/opt/ml/model",
                "training_data_name": "train",
                "validation_data_name": "val",
                "validation_data_path": "/data/val.jsonl",
            },
        }
    )

    mock_recorder = MagicMock()
    with patch.object(HpCliValidationLauncher, "__init__", lambda self, *a, **kw: None):
        launcher = HpCliValidationLauncher.__new__(HpCliValidationLauncher)
        launcher.job_recorder = mock_recorder
        launcher.config = config
        launcher.logger = MagicMock()
        launcher.hpcli_config = config.hpcli_config
        launcher.endpoint_url = config.hpcli_config.endpoint
        launcher.model_arn = config.hpcli_config.model_arn
        launcher._namespace = config.hpcli_config.namespace
        launcher.boto_session = MagicMock()
        launcher.aws_env = os.environ.copy()
        launcher._hub_overridden = False
        launcher._hub_override_lock = MagicMock()
    return launcher


class TestLauncherBuildJob:
    """Test that _build_job constructs a correctly-configured builder."""

    def test_build_job_returns_builder(self):
        launcher = _make_launcher()
        builder = launcher._build_job("fine-tuning/llama/llmft_llama_sft_lora.yaml")
        assert isinstance(builder, HpCliRecipeJobBuilder)

    def test_build_job_sets_model(self):
        launcher = _make_launcher()
        builder = launcher._build_job("some-recipe.yaml")
        assert builder._model_arn == launcher.model_arn

    def test_build_job_sets_technique_from_recipe(self):
        launcher = _make_launcher()
        builder = launcher._build_job("fine-tuning/llama/llmft_llama_dpo.yaml")
        assert builder._technique == "DPO"

    def test_build_job_sets_data_config(self):
        launcher = _make_launcher()
        builder = launcher._build_job("recipe.yaml")
        assert builder._data_path == "/data/train.jsonl"
        assert builder._output_path == "/opt/ml/model"


class TestLauncherExtractTechnique:
    def test_detects_dpo(self):
        launcher = _make_launcher()
        assert launcher._extract_technique("fine-tuning/llama/llmft_llama_dpo.yaml") == "DPO"

    def test_detects_rlaif(self):
        launcher = _make_launcher()
        assert launcher._extract_technique("verl-grpo-rlaif-llama.yaml") == "RLAIF"

    def test_detects_rlvr(self):
        launcher = _make_launcher()
        assert launcher._extract_technique("verl-grpo-rlvr-qwen.yaml") == "RLVR"

    def test_detects_ppo(self):
        launcher = _make_launcher()
        assert launcher._extract_technique("verl-ppo-llama.yaml") == "PPO"

    def test_detects_cpt(self):
        launcher = _make_launcher()
        assert launcher._extract_technique("llmft-cpt-llama.yaml") == "CPT"

    def test_defaults_to_sft(self):
        launcher = _make_launcher()
        assert launcher._extract_technique("fine-tuning/qwen/llmft_qwen_sft_lora.yaml") == "SFT"

    def test_defaults_from_config(self):
        launcher = _make_launcher()
        assert launcher._extract_technique("some-recipe") == "SFT"


class TestLauncherHubOverride:
    def test_skips_when_no_expected_image(self):
        launcher = _make_launcher()
        launcher.hpcli_config = OmegaConf.create(
            {
                **OmegaConf.to_container(launcher.hpcli_config),
                "expected_container_image": "",
            }
        )
        launcher._override_hub_container_image()
        launcher.boto_session.client.assert_not_called()

    def test_skips_when_image_already_matches(self):
        launcher = _make_launcher()
        expected = launcher.hpcli_config.expected_container_image

        mock_sm = MagicMock()
        launcher.boto_session.client.return_value = mock_sm
        mock_sm.describe_hub_content.return_value = {
            "HubContentDocument": f'{{"RecipeCollection": [{{"SmtjImageUri": "{expected}"}}]}}'
        }

        launcher._override_hub_container_image()
        # Always re-imports to ensure hub stays in sync
        mock_sm.import_hub_content.assert_called_once()

    def test_overrides_when_image_differs(self):
        launcher = _make_launcher()

        mock_sm = MagicMock()
        launcher.boto_session.client.return_value = mock_sm
        mock_sm.describe_hub_content.return_value = {
            "HubContentDocument": '{"RecipeCollection": [{"SmtjImageUri": "old-image:v1"}]}'
        }

        launcher._override_hub_container_image()

        mock_sm.import_hub_content.assert_called_once()
        call_kwargs = mock_sm.import_hub_content.call_args[1]
        assert launcher.hpcli_config.expected_container_image in call_kwargs["HubContentDocument"]


class TestLauncherLaunchJob:
    """Test the full launch_job pipeline (all steps mocked)."""

    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_launch_job_calls_all_steps(self, mock_cmd):
        """Verify init → configure → validate → create → describe (monitor) → delete (cleanup)."""
        # First 4 calls succeed (init, configure, validate, create)
        # 5th call is describe (returns "Succeeded")
        # 6th call is delete (cleanup)
        mock_cmd.side_effect = [
            CLIResult([], 0, "", ""),  # init
            CLIResult([], 0, "", ""),  # configure
            CLIResult([], 0, "", ""),  # validate
            CLIResult([], 0, "", ""),  # create
            CLIResult([], 0, "Status: Succeeded", ""),  # describe
            CLIResult([], 0, "", ""),  # delete (cleanup)
        ]

        launcher = _make_launcher()
        launcher._hub_overridden = True  # skip hub override

        result = launcher.launch_job("some-recipe.yaml")
        assert result is True
        assert mock_cmd.call_count == 6  # 4 pipeline + 1 describe + 1 delete

    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_launch_job_fails_on_init_failure(self, mock_cmd):
        mock_cmd.return_value = CLIResult([], 1, "", "init error")

        launcher = _make_launcher()
        launcher._hub_overridden = True

        result = launcher.launch_job("recipe.yaml")
        assert result is False
        launcher.job_recorder.update_job.assert_called()

    @patch("scripts.validations.validation_launchers.hpcli_launcher.run_hyp_command")
    def test_launch_job_fails_on_validate_failure(self, mock_cmd):
        mock_cmd.side_effect = [
            CLIResult([], 0, "", ""),  # init
            CLIResult([], 0, "", ""),  # configure
            CLIResult([], 1, "", "validation error"),  # validate fails
        ]

        launcher = _make_launcher()
        launcher._hub_overridden = True

        result = launcher.launch_job("recipe.yaml")
        assert result is False
