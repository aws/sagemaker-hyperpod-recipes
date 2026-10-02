"""Offline dry-run for template/sm_jobs_model_trainer.py.

Proves the use_training_recipe routing that fixes verl SMJOBS submissions on
sagemaker>=2.250.0 (SDK from_recipe hard-requires recipe.model.model_type, which
verl recipes omit). No AWS calls: the SageMaker session, execution role, and
ModelTrainer are all mocked.

  * use_training_recipe: true (verl)  -> ModelTrainer(...) built directly with a
    `recipe` input channel + sagemaker_recipe_local_path hyperparameter and NO
    source_code; from_recipe is never called (so the model_type check is never hit).
  * use_training_recipe absent (NeMo) -> ModelTrainer.from_recipe(...) is called
    (regression guard: the legacy path is unchanged).
"""

import importlib.util
import os
import sys
from unittest import mock

_TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "..", "template", "sm_jobs_model_trainer.py")


def _load_template_module():
    spec = importlib.util.spec_from_file_location("sm_jobs_model_trainer_under_test", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path, recipe_yaml, sm_jobs_yaml):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(recipe_yaml)
    sm_jobs = tmp_path / "sm_jobs_config.yaml"
    sm_jobs.write_text(sm_jobs_yaml)
    return str(recipe), str(sm_jobs)


def _run_main(mod, recipe_path, sm_jobs_path):
    """Run main() with the SDK boundary mocked; return the mocked ModelTrainer."""
    argv = [
        "prog",
        "--recipe",
        recipe_path,
        "--sm_jobs_config",
        sm_jobs_path,
        "--job_name",
        "verl-test-job",
        "--instance_type",
        "ml.p5.48xlarge",
    ]
    fake_session = mock.MagicMock()
    fake_session.default_bucket.return_value = "test-bucket"
    with mock.patch.object(sys, "argv", argv), mock.patch.object(
        mod.sagemaker, "Session", return_value=fake_session
    ), mock.patch.object(
        mod.sagemaker, "get_execution_role", return_value="arn:aws:iam::111122223333:role/test"
    ), mock.patch.object(
        mod, "ModulesSession", mock.MagicMock()
    ), mock.patch.object(
        mod, "ModelTrainer"
    ) as MockTrainer:
        mod.main()
    return MockTrainer


IMAGE = "123456789012.dkr.ecr.us-west-2.amazonaws.com/verl:latest"


def test_verl_uses_container_entrypoint_not_from_recipe(tmp_path):
    recipe_path, sm_jobs_path = _write(
        tmp_path,
        recipe_yaml="run:\n  model_type: verl\ntrainer:\n  num_nodes: 1\n",
        sm_jobs_yaml=(
            "output_path: s3://test/out\n"
            "inputs:\n  s3:\n    train: s3://test/train\n"
            "additional_estimator_kwargs:\n"
            "  use_training_recipe: true\n"
            f"  training_image: {IMAGE}\n"
            "wait: false\n"
        ),
    )
    mod = _load_template_module()
    MockTrainer = _run_main(mod, recipe_path, sm_jobs_path)

    # from_recipe (the model_type-tripping path) is NEVER used for verl.
    MockTrainer.from_recipe.assert_not_called()

    # ModelTrainer is constructed directly, exactly once, and trained.
    MockTrainer.assert_called_once()
    _, kwargs = MockTrainer.call_args

    # No source_code -> the container's own ENTRYPOINT runs.
    assert "source_code" not in kwargs

    assert kwargs["training_image"] == IMAGE

    # The recipe is shipped as the `recipe` input channel.
    channels = {getattr(c, "channel_name", None) for c in kwargs["input_data_config"]}
    assert mod.SM_RECIPE in channels

    # ...and pointed at via the hyperparameter the container reads.
    assert kwargs["hyperparameters"]["sagemaker_recipe_local_path"] == mod.SM_RECIPE_CONTAINER_PATH

    MockTrainer.return_value.train.assert_called_once()


def test_nemo_still_uses_from_recipe(tmp_path):
    recipe_path, sm_jobs_path = _write(
        tmp_path,
        recipe_yaml="model:\n  model_type: llama\n",
        sm_jobs_yaml=(
            "output_path: s3://test/out\n"
            "inputs:\n  s3:\n    train: s3://test/train\n"
            "additional_estimator_kwargs:\n"
            f"  training_image: {IMAGE}\n"
            "wait: false\n"
        ),
    )
    mod = _load_template_module()
    MockTrainer = _run_main(mod, recipe_path, sm_jobs_path)

    # Legacy NeMo/LLMFT path is unchanged: from_recipe drives the submission.
    MockTrainer.from_recipe.assert_called_once()
    # ModelTrainer() is not constructed directly on this path.
    MockTrainer.assert_not_called()
    MockTrainer.from_recipe.return_value.train.assert_called_once()
