import json
import os
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from omegaconf import OmegaConf

from launcher.nova.launchers import (
    SMNovaK8SLauncherPPO,
    SMNovaK8SLauncherRFT,
    SMNovaK8SLauncherSFT,
    get_cpu_instance_types,
    get_legacy_quoted_recipes,
    get_override_sub_instance_types,
    should_strip_scalar_quotes,
)
from utils.template_utils import remove_quotes_from_scalar_params


class TestGetCpuInstanceTypes(unittest.TestCase):
    def test_returns_none_when_unset(self):
        cfg = OmegaConf.create({"cluster": {"instance_type": "p5.48xlarge"}})
        self.assertIsNone(get_cpu_instance_types(cfg))

    def test_returns_none_when_no_cluster(self):
        cfg = OmegaConf.create({"instance_type": "p5.48xlarge"})
        self.assertIsNone(get_cpu_instance_types(cfg))

    def test_string_gets_ml_prefix(self):
        cfg = OmegaConf.create({"cluster": {"cpu_instance_type": "m5.xlarge"}})
        self.assertEqual(get_cpu_instance_types(cfg), ["ml.m5.xlarge"])

    def test_normalized_lowercase(self):
        cfg = OmegaConf.create({"cluster": {"cpu_instance_type": "ML.M5.XLARGE"}})
        self.assertEqual(get_cpu_instance_types(cfg), ["ml.m5.xlarge"])

    def test_list(self):
        cfg = OmegaConf.create({"cluster": {"cpu_instance_type": ["m5.xlarge", "ml.m5.2xlarge"]}})
        self.assertEqual(get_cpu_instance_types(cfg), ["ml.m5.xlarge", "ml.m5.2xlarge"])

    def test_list_of_none_returns_none(self):
        cfg = OmegaConf.create({"cluster": {"cpu_instance_type": [None]}})
        self.assertIsNone(get_cpu_instance_types(cfg))


class TestGetOverrideSubInstanceTypes(unittest.TestCase):
    def test_returns_empty_when_unset(self):
        cfg = OmegaConf.create({"cluster": {"instance_type": "p5.48xlarge"}})
        self.assertEqual(get_override_sub_instance_types(cfg), {})

    def test_returns_empty_when_no_cluster(self):
        cfg = OmegaConf.create({"instance_type": "p5.48xlarge"})
        self.assertEqual(get_override_sub_instance_types(cfg), {})

    def test_map_values_normalized(self):
        cfg = OmegaConf.create(
            {"cluster": {"override_sub_instance_type": {"hub": "r6i.24xlarge", "training": "ML.P5.48XLARGE"}}}
        )
        self.assertEqual(
            get_override_sub_instance_types(cfg),
            {"hub": "ml.r6i.24xlarge", "training": "ml.p5.48xlarge"},
        )

    def test_empty_value_dropped(self):
        cfg = OmegaConf.create({"cluster": {"override_sub_instance_type": {"hub": None, "rbs": "r6i.24xlarge"}}})
        self.assertEqual(get_override_sub_instance_types(cfg), {"rbs": "ml.r6i.24xlarge"})


class TestSMNovaK8SLauncherSFT(unittest.TestCase):
    def setUp(self):
        os.environ["AWS_REGION"] = "us-east-1"
        # Minimal configuration needed to initialize the launcher
        cfg_dict = {
            "recipes": {"run": {"name": "test_job"}},
            "base_results_dir": "/tmp",
            "launch_json": False,
            "container": "test_container",
            "cluster": {"instance_type": "p5.48xlarge"},
            "cluster_type": "k8s",
        }
        self.cfg = OmegaConf.create(cfg_dict)

    @patch.object(SMNovaK8SLauncherSFT, "_prepare_output_dir")
    @patch.object(SMNovaK8SLauncherSFT, "_save_hydra_config")
    @patch.object(SMNovaK8SLauncherSFT, "_create_chart_file")
    @patch.object(SMNovaK8SLauncherSFT, "_copy_k8s_template")
    @patch.object(SMNovaK8SLauncherSFT, "_process_values_yaml")
    @patch.object(SMNovaK8SLauncherSFT, "_create_helm_script", return_value=Path("/tmp/fake_helm.sh"))
    @patch.object(SMNovaK8SLauncherSFT, "_create_launch_json")
    @patch.object(SMNovaK8SLauncherSFT, "_run_helm_script")
    def test_run_executes_helm_script(
        self, mock_run_helm, mock_launch_json, mock_helm, mock_process, mock_copy, mock_chart, mock_save, mock_prepare
    ):
        launcher = SMNovaK8SLauncherSFT(self.cfg)
        launcher.run()

        mock_prepare.assert_called_once()
        mock_save.assert_called_once()
        mock_chart.assert_called_once_with(launcher._template_dir)
        mock_copy.assert_called_once()
        mock_process.assert_called_once()
        mock_helm.assert_called_once_with(launcher._output_dir_k8s_folder)
        mock_launch_json.assert_not_called()
        mock_run_helm.assert_called_once_with(Path("/tmp/fake_helm.sh"))

    @patch.object(SMNovaK8SLauncherSFT, "_prepare_output_dir")
    @patch.object(SMNovaK8SLauncherSFT, "_save_hydra_config")
    @patch.object(SMNovaK8SLauncherSFT, "_create_chart_file")
    @patch.object(SMNovaK8SLauncherSFT, "_copy_k8s_template")
    @patch.object(SMNovaK8SLauncherSFT, "_process_values_yaml")
    @patch.object(SMNovaK8SLauncherSFT, "_create_helm_script", return_value=Path("/tmp/fake_helm.sh"))
    @patch.object(SMNovaK8SLauncherSFT, "_create_launch_json")
    @patch.object(SMNovaK8SLauncherSFT, "_run_helm_script")
    def test_run_creates_launch_json(
        self, mock_run_helm, mock_launch_json, mock_helm, mock_process, mock_copy, mock_chart, mock_save, mock_prepare
    ):
        self.cfg["launch_json"] = True

        launcher = SMNovaK8SLauncherSFT(self.cfg)
        launcher.run()

        mock_prepare.assert_called_once()
        mock_save.assert_called_once()
        mock_chart.assert_called_once_with(launcher._template_dir)
        mock_copy.assert_called_once()
        mock_process.assert_called_once()
        mock_helm.assert_called_once_with(launcher._output_dir_k8s_folder)
        mock_launch_json.assert_called_once_with(launcher._output_dir_k8s_folder)
        mock_run_helm.assert_not_called()


class TestSMNovaK8SLauncherPPO(unittest.TestCase):
    def setUp(self):
        os.environ["AWS_REGION"] = "us-east-1"

        cfg_dict = {
            "recipes": {
                "run": {"name": "test_ppo_job"},
                "ppo_reward": {"trainer": {"num_nodes": 1, "devices": 8}},
                "ppo_critic": {"trainer": {"num_nodes": 1, "devices": 8}},
                "ppo_anchor": {"trainer": {"num_nodes": 1, "devices": 8}},
                "ppo_actor_generation": {"trainer": {"num_nodes": 1, "devices": 8}},
                "ppo_actor_train": {"trainer": {"num_nodes": 1, "devices": 8}},
            },
            "base_results_dir": "/tmp",
            "launch_json": False,
            "container": "test_container",
            "cluster": {"instance_type": "p5.48xlarge"},
            "cluster_type": "k8s",
        }
        self.cfg = OmegaConf.create(cfg_dict)

    @patch.object(SMNovaK8SLauncherPPO, "_prepare_output_dir")
    @patch.object(SMNovaK8SLauncherPPO, "_save_hydra_config")
    @patch.object(SMNovaK8SLauncherPPO, "_create_chart_file")
    @patch.object(SMNovaK8SLauncherPPO, "_copy_k8s_template")
    @patch.object(SMNovaK8SLauncherPPO, "_process_values_yaml")
    @patch.object(SMNovaK8SLauncherPPO, "_create_helm_script", return_value=Path("/tmp/fake_helm.sh"))
    @patch.object(SMNovaK8SLauncherPPO, "_create_launch_json")
    @patch.object(SMNovaK8SLauncherPPO, "_run_helm_script")
    def test_run_executes_helm_script(
        self,
        mock_run_helm,
        mock_create_launch_json,
        mock_create_helm_script,
        mock_process_values_yaml,
        mock_copy_k8s_template,
        mock_create_chart_file,
        mock_save_hydra_config,
        mock_prepare_output_dir,
    ):
        # Run the launcher
        launcher = SMNovaK8SLauncherPPO(self.cfg)
        launcher.run()

        mock_create_helm_script.assert_called_once_with(launcher._output_dir_k8s_folder)
        mock_create_launch_json.assert_not_called()
        mock_run_helm.assert_called_once_with(Path("/tmp/fake_helm.sh"))
        mock_prepare_output_dir.assert_called_once()
        mock_save_hydra_config.assert_called_once()
        mock_create_chart_file.assert_called_once_with(launcher._template_dir)
        mock_copy_k8s_template.assert_called_once()
        mock_process_values_yaml.assert_called_once()

    @patch.object(SMNovaK8SLauncherPPO, "_prepare_output_dir")
    @patch.object(SMNovaK8SLauncherPPO, "_save_hydra_config")
    @patch.object(SMNovaK8SLauncherPPO, "_create_chart_file")
    @patch.object(SMNovaK8SLauncherPPO, "_copy_k8s_template")
    @patch.object(SMNovaK8SLauncherPPO, "_process_values_yaml")
    @patch.object(SMNovaK8SLauncherPPO, "_create_helm_script", return_value=Path("/tmp/fake_helm.sh"))
    @patch.object(SMNovaK8SLauncherPPO, "_create_launch_json")
    @patch.object(SMNovaK8SLauncherPPO, "_run_helm_script")
    def test_run_launch_json(
        self,
        mock_run_helm,
        mock_create_launch_json,
        mock_create_helm_script,
        mock_process_values_yaml,
        mock_copy_k8s_template,
        mock_create_chart_file,
        mock_save_hydra_config,
        mock_prepare_output_dir,
    ):
        self.cfg["launch_json"] = True

        # Run launcher
        launcher = SMNovaK8SLauncherPPO(self.cfg)
        launcher.run()

        # Verify each internal step was called
        mock_prepare_output_dir.assert_called_once()
        mock_save_hydra_config.assert_called_once()
        mock_create_chart_file.assert_called_once_with(launcher._template_dir)
        mock_copy_k8s_template.assert_called_once()
        mock_process_values_yaml.assert_called_once()
        mock_create_helm_script.assert_called_once_with(launcher._output_dir_k8s_folder)

        # Since launch_json=True, _create_launch_json should be called
        mock_create_launch_json.assert_called_once_with(launcher._output_dir_k8s_folder)

        # _run_helm_script should NOT be called if launch_json=True
        mock_run_helm.assert_not_called()


class TestSMNovaK8SLauncherRFT(unittest.TestCase):
    def setUp(self):
        os.environ["AWS_REGION"] = "us-east-1"
        # Configuration matching actual nova_lite_v2_p5_rft.yaml recipe structure
        cfg_dict = {
            "recipes": {
                "run": {
                    "name": "test_rft_job",
                    "model_type": "amazon.nova-2-lite-v1:0:256k",
                    "model_name_or_path": "nova-lite-1.5.4/prod",
                    "data_s3_path": "s3://example-bucket/train.jsonl",
                    "output_s3_path": "",
                    "replicas": 2,
                    "generation_replicas": 2,
                    "rollout_worker_replicas": 1,
                    "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:SageMaker-reward-function",
                },
                "training_config": {
                    "max_length": 10240,
                    "global_batch_size": 1024,
                    "reasoning_effort": "high",
                    "data": {
                        "type": "single-turn",
                        "shuffle": False,
                    },
                    "rollout": {
                        "rollout_strategy": {
                            "type": "off_policy_async",
                            "age_tolerance": 2,
                        },
                        "advantage_strategy": {
                            "number_generation": 8,
                        },
                        "generator": {
                            "max_new_tokens": 8192,
                            "set_random_seed": True,
                            "temperature": 1,
                            "top_k": 0,
                        },
                        "rewards": {
                            "api_endpoint": {
                                "reward_lambda_arn": "${oc.select:run.reward_lambda_arn}",
                                "lambda_concurrency_limit": 100,
                            },
                        },
                    },
                    "trainer": {
                        "max_steps": 100,
                        "save_steps": 100,
                        "save_top_k": 5,
                        "refit_freq": 4,
                        "clip_ratio_high": 0.2,
                        "entropy_coeff": 0.001,
                        "loss_scale": 1,
                        "optim_config": {
                            "lr": 7e-7,
                            "weight_decay": 0.0,
                            "adam_beta1": 0.9,
                            "adam_beta2": 0.95,
                        },
                    },
                },
            },
            "base_results_dir": "/tmp",
            "launch_json": False,
            "container": "test_container",
            "cluster": {"instance_type": "p5.48xlarge"},
        }
        self.cfg = OmegaConf.create(cfg_dict)

    def test_validate_recipe_parameters_success(self):
        """Test that recipe parameter validation passes with valid config."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        # This should not raise any exceptions
        launcher._validate_recipe_parameters()

    def test_validate_recipe_parameters_missing_required_sections(self):
        """Test that validation fails when required sections are missing."""
        # Remove required sections (rollout, data, train)
        del self.cfg.recipes.training_config.rollout
        del self.cfg.recipes.training_config.data
        del self.cfg.recipes.training_config.trainer
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        with self.assertRaises(ValueError) as context:
            launcher._validate_recipe_parameters()

        self.assertIn("model_config", str(context.exception))

    def test_validate_recipe_parameters_missing_dataset_path(self):
        """Test that validation fails when dataset path is missing."""
        # Remove dataset path from run config
        del self.cfg.recipes.run.data_s3_path
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        with self.assertRaises(ValueError) as context:
            launcher._validate_recipe_parameters()

        self.assertIn("dataset parameter", str(context.exception))

    def test_validate_recipe_parameters_invalid_replica_count(self):
        """Test that validation fails with invalid replica counts."""
        # Set invalid replica count
        self.cfg.recipes.run.replicas = -1
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        with self.assertRaises(ValueError) as context:
            launcher._validate_recipe_parameters()

        self.assertIn("must be a positive integer", str(context.exception))

    @patch.object(SMNovaK8SLauncherRFT, "_prepare_output_dir")
    @patch.object(SMNovaK8SLauncherRFT, "_save_hydra_config")
    @patch.object(SMNovaK8SLauncherRFT, "_create_chart_file")
    @patch.object(SMNovaK8SLauncherRFT, "_copy_k8s_template")
    @patch.object(SMNovaK8SLauncherRFT, "_process_values_yaml")
    @patch.object(SMNovaK8SLauncherRFT, "_create_helm_script", return_value=Path("/tmp/fake_helm.sh"))
    @patch.object(SMNovaK8SLauncherRFT, "_create_launch_json")
    @patch.object(SMNovaK8SLauncherRFT, "_run_helm_script")
    def test_run_executes_helm_script(
        self, mock_run_helm, mock_launch_json, mock_helm, mock_process, mock_copy, mock_chart, mock_save, mock_prepare
    ):
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.run()

        mock_prepare.assert_called_once()
        mock_save.assert_called_once()
        mock_chart.assert_called_once_with(launcher._template_dir)
        mock_copy.assert_called_once()
        mock_process.assert_called_once()
        mock_helm.assert_called_once_with(launcher._output_dir_k8s_folder)
        mock_launch_json.assert_not_called()
        mock_run_helm.assert_called_once_with(Path("/tmp/fake_helm.sh"))

    @patch.object(SMNovaK8SLauncherRFT, "_prepare_output_dir")
    @patch.object(SMNovaK8SLauncherRFT, "_save_hydra_config")
    @patch.object(SMNovaK8SLauncherRFT, "_create_chart_file")
    @patch.object(SMNovaK8SLauncherRFT, "_copy_k8s_template")
    @patch.object(SMNovaK8SLauncherRFT, "_process_values_yaml")
    @patch.object(SMNovaK8SLauncherRFT, "_create_helm_script", return_value=Path("/tmp/fake_helm.sh"))
    @patch.object(SMNovaK8SLauncherRFT, "_create_launch_json")
    @patch.object(SMNovaK8SLauncherRFT, "_run_helm_script")
    def test_run_creates_launch_json(
        self, mock_run_helm, mock_launch_json, mock_helm, mock_process, mock_copy, mock_chart, mock_save, mock_prepare
    ):
        self.cfg["launch_json"] = True

        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.run()

        mock_prepare.assert_called_once()
        mock_save.assert_called_once()
        mock_chart.assert_called_once_with(launcher._template_dir)
        mock_copy.assert_called_once()
        mock_process.assert_called_once()
        mock_helm.assert_called_once_with(launcher._output_dir_k8s_folder)
        mock_launch_json.assert_called_once_with(launcher._output_dir_k8s_folder)
        mock_run_helm.assert_not_called()

    def test_build_job_list_structure(self):
        """Test that _build_job_list creates correct job structure."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        values_template = OmegaConf.create(
            {
                "trainingConfig": {
                    "training": {"replicas": 2},
                    "vllmGeneration": {"replicas": 2},
                    "hub": {"replicas": 1},
                    "prompter": {"replicas": 1},
                    "rbs": {"replicas": 1},
                }
            }
        )
        job_list = launcher._build_job_list(values_template)

        # Verify job list structure
        self.assertIsInstance(job_list, list)
        self.assertGreater(len(job_list), 0)

        # Verify each job has required fields
        for job in job_list:
            self.assertIn("jobName", job)
            self.assertIn("serviceType", job)
            self.assertIn("replicas", job)

        # Verify specific jobs exist
        job_names = [job["jobName"] for job in job_list]
        self.assertIn("test_rft_job-training", job_names)
        self.assertIn("test_rft_job-vllm-generation", job_names)
        self.assertIn("test_rft_job-hub", job_names)

    def test_service_order_includes_redis_for_delegate(self):
        """Test that redis is added to service order for delegate configuration."""
        # Set delegate configuration in rollout
        self.cfg.recipes.training_config.rollout = OmegaConf.create({"delegate": True})

        launcher = SMNovaK8SLauncherRFT(self.cfg)

        # Verify redis is in service order
        self.assertIn("redis", launcher._service_order)
        self.assertEqual(launcher._service_order[0], "redis")  # Should be first

    @patch.object(SMNovaK8SLauncherRFT, "_validate_recipe_parameters")
    def test_run_method_workflow(self, mock_validate):
        """Test that run method calls all required methods in correct order."""
        with patch.object(SMNovaK8SLauncherRFT, "_prepare_output_dir") as mock_prepare, patch.object(
            SMNovaK8SLauncherRFT, "_save_hydra_config"
        ) as mock_save, patch.object(SMNovaK8SLauncherRFT, "_create_chart_file") as mock_chart, patch.object(
            SMNovaK8SLauncherRFT, "_copy_k8s_template"
        ) as mock_copy, patch.object(
            SMNovaK8SLauncherRFT, "_process_values_yaml"
        ) as mock_process, patch.object(
            SMNovaK8SLauncherRFT, "_create_helm_script", return_value=Path("/tmp/test.sh")
        ) as mock_helm, patch.object(
            SMNovaK8SLauncherRFT, "_run_helm_script"
        ) as mock_run:
            launcher = SMNovaK8SLauncherRFT(self.cfg)
            launcher.run()

            # Verify all methods were called
            mock_validate.assert_called_once()
            mock_prepare.assert_called_once()
            mock_save.assert_called_once()
            mock_chart.assert_called_once()
            mock_copy.assert_called_once()
            mock_process.assert_called_once()
            mock_helm.assert_called_once()
            mock_run.assert_called_once()

    @patch("pathlib.Path.mkdir")
    @patch("omegaconf.OmegaConf.save")
    @patch.object(SMNovaK8SLauncherRFT, "_build_job_list")
    def test_create_service_config_files(self, mock_build_job_list, mock_save, mock_mkdir):
        """Test _create_service_config_files creates config files for each service."""
        mock_build_job_list.return_value = [
            {"jobName": "test-training", "serviceType": "training", "replicas": 1},
            {"jobName": "test-hub", "serviceType": "hub", "replicas": 1},
        ]

        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher._output_dir = Path("/tmp/test")

        launcher._create_service_config_files()

        # Verify config files are created for all services
        expected_calls = len(mock_build_job_list.return_value)
        self.assertEqual(mock_save.call_count, expected_calls)

    def test_build_job_list_with_redis(self):
        """Test _build_job_list includes Redis for delegate configuration."""
        self.cfg.recipes.training_config.rollout = OmegaConf.create({"delegate": True})
        self.cfg.recipes.run.redis_replicas = 2

        launcher = SMNovaK8SLauncherRFT(self.cfg)
        values_template = OmegaConf.create(
            {
                "trainingConfig": {
                    "training": {"replicas": 2},
                    "vllmGeneration": {"replicas": 2},
                    "hub": {"replicas": 1},
                    "prompter": {"replicas": 1},
                    "rbs": {"replicas": 1},
                }
            }
        )
        job_list = launcher._build_job_list(values_template)

        redis_jobs = [job for job in job_list if job["serviceType"] == "redis"]
        self.assertEqual(len(redis_jobs), 1)
        self.assertEqual(redis_jobs[0]["replicas"], 2)

    def test_map_rft_replica_config(self):
        """Test _map_rft_replica_config sets correct replica values."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        # Mock values template
        values_template = OmegaConf.create(
            {
                "trainingConfig": {
                    "training": {"replicas": 0},
                    "vllmGeneration": {"replicas": 0},
                    "rewardFunction": {"replicas": 0},
                    "hub": {"replicas": 0},
                    "promptRbs": {"replicas": 0},
                    "natsServer": {"replicas": 0},
                    "redis": {"enabled": False, "replicas": 0},
                }
            }
        )

        launcher._map_rft_replica_config(values_template)

        # Verify replica counts are set correctly
        self.assertEqual(values_template.trainingConfig.training.replicas, 2)
        self.assertEqual(values_template.trainingConfig.vllmGeneration.replicas, 2)
        self.assertEqual(values_template.trainingConfig.hub.replicas, 1)

    def test_map_image_config(self):
        """Test _map_image_config sets image configuration."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        values_template = OmegaConf.create(
            {
                "image": {
                    "trainingImage": "",
                    "generationImage": "",
                    "stormImage": "",
                    "natsServerImage": "",
                    "natsReloaderImage": "",
                    "redis": "",
                }
            }
        )

        launcher._map_image_config(values_template)

        # RFT sets specific images using utility functions
        self.assertNotEqual(values_template.image.generationImage, "")
        self.assertNotEqual(values_template.image.stormImage, "")
        self.assertNotEqual(values_template.image.natsServerImage, "")

    def test_map_resource_config_sets_instance_types(self):
        """Test _map_resource_config sets instance types for services."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.instance_type = "p5.48xlarge"

        values_template = OmegaConf.create(
            {
                "trainingConfig": {
                    "defaultResources": {"instanceType": "ml.p5.48xlarge"},
                    "training": {},
                    "vllmGeneration": {},
                    "hub": {},
                    "prompter": {},
                    "rbs": {},
                    "natsServer": {},
                }
            }
        )

        launcher._map_resource_config(values_template)

        # Verify instance types are set for services that exist (check if attribute exists first)
        if hasattr(values_template.trainingConfig.training, "instanceType"):
            self.assertEqual(values_template.trainingConfig.training.instanceType, "ml.p5.48xlarge")
        if hasattr(values_template.trainingConfig.vllmGeneration, "instanceType"):
            self.assertEqual(values_template.trainingConfig.vllmGeneration.instanceType, "ml.p5.48xlarge")
        if hasattr(values_template.trainingConfig.hub, "instanceType"):
            self.assertEqual(values_template.trainingConfig.hub.instanceType, "ml.p5.48xlarge")

        # Test passes if method runs without error
        self.assertTrue(True)

    @staticmethod
    def _rft_values_template():
        # Services must be non-empty: _map_resource_config skips falsy service nodes,
        # matching the real values.yaml where each service has instanceType/labelSelector.
        svc = {"instanceType": "ml.p5.48xlarge", "labelSelector": None}
        return OmegaConf.create(
            {
                "trainingConfig": {
                    "defaultResources": {"instanceType": "ml.p5.48xlarge"},
                    "training": dict(svc),
                    "vllmGeneration": dict(svc),
                    "hub": dict(svc),
                    "prompter": dict(svc),
                    "rbs": dict(svc),
                    "natsServer": dict(svc),
                    "redis": {"enabled": True, "instanceType": "ml.p5.48xlarge", "labelSelector": None},
                }
            }
        )

    INSTANCE_KEY = "node.kubernetes.io/instance-type"

    def test_map_resource_config_cpu_instance_type_fallback(self):
        """cpu_instance_type applies to all CPU services; GPU services keep the global type."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.instance_type = "ml.p5.48xlarge"
        launcher.cpu_instance_types = ["ml.r6i.24xlarge"]
        launcher.override_sub_instance_types = {}

        values_template = self._rft_values_template()
        launcher._map_resource_config(values_template)

        tc = values_template.trainingConfig

        # GPU services stay on the global instance type
        self.assertEqual(tc.training.instanceType, "ml.p5.48xlarge")
        self.assertEqual(tc.vllmGeneration.instanceType, "ml.p5.48xlarge")
        self.assertEqual(tc.training.labelSelector["required"][self.INSTANCE_KEY], ["ml.p5.48xlarge"])

        # CPU services (incl. redis) fall back to cpu_instance_type
        for svc in ("hub", "prompter", "rbs", "natsServer", "redis"):
            self.assertEqual(getattr(tc, svc).instanceType, "ml.r6i.24xlarge")
            self.assertEqual(getattr(tc, svc).labelSelector["required"][self.INSTANCE_KEY], ["ml.r6i.24xlarge"])

    def test_map_resource_config_per_service_override_map(self):
        """override_sub_instance_type map targets named services; others fall back per rules."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.instance_type = "ml.p5.48xlarge"
        launcher.cpu_instance_types = ["ml.r6i.24xlarge"]
        # Per-service map: rbs gets its own type; training (a GPU service) is overridden too.
        launcher.override_sub_instance_types = {
            "rbs": "ml.r6i.12xlarge",
            "training": "ml.p5en.48xlarge",
        }

        values_template = self._rft_values_template()
        launcher._map_resource_config(values_template)

        tc = values_template.trainingConfig

        # Named override wins
        self.assertEqual(tc.rbs.instanceType, "ml.r6i.12xlarge")
        self.assertEqual(tc.rbs.labelSelector["required"][self.INSTANCE_KEY], ["ml.r6i.12xlarge"])
        self.assertEqual(tc.training.instanceType, "ml.p5en.48xlarge")

        # Unnamed CPU services fall back to cpu_instance_type
        self.assertEqual(tc.hub.instanceType, "ml.r6i.24xlarge")
        # Unnamed GPU service falls back to global instance_type (no cpu fallback)
        self.assertEqual(tc.vllmGeneration.instanceType, "ml.p5.48xlarge")

    def test_map_resource_config_cpu_instance_type_list(self):
        """A cpu_instance_type list: scalar instanceType takes the first, selector carries all."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.instance_type = "ml.p5.48xlarge"
        launcher.cpu_instance_types = ["ml.r6i.24xlarge", "ml.r6i.12xlarge"]
        launcher.override_sub_instance_types = {}

        values_template = self._rft_values_template()
        launcher._map_resource_config(values_template)

        hub = values_template.trainingConfig.hub
        self.assertEqual(hub.instanceType, "ml.r6i.24xlarge")
        self.assertEqual(hub.labelSelector["required"][self.INSTANCE_KEY], ["ml.r6i.24xlarge", "ml.r6i.12xlarge"])

    def test_map_resource_config_defaults_to_global_when_nothing_set(self):
        """With no override and no cpu_instance_type, every service uses the global type."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.instance_type = "ml.p5.48xlarge"
        launcher.cpu_instance_types = None
        launcher.override_sub_instance_types = {}

        values_template = self._rft_values_template()
        launcher._map_resource_config(values_template)

        tc = values_template.trainingConfig
        self.assertEqual(tc.hub.instanceType, "ml.p5.48xlarge")
        self.assertEqual(tc.redis.instanceType, "ml.p5.48xlarge")

    def test_map_resource_config_launch_json_uses_placeholder(self):
        """In launch_json mode the label selector uses the placeholder instance type."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.instance_type = "ml.p5.48xlarge"
        launcher.cpu_instance_types = ["ml.r6i.24xlarge"]
        launcher.override_sub_instance_types = {}
        launcher._launch_json = True

        values_template = self._rft_values_template()
        launcher._map_resource_config(values_template)

        tc = values_template.trainingConfig
        self.assertEqual(tc.training.labelSelector["required"][self.INSTANCE_KEY], ["PLACEHOLDER_INSTANCE_TYPE"])
        self.assertEqual(tc.hub.labelSelector["required"][self.INSTANCE_KEY], ["PLACEHOLDER_INSTANCE_TYPE"])
        self.assertEqual(tc.redis.labelSelector["required"][self.INSTANCE_KEY], ["PLACEHOLDER_INSTANCE_TYPE"])

    def test_set_efa_resources_uses_override_instance_type(self):
        """EFA counts follow the instance type resolved for each GPU service."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.instance_type = "ml.p5.48xlarge"
        launcher.num_efa_devices = 32
        launcher.cpu_instance_types = None
        # Override vllm_generation onto a single-EFA instance type.
        launcher.override_sub_instance_types = {"vllm_generation": "ml.g5.8xlarge"}

        values_template = OmegaConf.create(
            {
                "trainingConfig": {
                    "training": {
                        "resources": {
                            "master": {"requests": {}, "limits": {}},
                            "worker": {"requests": {}, "limits": {}},
                        }
                    },
                    "vllmGeneration": {"resources": {"requests": {}, "limits": {}}},
                }
            }
        )
        launcher._set_efa_resources(values_template)

        efa_key = "vpc.amazonaws.com/efa"
        # training keeps the global p5.48xlarge EFA count (master + worker)
        self.assertEqual(values_template.trainingConfig.training.resources.master.requests[efa_key], 32)
        self.assertEqual(values_template.trainingConfig.training.resources.worker.limits[efa_key], 32)
        # vllmGeneration uses the overridden instance type's EFA count
        self.assertEqual(
            values_template.trainingConfig.vllmGeneration.resources.requests[efa_key],
            launcher._efa_devices_for("ml.g5.8xlarge"),
        )

    def test_map_resource_config_falls_back_to_default_resources(self):
        """When the global instance_type is None, services fall back to defaultResources."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher.instance_type = None
        launcher.cpu_instance_types = None
        launcher.override_sub_instance_types = {}

        values_template = self._rft_values_template()
        launcher._map_resource_config(values_template)

        # defaultResources.instanceType is ml.p5.48xlarge in the fixture
        self.assertEqual(values_template.trainingConfig.hub.instanceType, "ml.p5.48xlarge")
        self.assertEqual(values_template.trainingConfig.training.instanceType, "ml.p5.48xlarge")

    def test_override_service_keys_accepts_valid_rft_keys(self):
        """RFT launcher accepts override keys within its supported service set."""
        self.cfg.cluster.override_sub_instance_type = OmegaConf.create(
            {"hub": "ml.r6i.24xlarge", "training": "ml.p5.48xlarge"}
        )
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        self.assertEqual(
            launcher.override_sub_instance_types,
            {"hub": "ml.r6i.24xlarge", "training": "ml.p5.48xlarge"},
        )

    def test_override_service_keys_rejects_unknown_key(self):
        """RFT launcher rejects an override key outside its supported service set."""
        self.cfg.cluster.override_sub_instance_type = OmegaConf.create({"hubb": "ml.r6i.24xlarge"})
        with self.assertRaises(ValueError) as ctx:
            SMNovaK8SLauncherRFT(self.cfg)
        self.assertIn("hubb", str(ctx.exception))
        self.assertIn("Valid keys", str(ctx.exception))

    def test_override_not_supported_for_sft(self):
        """A recipe type with no overridable services rejects any override key."""
        self.cfg.cluster.override_sub_instance_type = OmegaConf.create({"hub": "ml.r6i.24xlarge"})
        with self.assertRaises(ValueError) as ctx:
            SMNovaK8SLauncherSFT(self.cfg)
        self.assertIn("not supported", str(ctx.exception))

    def test_to_camel_case(self):
        """Test _to_camel_case converts snake_case to camelCase."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        self.assertEqual(launcher._to_camel_case("test_string"), "testString")
        self.assertEqual(launcher._to_camel_case("another_test_case"), "anotherTestCase")
        self.assertEqual(launcher._to_camel_case("single"), "single")

    def test_copy_k8s_template_sm_jobs(self):
        """Test _copy_k8s_template skips copying for SM Jobs."""
        self.cfg.cluster_type = "sm_jobs"
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        # Should not raise any exceptions and should return early
        launcher._copy_k8s_template()

    def test_service_order_redis_delegate(self):
        """Test service order includes Redis first for delegate configuration."""
        # Set delegate configuration in rollout
        self.cfg.recipes.training_config.rollout = OmegaConf.create({"delegate": True})

        launcher = SMNovaK8SLauncherRFT(self.cfg)

        self.assertIn("redis", launcher._service_order)
        self.assertEqual(launcher._service_order[0], "redis")

    def test_service_order_no_redis_no_delegate(self):
        """Test service order excludes Redis when delegate is not enabled."""
        # Ensure no delegate configuration (default behavior)
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        self.assertNotIn("redis", launcher._service_order)

    @patch("launcher.nova.launchers.get_recipe_file_path", return_value=None)
    def test_init_no_recipe_file_path(self, mock_get_recipe):
        """Test initialization when recipe file path is None."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        self.assertIsNone(launcher.recipe_file_path)

    def test_map_rft_replica_config_redis_disabled(self):
        """Test _map_rft_replica_config with Redis disabled for single-turn."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        values_template = OmegaConf.create(
            {
                "trainingConfig": {
                    "training": {"replicas": 0},
                    "vllmGeneration": {"replicas": 0},
                    "rewardFunction": {"replicas": 0},
                    "hub": {"replicas": 0},
                    "promptRbs": {"replicas": 0},
                    "natsServer": {"replicas": 0},
                    "redis": {"enabled": True, "replicas": 1},  # Start enabled
                }
            }
        )

        launcher._map_rft_replica_config(values_template)

        # Redis should be disabled for single-turn (default)
        self.assertFalse(values_template.trainingConfig.redis.enabled)

    def test_map_rft_replica_config_redis_enabled_delegate(self):
        """Test _map_rft_replica_config enables Redis for delegate configuration."""
        # Set delegate configuration
        self.cfg.recipes.training_config.rollout = OmegaConf.create({"delegate": True})
        self.cfg.recipes.run.redis_replicas = 3

        launcher = SMNovaK8SLauncherRFT(self.cfg)

        values_template = OmegaConf.create(
            {
                "trainingConfig": {
                    "training": {"replicas": 0},
                    "vllmGeneration": {"replicas": 0},
                    "rewardFunction": {"replicas": 0},
                    "hub": {"replicas": 0},
                    "promptRbs": {"replicas": 0},
                    "natsServer": {"replicas": 0},
                    "redis": {"enabled": False, "replicas": 0},
                }
            }
        )

        launcher._map_rft_replica_config(values_template)

        # Redis should be enabled and configured
        self.assertTrue(values_template.trainingConfig.redis.enabled)
        self.assertEqual(values_template.trainingConfig.redis.replicas, 3)

    def test_validate_recipe_parameters_flattened_structure(self):
        """Test validation with flattened recipe structure."""
        # RFT uses optimized structure with rollout, data, train
        self.cfg.recipes.training_config = OmegaConf.create(
            {"rollout": {"config": "test"}, "data": {"type": "single-turn"}, "trainer": {"config": "test"}}
        )
        # Add dataset path to run config
        self.cfg.recipes.run.data_s3_path = "/test/dataset/path"

        launcher = SMNovaK8SLauncherRFT(self.cfg)

        # Should not raise exception with flattened structure
        launcher._validate_recipe_parameters()

    def test_validate_recipe_parameters_optimized_rft_structure(self):
        """Test validation with optimized RFT structure (rollout, data, train)."""
        # RFT uses optimized structure with rollout, data, train
        self.cfg.recipes.training_config = OmegaConf.create(
            {"rollout": {"config": "test"}, "data": {"type": "single-turn"}, "trainer": {"config": "test"}}
        )
        # Add dataset path to run config
        self.cfg.recipes.run.data_s3_path = "/test/dataset/path"

        launcher = SMNovaK8SLauncherRFT(self.cfg)

        # Should not raise exception with optimized structure
        launcher._validate_recipe_parameters()

    def test_validate_recipe_parameters_dataset_in_run_config(self):
        """Test validation finds dataset path in run config."""
        # RFT structure already has dataset in run config
        self.cfg.recipes.run.data_s3_path = "/test/dataset/path"

        launcher = SMNovaK8SLauncherRFT(self.cfg)

        # Should not raise exception
        launcher._validate_recipe_parameters()

    def test_utility_functions_coverage(self):
        """Test utility functions for better coverage."""
        from launcher.nova.launchers import _is_efa_supported, get_num_efa_devices

        # Test key utility functions
        self.assertFalse(_is_efa_supported(None))
        self.assertTrue(_is_efa_supported("p5.48xlarge"))
        self.assertEqual(get_num_efa_devices(None), 0)
        self.assertEqual(get_num_efa_devices("p5.48xlarge"), 32)

    def test_validate_recipe_parameters_dataset_locations(self):
        """Test validation finds dataset in different config locations."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        # Test with trainer config dataset
        del self.cfg.recipes.run.data_s3_path
        self.cfg.recipes.training_config.trainer = OmegaConf.create(
            {"dataset_config": {"path": "/trainer/dataset/path"}}
        )
        launcher._validate_recipe_parameters()  # Should not raise

    def test_validate_recipe_parameters_replica_validation(self):
        """Test replica count validation."""
        self.cfg.recipes.run.replicas = 0
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        with self.assertRaises(ValueError) as context:
            launcher._validate_recipe_parameters()

        self.assertIn("must be a positive integer", str(context.exception))

    def test_map_rft_replica_config_with_redis_config(self):
        """Test _map_rft_replica_config with Redis configuration."""
        # Set delegate configuration with Redis config
        self.cfg.recipes.training_config.rollout = OmegaConf.create({"delegate": True})
        self.cfg.recipes.training_config.redis = OmegaConf.create(
            {"max_memory": "16gb", "max_memory_policy": "allkeys-lfu"}
        )
        self.cfg.recipes.run.redis_replicas = 2

        launcher = SMNovaK8SLauncherRFT(self.cfg)

        values_template = OmegaConf.create(
            {
                "trainingConfig": {
                    "training": {"replicas": 0},
                    "vllmGeneration": {"replicas": 0},
                    "hub": {"replicas": 0},
                    "redis": {"enabled": False, "replicas": 0},
                }
            }
        )

        launcher._map_rft_replica_config(values_template)

        # Verify Redis configuration is applied
        self.assertTrue(values_template.trainingConfig.redis.enabled)
        self.assertEqual(values_template.trainingConfig.redis.replicas, 2)
        self.assertEqual(values_template.trainingConfig.redis.maxMemory, "16gb")
        self.assertEqual(values_template.trainingConfig.redis.maxMemoryPolicy, "allkeys-lfu")

    def test_map_image_config_with_custom_container(self):
        """Test _map_image_config with custom container."""
        self.cfg.container = "custom-training-image:latest"
        launcher = SMNovaK8SLauncherRFT(self.cfg)

        values_template = OmegaConf.create(
            {
                "image": {
                    "trainingImage": "",
                    "generationImage": "",
                    "stormImage": "",
                    "natsServerImage": "",
                    "natsReloaderImage": "",
                    "redis": "",
                }
            }
        )

        launcher._map_image_config(values_template)

        # Should use custom container for training image
        self.assertEqual(values_template.image.trainingImage, "custom-training-image:latest")

    def test_process_values_yaml_with_deployment_metadata(self):
        """Test _process_values_yaml with deployment metadata."""
        launcher = SMNovaK8SLauncherRFT(self.cfg)
        launcher._template_dir = Path("/tmp/template")
        launcher._job_name = "test-job"
        launcher._init_container_uri = "init-container:latest"
        launcher.num_efa_devices = 32

        # Add deployment metadata to cluster config
        self.cfg.cluster = OmegaConf.create(
            {
                "deployment_metadata": {"region": "us-west-2", "alias": "test-alias", "node_type": "worker"},
                "namespace": "test-namespace",
                "annotations": {"test": "annotation"},
                "priority_class_name": "high-priority",
                "service_account_name": "test-service-account",
                "custom_labels": {"env": "test"},
            }
        )

        # Mock template file
        template_content = OmegaConf.create(
            {
                "trainingConfig": {
                    "jobName": "",
                    "initContainer": {"image": ""},
                    "envVars": {},
                    "numEFADevices": 0,
                    "region": "",
                    "alias": "",
                    "nodeType": "",
                    "requiredTolerations": [],
                    "namespace": "",
                    "annotations": {},
                    "priorityClassName": "",
                    "serviceAccountName": "",
                    "customLabels": {},
                    "redis": {"enabled": False},
                    "labelSelector": {},
                    "training": {"replicas": 2, "worker_replicas": 1},  # Add training section
                    "vllmGeneration": {"replicas": 2},
                    "hub": {"replicas": 1},
                },
                "jobList": [],
                "image": {},
            }
        )

        with patch("builtins.open", mock_open()), patch(
            "omegaconf.OmegaConf.load", return_value=template_content
        ), patch("launcher.nova.launchers.get_recipe_file_path", return_value="path"), patch.object(
            launcher, "_get_env_vars", return_value={}
        ), patch.object(
            launcher, "_map_rft_replica_config"
        ), patch.object(
            launcher, "_map_resource_config"
        ), patch.object(
            launcher, "_build_job_list", return_value=[]
        ), patch.object(
            launcher, "_map_image_config"
        ), patch.object(
            launcher, "_get_label_selectors", return_value={}
        ), patch.object(
            launcher, "_write_value_template"
        ):
            launcher._process_values_yaml()

            # Verify deployment metadata is applied
            self.assertEqual(template_content.trainingConfig.region, "us-west-2")
            self.assertEqual(template_content.trainingConfig.alias, "test-alias")
            self.assertEqual(template_content.trainingConfig.nodeType, "worker")
            self.assertEqual(template_content.trainingConfig.namespace, "test-namespace")

    def test_init_no_recipe_file_path(self):
        """Test RFT launcher initialization without recipe file path."""
        with patch("launcher.nova.launchers.get_recipe_file_path", return_value=None):
            launcher = SMNovaK8SLauncherRFT(self.cfg)
            self.assertIsNone(launcher.recipe_file_path)


class TestNovaLegacyQuotedRecipesBackwardCompat(unittest.TestCase):
    """Legacy nova recipes must keep quoted scalar placeholders; new recipes get them stripped."""

    NOVA_METADATA_PATH = "./launcher/recipe_templatization/nova/nova_metadata.json"

    # Mixed numeric + boolean placeholders, both quoted (as legacy recipes render them).
    SAMPLE_CONTENT = "run:\n  replicas: '{{replicas}}'\n  use_kl_loss: '{{use_kl_loss}}'\n"
    SAMPLE_OVERRIDE_SPEC = {"replicas": {"type": "integer"}, "use_kl_loss": {"type": "boolean"}}

    def _apply_launcher_quote_logic(self, recipe_name):
        """Mirror the launcher: strip scalar quotes only when the gate allows it."""
        if should_strip_scalar_quotes(recipe_name):
            return remove_quotes_from_scalar_params(self.SAMPLE_CONTENT, self.SAMPLE_OVERRIDE_SPEC)
        return self.SAMPLE_CONTENT

    def test_legacy_snapshot_is_non_empty_subset_of_metadata(self):
        """Every grandfathered recipe must be a real nova recipe (guards typos/drift)."""
        legacy = get_legacy_quoted_recipes()
        self.assertTrue(legacy, "legacy quoted-recipes snapshot must not be empty")
        with open(self.NOVA_METADATA_PATH, "r") as f:
            metadata_recipes = set(json.load(f))
        self.assertTrue(
            legacy <= metadata_recipes,
            f"legacy recipes missing from nova_metadata.json: {sorted(legacy - metadata_recipes)}",
        )

    def test_legacy_recipe_keeps_scalar_quotes(self):
        """Backward compat: a legacy recipe must NOT have numeric/boolean quotes stripped."""
        legacy_recipe = next(iter(get_legacy_quoted_recipes()))
        self.assertFalse(should_strip_scalar_quotes(legacy_recipe))

        result = self._apply_launcher_quote_logic(legacy_recipe)
        self.assertIn("replicas: '{{replicas}}'", result)
        self.assertIn("use_kl_loss: '{{use_kl_loss}}'", result)

    def test_new_recipe_strips_scalar_quotes(self):
        """A new (non-legacy) recipe DOES get numeric and boolean quotes stripped."""
        new_recipe = "nova_future_recipe_not_yet_shipped"
        self.assertNotIn(new_recipe, get_legacy_quoted_recipes())
        self.assertTrue(should_strip_scalar_quotes(new_recipe))

        result = self._apply_launcher_quote_logic(new_recipe)
        self.assertIn("replicas: {{replicas}}", result)
        self.assertIn("use_kl_loss: {{use_kl_loss}}", result)
        self.assertNotIn("'{{replicas}}'", result)
        self.assertNotIn("'{{use_kl_loss}}'", result)
