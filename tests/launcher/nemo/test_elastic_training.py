import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from omegaconf import OmegaConf

# Mock nemo_launcher dependencies before importing
sys.modules["nemo_launcher"] = MagicMock()
sys.modules["nemo_launcher.utils"] = MagicMock()
sys.modules["nemo_launcher.utils.job_utils"] = MagicMock()
sys.modules["nemo_launcher.core"] = MagicMock()
sys.modules["nemo_launcher.core.stages"] = MagicMock()

from launcher.nemo.recipe_stages import SMTrainingGPURecipe, SMTrainingGPURecipeElastic
from main import get_training_stage

logger = logging.getLogger(__name__)


class TestElasticTraining:
    """Test suite for elastic training functionality"""

    def test_get_training_stage_elastic_gpu(self):
        """Test that elastic GPU training returns the correct stage class"""
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster_type": "k8s",
                "training": {
                    "elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4, "replica_increment_step": 1}
                },
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipeElastic

    def test_get_training_stage_non_elastic_gpu(self):
        """Test that non-elastic GPU training returns the standard stage class"""
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster_type": "k8s",
                "training": {"elastic_policy": {"is_elastic": False}},
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipe

    def test_get_training_stage_missing_elastic_policy(self):
        """Test that missing elastic policy defaults to non-elastic"""
        cfg = OmegaConf.create({"instance_type": "p5.48xlarge", "cluster_type": "k8s", "training": {}})

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipe

    def test_elastic_training_stage_inheritance(self):
        """Test that SMTrainingGPURecipeElastic inherits from SMTrainingGPURecipe"""
        assert issubclass(SMTrainingGPURecipeElastic, SMTrainingGPURecipe)

    @patch("omegaconf.OmegaConf.save")
    def test_save_stage_hydra_config_with_scale_config(self, mock_save):
        """Test that elastic training saves multiple config files for different node counts"""
        # Setup mock job paths
        temp_dir = Path(tempfile.mkdtemp())
        mock_job_paths_instance = Mock()
        mock_job_paths_instance.folder = temp_dir

        # Create test configuration with scale_config
        stage_cfg = OmegaConf.create(
            {
                "trainer": {"num_nodes": 1},
                "model": {"learning_rate": 0.001},
                "elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4},
                "scale_config": {
                    "1": {"trainer": {"num_nodes": 1}, "model": {"batch_size": 32}},
                    "2": {"trainer": {"num_nodes": 2}, "model": {"batch_size": 16}},
                    "4": {"trainer": {"num_nodes": 4}, "model": {"batch_size": 8}},
                },
            }
        )

        cfg = OmegaConf.create({})

        try:
            # Call the method
            SMTrainingGPURecipeElastic.save_stage_hydra_config(stage_cfg, mock_job_paths_instance, cfg)

            # Verify that save was called for each scale config
            expected_calls = 3  # One for each node count in scale_config
            assert mock_save.call_count >= expected_calls

            # Verify the file paths
            call_args_list = mock_save.call_args_list
            expected_files = [
                temp_dir / "train_config_n1.yaml",
                temp_dir / "train_config_n2.yaml",
                temp_dir / "train_config_n4.yaml",
            ]

            actual_files = [call[0][1] for call in call_args_list[-3:]]  # Last 3 calls
            for expected_file in expected_files:
                assert expected_file in actual_files

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)

    @patch("launcher.nemo.recipe_stages.shutil.copy")
    def test_copy_k8s_helm_chart_copies_train_configs(self, mock_copy):
        """Test that elastic training copies train_config_n*.yaml files to k8s config directory"""
        # Setup
        temp_dir = Path(tempfile.mkdtemp())
        job_path = Mock()
        job_path.folder = temp_dir

        # Create mock train config files
        train_config_files = [
            temp_dir / "train_config_n1.yaml",
            temp_dir / "train_config_n2.yaml",
            temp_dir / "train_config_n4.yaml",
        ]

        for file_path in train_config_files:
            file_path.touch()

        # Create k8s config directory
        k8s_config_dir = temp_dir / "k8s_template" / "config"
        k8s_config_dir.mkdir(parents=True)

        # Create proper configuration with required structure
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster": {"instance_type": "p5.48xlarge"},
                "training": {
                    "run": {"name": "test-job"},
                    "elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4},
                },
            }
        )

        # Mock the glob method to return our test files
        with patch.object(Path, "glob", return_value=train_config_files):
            # Create instance and call method
            stage = SMTrainingGPURecipeElastic(cfg=cfg)

            # Mock the parent method
            with patch.object(SMTrainingGPURecipe, "_copy_k8s_helm_chart"):
                stage._copy_k8s_helm_chart("template_root", job_path)

        # Verify that copy was called for each train config file
        assert mock_copy.call_count == len(train_config_files)

        # Verify the copy calls
        for i, train_config_file in enumerate(train_config_files):
            mock_copy.assert_any_call(train_config_file, k8s_config_dir)

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_generate_default_k8s_value_template_sets_elastic_policy(self):
        """Test that elastic training sets elastic policy values in k8s template"""
        # Setup configuration
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster": {"instance_type": "p5.48xlarge"},
                "training": {
                    "run": {"name": "test-job"},
                    "elastic_policy": {"is_elastic": True, "min_nodes": 2, "max_nodes": 8, "replica_increment_step": 2},
                },
            }
        )

        # Create stage instance
        stage = SMTrainingGPURecipeElastic(cfg=cfg)

        # Mock the parent method to return a basic template
        mock_template = Mock()
        mock_template.elastic_policy = Mock()

        with patch.object(SMTrainingGPURecipe, "generate_default_k8s_value_template", return_value=mock_template):
            result = stage.generate_default_k8s_value_template("template_root", {})

        # Verify elastic policy values are set
        assert result.trainingConfig.elastic_policy.is_elastic == True
        assert result.trainingConfig.elastic_policy.min_nodes == 2
        assert result.trainingConfig.elastic_policy.max_nodes == 8
        assert result.trainingConfig.elastic_policy.replica_increment_step == 2

    def test_get_script_args_str_k8s_cluster(self):
        """Test that elastic training returns correct script args for k8s cluster"""
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster": {"instance_type": "p5.48xlarge"},
                "training": {"run": {"name": "test-job"}, "elastic_policy": {"is_elastic": True}},
            }
        )

        stage = SMTrainingGPURecipeElastic(cfg=cfg)
        stage.cluster = "k8s"

        stage_cfg_path = Path("/some/path/config.yaml")
        result = stage.get_script_args_str(stage_cfg_path)

        assert result == "--config-path=/config"

    def test_get_script_args_str_non_k8s_cluster(self):
        """Test that elastic training returns correct script args for non-k8s cluster"""
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster": {"instance_type": "p5.48xlarge"},
                "training": {"run": {"name": "test-job"}, "elastic_policy": {"is_elastic": True}},
            }
        )

        stage = SMTrainingGPURecipeElastic(cfg=cfg)
        stage.cluster = "slurm"

        stage_cfg_path = Path("/some/path/config.yaml")
        result = stage.get_script_args_str(stage_cfg_path)

        expected = f"--config-path={stage_cfg_path.parents[0]} --config-name={stage_cfg_path.name}"
        assert result == expected

    def test_save_stage_hydra_config_removes_elastic_keys(self):
        """Test that elastic training removes elastic-specific keys from basic config"""
        # This test verifies that scale_config and elastic_policy are removed
        # from the basic config when creating individual node configs

        stage_cfg = OmegaConf.create(
            {
                "trainer": {"num_nodes": 1},
                "model": {"learning_rate": 0.001},
                "elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4},
                "scale_config": {"1": {"trainer": {"num_nodes": 1}}},
                "other_config": {"value": "should_remain"},
            }
        )

        temp_dir = Path(tempfile.mkdtemp())
        job_path = Mock()
        job_path.folder = temp_dir
        cfg = OmegaConf.create({})

        saved_configs = []

        def capture_save(config, path):
            saved_configs.append((config, path))

        try:
            with patch("launcher.nemo.recipe_stages.omegaconf.OmegaConf.save", side_effect=capture_save):
                SMTrainingGPURecipeElastic.save_stage_hydra_config(stage_cfg, job_path, cfg)

            # Verify that saved configs don't contain elastic-specific keys
            for config, path in saved_configs:
                if "train_config_n" in str(path):
                    assert "elastic_policy" not in config
                    assert "scale_config" not in config
                    assert "other_config" in config  # Other configs should remain

        finally:
            shutil.rmtree(temp_dir)
