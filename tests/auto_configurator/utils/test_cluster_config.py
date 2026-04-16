from unittest.mock import Mock, patch

import pytest
from omegaconf import ListConfig, OmegaConf

from auto_configurator.utils.cluster_config import (
    DEFAULT_INSTANCE_TYPES,
    get_instance_type_list,
    validate_cluster,
)


class TestGetInstanceTypeList:
    def test_auto_returns_defaults(self):
        cfg = OmegaConf.create({"autotune_config": {"instance_type_list": "auto"}})
        assert get_instance_type_list(cfg) == DEFAULT_INSTANCE_TYPES

    def test_auto_default_when_missing(self):
        cfg = OmegaConf.create({"autotune_config": {}})
        assert get_instance_type_list(cfg) == DEFAULT_INSTANCE_TYPES

    def test_single_value(self):
        cfg = OmegaConf.create({"autotune_config": {"instance_type_list": "ml.p5.48xlarge"}})
        assert get_instance_type_list(cfg) == ["ml.p5.48xlarge"]

    def test_list_value(self):
        cfg = OmegaConf.create({"autotune_config": {"instance_type_list": ["ml.p5.48xlarge", "ml.p4d.24xlarge"]}})
        result = get_instance_type_list(cfg)
        assert result == ["ml.p5.48xlarge", "ml.p4d.24xlarge"]

    def test_listconfig_handled(self):
        """OmegaConf returns ListConfig, not plain list — must be handled."""
        cfg = OmegaConf.create({"autotune_config": {"instance_type_list": ["ml.p5.48xlarge"]}})
        instance_types = cfg.autotune_config.instance_type_list
        assert isinstance(instance_types, ListConfig)
        result = get_instance_type_list(cfg)
        assert result == ["ml.p5.48xlarge"]
        # Elements should be usable as dict keys (not ListConfig)
        for item in result:
            assert isinstance(item, str)


class TestValidateCluster:
    def test_raises_when_no_context_map(self):
        cfg = OmegaConf.create({})
        with pytest.raises(ValueError, match="No kubectl context mapped"):
            validate_cluster("ml.p5.48xlarge", cfg)

    def test_raises_when_instance_type_not_in_map(self):
        cfg = OmegaConf.create({"k8": {"cluster_context_map": {"ml.p4d.24xlarge": "some-context"}}})
        with pytest.raises(ValueError, match="No kubectl context mapped"):
            validate_cluster("ml.p5.48xlarge", cfg)

    @patch("auto_configurator.utils.cluster_config.subprocess.run")
    def test_raises_on_kubectl_failure(self, mock_run):
        mock_run.return_value = Mock(returncode=1, stderr="connection refused")
        cfg = OmegaConf.create({"k8": {"cluster_context_map": {"ml.p5.48xlarge": "my-context"}}})
        with pytest.raises(RuntimeError, match="Failed to query cluster nodes"):
            validate_cluster("ml.p5.48xlarge", cfg)

    @patch("auto_configurator.utils.cluster_config.subprocess.run")
    def test_raises_when_no_nodes_found(self, mock_run):
        mock_run.return_value = Mock(returncode=0, stdout="")
        cfg = OmegaConf.create({"k8": {"cluster_context_map": {"ml.p5.48xlarge": "my-context"}}})
        with pytest.raises(RuntimeError, match="not found in cluster"):
            validate_cluster("ml.p5.48xlarge", cfg)

    @patch("auto_configurator.utils.cluster_config.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = Mock(returncode=0, stdout="node-1 Ready\nnode-2 Ready\n")
        cfg = OmegaConf.create({"k8": {"cluster_context_map": {"ml.p5.48xlarge": "my-context"}}})
        validate_cluster("ml.p5.48xlarge", cfg)
        cmd = mock_run.call_args[0][0]
        assert "--context" in cmd
        assert "my-context" in cmd
