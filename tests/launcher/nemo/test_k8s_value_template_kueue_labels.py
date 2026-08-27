import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from launcher.nemo.stages import SMTraining

VALUES_TEMPLATE = """\
image:
  trainingImage: placeholder
  pullPolicy: IfNotPresent
trainingConfig:
  jobName: "nil"
  namespace: "default"
  customLabels: null
  priorityClassName: null
  numEFADevices: 0
  envVars: {}
"""


def _make_stage():
    """Build an SMTraining instance without running the heavy __init__."""
    stage = SMTraining.__new__(SMTraining)
    stage.cfg = OmegaConf.create({})
    stage.stage_cfg = OmegaConf.create({"run": {"name": "test-job"}})
    stage.num_efa_devices = 0
    stage.device = "gpu"
    stage.instance_type = "p5.48xlarge"
    stage.cluster = "k8s"
    return stage


def _render(cluster_parameters):
    stage = _make_stage()
    with tempfile.TemporaryDirectory() as tmp:
        template_root = Path(tmp)
        (template_root / "values.yaml").write_text(VALUES_TEMPLATE)
        params = {"container_image": "test-image", **cluster_parameters}
        values = stage.generate_default_k8s_value_template(template_root, params)
    labels = values.trainingConfig.customLabels
    if labels is None:
        return None
    return OmegaConf.to_container(labels, resolve=True)


class TestPriorityClassLabel:
    def test_priority_class_sets_label_without_existing_labels(self):
        labels = _render({"priority_class": "high-priority"})
        assert labels == {"kueue.x-k8s.io/priority-class": "high-priority"}

    def test_priority_class_merges_with_existing_custom_labels(self):
        labels = _render({"priority_class": "low-priority", "custom_labels": {"team": "ml"}})
        assert labels == {"team": "ml", "kueue.x-k8s.io/priority-class": "low-priority"}

    def test_no_priority_class_leaves_labels_untouched(self):
        labels = _render({})
        assert labels is None


class TestQueueNameLabel:
    def test_queue_name_sets_label_without_existing_labels(self):
        labels = _render({"queue_name": "hyperpod-ns-default-localqueue"})
        assert labels == {"kueue.x-k8s.io/queue-name": "hyperpod-ns-default-localqueue"}

    def test_queue_name_merges_with_existing_custom_labels(self):
        labels = _render({"queue_name": "team-localqueue", "custom_labels": {"team": "ml"}})
        assert labels == {"team": "ml", "kueue.x-k8s.io/queue-name": "team-localqueue"}


def test_priority_class_and_queue_name_together():
    labels = _render(
        {"priority_class": "high-priority", "queue_name": "team-localqueue", "custom_labels": {"team": "ml"}}
    )
    assert labels == {
        "team": "ml",
        "kueue.x-k8s.io/priority-class": "high-priority",
        "kueue.x-k8s.io/queue-name": "team-localqueue",
    }
