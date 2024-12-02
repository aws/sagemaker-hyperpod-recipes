import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List

CW_NAME_SPACE = "RecipesTelemetry"


@dataclass
class Metric:
    Name: str = None
    Unit: str = None


@dataclass
class MetricDirective:
    Namespace: str = ""
    Dimensions: List[List[str]] = None
    Metrics: List[Metric] = None


@dataclass
class Metadata:
    CloudWatchMetrics: List[MetricDirective] = field(default_factory=lambda: [MetricDirective])
    Timestamp: int = None


@dataclass
class CWTelemetryStart:
    account_id: str = ""
    training_start_time: int = 0
    num_nodes: int = 0
    job_name: str = ""
    cluster_type: str = ""
    instance_type: str = ""
    _aws: Metadata = None
    job_id: int = 0
    recipe: str = ""
    container: str = ""


class Telemetry:
    def __init__(self, log_path="/var/log/aws/clusters/sagemaker-hyperpod-recipes-telemetry.log"):
        self.log_path = log_path

    def get_account_id(self):
        import boto3

        client = boto3.client("sts")
        return client.get_caller_identity()["Account"]

    def publish_cw_log(self, log):
        save_log = asdict(log)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(save_log, separators=(",", ":")) + "\n")

    def start(
        self,
        cluster_type=None,
        instance_type=None,
        num_nodes=None,
        job_id=None,
        container=None,
    ):
        if not os.path.exists(self.log_path):
            return
        account_id = self.get_account_id()
        cw_telemetry_start = CWTelemetryStart(account_id=account_id)
        cw_telemetry_start.training_start_time = int(time.time() * 1000)
        cw_telemetry_start.num_nodes = int(num_nodes)
        cw_telemetry_start.cluster_type = cluster_type
        cw_telemetry_start.instance_type = instance_type
        cw_telemetry_start.job_id = job_id
        cw_telemetry_start.container = container

        recipe = ""
        for arg in sys.argv:
            if arg.startswith("recipes="):
                recipe = arg.split("=")[1]
        cw_telemetry_start.recipe = recipe

        metadata = Metadata(
            Timestamp=int(time.time() * 1000),
            CloudWatchMetrics=[
                MetricDirective(
                    Namespace=CW_NAME_SPACE,
                    Dimensions=[[]],
                    Metrics=[Metric(Name="num_nodes", Unit="Count")],
                )
            ],
        )
        cw_telemetry_start._aws = metadata
        self.publish_cw_log(cw_telemetry_start)
