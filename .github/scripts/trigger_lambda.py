import argparse
import json
import os
from datetime import datetime

import boto3
from botocore.config import Config

region = os.environ.get("AWS_REGION", "us-west-2")


class CloudWatchHandler:
    def __init__(self):
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self.logs = boto3.client("logs", region_name=region)
        self.namespace = "HyperpodRecipes/github"
        self.log_group = "github-actions/update-recipes-JS"

    def create_log_stream(self):
        """Create a new log stream with timestamp"""
        stream_name = f'recipe-update-{datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")}'
        try:
            self.logs.create_log_stream(logGroupName=self.log_group, logStreamName=stream_name)
            return stream_name
        except Exception as e:
            print(f"Error creating log stream: {e}")
            return None

    def put_metric(self, status_code, recipe_count):
        """Put metrics based on lambda response"""
        try:
            metric_data = [
                {
                    "MetricName": "GithubActions",
                    "Value": 1,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "Workflow", "Value": "process-recipes-changes"},
                        {"Name": "RecipeUpdateStatus", "Value": str(status_code)},
                    ],
                }
            ]

            self.cloudwatch.put_metric_data(Namespace=self.namespace, MetricData=metric_data)
        except Exception as e:
            print(f"Error putting metrics: {e}")

    def put_log(self, stream_name, message, metadata=None):
        """Log events to CloudWatch Logs"""
        try:
            log_entry = {
                "timestamp": int(datetime.utcnow().timestamp() * 1000),
                **message,
                "metadata": metadata,
            }

            self.logs.put_log_events(
                logGroupName=self.log_group,
                logStreamName=stream_name,
                logEvents=[{"timestamp": log_entry["timestamp"], "message": json.dumps(log_entry)}],
            )
        except Exception as e:
            print(f"Error logging event: {e}")


def invoke_lambda(lambda_client, function_name, payload):
    """Invoke Lambda function once"""
    try:
        response = lambda_client.invoke(FunctionName=function_name, Payload=payload)
        status_code = response["StatusCode"]  # invoke() returns StatusCode

        if status_code == 200:
            return json.loads(response["Payload"].read())

        else:
            return {"statusCode": status_code, "Error": "Lambda invocation failed"}
    except Exception as e:
        return {"statusCode": 500, "Error": str(e)}


def handle_lambda_response(response, recipe_metadata):
    """Create Cloudwatch entries based on JS lambda response"""
    handler = CloudWatchHandler()

    stream_name = handler.create_log_stream()

    status_code = response.get("statusCode")
    recipe_count = len(json.loads(recipe_metadata)["updatedRecipes"])

    # Put metrics
    handler.put_metric(status_code, recipe_count)

    # Log the event
    log_message = {
        "status_code": status_code,
        "recipe_count": recipe_count,
        "operation": "recipe_update",
        "response": response,
    }

    handler.put_log(stream_name, log_message, recipe_metadata)

    # Need to handle different status codes
    if status_code in [502, 503, 504] or (400 <= status_code < 500):
        print(f"Recipe update failed with status code {status_code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JumpStart Lambda response and update CloudWatch")
    parser.add_argument("--metadata", required=True, help="Recipe metadata JSON string")
    parser.add_argument("--lambda-name", required=True, help="Lambda function name")
    args = parser.parse_args()

    # Configure the inbuilt retry behavior
    config = Config(retries=dict(max_attempts=3, mode="standard"), read_timeout=960)  # 16 minutes

    lambda_client = boto3.client("lambda", region_name=region, config=config)
    recipe_metadata = args.metadata

    response = invoke_lambda(lambda_client, args.lambda_name, recipe_metadata)
    print(f"Lambda response: {response}")

    handle_lambda_response(response, recipe_metadata)
