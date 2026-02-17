import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime

import boto3
from botocore.config import Config

region = os.environ.get("AWS_REGION", "us-west-2")

# Enable verbose debug logging for boto3, botocore, and urllib3
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def call_lambda(lambda_client, function_name, payload):
    """Call Lambda function once with detailed logging"""
    start_time = datetime.utcnow()
    logger.info(f"=== Lambda Invocation Start ===")
    logger.info(f"Function: {function_name}")
    logger.info(f"Start time: {start_time.isoformat()}")
    logger.info(f"Payload size: {len(payload)} bytes")

    # Log the client configuration
    client_config = lambda_client._client_config
    logger.info(f"Client config - read_timeout: {client_config.read_timeout}")
    logger.info(f"Client config - connect_timeout: {client_config.connect_timeout}")
    logger.info(f"Client config - retries: {client_config.retries}")
    logger.info(f"Client config - region: {client_config.region_name}")

    try:
        logger.info("Calling lambda_client.invoke()...")
        response = lambda_client.invoke(FunctionName=function_name, Payload=payload)

        end_time = datetime.utcnow()
        elapsed = (end_time - start_time).total_seconds()

        status_code = response["StatusCode"]
        logger.info(f"=== Lambda Invocation Success ===")
        logger.info(f"End time: {end_time.isoformat()}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"HTTP Status Code: {status_code}")
        logger.info(f"Response metadata: {response.get('ResponseMetadata', {})}")

        if status_code == 200:
            payload_response = json.loads(response["Payload"].read())
            logger.info(f"Payload response: {json.dumps(payload_response)[:500]}...")  # First 500 chars
            return payload_response
        else:
            return {"statusCode": status_code, "Error": "Lambda invocation failed"}

    except Exception as e:
        end_time = datetime.utcnow()
        elapsed = (end_time - start_time).total_seconds()

        logger.error(f"=== Lambda Invocation Failed ===")
        logger.error(f"End time: {end_time.isoformat()}")
        logger.error(f"Elapsed time: {elapsed:.2f} seconds")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # Log exception chain if present
        cause = e.__cause__
        if cause:
            logger.error(f"Caused by: {type(cause).__name__}: {cause}")

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

    if "statusCode" is None:
        print("Error in JS Lambda response: ", response)
        sys.exit(1)

    # Need to handle different status codes
    if status_code in [502, 503, 504] or (400 <= status_code <= 500):
        print(f"Recipe update failed with status code {status_code}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JumpStart Lambda response and update CloudWatch")
    parser.add_argument("--metadata", required=True, help="Recipe metadata JSON string")
    parser.add_argument("--lambda-name", required=True, help="Lambda function name")
    args = parser.parse_args()

    logger.info("=== Script Configuration ===")
    logger.info(f"AWS Region: {region}")
    logger.info(f"Lambda function: {args.lambda_name}")

    # Configure with extended timeouts
    config = Config(
        retries={"max_attempts": 1, "mode": "standard"},
        read_timeout=960,  # 16 minutes
        connect_timeout=60,  # 1 minute for connection
        tcp_keepalive=True,
    )

    lambda_client = boto3.client("lambda", region_name=region, config=config)
    recipe_metadata = args.metadata

    response = call_lambda(lambda_client, args.lambda_name, recipe_metadata)
    print(f"Lambda response: {response}")

    handle_lambda_response(response, recipe_metadata)
