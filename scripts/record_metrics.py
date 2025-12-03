import json
import logging
import os
from datetime import datetime, timedelta, timezone

import boto3
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GitHub API configuration
github_token = os.environ["GH_TOKEN"]

# List of repositories to track
repos = [
    {"owner": "aws", "name": "sagemaker-hyperpod-recipes"},
    {"owner": "aws", "name": "sagemaker-hyperpod-training-adapter-for-nemo"},
]

# AWS CloudWatch configuration
cloudwatch = boto3.client("cloudwatch", region_name="us-west-2")


def get_traffic_data(owner, repo):
    headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}

    logger.info(f"Fetching traffic data for {owner}/{repo}")

    # Get views
    views_url = f"https://api.github.com/repos/{owner}/{repo}/traffic/views"
    views_response = requests.get(views_url, headers=headers)
    views_data = views_response.json()

    if views_response.status_code != 200:
        logger.error(f"Failed to get views data: {views_data}")
    else:
        logger.info(f"Views data: {json.dumps(views_data, indent=2)}")

    # Get clones
    clones_url = f"https://api.github.com/repos/{owner}/{repo}/traffic/clones"
    clones_response = requests.get(clones_url, headers=headers)
    clones_data = clones_response.json()

    if clones_response.status_code != 200:
        logger.error(f"Failed to get clones data: {clones_data}")
    else:
        logger.info(f"Clones data: {json.dumps(clones_data, indent=2)}")

    return views_data, clones_data


def get_yesterday_data(data_array):
    """Get data from yesterday since it is guaranteed to be complete"""
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_str = yesterday.strftime("%Y-%m-%dT00:00:00Z")

    if data_array:
        for entry in reversed(data_array):
            if entry["timestamp"] == yesterday_str:
                logger.info(f"Found latest entry {entry}")
                return entry, yesterday

    logger.warning(f"No data found for yesterday ({yesterday_str}). Using 0 values.")
    return {"timestamp": yesterday_str, "count": 0, "uniques": 0}, yesterday


def publish_to_cloudwatch(owner, repo, views_data, clones_data):
    metrics = []

    # Process views
    if "views" in views_data:
        last_view, last_timestamp = get_yesterday_data(views_data["views"])

        logger.info(f"Publishing views data for date: {last_view['timestamp']}")

        # Daily metrics
        metrics.append(
            {
                "MetricName": "Views",
                "Value": last_view["count"],
                "Unit": "Count",
                "Timestamp": last_timestamp,
                "Dimensions": [
                    {"Name": "Repository", "Value": f"{owner}/{repo}"},
                    {"Name": "TimeFrame", "Value": "Daily"},
                ],
            }
        )

        metrics.append(
            {
                "MetricName": "UniqueVisitors",
                "Value": last_view["uniques"],
                "Unit": "Count",
                "Timestamp": last_timestamp,
                "Dimensions": [
                    {"Name": "Repository", "Value": f"{owner}/{repo}"},
                    {"Name": "TimeFrame", "Value": "Daily"},
                ],
            }
        )

        # Two-week totals using the same timestamp as daily metrics
        metrics.append(
            {
                "MetricName": "Views",
                "Value": views_data.get("count", 0),
                "Unit": "Count",
                "Timestamp": last_timestamp,
                "Dimensions": [
                    {"Name": "Repository", "Value": f"{owner}/{repo}"},
                    {"Name": "TimeFrame", "Value": "TwoWeek"},
                ],
            }
        )

        metrics.append(
            {
                "MetricName": "UniqueVisitors",
                "Value": views_data.get("uniques", 0),
                "Unit": "Count",
                "Timestamp": last_timestamp,
                "Dimensions": [
                    {"Name": "Repository", "Value": f"{owner}/{repo}"},
                    {"Name": "TimeFrame", "Value": "TwoWeek"},
                ],
            }
        )

    # Process clones
    if "clones" in clones_data:
        last_clone, last_timestamp = get_yesterday_data(clones_data["clones"])

        logger.info(f"Publishing clones data for date: {last_clone['timestamp']}")

        # Daily metrics
        metrics.append(
            {
                "MetricName": "Clones",
                "Value": last_clone["count"],
                "Unit": "Count",
                "Timestamp": last_timestamp,
                "Dimensions": [
                    {"Name": "Repository", "Value": f"{owner}/{repo}"},
                    {"Name": "TimeFrame", "Value": "Daily"},
                ],
            }
        )

        metrics.append(
            {
                "MetricName": "UniqueCloners",
                "Value": last_clone["uniques"],
                "Unit": "Count",
                "Timestamp": last_timestamp,
                "Dimensions": [
                    {"Name": "Repository", "Value": f"{owner}/{repo}"},
                    {"Name": "TimeFrame", "Value": "Daily"},
                ],
            }
        )

        # Two-week totals using the same timestamp as daily metrics
        metrics.append(
            {
                "MetricName": "Clones",
                "Value": clones_data.get("count", 0),
                "Unit": "Count",
                "Timestamp": last_timestamp,
                "Dimensions": [
                    {"Name": "Repository", "Value": f"{owner}/{repo}"},
                    {"Name": "TimeFrame", "Value": "TwoWeek"},
                ],
            }
        )

        metrics.append(
            {
                "MetricName": "UniqueCloners",
                "Value": clones_data.get("uniques", 0),
                "Unit": "Count",
                "Timestamp": last_timestamp,
                "Dimensions": [
                    {"Name": "Repository", "Value": f"{owner}/{repo}"},
                    {"Name": "TimeFrame", "Value": "TwoWeek"},
                ],
            }
        )

    if metrics:
        logger.info(f"Publishing metrics to CloudWatch: {json.dumps(metrics, indent=2, default=str)}")
        try:
            response = cloudwatch.put_metric_data(Namespace="HyperPodRecipes/Traffic/v2", MetricData=metrics)
            logger.info(f"Successfully published metrics: {response}")
        except Exception as e:
            logger.error(f"Failed to publish metrics: {str(e)}")
    else:
        logger.warning("No metrics to publish")


def main():
    logger.info("Starting traffic metrics collection")
    for repo in repos:
        logger.info(f"Processing repository: {repo['owner']}/{repo['name']}")
        views_data, clones_data = get_traffic_data(repo["owner"], repo["name"])
        publish_to_cloudwatch(repo["owner"], repo["name"], views_data, clones_data)
    logger.info("Completed traffic metrics collection")


if __name__ == "__main__":
    main()
