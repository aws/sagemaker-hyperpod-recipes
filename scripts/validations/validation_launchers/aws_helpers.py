"""AWS-specific helper functions for validation utilities."""

import logging
import re
import time
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# Default regex patterns for common log metrics
DEFAULT_LOG_PATTERNS = {
    "output_model_arn": r"OutputModelPackageArn:\s*(arn:aws:sagemaker:[^\s,]+)",
    "training_loss": r"(?:training_loss|train_loss):\s*([\d.]+)",
    "validation_loss": r"(?:validation_loss|val_loss|eval_loss):\s*([\d.]+)",
    "epoch": r"epoch:\s*(\d+)",
    "learning_rate": r"learning_rate:\s*([\d.e-]+)",
    "tokens_per_sec": r"tokens[/_]per[/_]sec(?:ond)?:\s*([\d.]+)",
    "error": r"(?:Error|ERROR|Exception|EXCEPTION):\s*(.+)",
    "warning": r"(?:Warning|WARNING):\s*(.+)",
}

# Regex patterns for checkpoint-related log messages
CHECKPOINT_PATTERNS = {
    "found": r"Found checkpoint:.*?(/opt/ml/model/global_step_\d+)",
    "load": r"Load from checkpoint folder:\s*(/opt/ml/model/global_step_\d+)",
    "global_step": r"Setting global step to\s*(\d+)",
    "resume": r"Resuming from\s*(/opt/ml/model/global_step_\d+)",
}


def extract_training_job_name_from_output_path(output_path: str) -> Optional[str]:
    """
    Extract training job name from S3 output path.

    Args:
        output_path: S3 output path (e.g., 's3://bucket/path/job-name/output')

    Returns:
        str or None: Training job name if found
    """
    match = re.search(r"/([^/]+)/output", output_path)
    return match.group(1) if match else None


def extract_training_job_name_from_log(output_log: str) -> Optional[str]:
    """
    Extract training job name from CloudWatch log stream pattern.

    Args:
        output_log: Log output containing stream information

    Returns:
        str or None: Training job name if found
    """
    # Look for pattern: "stream: <job-name>/algo-1-..."
    match = re.search(r"stream:\s*([^/\s]+)/algo", output_log)
    return match.group(1) if match else None


def parse_checkpoint_patterns(message: str) -> Optional[Dict[str, str]]:
    """
    Parse a checkpoint-related log message to extract structured information.

    Args:
        message: Log message string

    Returns:
        dict: Parsed checkpoint information with keys: type, step, path
              Returns None if message doesn't contain recognizable checkpoint patterns
    """
    checkpoint_info = {}

    # Pattern 1: Saving checkpoint
    save_match = re.search(
        r"(?:saving|saved|save)\s+(?:the\s+)?checkpoint(?:\s+to)?\s+([^\s]+)?", message, re.IGNORECASE
    )
    if save_match:
        checkpoint_info["type"] = "saving"
        path = save_match.group(1) if save_match.group(1) else None
        if path:
            checkpoint_info["path"] = path
            step_match = re.search(r"global_step_(\d+)", path)
            if step_match:
                checkpoint_info["step"] = step_match.group(1)
        return checkpoint_info

    # Pattern 2: Loading checkpoint
    load_match = re.search(
        r"(?:loading|loaded|load)\s+(?:from\s+)?checkpoint(?:\s+from)?\s+([^\s]+)?", message, re.IGNORECASE
    )
    if load_match:
        checkpoint_info["type"] = "loading"
        path = load_match.group(1) if load_match.group(1) else None
        if path:
            checkpoint_info["path"] = path
            step_match = re.search(r"global_step_(\d+)", path)
            if step_match:
                checkpoint_info["step"] = step_match.group(1)
        return checkpoint_info

    # Pattern 3: Global step reference
    step_match = re.search(r"global_step_(\d+)", message)
    if step_match:
        checkpoint_info["type"] = "reference"
        checkpoint_info["step"] = step_match.group(1)
        return checkpoint_info

    # Pattern 4: Checkpoint path reference
    path_match = re.search(r"/opt/ml/model/global_step_(\d+)", message)
    if path_match:
        checkpoint_info["type"] = "path_reference"
        checkpoint_info["step"] = path_match.group(1)
        checkpoint_info["path"] = f"/opt/ml/model/global_step_{path_match.group(1)}"
        return checkpoint_info

    return None


def extract_arn_from_log(log_text: str) -> Optional[str]:
    """
    Extract OutputModelPackageArn from log text.

    Args:
        log_text: Log text to search

    Returns:
        str or None: ARN if found, None otherwise
    """
    pattern = r"OutputModelPackageArn:\s*(arn:aws:sagemaker:[^\s,]+)"
    matches = re.findall(pattern, log_text)
    if matches:
        # Remove trailing punctuation and return last match
        return matches[-1].rstrip(".,;")
    return None


def extract_output_model_arn(job_recorder, allow_incomplete: bool = False) -> str:
    """
    Extract the OutputModelPackageArn from job results for use as SourceModelPackageArn in next job.

    Args:
        job_recorder: JobRecorder object with validation results
        allow_incomplete: If True, allows extraction from jobs that were stopped early (status != "Complete")

    Returns:
        str: OutputModelPackageArn from the job output

    Raises:
        ValueError: If no jobs found or OutputModelPackageArn not found
    """
    if not job_recorder or not hasattr(job_recorder, "jobs") or len(job_recorder.jobs) == 0:
        raise ValueError("No jobs found in job recorder")

    if len(job_recorder.jobs) > 1:
        logger.warning(f"Multiple jobs found ({len(job_recorder.jobs)}), using first job for output extraction")

    # Get the job
    job_name, job = next(iter(job_recorder.jobs.items()))

    # Check if job completed successfully (unless allow_incomplete is True)
    if not allow_incomplete and job.status != "Complete":
        raise ValueError(f"Job {job_name} did not complete successfully (status: {job.status})")

    # Log warning if job was stopped early but we're allowing incomplete extraction
    if allow_incomplete and job.status != "Complete":
        logger.warning(f"Extracting ARN from job with status '{job.status}' (allow_incomplete=True)")

    # Parse logs for OutputModelPackageArn
    if hasattr(job, "output_log") and job.output_log:
        output_model_arn = extract_arn_from_log(job.output_log)
        if output_model_arn:
            logger.info(f"Found OutputModelPackageArn in logs: {output_model_arn}")
            return output_model_arn

    # If not found, raise an error
    raise ValueError(f"Could not find OutputModelPackageArn in job {job_name} output logs")


def parse_job_logs(job_recorder, parse_patterns: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Parse job logs for specific patterns and extract useful information.

    Args:
        job_recorder: JobRecorder object with validation results
        parse_patterns: Optional dict of {pattern_name: regex_pattern} to search for
                       If None, uses default patterns for common metrics

    Returns:
        dict: Parsed log information for each job
    """
    # Default patterns if none provided
    if parse_patterns is None:
        parse_patterns = DEFAULT_LOG_PATTERNS

    parsed_results = {}

    for job_name, job in job_recorder.jobs.items():
        job_info = {
            "job_name": job_name,
            "status": job.status,
            "platform": getattr(job, "platform", "N/A"),
            "instance_type": getattr(job, "instance_type", "N/A"),
            "output_path": getattr(job, "output_path", "N/A"),
            "parsed_data": {},
        }

        # Parse logs if available
        if hasattr(job, "output_log") and job.output_log:
            for pattern_name, pattern in parse_patterns.items():
                matches = re.findall(pattern, job.output_log, re.IGNORECASE)
                if matches:
                    # Store all matches or just the last one depending on pattern
                    if pattern_name in ["error", "warning"]:
                        job_info["parsed_data"][pattern_name] = matches  # Keep all errors/warnings
                    else:
                        job_info["parsed_data"][pattern_name] = matches[-1]  # Keep last match

                    logger.info(f"  [{job_name}] Found {pattern_name}: {job_info['parsed_data'][pattern_name]}")
        else:
            logger.warning(f"  [{job_name}] No output log available for parsing")

        parsed_results[job_name] = job_info

    return parsed_results


def fetch_cloudwatch_logs(
    log_group_name: str, log_stream_name: str, region: str = "us-west-2", max_events: int = 1000
) -> list:
    """
    Fetch CloudWatch logs for a SageMaker training job.

    Args:
        log_group_name: CloudWatch log group name (e.g., '/aws/sagemaker/TrainingJobs')
        log_stream_name: CloudWatch log stream name (training job name)
        region: AWS region
        max_events: Maximum number of log events to fetch

    Returns:
        list: List of log events with timestamp and message
    """

    try:
        logs_client = boto3.client("logs", region_name=region)

        logger.info(f"Fetching CloudWatch logs from {log_group_name}/{log_stream_name}")

        response = logs_client.get_log_events(
            logGroupName=log_group_name, logStreamName=log_stream_name, limit=max_events, startFromHead=True
        )

        events = response.get("events", [])
        logger.info(f"Fetched {len(events)} log events from CloudWatch")

        return events
    except Exception as e:
        logger.error(f"Failed to fetch CloudWatch logs: {e}")
        return []


def parse_cloudwatch_logs_for_checkpoints(log_events: list) -> Dict[str, Any]:
    """
    Parse CloudWatch log events for checkpoint-related information.

    Looks for patterns like:
    - "Found checkpoint: %s /opt/ml/model/global_step_7"
    - "Load from checkpoint folder: /opt/ml/model/global_step_7"
    - "Setting global step to 7"
    - "Resuming from /opt/ml/model/global_step_7"

    Args:
        log_events: List of CloudWatch log events

    Returns:
        dict: Parsed checkpoint information
    """
    checkpoint_info = {
        "found_checkpoints": [],
        "loaded_checkpoints": [],
        "global_steps": [],
        "resume_paths": [],
        "all_checkpoint_logs": [],
    }

    checkpoint_patterns = CHECKPOINT_PATTERNS

    for event in log_events:
        message = event.get("message", "")
        timestamp = event.get("timestamp", 0)

        # Check for checkpoint-related messages
        if "checkpoint" in message.lower() or "global_step" in message.lower():
            checkpoint_info["all_checkpoint_logs"].append({"timestamp": timestamp, "message": message})

            # Parse specific patterns
            for pattern_name, pattern in checkpoint_patterns.items():
                matches = re.findall(pattern, message, re.IGNORECASE)
                if matches:
                    if pattern_name == "found":
                        checkpoint_info["found_checkpoints"].extend(matches)
                    elif pattern_name == "load":
                        checkpoint_info["loaded_checkpoints"].extend(matches)
                    elif pattern_name == "global_step":
                        checkpoint_info["global_steps"].extend(matches)
                    elif pattern_name == "resume":
                        checkpoint_info["resume_paths"].extend(matches)

    logger.info(f"Parsed checkpoint info: {len(checkpoint_info['all_checkpoint_logs'])} checkpoint-related logs")
    logger.info(f"  Found checkpoints: {checkpoint_info['found_checkpoints']}")
    logger.info(f"  Loaded checkpoints: {checkpoint_info['loaded_checkpoints']}")
    logger.info(f"  Global steps: {checkpoint_info['global_steps']}")
    logger.info(f"  Resume paths: {checkpoint_info['resume_paths']}")

    return checkpoint_info


def stop_sagemaker_training_job(training_job_name: str, region: str = "us-west-2") -> Optional[Dict[str, Any]]:
    """
    Stop a SageMaker training job.

    Args:
        training_job_name: Name of the SageMaker training job
        region: AWS region

    Returns:
        dict: Response from DescribeTrainingJob API, or None if job not found
    """

    try:
        sagemaker_client = boto3.client("sagemaker", region_name=region)

        logger.info(f"Attempting to stop SageMaker training job: {training_job_name}")

        # First check if job exists and its current status
        try:
            job_status = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
            current_status = job_status["TrainingJobStatus"]
            logger.info(f"Current job status: {current_status}")

            # If already in terminal state, no need to stop
            if current_status in ["Stopped", "Failed", "Completed"]:
                logger.info(f"Job already in terminal state: {current_status}. No stop action needed.")
                return job_status

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                logger.warning(f"Job {training_job_name} not found or already cleaned up")
                return None
            raise

        # Try to stop the job
        try:
            response = sagemaker_client.stop_training_job(TrainingJobName=training_job_name)
            logger.info(f"Stop request sent for job: {training_job_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                logger.warning(f"Job {training_job_name} cannot be stopped (may have already completed)")
                return job_status
            raise

        # Wait for job to stop
        max_wait_time = 300  # 5 minutes
        wait_interval = 10
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            job_status = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)

            status = job_status["TrainingJobStatus"]
            logger.info(f"Job status: {status}")

            if status in ["Stopped", "Failed", "Completed"]:
                logger.info(f"Job reached terminal state: {status}")
                return job_status

            time.sleep(wait_interval)
            elapsed_time += wait_interval

        logger.warning(f"Job did not stop within {max_wait_time} seconds")
        return sagemaker_client.describe_training_job(TrainingJobName=training_job_name)

    except Exception as e:
        logger.error(f"Failed to stop training job: {e}")
        raise


def stop_job_early(
    job_recorder,
    stop_reason: str = "Manual stop for continuous validation",
    region: str = "us-west-2",
    training_job_name: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    """
    Stop a running SageMaker job early and extract its current state/ARN.

    Args:
        job_recorder: JobRecorder object with job information
        stop_reason: Reason for stopping the job
        region: AWS region
        training_job_name: Optional training job name. If not provided, will try to extract from job recorder
        endpoint: Optional custom SageMaker endpoint (for serverless jobs)

    Returns:
        str: Source ARN from the stopped job (if available)
    """
    logger.warning(f"Early job stopping requested: {stop_reason}")

    if not job_recorder or not hasattr(job_recorder, "jobs") or len(job_recorder.jobs) == 0:
        raise ValueError("No jobs found in job recorder")

    # Get the first job
    job_name, job = next(iter(job_recorder.jobs.items()))

    # If training_job_name not provided, try to extract from output_log
    if not training_job_name:
        if hasattr(job, "output_log") and job.output_log:
            training_job_name = extract_training_job_name_from_log(job.output_log)

        if not training_job_name:
            logger.error("Could not extract training job name from job recorder output_log")
            raise ValueError("Training job name not found in job recorder")

    logger.info(f"Using training job name: {training_job_name}")

    # For serverless jobs with custom endpoint, use AWS CLI to get job details
    if endpoint:
        logger.info(f"Using custom endpoint for serverless job: {endpoint}")
        import json
        import subprocess
        import time

        try:
            describe_command = [
                "aws",
                "sagemaker",
                "describe-training-job",
                "--endpoint",
                endpoint,
                "--training-job-name",
                training_job_name,
                "--region",
                region,
            ]

            # Poll for OutputModelPackageArn with retry logic
            max_retries = 60  # Poll for up to 10 minutes
            retry_interval = 10  # Check every 10 seconds

            for attempt in range(max_retries):
                result = subprocess.run(describe_command, capture_output=True, text=True, check=True)
                job_details = json.loads(result.stdout)

                # Extract OutputModelPackageArn from job details
                output_model_arn = job_details.get("OutputModelPackageArn")
                if output_model_arn:
                    logger.info(f"Extracted OutputModelPackageArn from serverless job: {output_model_arn}")
                    return output_model_arn

                # Check job status
                job_status = job_details.get("TrainingJobStatus", "Unknown")
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries}: OutputModelPackageArn not yet available (job status: {job_status})"
                )

                # If job is in terminal state without ARN, something is wrong
                if job_status in ["Failed", "Stopped"]:
                    failure_reason = job_details.get("FailureReason", "Unknown")
                    raise ValueError(
                        f"Job {training_job_name} reached terminal state {job_status} without OutputModelPackageArn. Reason: {failure_reason}"
                    )

                # Wait before next attempt
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)

            # If we exhausted all retries
            raise ValueError(
                f"OutputModelPackageArn not found after {max_retries} attempts for job {training_job_name}"
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to describe serverless job: {e.stderr}")
            raise ValueError(f"Could not describe serverless job {training_job_name}: {e.stderr}")
        except Exception as e:
            logger.error(f"Error getting serverless job details: {e}")
            raise

    # For standard SageMaker jobs, use boto3
    # Stop the training job
    job_status = stop_sagemaker_training_job(training_job_name, region)

    # Extract model artifact location from stopped job (if available)
    if job_status and "ModelArtifacts" in job_status and "S3ModelArtifacts" in job_status["ModelArtifacts"]:
        model_artifacts = job_status["ModelArtifacts"]["S3ModelArtifacts"]
        logger.info(f"Model artifacts location: {model_artifacts}")
    elif not job_status:
        logger.warning("Job status not available - job may have already completed or been cleaned up")

    # Try to extract output model ARN from job recorder logs
    # Use allow_incomplete=True since we're stopping the job early
    try:
        output_arn = extract_output_model_arn(job_recorder, allow_incomplete=True)
        logger.info(f"Extracted ARN from job logs: {output_arn}")
        return output_arn
    except Exception as e:
        logger.error(f"Failed to extract ARN from job logs: {e}")
        raise


def _describe_training_job_with_retry(
    training_job_name: str, region: str, endpoint: Optional[str] = None, max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Describe a SageMaker training job with retry logic.

    Args:
        training_job_name: Name of the SageMaker training job
        region: AWS region
        endpoint: Optional custom SageMaker endpoint (for serverless jobs)
        max_retries: Maximum number of retries

    Returns:
        dict: Job status information, or None if job not found after retries
    """
    import json
    import subprocess
    import time

    retry_count = 0
    while retry_count < max_retries:
        try:
            describe_command = [
                "aws",
                "sagemaker",
                "describe-training-job",
                "--training-job-name",
                training_job_name,
                "--region",
                region,
            ]

            if endpoint:
                describe_command.insert(3, "--endpoint")
                describe_command.insert(4, endpoint)

            result = subprocess.run(describe_command, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            if "ValidationException" in e.stderr:
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"⏳ Job not yet registered in SageMaker, retry {retry_count}/{max_retries}...")
                    time.sleep(10)
                else:
                    raise
            else:
                raise
    return None


def _get_log_streams_for_job(training_job_name: str, region: str) -> list:
    """
    Get CloudWatch log streams for a training job using AWS CLI.

    Args:
        training_job_name: Name of the SageMaker training job
        region: AWS region

    Returns:
        list: List of log streams
    """
    import json
    import subprocess

    log_group_name = "/aws/sagemaker/TrainingJobs"
    describe_streams_cmd = [
        "aws",
        "logs",
        "describe-log-streams",
        "--log-group-name",
        log_group_name,
        "--log-stream-name-prefix",
        training_job_name,
        "--max-items",
        "5",
        "--region",
        region,
    ]

    try:
        result = subprocess.run(describe_streams_cmd, capture_output=True, text=True, check=True)
        log_streams_data = json.loads(result.stdout)
        return log_streams_data.get("logStreams", [])
    except subprocess.CalledProcessError as e:
        if "ResourceNotFoundException" in e.stderr:
            logger.debug(f"Log stream not yet available for {training_job_name}")
        else:
            logger.warning(f"Error accessing logs: {e.stderr}")
        return []


def _get_log_events_since_timestamp(stream_name: str, region: str, last_timestamp: int) -> tuple[list, int]:
    """
    Get log events from a stream since the last timestamp.

    Args:
        stream_name: CloudWatch log stream name
        region: AWS region
        last_timestamp: Last checked timestamp

    Returns:
        tuple: (list of events, new last timestamp)
    """
    import json
    import subprocess

    log_group_name = "/aws/sagemaker/TrainingJobs"
    get_events_cmd = [
        "aws",
        "logs",
        "get-log-events",
        "--log-group-name",
        log_group_name,
        "--log-stream-name",
        stream_name,
        "--start-time",
        str(last_timestamp),
        "--limit",
        "100",
        "--region",
        region,
    ]

    try:
        result = subprocess.run(get_events_cmd, capture_output=True, text=True, check=True)
        log_events_data = json.loads(result.stdout)
        events = log_events_data.get("events", [])

        new_timestamp = last_timestamp
        if events:
            new_timestamp = max(e["timestamp"] for e in events) + 1

        return events, new_timestamp
    except subprocess.CalledProcessError:
        return [], last_timestamp


def _check_checkpoint_in_events(events: list, target_checkpoint_step: Optional[int] = None) -> bool:
    """
    Check if any events contain checkpoint patterns matching the target.
    Uses the existing parse_checkpoint_patterns function for consistency.

    Args:
        events: List of CloudWatch log events
        target_checkpoint_step: Optional specific checkpoint step to look for

    Returns:
        bool: True if matching checkpoint detected
    """
    for event in events:
        message = event.get("message", "")
        timestamp = event.get("timestamp", 0)

        if "checkpoint" in message.lower() or "global_step" in message.lower():
            checkpoint_info = parse_checkpoint_patterns(message)

            if checkpoint_info:
                logger.info(f"📋 Checkpoint log detected:")
                logger.info(f"   Time: {timestamp}")
                logger.info(f"   Type: {checkpoint_info.get('type', 'unknown')}")
                logger.info(f"   Step: {checkpoint_info.get('step', 'N/A')}")
                logger.info(f"   Path: {checkpoint_info.get('path', 'N/A')}")
                logger.info(f"   Message: {message[:150]}")

                if target_checkpoint_step:
                    if checkpoint_info.get("step") == str(target_checkpoint_step):
                        logger.info(f"✓ Target checkpoint global_step_{target_checkpoint_step} detected!")
                        return True
                else:
                    if checkpoint_info.get("type") in ["save", "saved", "saving"]:
                        logger.info(f"✓ Checkpoint save detected!")
                        return True
            else:
                # Fallback to simple pattern matching
                logger.debug(f"Checkpoint-related log: {message[:100]}")

                if target_checkpoint_step:
                    pattern = rf"global_step_{target_checkpoint_step}\b"
                    if re.search(pattern, message):
                        logger.info(f"✓ Target checkpoint global_step_{target_checkpoint_step} detected!")
                        return True
                else:
                    if re.search(r"(?:saving|saved|save).*checkpoint", message, re.IGNORECASE):
                        logger.info(f"✓ Checkpoint save detected!")
                        return True
    return False


def _stop_training_job_with_cli_support(
    training_job_name: str, region: str, endpoint: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Stop a SageMaker training job using AWS CLI for both standard and serverless jobs.

    Args:
        training_job_name: Name of the SageMaker training job
        region: AWS region
        endpoint: Optional custom SageMaker endpoint (for serverless jobs)

    Returns:
        dict: Job status after stopping, or None if job not found
    """
    import subprocess

    stop_command = [
        "aws",
        "sagemaker",
        "stop-training-job",
        "--training-job-name",
        training_job_name,
        "--region",
        region,
    ]

    # Add endpoint if provided (for serverless jobs)
    if endpoint:
        stop_command.insert(3, "--endpoint")
        stop_command.insert(4, endpoint)

    try:
        subprocess.run(stop_command, capture_output=True, text=True, check=True)
        logger.info("Stop request sent successfully")
        # Return job status using the retry function
        return _describe_training_job_with_retry(training_job_name, region, endpoint, max_retries=1)
    except subprocess.CalledProcessError as e:
        if "ValidationException" in e.stderr:
            logger.warning(f"Job may have already completed: {e.stderr}")
            return None
        else:
            raise


def monitor_job_and_stop_at_checkpoint(
    training_job_name: str,
    region: str = "us-west-2",
    target_checkpoint_step: Optional[int] = None,
    max_wait_minutes: int = 30,
    endpoint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Monitor a SageMaker training job's CloudWatch logs and stop it when a checkpoint is detected.
    Uses AWS CLI for all operations.

    Args:
        training_job_name: Name of the SageMaker training job
        region: AWS region
        target_checkpoint_step: Optional specific checkpoint step to wait for (e.g., 7 for global_step_7)
                               If None, stops at first checkpoint detected
        max_wait_minutes: Maximum time to wait for checkpoint before giving up
        endpoint: Optional custom SageMaker endpoint (for serverless jobs)

    Returns:
        dict: Job status after stopping, or None if no checkpoint detected
    """
    import time

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    poll_interval = 10  # Check every 10 seconds

    logger.info(f"Monitoring job {training_job_name} for checkpoint detection")
    if target_checkpoint_step:
        logger.info(f"  Target checkpoint: global_step_{target_checkpoint_step}")
    else:
        logger.info(f"  Will stop at first checkpoint detected")

    last_checked_timestamp = 0
    last_logged_status = None
    poll_count = 0

    # Add initial delay to allow job to be registered in SageMaker
    logger.info("Waiting for job to be registered in SageMaker...")
    time.sleep(10)

    while time.time() - start_time < max_wait_seconds:
        try:
            poll_count += 1
            elapsed_minutes = (time.time() - start_time) / 60

            # Check job status
            job_status = _describe_training_job_with_retry(training_job_name, region, endpoint)
            if not job_status:
                logger.warning(f"Could not describe job {training_job_name} after retries")
                time.sleep(poll_interval)
                continue

            current_status = job_status.get("TrainingJobStatus", "Unknown")
            secondary_status = job_status.get("SecondaryStatus", "N/A")

            # Log status changes or every 6 polls (1 minute)
            if current_status != last_logged_status or poll_count % 6 == 0:
                logger.info(
                    f"Job Status: {current_status} | Secondary: {secondary_status} | Elapsed: {elapsed_minutes:.1f}min"
                )
                last_logged_status = current_status

            # If job completed or failed (but not stopped), return
            # For "Stopped" status, we need to continue polling to get OutputModelPackageArn
            if current_status in ["Completed", "Failed"]:
                logger.info(f"Job reached terminal state: {current_status}")
                return job_status

            # Get log streams using AWS CLI (similar to existing fetch_cloudwatch_logs but for streams)
            log_streams = _get_log_streams_for_job(training_job_name, region)
            if not log_streams:
                logger.debug(f"No log streams yet for {training_job_name}")
                time.sleep(poll_interval)
                continue

            # Check logs from all streams for checkpoint patterns
            checkpoint_detected = False
            for stream in log_streams:
                stream_name = stream["logStreamName"]
                events, last_checked_timestamp = _get_log_events_since_timestamp(
                    stream_name, region, last_checked_timestamp
                )

                if events and _check_checkpoint_in_events(events, target_checkpoint_step):
                    checkpoint_detected = True
                    break

            if checkpoint_detected:
                # Stop the job using existing function with CLI support
                logger.info(f"Stopping job {training_job_name} after checkpoint detection")
                final_job_status = _stop_training_job_with_cli_support(training_job_name, region, endpoint)

                # Wait a bit for the job to fully stop if needed
                if final_job_status and final_job_status.get("TrainingJobStatus") not in [
                    "Stopped",
                    "Failed",
                    "Completed",
                ]:
                    import time

                    for _ in range(30):  # Wait up to 5 minutes
                        time.sleep(10)
                        updated_status = _describe_training_job_with_retry(
                            training_job_name, region, endpoint, max_retries=1
                        )
                        if updated_status and updated_status.get("TrainingJobStatus") in [
                            "Stopped",
                            "Failed",
                            "Completed",
                        ]:
                            return updated_status

                return final_job_status

            time.sleep(poll_interval)

        except Exception as e:
            logger.error(f"Error monitoring job: {e}")
            raise

    logger.warning(f"No checkpoint detected within {max_wait_minutes} minutes")
    return None
