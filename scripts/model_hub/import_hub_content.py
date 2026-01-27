#!/usr/bin/env python3
"""
Import modified SageMaker Hub content from JSON file.
"""

import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import modified SageMaker Hub content from a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python import_hub_content.py --hub-name MyHub --input my-model_export.json
  python import_hub_content.py --hub-name MyHub --input my_content.json --region us-west-2
        """,
    )

    parser.add_argument("--hub-name", default="SageMakerPublicHub", help="Target Hub name for import")

    parser.add_argument("--input", required=True, help="Path to the JSON file containing Hub content data")

    parser.add_argument("--region", required=True, help="AWS region (defaults to AWS config default)")

    parser.add_argument(
        "--endpoint", choices=["beta", "gamma", "prod"], default="prod", help="Environment endpoint (default: prod)"
    )

    return parser.parse_args()


def load_json(file_path):
    """
    Load and parse JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        dict: Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON format in file '{file_path}' at line {e.lineno}, column {e.colno}: {e.msg}", e.doc, e.pos
        )


def validate_json_structure(data):
    """
    Validate that required fields are present in JSON data.

    Args:
        data: Parsed JSON data

    Returns:
        list: List of missing required fields (empty if valid)
    """
    required_fields = ["HubContentType", "HubContentName", "HubContentDocument"]
    missing_fields = []

    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)

    return missing_fields


def stringify_hub_content_document(data):
    """
    Convert HubContentDocument object to stringified JSON.

    Args:
        data: JSON data with HubContentDocument

    Returns:
        dict: Modified data with stringified HubContentDocument
    """
    if "HubContentDocument" in data and data["HubContentDocument"]:
        # If it's already a string, leave it as-is
        if not isinstance(data["HubContentDocument"], str):
            # Convert object to stringified JSON
            data["HubContentDocument"] = json.dumps(data["HubContentDocument"])

    return data


def import_hub_content(hub_name, content_data, region, endpoint=None):
    """
    Call AWS SageMaker ImportHubContent API.

    Args:
        hub_name: Target Hub name
        content_data: Hub content data from JSON
        region: AWS region (optional)
        endpoint: Environment endpoint (optional)

    Returns:
        dict: Response from ImportHubContent API
    """
    # Initialize boto3 SageMaker client
    client_kwargs = {}
    client_kwargs["region_name"] = region

    # Set endpoint URL based on environment
    if endpoint == "beta":
        client_kwargs["endpoint_url"] = f"https://sagemaker.beta.{region}.ml-platform.aws.a2z.com"
    elif endpoint == "gamma":
        client_kwargs["endpoint_url"] = f"https://sagemaker.gamma.{region}.ml-platform.aws.a2z.com"
    # For 'prod' or None, don't set endpoint_url (use default)

    client = boto3.client("sagemaker", **client_kwargs)

    # Prepare API parameters
    api_params = {
        "HubName": hub_name,
        "HubContentType": content_data["HubContentType"],
        "HubContentName": content_data["HubContentName"],
        "HubContentDocument": content_data["HubContentDocument"],
        "DocumentSchemaVersion": "2.4.0",
    }

    # Add optional fields if present
    optional_fields = [
        "HubContentVersion",
        "HubContentDisplayName",
        "HubContentDescription",
        "HubContentMarkdown",
        "HubContentSearchKeywords",
        "Tags",
    ]

    for field in optional_fields:
        if field in content_data and content_data[field] is not None:
            api_params[field] = content_data[field]

    # Call ImportHubContent API
    response = client.import_hub_content(**api_params)

    return response


def main():
    """Main orchestration function for import workflow."""
    try:
        # Parse command-line arguments
        args = parse_arguments()

        print(f"Loading JSON file: {args.input}")

        # Load JSON file
        content_data = load_json(args.input)

        print("JSON file loaded successfully.")

        # Validate JSON structure
        missing_fields = validate_json_structure(content_data)

        if missing_fields:
            print(f"Error: JSON file is missing required fields: {', '.join(missing_fields)}")
            print("\nRequired fields:")
            print("  - HubContentType")
            print("  - HubContentName")
            print("  - HubContentDocument")
            sys.exit(1)

        print("JSON validation passed.")

        # Stringify HubContentDocument
        content_data = stringify_hub_content_document(content_data)

        print(f"Importing Hub content '{content_data['HubContentName']}' to Hub '{args.hub_name}'...")

        # Call ImportHubContent API
        response = import_hub_content(
            hub_name=args.hub_name, content_data=content_data, region=args.region, endpoint=args.endpoint
        )

        # Display success message with ARN
        hub_content_arn = response.get("HubContentArn", "N/A")
        print(f"✓ Import complete!")
        print(f"Hub Content ARN: {hub_content_arn}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the input file path is correct.")
        sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Error: {e}")
        print("Please ensure the file contains valid JSON.")
        sys.exit(1)

    except NoCredentialsError:
        print("Error: AWS credentials not configured.")
        print("Please configure your AWS credentials using one of these methods:")
        print("  - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        print("  - Configure ~/.aws/credentials file")
        print("  - Use AWS_PROFILE environment variable")
        sys.exit(1)

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]

        print(f"AWS API Error ({error_code}): {error_message}")

        if error_code == "ResourceNotFound":
            print("The specified Hub may not exist.")
        elif error_code in ["AccessDenied", "UnauthorizedOperation"]:
            print("Please check your IAM permissions for SageMaker ImportHubContent.")

        sys.exit(1)

    except (IOError, PermissionError) as e:
        print(f"Error: Could not read file '{args.input}'.")
        print(f"Details: {e}")
        print("Please check file system permissions.")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
