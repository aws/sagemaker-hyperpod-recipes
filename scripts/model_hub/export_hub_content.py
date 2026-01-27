#!/usr/bin/env python3
"""
Export SageMaker Hub content metadata to JSON file.
"""

import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export SageMaker Hub content metadata to a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_hub_content.py --hub-name SageMakerPublicHub --content-name my-model
  python export_hub_content.py --hub-name MyHub --content-name my-notebook --content-type Notebook --output my_export.json
        """,
    )

    parser.add_argument("--hub-name", default="SageMakerPublicHub", help="Name of the SageMaker Hub")

    parser.add_argument("--content-name", required=True, help="Name of the Hub content to export")

    parser.add_argument("--content-type", default="Model", help="Type of Hub content (default: Model)")

    parser.add_argument("--region", help="AWS region (defaults to AWS config default)")

    parser.add_argument("--output", help="Output JSON file path (default: {content-name}_export.json)")

    parser.add_argument(
        "--endpoint", choices=["beta", "gamma", "prod"], default="prod", help="Environment endpoint (default: prod)"
    )

    return parser.parse_args()


def describe_hub_content(hub_name, content_name, content_type, version=None, region=None, endpoint=None):
    """
    Call AWS SageMaker DescribeHubContent API.

    Args:
        hub_name: Name of the SageMaker Hub
        content_name: Name of the Hub content
        content_type: Type of content (Model, Notebook, ModelReference)
        region: AWS region (optional)
        endpoint: Environment endpoint (optional)

    Returns:
        dict: Response from DescribeHubContent API
    """
    # Initialize boto3 SageMaker client
    client_kwargs = {}
    if region:
        client_kwargs["region_name"] = region

    # Set endpoint URL based on environment
    if endpoint == "beta":
        client_kwargs["endpoint_url"] = f"https://sagemaker.beta.{region}.ml-platform.aws.a2z.com"
    elif endpoint == "gamma":
        client_kwargs["endpoint_url"] = f"https://sagemaker.gamma.{region}.ml-platform.aws.a2z.com"
    # For 'prod' or None, don't set endpoint_url (use default)

    client = boto3.client("sagemaker", **client_kwargs)

    # Call DescribeHubContent API
    response = client.describe_hub_content(
        HubName=hub_name,
        HubContentType=content_type,
        HubContentName=content_name,
    )

    return response


def parse_hub_content_document(data):
    """
    Parse stringified HubContentDocument JSON into object.

    Args:
        data: Response data from DescribeHubContent API

    Returns:
        dict: Modified data with parsed HubContentDocument
    """
    if "HubContentDocument" in data and data["HubContentDocument"]:
        try:
            # Parse the stringified JSON into a proper object
            if isinstance(data["HubContentDocument"], str):
                data["HubContentDocument"] = json.loads(data["HubContentDocument"])
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse HubContentDocument as JSON: {e}")
            print("HubContentDocument will be saved as-is.")

    return data


def save_to_json(data, output_path):
    """
    Save data to JSON file with proper formatting.

    Args:
        data: Data to save
        output_path: Path to output file
    """
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    """Main orchestration function for export workflow."""
    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Generate default output filename if not provided
        output_path = args.output or f"{args.content_name}_export.json"

        print(f"Exporting Hub content '{args.content_name}' from Hub '{args.hub_name}'...")

        # Call DescribeHubContent API
        response = describe_hub_content(
            hub_name=args.hub_name,
            content_name=args.content_name,
            content_type=args.content_type,
            region=args.region,
            endpoint=args.endpoint,
        )

        print("Hub content retrieved successfully.")

        # Parse HubContentDocument
        response = parse_hub_content_document(response)

        # Save to JSON file
        save_to_json(response, output_path)

        print(f"✓ Export complete! Saved to: {output_path}")

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

        if error_code == "ResourceNotFound":
            print(f"Error: Hub content not found.")
            print(f"Details: {error_message}")
        elif error_code in ["AccessDenied", "UnauthorizedOperation"]:
            print(f"Error: Access denied.")
            print(f"Details: {error_message}")
            print("Please check your IAM permissions for SageMaker DescribeHubContent.")
        else:
            print(f"AWS API Error: {error_message}")

        sys.exit(1)

    except (IOError, PermissionError) as e:
        print(f"Error: Could not write to file '{output_path}'.")
        print(f"Details: {e}")
        print("Please check file system permissions.")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
