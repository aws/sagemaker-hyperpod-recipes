import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import boto3
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Update private hub by exporting content and generating launch JSONs")
    parser.add_argument("--hub-name", required=True, help="Private hub name")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket for artifacts")
    parser.add_argument("--region", required=True, help="AWS region")
    parser.add_argument(
        "--output-dir", default="results/hub_update_output", help="Output directory for generated files"
    )
    parser.add_argument(
        "--model-id-map",
        default="launcher/recipe_templatization/jumpstart_model-id_map.json",
        help="Path to model ID mapping file",
    )
    parser.add_argument("--version", default="1.0.0", help="Version for the hub content (format: x.xx.xx).")
    parser.add_argument("--endpoint", default="prod", help="SageMaker endpoint, beta/prod")

    # Recipe input options (mutually exclusive)
    recipe_group = parser.add_mutually_exclusive_group(required=True)
    recipe_group.add_argument(
        "--recipe-regex",
        help="Regex pattern to match recipe files in recipes_collection/recipes/ (e.g., '.*llama.*\\.yaml$' or 'fine-tuning/deepseek/.*')",
    )
    recipe_group.add_argument(
        "--recipe-files",
        help="Comma-separated list of recipe file paths relative to recipes_collection/recipes/ (e.g., 'fine-tuning/llama/recipe1.yaml,fine-tuning/qwen/recipe2.yaml')",
    )

    return parser.parse_args()


def load_recipes_from_regex(regex_pattern):
    """Load recipe files matching a regex pattern from recipes_collection/recipes/"""
    base_path = Path("recipes_collection/recipes")

    print(f"Searching for recipes matching regex: {regex_pattern}")

    # Compile the regex pattern
    try:
        pattern = re.compile(regex_pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{regex_pattern}': {e}")

    recipe_files = []

    # Walk through all files in recipes_collection/recipes/
    for file_path in base_path.rglob("*.yaml"):
        # Get the relative path from base_path for matching
        relative_path = str(file_path.relative_to(base_path))

        # Also try matching against the full path from recipes_collection/recipes/
        full_relative_path = str(file_path)

        if pattern.search(relative_path) or pattern.search(full_relative_path):
            recipe_files.append(str(file_path))

    if not recipe_files:
        print(f"Warning: No recipe files found matching regex: {regex_pattern}")
    else:
        print(f"Found {len(recipe_files)} recipes matching regex pattern:")
        for f in recipe_files:
            print(f"  - {f}")

    return recipe_files


def load_recipes_from_file_list(file_list_str):
    """Load recipe files from a comma-separated list of file paths"""
    base_path = "recipes_collection/recipes/"

    print(f"Loading recipes from file list")

    # Split by comma and strip whitespace
    file_paths = [f.strip() for f in file_list_str.split(",") if f.strip()]

    recipe_files = []

    for file_path in file_paths:
        # Handle both relative paths (from recipes_collection/recipes/) and full paths
        if file_path.startswith("recipes_collection/recipes/"):
            full_path = file_path
        else:
            full_path = os.path.join(base_path, file_path)

        if os.path.exists(full_path):
            recipe_files.append(full_path)
            print(f"  - {full_path}")
        else:
            print(f"Warning: Recipe file not found: {full_path}")

    print(f"Loaded {len(recipe_files)} valid recipes from file list")
    return recipe_files


def load_recipes(args):
    """Load recipe files based on CLI arguments (regex or file list)"""
    if args.recipe_regex:
        return load_recipes_from_regex(args.recipe_regex)
    elif args.recipe_files:
        return load_recipes_from_file_list(args.recipe_files)
    else:
        raise ValueError("Either --recipe-regex or --recipe-files must be provided")


def get_model_name_from_recipe(recipe_file):
    """Get model name for JS model id mapping"""

    with open(recipe_file, "r") as f:
        recipe_data = yaml.safe_load(f)

    if is_nova_recipe(recipe_path=recipe_file):
        recipe_name = recipe_data.get("run", {}).get("model_type")
    else:
        recipe_name = recipe_data.get("run", {}).get("name")

    if not recipe_name:
        raise ValueError(f"Recipe name not found in file: {recipe_file}")

    return recipe_name


def get_recipe_name_from_path(recipe_file_path):
    """Get recipe from recipe path"""
    # path_string = "recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/SFT/nova_micro_1_0_p5_p4d_gpu_lora_sft.yaml"
    # return # nova_micro_1_0_p5_p4d_gpu_lora_sft
    return Path(recipe_file_path).stem


def generate_launch_json(recipe_file, recipe_name, output_dir):
    # Convert file path to recipe path for launch json
    # e.g., "recipes_collection/recipes/fine-tuning/deepseek/recipe.yaml" -> "fine-tuning/deepseek/recipe"
    recipe_path = get_recipe_path_from_file(recipe_file)

    cmd = build_launch_json_command(recipe_path, recipe_name, output_dir)

    print(f"Generating launch.json for {recipe_name}...")
    print(f"Command: {' '.join(cmd)}")

    # Execute command
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Failed to generate launch.json for {recipe_name}: {result.stderr}")
        return None

    launch_json_path = extract_launch_json_path_from_output(result.stdout)
    return launch_json_path


def get_recipe_path_from_file(recipe_file):
    """Convert recipe file path to recipe path for main.py"""
    # Remove recipes_collection/ prefix and .yaml suffix
    path = recipe_file.replace("recipes_collection/recipes/", "")
    if path.endswith(".yaml"):
        path = path[:-5]
    return path


def build_launch_json_command(recipe_path, recipe_name, results_dir):
    """Build command to generate launch.json (inspired by your example)"""

    # Base command - similar to _build_standard_command
    cmd = [
        "python3",
        "main.py",
        f"recipes={recipe_path}",
        "cluster_type=k8s",
        "cluster=k8s",
        f"base_results_dir={results_dir}",
        "container=test_container",
        "git.use_default=false",
        "git.entry_script=/app/src/train_hp.py",
        "launch_json=true",
        "++recipes.training_config.model_config.model_name_or_path=''",
        "++recipes.training_config.training_args.training_dir=''",
        "++recipes.training_config.datasets.train_data.name=''",
        "++recipes.training_config.datasets.train_data.file_path=''",
        "++recipes.training_config.datasets.val_data.name=''",
        "++recipes.training_config.datasets.val_data.file_path=''",
    ]

    # Add recipe-specific parameters based on recipe type
    if is_nova_recipe(recipe_path):
        cmd.extend(["instance_type=p5.48xlarge"])
    else:
        cmd.extend(
            [
                "+cluster.persistent_volume_claims.0.claimName=fsx-claim",
                "+cluster.persistent_volume_claims.0.mountPath=/data",
            ]
        )

    return cmd


def is_nova_recipe(recipe_path):
    """Check if recipe is a Nova recipe"""
    return "nova" in recipe_path.lower()


def extract_launch_json_path_from_output(stdout):
    """Extract the launch.json file path from command output"""
    for line in stdout.split("\n"):
        if "launch.json" in line:
            path = line.strip().split()[-1]
            if os.path.exists(path):
                return path

    return None


def upload_artifacts_to_s3(launch_json_path, recipe_name, s3_bucket, region, version):
    """Upload recipe artifacts to S3 and return S3 URIs"""
    with open(launch_json_path, "r") as f:
        launch_data = json.load(f)

    yaml_content = extract_yaml_content(launch_data)
    k8s_yaml_content = yaml_content.replace("test_container", "{{container_image}}")
    sm_jobs_yaml_content = extract_sm_jobs_yaml_content(launch_json_path, recipe_name)
    override_params = launch_data.get("recipe_override_parameters", {})

    s3_keys = {
        "k8s_yaml": f"recipes/{recipe_name}_payload_template_k8s_{version}.yaml",
        "k8s_json": f"recipes/{recipe_name}_override_params_k8s_{version}.json",
        "sm_jobs_yaml": f"recipes/{recipe_name}_payload_template_sm_jobs_{version}.yaml",
        "sm_jobs_json": f"recipes/{recipe_name}_override_params_sm_jobs_{version}.json",
    }

    s3_client = boto3.client("s3", region_name=region)
    s3_uris = {}

    # Upload k8s artifacts
    upload_yaml_to_s3(s3_client, k8s_yaml_content, s3_bucket, s3_keys["k8s_yaml"])
    upload_json_to_s3(s3_client, override_params, s3_bucket, s3_keys["k8s_json"])

    # Upload sm_jobs artifacts
    upload_yaml_to_s3(s3_client, sm_jobs_yaml_content, s3_bucket, s3_keys["sm_jobs_yaml"])
    upload_json_to_s3(s3_client, override_params, s3_bucket, s3_keys["sm_jobs_json"])

    # Generate S3 URIs
    for key_name, s3_key in s3_keys.items():
        s3_uris[key_name] = f"s3://{s3_bucket}/{s3_key}"

    print(f"Uploaded artifacts for recipe={recipe_name} to S3")
    return s3_uris


def extract_yaml_content(launch_data):
    """Extract and combine YAML content from launch.json"""
    yaml_fields = []
    for key, value in launch_data.items():
        if key.endswith(".yaml"):
            yaml_fields.append(value)
    return "".join(yaml_fields)


def extract_sm_jobs_yaml_content(launch_json_path, recipe_name):
    recipe_dir = os.path.dirname(launch_json_path)

    print(f"Searching *_hydra.yaml under {recipe_dir}")
    yaml_path = list(Path(recipe_dir).rglob("*_hydra.yaml"))

    if yaml_path:
        with open(yaml_path[0], "r") as f:
            yaml_content = f.read()
    else:
        print(f"SM Jobs YAML file not found: {yaml_path}")
        return None

    return yaml_content


def upload_yaml_to_s3(s3_client, content, bucket, key):
    s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"), Tagging="SageMaker=True")


def upload_json_to_s3(s3_client, data, bucket, key):
    content = json.dumps(data, indent=2)
    s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"), Tagging="SageMaker=True")


def process_recipe_metadata(launch_json_path, s3_uris, region):
    """Use launch.json to create recipe collection metadata"""
    with open(launch_json_path, "r") as f:
        launch_data = json.load(f)

    metadata = launch_data["metadata"]

    # Build RecipeCollection entry
    recipe_entry = {
        "DisplayName": metadata.get("DisplayName"),
        "Name": metadata.get("Name"),
        "CustomizationTechnique": metadata.get("CustomizationTechnique", "").upper(),
        "InstanceCount": metadata.get("InstanceCount", 1),
        "ServerlessMeteringType": metadata.get("ServerlessMeteringType", "Token-based"),
        "Type": metadata.get("Type"),
        "Versions": metadata.get("Versions", ["1.0.0"]),
        "Hardware": metadata.get("Hardware", "").upper(),
        "SupportedInstanceTypes": metadata.get("InstanceTypes", []),
        "RecipeFilePath": metadata.get("RecipeFilePath"),
        "SequenceLength": metadata.get("SequenceLength"),
        "HostingConfigs": metadata.get("HostingConfigs", []),
        # Add S3 URIs
        "HpEksPayloadTemplateS3Uri": s3_uris["k8s_yaml"],
        "HpEksOverrideParamsS3Uri": s3_uris["k8s_json"],
        "SmtjRecipeTemplateS3Uri": s3_uris["sm_jobs_yaml"],
        "SmtjOverrideParamsS3Uri": s3_uris["sm_jobs_json"],
        # Add regional ECR URI
        "SmtjImageUri": get_regional_ecr_uri(launch_data, region),
    }

    if metadata.get("Peft"):
        recipe_entry["Peft"] = metadata.get("Peft", "").upper()

    return recipe_entry


def get_regional_ecr_uri(launch_data, region):
    """Extract regional ECR URI from launch data"""
    regional_params = launch_data.get("regional_parameters", {})
    ecr_uris = regional_params.get("smtj_regional_ecr_uri", {})
    prod_uris = ecr_uris.get("prod", {})
    return prod_uris.get(region)


def create_new_recipecollection(exported_json_path):
    with open(exported_json_path, "r") as f:
        hub_content = json.load(f)

    hub_content["HubContentDocument"]["RecipeCollection"] = []

    with open(exported_json_path, "w") as f:
        json.dump(hub_content, f, indent=2)

    print(f"✓ Successfully initialized RecipeCollection in {exported_json_path}")
    return True


def update_exported_json(exported_json_path, new_recipe_entries, model_id, version):
    """Add new recipe entries to exported hub content JSON"""
    with open(exported_json_path, "r") as f:
        hub_content = json.load(f)

    hub_content["HubContentDocument"]["RecipeCollection"].extend(new_recipe_entries)

    hub_content["HubContentVersion"] = version

    # Save updated JSON
    with open(exported_json_path, "w") as f:
        json.dump(hub_content, f, indent=2)

    print(f"Updated {exported_json_path} with {len(new_recipe_entries)} new recipes")


def export_hub_content(model_id, region, output_dir, endpoint):
    export_script_path = os.path.join("scripts", "model_hub", "export_hub_content.py")
    abs_export_script_path = os.path.abspath(export_script_path)

    # Build command
    cmd = [
        "python3",
        abs_export_script_path,
        "--hub-name",
        "SageMakerPublicHub",
        "--content-name",
        model_id,
        "--region",
        region,
        "--endpoint",
        endpoint,
    ]

    print(f"Exporting hub content for model: {model_id}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=output_dir)

        export_filename = f"{model_id}_export.json"
        export_path = os.path.join(output_dir, export_filename)

        if os.path.exists(export_path):
            print(f"Exported hub content to: {export_path}")
            return export_path
        else:
            print(f"Export file {export_filename} not found in {output_dir}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Failed to export hub content for {model_id}: {e.stderr}")
        return None


def import_hub_content(exported_json_path, private_hub_name, region, endpoint):
    import_script_path = os.path.join("scripts", "model_hub", "import_hub_content.py")

    cmd = [
        "python3",
        import_script_path,
        "--hub-name",
        private_hub_name,
        "--input",
        exported_json_path,
        "--region",
        region,
        "--endpoint",
        endpoint,
    ]

    print(f"Importing hub content from: {exported_json_path}")
    print(f"Target hub: {private_hub_name}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.returncode != 0:
        print(f"Failed to import hub content due to: {result.stderr}")
        return False

    print(f"Successfully imported hub content to {private_hub_name}")

    print(f"Output: {result.stdout}")

    return True


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ["AWS_REGION"] = args.region

    # Load jumpstart recipe name -> model id map
    model_id_mapping = json.load(open(args.model_id_map))

    recipe_files = load_recipes(args)

    if not recipe_files:
        print("No recipe files found. Exiting.")
        return

    processed_models = set()

    recipe_metadata_by_model = {}

    results = {
        "exported_models": {},
        "generated_launch_jsons": {},
        "uploaded_artifacts": {},
        "recipe_metadata": {},
    }

    for recipe_file in recipe_files:
        print(f"\nProcessing recipe file: {recipe_file}")

        recipe_name = get_recipe_name_from_path(recipe_file)
        model_name = get_model_name_from_recipe(recipe_file)

        model_id = model_id_mapping.get(model_name)

        # Get hub content from JumpStart's hub
        if model_id not in processed_models:
            exported_json_path = export_hub_content(model_id, args.region, args.output_dir, args.endpoint)
            if exported_json_path:
                results["exported_models"][model_id] = exported_json_path
                if create_new_recipecollection(exported_json_path):
                    print(f"Prepared {model_id} for new recipe collection")
                else:
                    print(f"Failed to create {model_id} new recipe collection")

                processed_models.add(model_id)

        # Generate launch.jsons
        launch_json_path = generate_launch_json(recipe_file, recipe_name, args.output_dir)
        if launch_json_path:
            results["generated_launch_jsons"][recipe_name] = launch_json_path

            s3_uris = upload_artifacts_to_s3(launch_json_path, recipe_name, args.s3_bucket, args.region, args.version)
            results["uploaded_artifacts"][recipe_name] = s3_uris

            recipe_metadata = process_recipe_metadata(launch_json_path, s3_uris, args.region)
            results["recipe_metadata"][recipe_name] = recipe_metadata

            if model_id not in recipe_metadata_by_model:
                recipe_metadata_by_model[model_id] = []
            recipe_metadata_by_model[model_id].append(recipe_metadata)

    # Update exported hub content with recipe metadata from launch.jsons
    print("TEST")
    for model_id, recipe_entries in recipe_metadata_by_model.items():
        if model_id in results["exported_models"]:
            exported_json_path = results["exported_models"][model_id]
            update_exported_json(exported_json_path, recipe_entries, model_id, args.version)
            print(f"Updated {model_id} with {len(recipe_entries)} recipe entries")

    # Import exported hub content to private hub
    for model_id in recipe_metadata_by_model:
        exported_json_path = results["exported_models"][model_id]
        import_hub_content(
            exported_json_path, private_hub_name=args.hub_name, region=args.region, endpoint=args.endpoint
        )


if __name__ == "__main__":
    main()
