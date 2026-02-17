"""Unit tests for update_private_hub.py"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.model_hub.update_private_hub import (
    bump_version,
    create_new_recipecollection,
    extract_sm_jobs_yaml_content,
    extract_yaml_content,
    get_model_name_from_recipe,
    get_recipe_name_from_path,
    get_regional_ecr_uri,
    get_version_from_export,
    is_nova_recipe,
    load_recipes,
    load_recipes_from_file_list,
    load_recipes_from_regex,
    process_recipe_metadata,
    update_exported_json,
    upload_json_to_s3,
    upload_yaml_to_s3,
)


class TestLoadRecipesFromRegex(unittest.TestCase):
    """Test suite for load_recipes_from_regex function."""

    def test_invalid_regex_pattern(self):
        """Test that invalid regex raises ValueError."""
        with self.assertRaises(ValueError) as context:
            load_recipes_from_regex("[invalid")
        self.assertIn("Invalid regex pattern", str(context.exception))

    @patch("scripts.model_hub.update_private_hub.Path")
    def test_no_matches_returns_empty_list(self, mock_path):
        """Test that no matches returns empty list."""
        mock_base_path = MagicMock()
        mock_base_path.rglob.return_value = []
        mock_path.return_value = mock_base_path

        result = load_recipes_from_regex("nonexistent.*")
        self.assertEqual(result, [])

    @patch("scripts.model_hub.update_private_hub.Path")
    def test_regex_matches_files(self, mock_path):
        """Test that regex properly matches files."""
        mock_base_path = MagicMock()

        # Create mock file paths
        mock_file1 = MagicMock()
        mock_file1.relative_to.return_value = Path("fine-tuning/llama/recipe1.yaml")
        mock_file1.__str__ = lambda self: "recipes_collection/recipes/fine-tuning/llama/recipe1.yaml"

        mock_file2 = MagicMock()
        mock_file2.relative_to.return_value = Path("fine-tuning/qwen/recipe2.yaml")
        mock_file2.__str__ = lambda self: "recipes_collection/recipes/fine-tuning/qwen/recipe2.yaml"

        mock_base_path.rglob.return_value = [mock_file1, mock_file2]
        mock_path.return_value = mock_base_path

        result = load_recipes_from_regex(".*llama.*")
        self.assertEqual(len(result), 1)
        self.assertIn("llama", result[0])


class TestLoadRecipesFromFileList(unittest.TestCase):
    """Test suite for load_recipes_from_file_list function."""

    @patch("os.path.exists")
    def test_file_list_parsing(self, mock_exists):
        """Test parsing comma-separated file list."""
        mock_exists.return_value = True

        result = load_recipes_from_file_list("recipe1.yaml, recipe2.yaml, recipe3.yaml")
        self.assertEqual(len(result), 3)

    @patch("os.path.exists")
    def test_nonexistent_files_not_included(self, mock_exists):
        """Test that non-existent files are not included."""
        mock_exists.side_effect = lambda path: "existing" in path

        result = load_recipes_from_file_list("existing.yaml,nonexistent.yaml")
        self.assertEqual(len(result), 1)

    @patch("os.path.exists")
    def test_handles_full_path(self, mock_exists):
        """Test handling of full paths starting with recipes_collection/recipes/."""
        mock_exists.return_value = True

        result = load_recipes_from_file_list("recipes_collection/recipes/recipe1.yaml")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "recipes_collection/recipes/recipe1.yaml")


class TestLoadRecipes(unittest.TestCase):
    """Test suite for load_recipes function."""

    @patch("scripts.model_hub.update_private_hub.load_recipes_from_regex")
    def test_dispatches_to_regex_loader(self, mock_regex_loader):
        """Test that load_recipes dispatches to regex loader when recipe_regex is set."""
        mock_regex_loader.return_value = ["recipe1.yaml"]
        args = MagicMock()
        args.recipe_regex = ".*llama.*"
        args.recipe_files = None

        result = load_recipes(args)
        mock_regex_loader.assert_called_once_with(".*llama.*")
        self.assertEqual(result, ["recipe1.yaml"])

    @patch("scripts.model_hub.update_private_hub.load_recipes_from_file_list")
    def test_dispatches_to_file_list_loader(self, mock_file_list_loader):
        """Test that load_recipes dispatches to file list loader when recipe_files is set."""
        mock_file_list_loader.return_value = ["recipe1.yaml"]
        args = MagicMock()
        args.recipe_regex = None
        args.recipe_files = "recipe1.yaml"

        result = load_recipes(args)
        mock_file_list_loader.assert_called_once_with("recipe1.yaml")
        self.assertEqual(result, ["recipe1.yaml"])

    def test_raises_error_when_no_option_provided(self):
        """Test that error is raised when neither option is provided."""
        args = MagicMock()
        args.recipe_regex = None
        args.recipe_files = None

        with self.assertRaises(ValueError) as context:
            load_recipes(args)
        self.assertIn("Either --recipe-regex or --recipe-files must be provided", str(context.exception))


class TestGetModelNameFromRecipe(unittest.TestCase):
    """Test suite for get_model_name_from_recipe function."""

    def test_extract_name_from_standard_recipe(self):
        """Test extracting name from standard recipe."""
        yaml_content = """
run:
  name: llama-3-8b
  epochs: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                result = get_model_name_from_recipe(f.name)
                self.assertEqual(result, "llama-3-8b")
            finally:
                os.unlink(f.name)

    def test_extract_model_type_from_nova_recipe(self):
        """Test extracting model_type from Nova recipe."""
        yaml_content = """
run:
  model_type: nova-lite
  epochs: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                # We need to mock is_nova_recipe to return True
                with patch("scripts.model_hub.update_private_hub.is_nova_recipe", return_value=True):
                    result = get_model_name_from_recipe(f.name)
                    self.assertEqual(result, "nova-lite")
            finally:
                os.unlink(f.name)

    def test_raises_error_when_name_not_found(self):
        """Test that ValueError is raised when name is not found."""
        yaml_content = """
run:
  epochs: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                with self.assertRaises(ValueError) as context:
                    get_model_name_from_recipe(f.name)
                self.assertIn("Recipe name not found", str(context.exception))
            finally:
                os.unlink(f.name)


class TestIsNovaRecipe(unittest.TestCase):
    """Test suite for is_nova_recipe function."""

    def test_nova_recipe_detection(self):
        """Test detection of Nova recipes."""
        self.assertTrue(is_nova_recipe("/path/to/nova/recipe.yaml"))
        self.assertTrue(is_nova_recipe("/path/to/NOVA/recipe.yaml"))
        self.assertTrue(is_nova_recipe("recipes_collection/recipes/nova-lite-sft.yaml"))

    def test_non_nova_recipe_detection(self):
        """Test detection of non-Nova recipes."""
        self.assertFalse(is_nova_recipe("/path/to/llama/recipe.yaml"))
        self.assertFalse(is_nova_recipe("recipes_collection/recipes/llama-8b.yaml"))


class TestGetRecipeNameFromPath(unittest.TestCase):
    """Test suite for get_recipe_name_from_path function."""

    def test_extract_recipe_name(self):
        """Test extraction of recipe name from path."""
        self.assertEqual(get_recipe_name_from_path("/path/to/llama-8b.yaml"), "llama-8b")
        self.assertEqual(get_recipe_name_from_path("recipes/nova-lite-sft.yaml"), "nova-lite-sft")
        self.assertEqual(get_recipe_name_from_path("recipe.yaml"), "recipe")


class TestExtractYamlContent(unittest.TestCase):
    """Test suite for extract_yaml_content function."""

    def test_extract_yaml_fields(self):
        """Test extraction of YAML fields from launch data."""
        launch_data = {
            "config.yaml": "key1: value1\n",
            "training.yaml": "key2: value2\n",
            "other_field": "not yaml content",
        }
        result = extract_yaml_content(launch_data)
        self.assertIn("key1: value1", result)
        self.assertIn("key2: value2", result)
        self.assertNotIn("not yaml content", result)

    def test_empty_launch_data(self):
        """Test with empty launch data."""
        result = extract_yaml_content({})
        self.assertEqual(result, "")


class TestExtractSmJobsYamlContent(unittest.TestCase):
    """Test suite for extract_sm_jobs_yaml_content function."""

    def test_finds_hydra_yaml(self):
        """Test finding *_hydra.yaml file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a hydra yaml file
            hydra_file = Path(tmpdir) / "test_hydra.yaml"
            hydra_file.write_text("test: content\n")

            launch_json_path = Path(tmpdir) / "launch.json"

            result = extract_sm_jobs_yaml_content(str(launch_json_path), "test_recipe")
            self.assertEqual(result, "test: content\n")

    def test_returns_none_when_not_found(self):
        """Test returns None when hydra yaml not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            launch_json_path = Path(tmpdir) / "launch.json"

            result = extract_sm_jobs_yaml_content(str(launch_json_path), "test_recipe")
            self.assertIsNone(result)


class TestUploadYamlToS3(unittest.TestCase):
    """Test suite for upload_yaml_to_s3 function."""

    def test_upload_with_tagging(self):
        """Test that upload includes SageMaker tagging."""
        mock_client = MagicMock()
        content = "key: value"
        bucket = "test-bucket"
        key = "test-key.yaml"

        upload_yaml_to_s3(mock_client, content, bucket, key)

        mock_client.put_object.assert_called_once_with(
            Bucket=bucket, Key=key, Body=content.encode("utf-8"), Tagging="SageMaker=True"
        )


class TestUploadJsonToS3(unittest.TestCase):
    """Test suite for upload_json_to_s3 function."""

    def test_upload_json_with_tagging(self):
        """Test that JSON upload includes proper formatting and tagging."""
        mock_client = MagicMock()
        data = {"key": "value", "nested": {"inner": "data"}}
        bucket = "test-bucket"
        key = "test-key.json"

        upload_json_to_s3(mock_client, data, bucket, key)

        mock_client.put_object.assert_called_once()
        call_kwargs = mock_client.put_object.call_args[1]
        self.assertEqual(call_kwargs["Bucket"], bucket)
        self.assertEqual(call_kwargs["Key"], key)
        self.assertEqual(call_kwargs["Tagging"], "SageMaker=True")

        # Verify JSON content
        uploaded_data = json.loads(call_kwargs["Body"].decode("utf-8"))
        self.assertEqual(uploaded_data, data)


class TestProcessRecipeMetadata(unittest.TestCase):
    """Test suite for process_recipe_metadata function."""

    def test_build_recipe_entry(self):
        """Test building recipe entry from launch.json."""
        launch_data = {
            "metadata": {
                "DisplayName": "Test Recipe",
                "Name": "test-recipe",
                "CustomizationTechnique": "sft",
                "InstanceCount": 2,
                "Type": "fine-tuning",
                "Versions": ["1.0.0"],
                "Hardware": "gpu",
                "InstanceTypes": ["ml.p4d.24xlarge"],
                "RecipeFilePath": "/path/to/recipe.yaml",
                "SequenceLength": 4096,
                "HostingConfigs": [],
            },
            "regional_parameters": {
                "smtj_regional_ecr_uri": {
                    "prod": {"us-west-2": "123456789012.dkr.ecr.us-west-2.amazonaws.com/image:latest"}
                }
            },
        }
        s3_uris = {
            "k8s_yaml": "s3://bucket/k8s.yaml",
            "k8s_json": "s3://bucket/k8s.json",
            "sm_jobs_yaml": "s3://bucket/sm_jobs.yaml",
            "sm_jobs_json": "s3://bucket/sm_jobs.json",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(launch_data, f)
            f.flush()
            try:
                result = process_recipe_metadata(f.name, s3_uris, "us-west-2", "prod")

                self.assertEqual(result["DisplayName"], "Test Recipe")
                self.assertEqual(result["Name"], "test-recipe")
                self.assertEqual(result["CustomizationTechnique"], "SFT")
                self.assertEqual(result["HpEksPayloadTemplateS3Uri"], "s3://bucket/k8s.yaml")
                self.assertEqual(result["SmtjImageUri"], "123456789012.dkr.ecr.us-west-2.amazonaws.com/image:latest")
            finally:
                os.unlink(f.name)

    def test_peft_field_included(self):
        """Test that Peft field is included when present."""
        launch_data = {
            "metadata": {"DisplayName": "Test Recipe", "Name": "test-recipe", "Peft": "lora"},
            "regional_parameters": {},
        }
        s3_uris = {
            "k8s_yaml": "s3://bucket/k8s.yaml",
            "k8s_json": "s3://bucket/k8s.json",
            "sm_jobs_yaml": "s3://bucket/sm_jobs.yaml",
            "sm_jobs_json": "s3://bucket/sm_jobs.json",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(launch_data, f)
            f.flush()
            try:
                result = process_recipe_metadata(f.name, s3_uris, "us-west-2", "prod")
                self.assertEqual(result["Peft"], "LORA")
            finally:
                os.unlink(f.name)


class TestGetRegionalEcrUri(unittest.TestCase):
    """Test suite for get_regional_ecr_uri function."""

    def test_extract_ecr_uri(self):
        """Test extracting ECR URI for region and endpoint."""
        launch_data = {
            "regional_parameters": {
                "smtj_regional_ecr_uri": {
                    "prod": {"us-west-2": "prod-ecr-uri", "us-east-1": "prod-ecr-uri-east"},
                    "beta": {"us-west-2": "beta-ecr-uri"},
                }
            }
        }

        result = get_regional_ecr_uri(launch_data, "us-west-2", "prod")
        self.assertEqual(result, "prod-ecr-uri")

        result = get_regional_ecr_uri(launch_data, "us-east-1", "prod")
        self.assertEqual(result, "prod-ecr-uri-east")

        result = get_regional_ecr_uri(launch_data, "us-west-2", "beta")
        self.assertEqual(result, "beta-ecr-uri")

    def test_missing_region_returns_none(self):
        """Test that missing region returns None."""
        launch_data = {"regional_parameters": {"smtj_regional_ecr_uri": {"prod": {"us-west-2": "prod-ecr-uri"}}}}

        result = get_regional_ecr_uri(launch_data, "eu-west-1", "prod")
        self.assertIsNone(result)


class TestCreateNewRecipecollection(unittest.TestCase):
    """Test suite for create_new_recipecollection function."""

    def test_initializes_empty_recipe_collection(self):
        """Test that RecipeCollection is initialized as empty list."""
        hub_content = {"HubContentDocument": {"ExistingField": "value"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hub_content, f)
            f.flush()
            try:
                result = create_new_recipecollection(f.name)
                self.assertTrue(result)

                # Verify file was updated
                with open(f.name, "r") as rf:
                    updated_content = json.load(rf)
                    self.assertEqual(updated_content["HubContentDocument"]["RecipeCollection"], [])
                    self.assertEqual(updated_content["HubContentDocument"]["ExistingField"], "value")
            finally:
                os.unlink(f.name)


class TestUpdateExportedJson(unittest.TestCase):
    """Test suite for update_exported_json function."""

    def test_adds_recipe_entries(self):
        """Test adding recipe entries to exported JSON."""
        hub_content = {
            "HubContentDocument": {"RecipeCollection": [{"Name": "existing-recipe"}]},
            "HubContentVersion": "1.0.0",
        }
        new_entries = [{"Name": "new-recipe-1"}, {"Name": "new-recipe-2"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hub_content, f)
            f.flush()
            try:
                update_exported_json(f.name, new_entries, "test-model", "2.0.0")

                with open(f.name, "r") as rf:
                    updated_content = json.load(rf)
                    self.assertEqual(len(updated_content["HubContentDocument"]["RecipeCollection"]), 3)
                    self.assertEqual(updated_content["HubContentVersion"], "2.0.0")
            finally:
                os.unlink(f.name)


class TestGetVersionFromExport(unittest.TestCase):
    """Test suite for get_version_from_export function."""

    def test_extract_version(self):
        """Test extracting HubContentVersion from exported JSON."""
        hub_content = {"HubContentVersion": "2.36.0"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hub_content, f)
            f.flush()
            try:
                result = get_version_from_export(f.name)
                self.assertEqual(result, "2.36.0")
            finally:
                os.unlink(f.name)

    def test_default_version_when_missing(self):
        """Test default version when HubContentVersion is missing."""
        hub_content = {"OtherField": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hub_content, f)
            f.flush()
            try:
                result = get_version_from_export(f.name)
                self.assertEqual(result, "1.0.0")
            finally:
                os.unlink(f.name)


class TestBumpVersion(unittest.TestCase):
    """Test suite for bump_version function."""

    def test_bump_patch_version(self):
        """Test bumping patch version."""
        self.assertEqual(bump_version("2.36.0"), "2.36.1")
        self.assertEqual(bump_version("1.0.0"), "1.0.1")
        self.assertEqual(bump_version("0.0.9"), "0.0.10")

    def test_bump_complex_version(self):
        """Test bumping version with larger numbers."""
        self.assertEqual(bump_version("10.20.30"), "10.20.31")
        self.assertEqual(bump_version("1.2.99"), "1.2.100")


class TestGenerateLaunchJson(unittest.TestCase):
    """Test suite for generate_launch_json function."""

    @patch("scripts.model_hub.update_private_hub.LaunchJsonGenerator")
    def test_successful_generation(self, mock_generator_class):
        """Test successful launch.json generation."""
        from scripts.model_hub.update_private_hub import generate_launch_json

        mock_generator = MagicMock()
        mock_generator.generate_launch_json.return_value = ("/path/to/launch.json", "success")
        mock_generator_class.return_value = mock_generator

        result = generate_launch_json("/path/to/recipe.yaml", "test-recipe", "/output")

        self.assertEqual(result, "/path/to/launch.json")

    @patch("scripts.model_hub.update_private_hub.LaunchJsonGenerator")
    def test_skipped_generation(self, mock_generator_class):
        """Test skipped launch.json generation."""
        from scripts.model_hub.update_private_hub import generate_launch_json

        mock_generator = MagicMock()
        mock_generator.generate_launch_json.return_value = (None, "skipped")
        mock_generator_class.return_value = mock_generator

        result = generate_launch_json("/path/to/recipe.yaml", "test-recipe", "/output")

        self.assertIsNone(result)

    @patch("scripts.model_hub.update_private_hub.LaunchJsonGenerator")
    def test_failed_generation(self, mock_generator_class):
        """Test failed launch.json generation."""
        from scripts.model_hub.update_private_hub import generate_launch_json

        mock_generator = MagicMock()
        mock_generator.generate_launch_json.return_value = (None, "error: some failure")
        mock_generator_class.return_value = mock_generator

        result = generate_launch_json("/path/to/recipe.yaml", "test-recipe", "/output")

        self.assertIsNone(result)


class TestExportHubContent(unittest.TestCase):
    """Test suite for export_hub_content function."""

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_successful_export(self, mock_exists, mock_run):
        """Test successful hub content export."""
        from scripts.model_hub.update_private_hub import export_hub_content

        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        mock_exists.return_value = True

        result = export_hub_content("test-hub", "test-model", "us-west-2", "/output", "prod")

        self.assertEqual(result, "/output/test-model_export.json")
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_failed_export(self, mock_run):
        """Test failed hub content export."""
        from subprocess import CalledProcessError

        from scripts.model_hub.update_private_hub import export_hub_content

        mock_run.side_effect = CalledProcessError(1, "cmd", stderr="Error")

        result = export_hub_content("test-hub", "test-model", "us-west-2", "/output", "prod")

        self.assertIsNone(result)


class TestImportHubContent(unittest.TestCase):
    """Test suite for import_hub_content function."""

    @patch("subprocess.run")
    def test_successful_import(self, mock_run):
        """Test successful hub content import."""
        from scripts.model_hub.update_private_hub import import_hub_content

        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        result = import_hub_content("/path/to/export.json", "test-hub", "us-west-2", "prod")

        self.assertTrue(result)

    @patch("subprocess.run")
    def test_failed_import_raises_exception(self, mock_run):
        """Test that failed import raises exception."""
        from subprocess import CalledProcessError

        from scripts.model_hub.update_private_hub import import_hub_content

        mock_run.side_effect = CalledProcessError(1, "cmd", stderr="Error")

        with self.assertRaises(CalledProcessError):
            import_hub_content("/path/to/export.json", "test-hub", "us-west-2", "prod")


class TestUploadArtifactsToS3(unittest.TestCase):
    """Test suite for upload_artifacts_to_s3 function."""

    @patch("scripts.model_hub.update_private_hub.boto3.client")
    @patch("scripts.model_hub.update_private_hub.extract_sm_jobs_yaml_content")
    @patch("scripts.model_hub.update_private_hub.extract_yaml_content")
    def test_uploads_all_artifacts(self, mock_extract_yaml, mock_extract_sm_jobs, mock_boto_client):
        """Test that all artifacts are uploaded to S3."""
        from scripts.model_hub.update_private_hub import upload_artifacts_to_s3

        launch_data = {"config.yaml": "test: content", "recipe_override_parameters": {"param1": "value1"}}
        mock_extract_yaml.return_value = "yaml_content"
        mock_extract_sm_jobs.return_value = "sm_jobs_yaml_content"

        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(launch_data, f)
            f.flush()
            try:
                result = upload_artifacts_to_s3(f.name, "test-recipe", "test-bucket", "us-west-2", "1.0.0")

                # Verify S3 client was created with correct region
                mock_boto_client.assert_called_once_with("s3", region_name="us-west-2")

                # Verify 4 artifacts were uploaded (k8s yaml, k8s json, sm_jobs yaml, sm_jobs json)
                self.assertEqual(mock_s3_client.put_object.call_count, 4)

                # Verify S3 URIs are returned
                self.assertIn("k8s_yaml", result)
                self.assertIn("k8s_json", result)
                self.assertIn("sm_jobs_yaml", result)
                self.assertIn("sm_jobs_json", result)

                # Verify S3 URI format
                self.assertTrue(result["k8s_yaml"].startswith("s3://test-bucket/"))
                self.assertIn("test-recipe", result["k8s_yaml"])
                self.assertIn("1.0.0", result["k8s_yaml"])
            finally:
                os.unlink(f.name)

    @patch("scripts.model_hub.update_private_hub.boto3.client")
    @patch("scripts.model_hub.update_private_hub.extract_sm_jobs_yaml_content")
    @patch("scripts.model_hub.update_private_hub.extract_yaml_content")
    def test_s3_key_format(self, mock_extract_yaml, mock_extract_sm_jobs, mock_boto_client):
        """Test that S3 keys follow the expected format."""
        from scripts.model_hub.update_private_hub import upload_artifacts_to_s3

        launch_data = {"config.yaml": "test: content", "recipe_override_parameters": {}}
        mock_extract_yaml.return_value = "yaml_content"
        mock_extract_sm_jobs.return_value = "sm_jobs_yaml_content"

        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(launch_data, f)
            f.flush()
            try:
                result = upload_artifacts_to_s3(f.name, "llama-8b-sft", "my-bucket", "us-east-1", "2.0.0")

                # Verify S3 key format: recipes/{recipe_name}_payload_template_k8s_{version}.yaml
                self.assertEqual(
                    result["k8s_yaml"], "s3://my-bucket/recipes/llama-8b-sft_payload_template_k8s_2.0.0.yaml"
                )
                self.assertEqual(
                    result["k8s_json"], "s3://my-bucket/recipes/llama-8b-sft_override_params_k8s_2.0.0.json"
                )
                self.assertEqual(
                    result["sm_jobs_yaml"], "s3://my-bucket/recipes/llama-8b-sft_payload_template_sm_jobs_2.0.0.yaml"
                )
                self.assertEqual(
                    result["sm_jobs_json"], "s3://my-bucket/recipes/llama-8b-sft_override_params_sm_jobs_2.0.0.json"
                )
            finally:
                os.unlink(f.name)

    @patch("scripts.model_hub.update_private_hub.boto3.client")
    @patch("scripts.model_hub.update_private_hub.extract_sm_jobs_yaml_content")
    @patch("scripts.model_hub.update_private_hub.extract_yaml_content")
    def test_k8s_yaml_container_replacement(self, mock_extract_yaml, mock_extract_sm_jobs, mock_boto_client):
        """Test that k8s YAML replaces test_container with {{container_image}} placeholder."""
        from scripts.model_hub.update_private_hub import upload_artifacts_to_s3

        launch_data = {"config.yaml": "image: test_container\n", "recipe_override_parameters": {}}
        mock_extract_yaml.return_value = "image: test_container\nother: value"
        mock_extract_sm_jobs.return_value = "sm_jobs_content"

        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(launch_data, f)
            f.flush()
            try:
                upload_artifacts_to_s3(f.name, "test-recipe", "test-bucket", "us-west-2", "1.0.0")

                # Find the k8s yaml upload call and verify content has replacement
                calls = mock_s3_client.put_object.call_args_list
                k8s_yaml_call = None
                for call in calls:
                    if "payload_template_k8s" in call[1].get("Key", ""):
                        k8s_yaml_call = call
                        break

                self.assertIsNotNone(k8s_yaml_call)
                body_content = k8s_yaml_call[1]["Body"].decode("utf-8")
                self.assertIn("{{container_image}}", body_content)
                self.assertNotIn("test_container", body_content)
            finally:
                os.unlink(f.name)

    @patch("scripts.model_hub.update_private_hub.boto3.client")
    @patch("scripts.model_hub.update_private_hub.extract_sm_jobs_yaml_content")
    @patch("scripts.model_hub.update_private_hub.extract_yaml_content")
    def test_sagemaker_tagging_applied(self, mock_extract_yaml, mock_extract_sm_jobs, mock_boto_client):
        """Test that SageMaker=True tagging is applied to all uploads."""
        from scripts.model_hub.update_private_hub import upload_artifacts_to_s3

        launch_data = {"config.yaml": "test: content", "recipe_override_parameters": {}}
        mock_extract_yaml.return_value = "yaml_content"
        mock_extract_sm_jobs.return_value = "sm_jobs_yaml_content"

        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(launch_data, f)
            f.flush()
            try:
                upload_artifacts_to_s3(f.name, "test-recipe", "test-bucket", "us-west-2", "1.0.0")

                # Verify all uploads have SageMaker tagging
                for call in mock_s3_client.put_object.call_args_list:
                    self.assertEqual(call[1]["Tagging"], "SageMaker=True")
            finally:
                os.unlink(f.name)
