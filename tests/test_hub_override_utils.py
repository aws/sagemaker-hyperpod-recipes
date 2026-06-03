"""Tests for hub_override_utils — shared hub content override logic."""

import json
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from scripts.validations.validation_launchers.hub_override_utils import (
    _update_s3_template_image,
    override_hub_recipe_images,
)


@pytest.fixture
def sm_client():
    return MagicMock()


@pytest.fixture
def s3_client():
    return MagicMock()


def _make_hub_response(recipes, extra_fields=None):
    """Helper to build a describe_hub_content response dict."""
    hub_doc = {"RecipeCollection": recipes}
    resp = {
        "HubContentDocument": json.dumps(hub_doc),
        "HubContentDisplayName": "Test Model",
        "HubContentDescription": "A test model",
    }
    if extra_fields:
        resp.update(extra_fields)
    return resp


class TestOverrideHubRecipeImages:
    def test_skips_when_no_expected_image(self, sm_client):
        result = override_hub_recipe_images(sm_client, "hub", "model", "")
        assert result is True
        sm_client.describe_hub_content.assert_not_called()

    def test_always_imports_even_when_images_match(self, sm_client):
        """import_hub_content is always called to ensure hub stays in sync."""
        sm_client.describe_hub_content.return_value = _make_hub_response([{"SmtjImageUri": "img:latest"}])
        result = override_hub_recipe_images(sm_client, "hub", "model", "img:latest")
        assert result is True
        sm_client.import_hub_content.assert_called_once()

    def test_overrides_mismatched_image(self, sm_client):
        sm_client.describe_hub_content.return_value = _make_hub_response([{"SmtjImageUri": "old:v1"}])
        result = override_hub_recipe_images(sm_client, "hub", "model", "new:v2")
        assert result is True
        sm_client.import_hub_content.assert_called_once()
        call_kwargs = sm_client.import_hub_content.call_args[1]
        doc = json.loads(call_kwargs["HubContentDocument"])
        assert doc["RecipeCollection"][0]["SmtjImageUri"] == "new:v2"

    def test_preserves_metadata_fields(self, sm_client):
        sm_client.describe_hub_content.return_value = _make_hub_response(
            [{"SmtjImageUri": "old:v1"}],
            extra_fields={
                "HubContentMarkdown": "# Readme",
                "HubContentSearchKeywords": ["llama"],
            },
        )
        override_hub_recipe_images(sm_client, "hub", "model", "new:v2")
        call_kwargs = sm_client.import_hub_content.call_args[1]
        assert call_kwargs["HubContentMarkdown"] == "# Readme"
        assert call_kwargs["HubContentSearchKeywords"] == ["llama"]

    def test_updates_k8s_templates_when_enabled(self, sm_client, s3_client):
        sm_client.describe_hub_content.return_value = _make_hub_response(
            [{"SmtjImageUri": "old:v1", "HpEksPayloadTemplateS3Uri": "s3://bucket/key.yaml"}]
        )
        s3_client.get_object.return_value = {"Body": BytesIO(b"image: old:v1\nother: stuff")}
        override_hub_recipe_images(
            sm_client,
            "hub",
            "model",
            "new:v2",
            s3_client=s3_client,
            update_k8s_templates=True,
        )
        s3_client.put_object.assert_called_once()
        put_body = s3_client.put_object.call_args[1]["Body"].decode()
        assert "new:v2" in put_body
        assert "old:v1" not in put_body

    def test_patches_k8s_template_even_when_smtj_image_matches(self, sm_client, s3_client):
        """K8s S3 template must be patched even when SmtjImageUri already equals expected_image.

        This covers the case where SmtjImageUri was updated in a prior run but the
        S3 template still contains {{container_image}} or a stale ECR URI.
        """
        sm_client.describe_hub_content.return_value = _make_hub_response(
            [{"SmtjImageUri": "new:v2", "HpEksPayloadTemplateS3Uri": "s3://bucket/k.yaml"}]
        )
        # Template still has the Jinja placeholder despite SmtjImageUri being correct
        s3_client.get_object.return_value = {"Body": BytesIO(b"image: {{container_image}}\nimagePullPolicy: Always")}
        override_hub_recipe_images(
            sm_client,
            "hub",
            "model",
            "new:v2",
            s3_client=s3_client,
            update_k8s_templates=True,
        )
        # S3 template should still get patched
        s3_client.put_object.assert_called_once()
        put_body = s3_client.put_object.call_args[1]["Body"].decode()
        assert "new:v2" in put_body
        assert "{{container_image}}" not in put_body

    def test_skips_k8s_update_when_disabled(self, sm_client, s3_client):
        sm_client.describe_hub_content.return_value = _make_hub_response(
            [{"SmtjImageUri": "old:v1", "HpEksPayloadTemplateS3Uri": "s3://b/k"}]
        )
        override_hub_recipe_images(
            sm_client,
            "hub",
            "model",
            "new:v2",
            s3_client=s3_client,
            update_k8s_templates=False,
        )
        s3_client.get_object.assert_not_called()


class TestUpdateS3TemplateImage:
    def test_replaces_old_image(self, s3_client):
        s3_client.get_object.return_value = {"Body": BytesIO(b"image: old:v1\nfoo: bar")}
        _update_s3_template_image(s3_client, "s3://bucket/template.yaml", "old:v1", "new:v2")
        s3_client.put_object.assert_called_once()
        body = s3_client.put_object.call_args[1]["Body"].decode()
        assert body == "image: new:v2\nfoo: bar"

    def test_skips_when_already_correct(self, s3_client):
        s3_client.get_object.return_value = {"Body": BytesIO(b"image: new:v2")}
        _update_s3_template_image(s3_client, "s3://bucket/t.yaml", "old:v1", "new:v2")
        s3_client.put_object.assert_not_called()

    def test_replaces_jinja_placeholder(self, s3_client):
        """When template has {{container_image}} placeholder, replace it with new_image."""
        template = "image: {{container_image}}\nimagePullPolicy: Always"
        s3_client.get_object.return_value = {"Body": BytesIO(template.encode())}
        _update_s3_template_image(s3_client, "s3://bucket/t.yaml", "old:v1", "new:v2")
        s3_client.put_object.assert_called_once()
        body = s3_client.put_object.call_args[1]["Body"].decode()
        assert "new:v2" in body
        assert "{{container_image}}" not in body

    def test_fallback_replaces_ecr_image_when_old_not_found(self, s3_client):
        """When old_image isn't in the template but a different ECR URI is, replace via regex fallback."""
        prod_image = "111222333444.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.0.0"
        template = f"image: {prod_image}\nother: stuff"
        s3_client.get_object.return_value = {"Body": BytesIO(template.encode())}
        _update_s3_template_image(s3_client, "s3://bucket/t.yaml", "old:v1", "new:v2")
        s3_client.put_object.assert_called_once()
        body = s3_client.put_object.call_args[1]["Body"].decode()
        assert "new:v2" in body
        assert prod_image not in body

    def test_fallback_skips_when_no_ecr_image(self, s3_client):
        """When template has no ECR URIs at all, skip gracefully."""
        s3_client.get_object.return_value = {"Body": BytesIO(b"image: busybox:latest\nother: stuff")}
        _update_s3_template_image(s3_client, "s3://bucket/t.yaml", "old:v1", "new:v2")
        s3_client.put_object.assert_not_called()

    def test_fallback_skips_when_only_new_image_ecr(self, s3_client):
        """When the only ECR URI in the template is already the new_image, skip."""
        new_img = "999888777666.dkr.ecr.us-east-1.amazonaws.com/repo:new-tag"
        template = f"image: {new_img}\nother: stuff"
        s3_client.get_object.return_value = {"Body": BytesIO(template.encode())}
        _update_s3_template_image(s3_client, "s3://bucket/t.yaml", "old:v1", new_img)
        # Already correct path should be hit first (new_image in content)
        s3_client.put_object.assert_not_called()

    def test_warns_on_s3_error(self, s3_client):
        s3_client.get_object.side_effect = Exception("access denied")
        # Should not raise
        _update_s3_template_image(s3_client, "s3://bucket/t.yaml", "old:v1", "new:v2")
        s3_client.put_object.assert_not_called()
