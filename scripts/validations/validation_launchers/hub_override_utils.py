import json
import logging
import re

logger = logging.getLogger(__name__)

# Metadata fields preserved across hub content updates
_PRESERVED_METADATA_FIELDS = [
    "HubContentDisplayName",
    "HubContentDescription",
    "HubContentMarkdown",
    "HubContentSearchKeywords",
]


def override_hub_recipe_images(
    sagemaker_client,
    hub_name: str,
    model_name: str,
    expected_image: str,
    s3_client=None,
    update_k8s_templates: bool = False,
) -> bool:
    """Override SmtjImageUri in a model's RecipeCollection on a private hub."""
    if not expected_image:
        logger.info("No expected_image provided, skipping hub override")
        return True

    response = sagemaker_client.describe_hub_content(
        HubName=hub_name, HubContentType="Model", HubContentName=model_name
    )
    hub_doc = json.loads(response.get("HubContentDocument", "{}"))
    recipes = hub_doc.get("RecipeCollection", [])

    changed = False
    for recipe in recipes:
        old_image = recipe.get("SmtjImageUri", "")
        if old_image and old_image != expected_image:
            logger.info(f"Overriding SmtjImageUri: {old_image} -> {expected_image}")
            recipe["SmtjImageUri"] = expected_image
            changed = True
        elif old_image == expected_image:
            logger.debug(f"Recipe image already matches: {expected_image}")

        # Always patch the K8s template in S3 — SmtjImageUri can match while
        # the template still contains {{container_image}} or a stale ECR URI.
        if update_k8s_templates and s3_client:
            k8s_uri = recipe.get("HpEksPayloadTemplateS3Uri", "")
            if k8s_uri:
                _update_s3_template_image(s3_client, k8s_uri, old_image, expected_image)

    if not changed:
        logger.info("Hub content already has expected container image, re-importing to ensure sync")

    # Always re-import the document to ensure hub is in sync with local config
    import_params = {
        "HubName": hub_name,
        "HubContentType": "Model",
        "HubContentName": model_name,
        "HubContentDocument": json.dumps(hub_doc),
        "DocumentSchemaVersion": "2.4.0",
    }
    for field in _PRESERVED_METADATA_FIELDS:
        if field in response:
            import_params[field] = response[field]

    sagemaker_client.import_hub_content(**import_params)
    logger.info(f"Successfully overrode hub content for {model_name}")
    return True


# Matches ECR image URIs like 123456789012.dkr.ecr.us-west-2.amazonaws.com/repo:tag
_ECR_IMAGE_PATTERN = re.compile(r"\d+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[^\s\"']+:[^\s\"':]+")

# Jinja placeholder used in K8s templates before image resolution
_CONTAINER_IMAGE_PLACEHOLDER = "{{container_image}}"


def _update_s3_template_image(s3_client, s3_uri: str, old_image: str, new_image: str):
    """Download a K8s YAML template from S3, replace an image URI, and re-upload."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        if old_image in content:
            # Exact match — best case
            content = content.replace(old_image, new_image)
        elif new_image in content:
            logger.info(f"K8s template already has correct image: {key}")
            return
        else:
            if _CONTAINER_IMAGE_PLACEHOLDER in content:
                content = content.replace(_CONTAINER_IMAGE_PLACEHOLDER, new_image)
                logger.info(
                    f"Replaced Jinja placeholder in K8s template: {_CONTAINER_IMAGE_PLACEHOLDER} -> {new_image}"
                )
            else:
                matches = list(set(_ECR_IMAGE_PATTERN.findall(content)))
                replaced = False
                for match in matches:
                    if match != new_image:
                        content = content.replace(match, new_image)
                        logger.info(f"Replaced ECR image in K8s template via fallback: {match} -> {new_image}")
                        replaced = True
                        break
                if not replaced:
                    logger.warning(f"K8s template doesn't contain any ECR image to replace, skipping: {key}")
                    return

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode("utf-8"),
        )
        logger.info(f"Updated K8s template in S3: {key}")
    except Exception as e:
        logger.warning(f"Failed to update K8s template {s3_uri}: {e}")
