"""Deploy a base model + LoRA-adapter students as SageMaker inference components (ICs).

Our student is a **LoRA adapter** on base Qwen/Qwen3.5-4B. We serve with the **DJL-LMI** container
(vLLM backend) and SageMaker **inference components**:

  * one BASE inference component hosts the base model with LoRA enabled;
  * each student is an ADAPTER inference component attached to that base IC — it references the
    base by name and points at the adapter artifacts in S3 (a loose checkpoints/hf/ prefix works;
    a .tar.gz also works), so the base model loads ONCE and adapters share it;
  * at invoke time the `InferenceComponentName` selects the base model or a specific adapter.

Why LMI/vLLM: the OpenAI /openai/v1 passthrough route does NOT support inference-component
routing (it 400s "Inference Component Name header is required" with no way to supply one on that
route). Routing to a specific IC is only honored via sagemaker-runtime InvokeEndpoint's
InferenceComponentName — which is how the eval harness's relay invokes these (see sm_openai_relay).
The DJL-LMI vLLM backend also maps Qwen3's per-projection q/k/v PEFT modules correctly.

Deployment uses the **SageMaker Core** resource API (Model / EndpointConfig / Endpoint /
InferenceComponent) with capacity-aware instance pools (CONFIGURED BY THE CALLER — the notebook
owns capacity config; no pool defaults are baked in here).

Scoring lives elsewhere: the eval notebook scores through the local OpenAI relay + ShoppingBench
harness, not this module. This module owns deploy / attach / teardown only.
"""
import re


def sanitize_name(name):
    """Make a string a valid SageMaker resource name: ^[a-zA-Z0-9](-*[a-zA-Z0-9])*$ — alphanumerics
    and dashes only. Any non-alphanumeric run becomes a single dash; leading/trailing dashes trimmed."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-")
    return s or "endpoint"


def build_instance_pools(instance_types):
    """Turn a caller-provided ordered list of instance types into SageMaker Core `InstancePool`
    objects, assigning priority by list position (first = priority 1, the preferred type).

    `instance_types` e.g. ["ml.g6e.2xlarge", "ml.g6.2xlarge", "ml.g5.2xlarge"]. SageMaker provisions
    from priority 1 first, falling back to the next pool on a capacity shortage. This module holds
    no default list — the preference list is CONFIG and lives in the notebook.
    """
    from sagemaker.core.shapes import InstancePool

    return [InstancePool(instance_type=t, priority=i + 1) for i, t in enumerate(instance_types)]


def resolve_adapter_uri(model_package_arn, region):
    """Resolve the S3 URI of the LoRA adapter inside a serverless training job's Model Package, for
    passing to `attach_adapter(adapter_s3_uri=...)`.

    Looks for an adapter directory (adapter_config.json + adapter_model.safetensors) under the Model
    Package's model-data root: checkpoints/hf/ first, else the latest global_step_*/lora_adapter/.
    Returns the loose prefix URI (the DJL-LMI container loads the adapter files directly from the
    prefix — no .tar.gz repackaging needed). boto3 S3 list only (no sagemaker-core S3-walk construct).
    """
    import boto3
    from sagemaker.core.resources import ModelPackage

    c = ModelPackage.get(model_package_name=model_package_arn, region=region).inference_specification.containers[0]
    root = c.model_data_source.s3_data_source.s3_uri.rstrip("/") + "/"
    bkt, key = root[5:].split("/", 1)
    s3 = boto3.client("s3", region_name=region)
    keys = [
        o["Key"][len(key) :]
        for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=bkt, Prefix=key)
        for o in pg.get("Contents", [])
    ]
    steps = sorted({k.split("/")[0] for k in keys if k.startswith("global_step_")}, key=lambda d: int(d.split("_")[-1]))
    for sub in ["checkpoints/hf/"] + [f"{d}/lora_adapter/" for d in reversed(steps)]:
        if f"{sub}adapter_config.json" in keys and f"{sub}adapter_model.safetensors" in keys:
            return root + sub
    raise RuntimeError(f"no LoRA adapter (adapter_config.json + adapter_model.safetensors) found under {root}")


def _variant(
    *,
    variant_name,
    instance_types,
    instance_type,
    training_plan_arn,
    provision_timeout_s,
    startup_timeout_s,
    download_timeout_s,
):
    """Build the ProductionVariant for an IC-based endpoint (no model on the variant — the
    inference components bring the models). A one-entry preference list has no fallback, so it
    collapses to a plain instance type; a multi-entry list uses capacity-aware instance_pools
    (priority = order). Timeouts bound each deploy phase so it FAILS rather than hangs (capacity
    shortage, crash-loop, stuck download).

    When training_plan_arn is given, the variant runs on a Flexible Training Plan reservation
    (CapacityReservationConfig, capacity-reservations-only). A reservation is a single instance
    type, so instance_pools (the on-demand multi-type fallback) is incompatible: the caller must
    pass exactly one instance type, which must match the plan's type — enforced in deploy_base_endpoint.
    """
    from sagemaker.core.shapes import (
        ProductionVariant,
        ProductionVariantCapacityReservationConfig,
    )

    kw = dict(
        variant_name=variant_name,
        initial_instance_count=1,
        variant_instance_provision_timeout_in_seconds=provision_timeout_s,
        container_startup_health_check_timeout_in_seconds=startup_timeout_s,
        model_data_download_timeout_in_seconds=download_timeout_s,
    )
    if instance_types and len(instance_types) == 1:
        kw["instance_type"] = instance_types[0]
    elif instance_types:
        kw["instance_pools"] = build_instance_pools(instance_types)
    else:
        kw["instance_type"] = instance_type
    if training_plan_arn:
        kw["capacity_reservation_config"] = ProductionVariantCapacityReservationConfig(
            ml_reservation_arn=training_plan_arn, capacity_reservation_preference="capacity-reservations-only"
        )
    return ProductionVariant(**kw)


def deploy_base_endpoint(
    *,
    endpoint_name,
    role,
    region,
    image_uri,
    base_model_id="Qwen/Qwen3.5-4B",
    instance_types=None,
    instance_type=None,
    training_plan_arn=None,
    max_lora_rank=64,
    max_loras=4,
    num_gpus=1,
    max_model_len=None,
    provision_timeout_s=900,
    startup_timeout_s=1800,
    download_timeout_s=1800,
    min_memory_mb=8192,
    wait=True,
):
    """Create a pooled endpoint hosting ONE base inference component (base model + LoRA enabled).

    Students are then attached as adapter ICs with `attach_adapter`. Returns
    (endpoint_name, base_ic_name).

    image_uri is REQUIRED and caller-provided — the serving container (and its version) is
    deployment config owned by the notebook, not baked into this module. Resolve one with
    sagemaker.core.image_uris.retrieve("djl-lmi", region=..., version=...) or pass a pinned ECR
    URI. Capacity config is likewise caller-provided — pass instance_types=["ml.g6e.2xlarge",
    "ml.g6.2xlarge", ...] (an ordered preference list; priority = order) or a single instance_type.

    training_plan_arn (optional): deploy onto a **Flexible Training Plan** reservation (target
    resource "endpoint") for predictable GPU capacity instead of on-demand. The plan must be
    Active, and the endpoint only serves during the reservation window. A reservation covers a
    SINGLE instance type, so you must pass exactly one instance type (via instance_types=[t] or
    instance_type=t) and it MUST match the plan's instance type — a multi-type fallback pool is
    rejected because it is incompatible with a single-type reservation.

    LMI LoRA env on the base container: OPTION_ENABLE_LORA=true, OPTION_MAX_LORA_RANK (must be
    >= the adapters' rank — our student trains at lora-r=64), OPTION_MAX_LORAS (adapters mixable
    per batch). OPTION_MAX_MODEL_LEN is set when max_model_len is given.
    """
    from sagemaker.core.resources import (
        Endpoint,
        EndpointConfig,
        InferenceComponent,
        Model,
    )
    from sagemaker.core.shapes import (
        ContainerDefinition,
        InferenceComponentComputeResourceRequirements,
        InferenceComponentRuntimeConfig,
        InferenceComponentSpecification,
        InferenceComponentStartupParameters,
    )

    if not instance_types and not instance_type:
        raise ValueError(
            "deploy_base_endpoint needs instance_types=[type, ...] (ordered preference "
            "list) or instance_type=... (capacity config lives in the notebook)."
        )
    if training_plan_arn and instance_types and len(instance_types) > 1:
        raise ValueError(
            "A Flexible Training Plan reservation covers a single instance type, so instance_pools "
            f"(a multi-type fallback) can't be used with training_plan_arn. Got {len(instance_types)} "
            "types: pass exactly one (instance_types=[the_plan's_type]) matching the plan."
        )

    endpoint_name = sanitize_name(endpoint_name)
    cfg_name = sanitize_name(endpoint_name + "-cfg")
    model_name = sanitize_name(endpoint_name + "-base-model")
    base_ic = sanitize_name(endpoint_name + "-base")

    env = {
        "HF_MODEL_ID": base_model_id,
        "OPTION_TRUST_REMOTE_CODE": "true",
        "OPTION_ENABLE_LORA": "true",
        "OPTION_MAX_LORA_RANK": str(max_lora_rank),
        "OPTION_MAX_LORAS": str(max_loras),
        "OPTION_TENSOR_PARALLEL_DEGREE": str(num_gpus),
    }
    if max_model_len:
        env["OPTION_MAX_MODEL_LEN"] = str(max_model_len)

    # Base model: LMI/vLLM serving the base with LoRA enabled.
    Model.create(
        model_name=model_name,
        execution_role_arn=role,
        region=region,
        primary_container=ContainerDefinition(image=image_uri, environment=env),
    )

    # IC-based endpoint: the production variant carries NO model (each IC brings its own); the
    # endpoint config carries the execution role so SageMaker can fetch component artifacts.
    EndpointConfig.create(
        endpoint_config_name=cfg_name,
        execution_role_arn=role,
        region=region,
        production_variants=[
            _variant(
                variant_name=endpoint_name,
                instance_types=instance_types,
                instance_type=instance_type,
                training_plan_arn=training_plan_arn,
                provision_timeout_s=provision_timeout_s,
                startup_timeout_s=startup_timeout_s,
                download_timeout_s=download_timeout_s,
            )
        ],
    )
    endpoint = Endpoint.create(endpoint_name=endpoint_name, endpoint_config_name=cfg_name, region=region)

    # If the deploy fails (e.g. InsufficientInstanceCapacity across all instance_pools, or the base IC
    # crash-looping), tear down the partial stack we just created so it doesn't accumulate as orphaned
    # Failed endpoints / configs / models, then re-raise so the caller sees the real failure.
    try:
        endpoint.wait_for_status("InService")
        InferenceComponent.create(
            inference_component_name=base_ic,
            endpoint_name=endpoint_name,
            variant_name=endpoint_name,
            region=region,
            specification=InferenceComponentSpecification(
                model_name=model_name,
                compute_resource_requirements=InferenceComponentComputeResourceRequirements(
                    number_of_accelerator_devices_required=num_gpus, min_memory_required_in_mb=min_memory_mb
                ),
                startup_parameters=InferenceComponentStartupParameters(
                    model_data_download_timeout_in_seconds=download_timeout_s,
                    container_startup_health_check_timeout_in_seconds=startup_timeout_s,
                ),
            ),
            runtime_config=InferenceComponentRuntimeConfig(copy_count=1),
        )
        if wait:
            InferenceComponent.get(inference_component_name=base_ic, region=region).wait_for_status("InService")
    except Exception:
        teardown_endpoint(endpoint_name, region=region)  # best-effort: ICs -> endpoint -> config
        for cls, kw in ((Model, "model_name"),):  # teardown_endpoint doesn't know the base Model name
            try:
                cls.get(**{kw: model_name, "region": region}).delete()
            except Exception:
                pass
        raise
    return endpoint_name, base_ic


def ic_status(name, region=None):
    """Status of an inference component (e.g. "InService"), or None if it does not exist (e.g. torn down).
    Generic — works for a base IC or an adapter IC."""
    from sagemaker.core.resources import InferenceComponent

    _reg = {"region": region} if region else {}
    try:
        return InferenceComponent.get(inference_component_name=name, **_reg).inference_component_status
    except Exception as e:
        if "Could not find" in str(e) or "ValidationException" in type(e).__name__:
            return None
        raise


def attach_adapter(
    *, endpoint_name, base_ic_name, adapter_component_name, adapter_s3_uri, region=None, wait=True, replace=False
):
    """Attach one LoRA student as an adapter inference component on the shared base IC.

    The adapter IC references the base IC by name and points at the adapter artifacts in S3 (a
    loose checkpoints/hf/ prefix or a .tar.gz); it uses the base IC's compute (no
    ComputeResourceRequirements). Invoke it later via InferenceComponentName=adapter_component_name.
    Returns the adapter IC name.

    IDEMPOTENT: if an IC with this name already exists, its current artifact is compared to
    adapter_s3_uri. When they MATCH, the existing IC is reused (waited to InService) — so a cell
    can be re-run safely without a create-already-exists crash, and without a delete/re-attach race
    that leaves invokes hitting a not-yet-InService component. When they DIFFER (a new training
    run), the stale IC is deleted and recreated so the adapter is never silently stale. Pass
    replace=True to force delete+recreate even when the artifact matches.
    """
    from sagemaker.core.resources import InferenceComponent
    from sagemaker.core.shapes import (
        InferenceComponentContainerSpecification,
        InferenceComponentSpecification,
    )

    name = sanitize_name(adapter_component_name)
    _reg = {"region": region} if region else {}

    existing = None
    try:
        existing = InferenceComponent.get(inference_component_name=name, **_reg)
    except Exception:
        existing = None  # not found -> fall through to create

    if existing is not None:
        try:
            cur = existing.specification.container.artifact_url
        except Exception:
            cur = None
        if not replace and cur == adapter_s3_uri:
            if wait:
                existing.wait_for_status("InService")  # safe re-run path: reuse the attached adapter
            return name
        existing.delete()
        existing.wait_for_delete()  # artifact changed/replace -> recreate

    InferenceComponent.create(
        inference_component_name=name,
        endpoint_name=sanitize_name(endpoint_name),
        **_reg,
        specification=InferenceComponentSpecification(
            base_inference_component_name=base_ic_name,
            container=InferenceComponentContainerSpecification(artifact_url=adapter_s3_uri),
        ),
    )
    if wait:
        InferenceComponent.get(inference_component_name=name, **_reg).wait_for_status("InService")
    return name


def teardown_endpoint(endpoint_name, *, region=None, delete_config=True):
    """Delete an IC-based endpoint in dependency order: adapter ICs (children) first, then the base
    IC, then the endpoint, then its config (best-effort, ignore missing).

    ICs do not cascade-delete, and an adapter IC references its base IC, so children must go before
    parents. Adapters are identified structurally by Specification.BaseInferenceComponentName.
    """
    import boto3
    from sagemaker.core.resources import Endpoint, EndpointConfig, InferenceComponent

    endpoint_name = sanitize_name(endpoint_name)
    smc = boto3.client("sagemaker", region_name=region) if region else boto3.client("sagemaker")
    _reg = {"region": region} if region else {}

    try:
        ics = smc.list_inference_components(EndpointNameEquals=endpoint_name, MaxResults=100).get(
            "InferenceComponents", []
        )
    except Exception:
        ics = []

    def _is_adapter(name):
        try:
            spec = smc.describe_inference_component(InferenceComponentName=name).get("Specification", {}) or {}
            return bool(spec.get("BaseInferenceComponentName"))
        except Exception:
            return False

    names = [ic["InferenceComponentName"] for ic in ics]
    children = [n for n in names if _is_adapter(n)]
    parents = [n for n in names if n not in children]
    for name in children + parents:
        try:
            ic = InferenceComponent.get(inference_component_name=name, **_reg)
            ic.delete()
            ic.wait_for_delete()
            print("  deleted component:", name, "(adapter)" if name in children else "(base)")
        except Exception as e:
            print("  skip component", name, "->", type(e).__name__)

    cfg = None
    if delete_config:
        try:
            cfg = smc.describe_endpoint(EndpointName=endpoint_name).get("EndpointConfigName")
        except Exception:
            pass
    try:
        ep = Endpoint.get(endpoint_name=endpoint_name, **_reg)
        ep.delete()
        ep.wait_for_delete()
        print("deleted endpoint:", endpoint_name)
    except Exception as e:
        print("skip endpoint", endpoint_name, "->", type(e).__name__)
    if cfg:
        try:
            EndpointConfig.get(endpoint_config_name=cfg, **_reg).delete()
            print("  deleted endpoint config:", cfg)
        except Exception as e:
            print("  skip config", cfg, "->", type(e).__name__)
