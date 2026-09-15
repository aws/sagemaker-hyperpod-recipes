# Per-service instance type overrides

Some recipes deploy more than one pod type in a single job — e.g. GPU-intensive
training and generation pods alongside lightweight CPU orchestration pods. By
default every pod runs on the single global `cluster.instance_type`, which forces
expensive GPU instances for CPU-only work.

Two optional cluster settings let you place a job's services on different instance
groups within the same cluster:

| Setting | Shape | Scope |
|---|---|---|
| `cluster.cpu_instance_type` | string, or list of strings | Coarse: applies to **all** CPU sub-services that are not named explicitly in `override_sub_instance_type`. Also used for the Ray submitter node. |
| `cluster.override_sub_instance_type` | map of `service -> string` | Fine-grained: targets **named** services. Each service takes exactly one instance type. |

## Resolution order

For each service, the instance type is resolved as:

1. `cluster.override_sub_instance_type[service]` — if the service is named in the map
2. `cluster.cpu_instance_type` — for CPU sub-services only
3. `cluster.instance_type` — the global default (used by GPU services and anything not covered above)

Each `override_sub_instance_type` service takes exactly one instance type.
`cpu_instance_type` may be a list, which means "schedule on any of these instance
types" (multi-value node affinity) for the CPU sub-services it covers. Override
values may be **CPU or GPU** instance types — the CPU/GPU grouping only decides
whether the coarse `cpu_instance_type` fallback applies; it does not pin a service
to CPU hardware.

Values are normalized to the `ml.`-prefixed, lowercase form (`r6i.24xlarge` and
`ml.R6I.24xlarge` both become `ml.r6i.24xlarge`).

## Overridable services per recipe type

Only the service names listed below are valid keys for a given recipe type.
Supplying an unknown key — or any key for a recipe type with no overridable
services — is a validation error.

### Nova RFT (Kubernetes)

Launcher: `SMNovaK8SLauncherRFT`.

| Service key | Kind | Default instance type |
|---|---|---|
| `training` | GPU | `cluster.instance_type` |
| `vllm_generation` | GPU | `cluster.instance_type` |
| `hub` | CPU | `cpu_instance_type` → `instance_type` |
| `prompter` | CPU | `cpu_instance_type` → `instance_type` |
| `rbs` | CPU | `cpu_instance_type` → `instance_type` |
| `nats_server` | CPU | `cpu_instance_type` → `instance_type` |
| `redis` | CPU (only when the delegate/rollout Redis is enabled) | `cpu_instance_type` → `instance_type` |

For GPU services (`training`, `vllm_generation`) the EFA device count is derived
from the instance type actually resolved for that service.

Example:

```yaml
cluster:
  instance_type: p5.48xlarge          # GPU services (training, vllm_generation)
  cpu_instance_type: r6i.24xlarge     # all CPU sub-services not named below
  override_sub_instance_type:
    rbs: r6i.12xlarge                 # rbs gets its own type
    redis: r6i.8xlarge
```

### Ray / verl recipes

`cluster.cpu_instance_type` targets the Ray submitter node (a single string or a
list).

### Other recipe types

Nova SFT / DPO / PPO, Nova RFT on SageMaker training jobs (`SMNovaSMTJLauncherRFT`),
evaluation, and other launcher types do **not** consume `override_sub_instance_type`
today. Setting it for these recipes raises a validation error rather than being
silently ignored. Use `cluster.instance_type` (and `cpu_instance_type` where a Ray
submitter node applies).
