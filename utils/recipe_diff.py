"""Recipe diff utilities — pure logic, no I/O dependencies.

Determines whether a recipe change is functional (affects training behavior)
or non-functional (only dataset paths, model paths, etc.).
"""

NON_FUNCTIONAL_PATHS = {
    "data.train_files",
    "data.val_files",
    "data.train_data",
    "data.val_data",
    "training_config.data.train_files",
    "training_config.data.val_files",
    "training_config.data.train_data",
    "training_config.data.val_data",
    "training_config.datasets.train_data.file_path",
    "training_config.datasets.val_data.file_path",
    "training_config.datasets.train_data.name",
    "training_config.datasets.val_data.name",
    "training_config.model_config.model_name_or_path",
    "training_config.actor_rollout_ref.model.path",
    "training_config.trainer.default_local_dir",
    "training_config.training_args.training_dir",
    "training_config.model.path",
    "run.results_dir",
    "run.hf_access_token",
    "version",
}


def flatten_yaml(d, prefix=""):
    """Flatten nested dict to dotted key→value pairs."""
    items = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_yaml(v, f"{key}."))
        else:
            items[key] = v
    return items


def is_functional_change(old_yaml: dict, new_yaml: dict) -> bool:
    """Determine if the diff between two recipe configs is functional.

    Returns True if any changed key is outside NON_FUNCTIONAL_PATHS.
    Returns False if all changes are non-functional (dataset paths, etc.).
    """
    old_flat = flatten_yaml(old_yaml)
    new_flat = flatten_yaml(new_yaml)

    for key in set(old_flat) | set(new_flat):
        if old_flat.get(key) != new_flat.get(key):
            if not any(key == p or key.startswith(p + ".") for p in NON_FUNCTIONAL_PATHS):
                return True
    return False
