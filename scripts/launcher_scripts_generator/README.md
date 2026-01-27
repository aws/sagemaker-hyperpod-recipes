# Launcher Script Generator

Generates shell scripts for launching training jobs from recipe YAML files.

## Usage

```bash
# Generate all launcher scripts
python -m scripts.launcher_scripts_generator.generate_launcher_scripts

# Check mode (exits non-zero if files would change, like formatter --check)
python -m scripts.launcher_scripts_generator.generate_launcher_scripts --check

# Check with diffs shown
python -m scripts.launcher_scripts_generator.generate_launcher_scripts --check --diff
```

## How It Works

1. **Discovery**: Scans `recipes_collection/recipes/` for all `*.yaml` files, excluding `excluded_recipe_dirs`
2. **Template Selection**: Reads `run.model_type` from each recipe (Hydra defaults are resolved) and selects matching template
3. **Generation**: Renders script from `header` + template with placeholders substituted
4. **Cleanup**: Removes scripts that no longer have backing recipes (preserves configured directories)

## Configuration

All customizable settings are in `launcher_scripts_config.yaml`:

### Templates

Templates are keyed by `model_type` value from recipes. The generator selects templates using:
- **Exact match**: If `run.model_type` matches a template key exactly
- **Prefix match**: Nova models (`amazon.nova-*`) use the `nova` template
- **Default fallback**: Unknown types use `default_template` (llm_finetuning_aws)

```yaml
templates:
  llm_finetuning_aws: |
    # Template for LLMFT recipes
    ...
  verl: |
    # Template for VERL recipes
    ...
  nova: |
    # Template for Nova models
    ...
```

### Placeholders

Available placeholders for templates:

| Placeholder | Description |
|-------------|-------------|
| `{recipe_path}` | Path to recipe (e.g., `fine-tuning/llama/recipe_name`) |
| `{recipe_name}` | Recipe filename without extension |
| `{run_name}` | Value from `run.name` in recipe, or derived from recipe_name with dashes |
| `{model_save_name}` | Value from `training_config.model_config.model_save_name` or `model_name_or_path` |

### Settings

```yaml
settings:
  # Directory containing recipe YAML files
  recipes_dir: recipes_collection/recipes

  # Output directory for generated scripts
  output_dir: launcher_scripts

  # Directories to preserve (not cleaned up)
  preserved_dirs:
    - custom_script
    - hydra_config

  # Recipe directories to exclude from generation (e.g., Hydra config fragments)
  excluded_recipe_dirs:
    - fine-tuning/hydra_config

  # Scripts that are manually maintained (not auto-generated)
  manually_maintained:
    "evaluation/open-source/open_source_deterministic_eval": "evaluation/run_open_source_deterministic_eval.sh"
```

## Adding Support for New Model Types

1. Add a new recipe with `run.model_type: my_new_type`
2. Add a matching template in `launcher_scripts_config.yaml`:

```yaml
templates:
  my_new_type: |
    MY_VAR="${MY_VAR}"
    HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
        recipes={recipe_path} \
        ...
```

3. Run the generator to create scripts for all recipes with that model type
