import threading
from pathlib import Path
from typing import Any, Union

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

RECIPES_COLLECTION_DIR = Path(__file__).resolve().parent.parent / "recipes_collection"

_hydra_lock = threading.Lock()


def load_recipe_with_hydra(
    recipe_file_path: Union[str, Path],
    return_dict: bool = False,
) -> Any:
    """
    Load a recipe file using Hydra's compose API, resolving all defaults.

    Args:
        recipe_file_path: Path to the recipe file. Can be:
            - Relative path from recipes_collection/recipes/ (e.g., "fine-tuning/llama/recipe.yaml")
            - Full Path object
        return_dict: If True, returns dict instead of OmegaConf

    Returns:
        Fully resolved recipe configuration as OmegaConf or dict
    """
    if isinstance(recipe_file_path, Path):
        recipes_dir = RECIPES_COLLECTION_DIR / "recipes"
        if recipe_file_path.is_absolute():
            relative_path = recipe_file_path.relative_to(recipes_dir)
        else:
            relative_path = recipe_file_path
        recipe_path_no_ext = str(relative_path).replace(".yaml", "").replace(".yml", "")
    else:
        recipe_path_no_ext = recipe_file_path.replace(".yaml", "").replace(".yml", "")

    hydra_config_searchpath = RECIPES_COLLECTION_DIR / "recipes" / "fine-tuning"

    with _hydra_lock:
        try:
            GlobalHydra.instance().clear()

            with initialize_config_dir(version_base=None, config_dir=str(RECIPES_COLLECTION_DIR.absolute())):
                searchpath_override = f"hydra.searchpath=[file://{hydra_config_searchpath}]"
                recipe_override = f"recipes={recipe_path_no_ext}"
                cfg = compose(config_name="config", overrides=[searchpath_override, recipe_override])

                if "recipes" in cfg and cfg.recipes is not None:
                    result = cfg.recipes
                else:
                    result = cfg

                if return_dict:
                    return OmegaConf.to_container(result, resolve=False, throw_on_missing=False)
                return result
        except Exception as e:
            raise RuntimeError(f"Failed to load recipe '{recipe_file_path}' with Hydra: {e}") from e
        finally:
            GlobalHydra.instance().clear()
