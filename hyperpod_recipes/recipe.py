import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_THIS_DIR = os.path.dirname(__file__)
RECIPES_DIR = os.path.abspath(os.path.join(_THIS_DIR, "recipes_src"))


class Recipe:
    def __init__(self, path):
        id, _ = os.path.splitext(os.path.relpath(path, RECIPES_DIR))
        self.name = id
        self.recipe_id = id
        self.path = path
        self._config = None

    def __repr__(self):
        return (
            f"Recipe(\n"
            f"    name={self.name!r},\n"
            f"    recipe_id={self.recipe_id!r},\n"
            f"    path={self.path!r}\n"
            f")"
        )

    @property
    def config(self):
        if self._config is None:
            # Determine the recipe's parent directory category (fine-tuning, training, evaluation)
            rel_path = os.path.relpath(self.path, RECIPES_DIR)
            parts = rel_path.split(os.sep)
            category = parts[0]  # e.g., "fine-tuning", "training", "evaluation"

            # Get the recipe file details
            recipe_name_with_ext = os.path.basename(self.path)
            recipe_dir = os.path.dirname(self.path)

            # Setup Hydra search paths to include the hydra_config directory
            hydra_config_path = os.path.join(RECIPES_DIR, category)

            # Create search path override for Hydra (must be absolute for file:// protocol)
            search_path_override = f"hydra.searchpath=[file://{hydra_config_path}]"

            # Use initialize_config_dir which accepts absolute paths
            with initialize_config_dir(version_base=None, config_dir=recipe_dir):
                cfg = compose(
                    config_name=recipe_name_with_ext, overrides=[search_path_override], return_hydra_config=False
                )

            # Unwrap @package recipes nesting
            # Hydra defaults use @package recipes which places their content under cfg.recipes
            # But the recipe file's own keys land at root level due to _self_
            # We need to merge them so everything is at root level
            if "recipes" in cfg:
                recipes_cfg = cfg.recipes
                # Get all non-recipes root keys (from the recipe file itself)
                root_overrides = OmegaConf.create({k: v for k, v in cfg.items() if k != "recipes"})
                # Deep merge: recipe file's values override defaults
                # Set struct=False to allow merging
                OmegaConf.set_struct(recipes_cfg, False)
                cfg = OmegaConf.merge(recipes_cfg, root_overrides)

            self._config = cfg

        return self._config

    def validate(self) -> None:
        """
        Verifies that the recipe has no problems or raises an Exception with the problem.
        This is done with a dry-run of the existing launch.
        """

    def launch(self) -> None:
        """
        Launches the recipe as configured
        """

    def launch_json(self) -> str:
        """
        Outputs the final launch configuration as a JSON output.
        The precise format will be platform dependent (SMTJ, EKS, slurm).
        """

    def write_recipe_dir(self, path, format):
        """
        This will write the files used in the recipe to a user dir for the user to edit.
        The files can be generated in different formats such as nemo_launcher, EKS, etc which will be in a StrEnum RecipeFormat.
        """

    @staticmethod
    def read_recipe_dir(path):
        """
        After a user writes to a directory and edits their files, this will allow reading their recipe back in.
        It requires parsing the format the recipe was written to to understand how to read it back in.
        """
