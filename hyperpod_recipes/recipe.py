import os

_THIS_DIR = os.path.dirname(__file__)
RECIPES_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "recipes_collection", "recipes"))


class Recipe:
    def __init__(self, path):
        id, _ = os.path.splitext(os.path.relpath(path, RECIPES_DIR))
        self.name = id
        self.recipe_id = id
        self.path = path

    def __repr__(self):
        return (
            f"Recipe(\n"
            f"    name={self.name!r},\n"
            f"    recipe_id={self.recipe_id!r},\n"
            f"    path={self.path!r}\n"
            f")"
        )

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
