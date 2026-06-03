"""Tests for utils/recipe_diff.py."""

from utils.recipe_diff import flatten_yaml, is_functional_change


class TestFlattenYaml:
    def test_flat(self):
        assert flatten_yaml({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self):
        assert flatten_yaml({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}

    def test_none(self):
        assert flatten_yaml(None) == {}

    def test_mixed(self):
        result = flatten_yaml({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
        assert result == {"a": 1, "b.c": 2, "b.d.e": 3}


class TestIsFunctionalChange:
    def test_functional(self):
        old = {"training_config": {"trainer": {"total_training_steps": 100}}}
        new = {"training_config": {"trainer": {"total_training_steps": 200}}}
        assert is_functional_change(old, new) is True

    def test_non_functional_data_path(self):
        old = {"data": {"train_files": "/old/path"}}
        new = {"data": {"train_files": "/new/path"}}
        assert is_functional_change(old, new) is False

    def test_non_functional_model_path(self):
        old = {"training_config": {"model": {"path": "/old/model"}}}
        new = {"training_config": {"model": {"path": "/new/model"}}}
        assert is_functional_change(old, new) is False

    def test_mixed_only_non_functional(self):
        old = {"data": {"train_files": "/old"}, "training_config": {"model": {"path": "/old/m"}}}
        new = {"data": {"train_files": "/new"}, "training_config": {"model": {"path": "/new/m"}}}
        assert is_functional_change(old, new) is False

    def test_mixed_with_functional(self):
        old = {"data": {"train_files": "/old"}, "training_config": {"trainer": {"lr": 0.001}}}
        new = {"data": {"train_files": "/new"}, "training_config": {"trainer": {"lr": 0.002}}}
        assert is_functional_change(old, new) is True

    def test_new_key_is_functional(self):
        old = {"a": 1}
        new = {"a": 1, "b": 2}
        assert is_functional_change(old, new) is True

    def test_removed_key_is_functional(self):
        old = {"a": 1, "b": 2}
        new = {"a": 1}
        assert is_functional_change(old, new) is True

    def test_identical(self):
        cfg = {"training_config": {"trainer": {"steps": 100}}}
        assert is_functional_change(cfg, cfg) is False

    def test_version_change_non_functional(self):
        old = {"version": "1.0", "training_config": {"trainer": {"steps": 100}}}
        new = {"version": "2.0", "training_config": {"trainer": {"steps": 100}}}
        assert is_functional_change(old, new) is False
