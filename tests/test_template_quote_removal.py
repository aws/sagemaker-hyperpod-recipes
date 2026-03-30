# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""
Unit tests for template variable quote removal based on parameter types.
"""

import yaml

from utils.template_utils import remove_quotes_from_numeric_params


class TestTemplateQuoteRemoval:
    """Test suite for removing quotes from numeric template variables."""

    def test_remove_quotes_from_integer_single_quotes(self):
        """Test removing single quotes from integer type parameters."""
        content = """
        training_config:
          global_batch_size: '{{global_batch_size}}'
          max_epochs: '{{max_epochs}}'
        """

        param_definitions = {
            "global_batch_size": {"type": "integer", "default": 8},
            "max_epochs": {"type": "integer", "default": 5},
        }

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert "global_batch_size: {{global_batch_size}}" in result
        assert "max_epochs: {{max_epochs}}" in result
        assert "'{{global_batch_size}}'" not in result
        assert "'{{max_epochs}}'" not in result

    def test_remove_quotes_from_float_double_quotes(self):
        """Test removing double quotes from float type parameters."""
        content = """
        training_config:
          learning_rate: "{{learning_rate}}"
          lr_warmup_ratio: "{{lr_warmup_ratio}}"
        """

        param_definitions = {
            "learning_rate": {"type": "float", "default": 0.0001},
            "lr_warmup_ratio": {"type": "float", "default": 0.1},
        }

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert "learning_rate: {{learning_rate}}" in result
        assert "lr_warmup_ratio: {{lr_warmup_ratio}}" in result
        assert '"{{learning_rate}}"' not in result
        assert '"{{lr_warmup_ratio}}"' not in result

    def test_keep_quotes_for_string_parameters(self):
        """Test that string parameters keep their quotes."""
        content = """
        training_config:
          model_name_or_path: '{{model_name_or_path}}'
          data_path: '{{data_path}}'
          name: '{{name}}'
        """

        param_definitions = {
            "model_name_or_path": {"type": "string", "default": "meta-llama/Llama-3.2-1B"},
            "data_path": {"type": "string", "default": ""},
            "name": {"type": "string", "default": "test-job"},
        }

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert "'{{model_name_or_path}}'" in result
        assert "'{{data_path}}'" in result
        assert "'{{name}}'" in result

    def test_mixed_types(self):
        """Test handling mixed parameter types in same content."""
        content = """
        training_config:
          model_config:
            model_name_or_path: '{{model_name_or_path}}'
            lora_alpha: '{{lora_alpha}}'
          training_args:
            learning_rate: '{{learning_rate}}'
            max_epochs: '{{max_epochs}}'
            output_path: '{{output_path}}'
        """

        param_definitions = {
            "model_name_or_path": {"type": "string"},
            "lora_alpha": {"type": "float"},
            "learning_rate": {"type": "float"},
            "max_epochs": {"type": "integer"},
            "output_path": {"type": "string"},
        }

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert "lora_alpha: {{lora_alpha}}" in result
        assert "learning_rate: {{learning_rate}}" in result
        assert "max_epochs: {{max_epochs}}" in result

        assert "'{{model_name_or_path}}'" in result
        assert "'{{output_path}}'" in result

    def test_parameter_not_in_content(self):
        """Test that parameters not in content don't cause issues."""
        content = """
        training_config:
          lora_alpha: '{{lora_alpha}}'
        """

        param_definitions = {
            "lora_alpha": {"type": "float"},
            "learning_rate": {"type": "float"},
            "max_epochs": {"type": "integer"},
        }

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert "lora_alpha: {{lora_alpha}}" in result

    def test_parameter_without_type_defaults_to_string(self):
        """Test that parameters without type field default to string behavior."""
        content = """
        training_config:
          unknown_param: '{{unknown_param}}'
        """

        param_definitions = {"unknown_param": {"default": "some_value"}}

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert "'{{unknown_param}}'" in result

    def test_validates_yaml_output_numeric_types(self):
        """Test that after substitution, numeric types are actually numeric in YAML."""
        content = """
        training_config:
          lora_alpha: {{lora_alpha}}
          learning_rate: {{learning_rate}}
          max_epochs: {{max_epochs}}
          model_name: '{{model_name}}'
        """

        substituted = content.replace("{{lora_alpha}}", "32")
        substituted = substituted.replace("{{learning_rate}}", "0.0001")
        substituted = substituted.replace("{{max_epochs}}", "5")
        substituted = substituted.replace("{{model_name}}", "test-model")

        config = yaml.safe_load(substituted)

        assert isinstance(config["training_config"]["lora_alpha"], int)
        assert config["training_config"]["lora_alpha"] == 32

        assert isinstance(config["training_config"]["learning_rate"], float)
        assert config["training_config"]["learning_rate"] == 0.0001

        assert isinstance(config["training_config"]["max_epochs"], int)
        assert config["training_config"]["max_epochs"] == 5

        assert isinstance(config["training_config"]["model_name"], str)
        assert config["training_config"]["model_name"] == "test-model"

    def test_validates_yaml_output_with_quotes_creates_strings(self):
        """Test that WITH quotes, numeric values become strings ."""
        content = """
        training_config:
          lora_alpha: '{{lora_alpha}}'
        """

        substituted = content.replace("{{lora_alpha}}", "32")

        config = yaml.safe_load(substituted)

        assert isinstance(config["training_config"]["lora_alpha"], str)
        assert config["training_config"]["lora_alpha"] == "32"

    def test_handles_both_quote_types_in_same_content(self):
        """Test handling content with mixed quote types."""
        content = """
        training_config:
          param1: '{{param1}}'
          param2: "{{param2}}"
          param3: '{{param3}}'
        """

        param_definitions = {
            "param1": {"type": "integer"},
            "param2": {"type": "float"},
            "param3": {"type": "integer"},
        }

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert "param1: {{param1}}" in result
        assert "param2: {{param2}}" in result
        assert "param3: {{param3}}" in result
        assert '"{{' not in result
        assert "'{{param1}}'" not in result
        assert "'{{param3}}'" not in result

    def test_empty_parameter_definitions(self):
        """Test handling empty parameter definitions."""
        content = """
        training_config:
          some_param: '{{some_param}}'
        """

        param_definitions = {}

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert result == content

    def test_none_parameter_definitions(self):
        """Test handling None parameter definitions."""
        content = """
        training_config:
          some_param: '{{some_param}}'
        """

        result = remove_quotes_from_numeric_params(content, None)

        assert result == content

    def test_template_variable_not_at_end_of_line(self):
        """Test template variables with additional content on same line (e.g., inline comments)."""
        content = """
        training_config:
          lora_alpha: '{{lora_alpha}}'  # LoRA scaling parameter
          learning_rate: '{{learning_rate}}'  # Initial learning rate
          model_name: '{{model_name}}'  # Model identifier
        """

        param_definitions = {
            "lora_alpha": {"type": "integer"},
            "learning_rate": {"type": "float"},
            "model_name": {"type": "string"},
        }

        result = remove_quotes_from_numeric_params(content, param_definitions)

        assert "lora_alpha: {{lora_alpha}}  # LoRA scaling parameter" in result
        assert "learning_rate: {{learning_rate}}  # Initial learning rate" in result

        assert "'{{model_name}}'  # Model identifier" in result

        assert "# LoRA scaling parameter" in result
        assert "# Initial learning rate" in result
        assert "# Model identifier" in result
