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

import logging
from difflib import SequenceMatcher
from typing import List

from launcher.nemo.constants import ROOT_DIR

logger = logging.getLogger(__name__)


def test_readme_table_links():
    readme_path = ROOT_DIR / "README.md"
    log_line = lambda line: logger.info(f"\nFailing line:\n{line}")

    def pluck_path_strings(line: str):
        paths_str: List[str] = []

        for chunk in line.split("|"):  # split by column delimeter
            if "[link]" in chunk:
                chunk = chunk.strip()
                chunk = chunk.replace("[link]", "")
                assert chunk[0] == "(" and chunk[-1] == ")", log_line(line)
                chunk = chunk[1:-1]  # remove parantheses
                paths_str.append(chunk)

        return paths_str

    with open(readme_path, "r") as fd:
        for line in fd:
            """
            Example:
            | Hugging Face | Llama 3.2 | 11b  | 8192            | 4     | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_2_11b_seq8k_gpu_p5x4_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_2_11b_seq8k_gpu_p5x4_pretrain.sh) |
            """
            if "[link]" in line:
                paths_str = pluck_path_strings(line)

                if len(paths_str) == 1:
                    file_path = ROOT_DIR / paths_str[0]
                    assert file_path.exists(), log_line(line)
                # there is a config and a script link
                elif len(paths_str) == 2:
                    config_file_path = ROOT_DIR / paths_str[0]
                    launcher_script_path = ROOT_DIR / paths_str[1]
                    # try to catch if a launch script is pointing to an incorrect config
                    str_distance_ratio = SequenceMatcher(None, config_file_path.stem, launcher_script_path.stem).ratio()

                    assert config_file_path.exists(), log_line(line)
                    assert launcher_script_path.exists(), log_line(line)
                    assert str_distance_ratio >= 0.75, log_line(line)
                else:
                    raise Exception("test condition not covered")
