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

import pytest

from launcher.config_validator.schema.verl_schema_validation import (
    VerlDpoTrainerValidator,
)


class TestVerlDpoTrainerValidator:
    """Covers VerlDpoTrainerValidator.validate_loss_type."""

    @pytest.mark.parametrize("loss_type", ["sigmoid", "hinge", "ipo", "kto_pair", None])
    def test_accepts_valid_loss_type(self, loss_type):
        validator = VerlDpoTrainerValidator(beta=0.1, loss_type=loss_type)
        assert validator.loss_type == loss_type

    def test_rejects_invalid_loss_type(self):
        with pytest.raises(ValueError, match="loss_type must be one of"):
            VerlDpoTrainerValidator(beta=0.1, loss_type="bogus")

    def test_rejects_empty_string_loss_type(self):
        # Empty string is not None, and not in the allowed set, so it must raise
        with pytest.raises(ValueError, match="loss_type must be one of"):
            VerlDpoTrainerValidator(beta=0.1, loss_type="")
