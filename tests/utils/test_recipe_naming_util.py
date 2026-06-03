"""Tests for utils/recipe_naming_util.py."""

from utils.recipe_naming_util import config_name, instance_short, seq_short


class TestInstanceShort:
    def test_p4de(self):
        assert instance_short("ml.p4de.24xlarge") == "p4de-24"

    def test_p5(self):
        assert instance_short("ml.p5.48xlarge") == "p5-48"

    def test_g5_12(self):
        assert instance_short("ml.g5.12xlarge") == "g5-12"

    def test_g5_48(self):
        assert instance_short("ml.g5.48xlarge") == "g5-48"

    def test_p4d(self):
        assert instance_short("ml.p4d.24xlarge") == "p4d-24"

    def test_family_only(self):
        assert instance_short("ml.trn1.xlarge") == "trn1"


class TestSeqShort:
    def test_4096(self):
        assert seq_short(4096) == "seq4k"

    def test_8192(self):
        assert seq_short(8192) == "seq8k"

    def test_16384(self):
        assert seq_short(16384) == "seq16k"

    def test_1536(self):
        assert seq_short(1536) == "seq1536"

    def test_1024(self):
        assert seq_short(1024) == "seq1k"


class TestConfigName:
    def test_fft(self):
        result = config_name("fine-tuning/qwen/verl-sft-qwen-3-dot-5-4b-fft.yaml", "ml.p4de.24xlarge", 4096)
        assert result == "verl-sft-qwen-3-dot-5-4b-p4de-24-seq4k-fft"

    def test_lora(self):
        result = config_name("verl-sft-qwen-2-5-7b-instruct-lora.yaml", "ml.p5.48xlarge", 8192)
        assert result == "verl-sft-qwen-2-5-7b-instruct-p5-48-seq8k-lora"

    def test_no_suffix(self):
        result = config_name("some-recipe.yaml", "ml.g5.12xlarge", 4096)
        assert result == "some-recipe-g5-12-seq4k"

    def test_strips_existing_instance(self):
        result = config_name("verl-sft-qwen-3-4b-p4de-24-fft.yaml", "ml.p5.48xlarge", 4096)
        assert result == "verl-sft-qwen-3-4b-p5-48-seq4k-fft"

    def test_strips_existing_seq(self):
        result = config_name("verl-sft-qwen-3-4b-seq4k-fft.yaml", "ml.p4de.24xlarge", 8192)
        assert result == "verl-sft-qwen-3-4b-p4de-24-seq8k-fft"
