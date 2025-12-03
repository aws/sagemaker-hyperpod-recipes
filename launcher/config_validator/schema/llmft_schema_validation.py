from pydantic import BaseModel, ConfigDict, Field, model_validator


### LLMFT Validators
class LLMFTTrainerValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    devices: int = Field(gt=0)
    num_nodes: int = Field(gt=0)


class LLMFTTrainingArgsValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    micro_train_batch_size: int | None = Field(default=None, gt=0)
    train_batch_size: int | None = Field(default=None, gt=0)
    learning_rate: float | None = Field(default=None, gt=0)
    lr_warmup_ratio: float | None = Field(default=None, ge=0, le=1)
    gradient_clipping: bool | None = None
    gradient_clipping_threshold: float | None = Field(default=None, gt=0)
    max_epochs: int | None = Field(default=None, gt=0)
    logging_steps: int | None = Field(default=None, gt=0)
    save_steps: int | None = Field(default=None, ge=0)
    eval_steps: int | None = Field(default=None, ge=-1)
    beta: float | None = Field(default=None, gt=0)
    nll_loss_coef: float | None = Field(default=None, ge=0)
    label_smoothing: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def check_batch_sizes(self):
        if self.micro_train_batch_size and self.train_batch_size:
            if self.train_batch_size < self.micro_train_batch_size:
                raise ValueError("train_batch_size must be >= micro_train_batch_size")
        return self


class LLMFTRecipeValidator(BaseModel):
    """Top-level validator for LLMFT recipes."""

    model_config = ConfigDict(extra="allow")

    trainer: LLMFTTrainerValidator | None = None
    training_config: dict | None = None

    @model_validator(mode="after")
    def validate_nested_fields(self):
        # Validate training_config.training_args if present
        if self.training_config and "training_args" in self.training_config:
            try:
                LLMFTTrainingArgsValidator(**self.training_config["training_args"])
            except Exception as e:
                raise ValueError(f"Error validating training_args: {str(e)}")

        return self
