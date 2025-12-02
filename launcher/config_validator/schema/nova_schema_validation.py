from pydantic import BaseModel, ConfigDict, Field, model_validator


### Nova Validators
class NovaTrainingConfigValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_length: int | None = Field(default=None, gt=0)
    global_batch_size: int = Field(gt=0)
    trainer: dict | None = None
    model: dict | None = None

    @model_validator(mode="after")
    def validate_all(self):
        # trainer checks
        trainer = self.trainer or {}
        max_epochs = trainer.get("max_epochs")
        if max_epochs and max_epochs <= 0:
            raise ValueError("trainer.max_epochs must be > 0")

        # model checks
        model = self.model or {}
        hidden_dropout = model.get("hidden_dropout")
        attention_dropout = model.get("attention_dropout")
        ffn_dropout = model.get("ffn_dropout")

        for name, val in {
            "hidden_dropout": hidden_dropout,
            "attention_dropout": attention_dropout,
            "ffn_dropout": ffn_dropout,
        }.items():
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        return self


class NovaRecipeValidator(BaseModel):
    """Top-level validator for Nova recipes."""

    model_config = ConfigDict(extra="allow")

    training_config: dict | None = None

    @model_validator(mode="after")
    def validate_nested_fields(self):
        # Validate training_config if present
        if self.training_config:
            try:
                NovaTrainingConfigValidator(**self.training_config)
            except Exception as e:
                raise ValueError(f"Error validating training_config: {str(e)}")

        return self
