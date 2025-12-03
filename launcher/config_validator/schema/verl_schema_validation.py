from pydantic import BaseModel, ConfigDict, Field, model_validator


### Verl Validators
class VerlModelOptimValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    lr: float | None = Field(gt=0)


class VerlModelConfigValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str | None = None


class VerlCriticValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    optim: VerlModelOptimValidator | dict | None = None
    model: VerlModelConfigValidator | dict | None = None
    ppo_micro_batch_size_per_gpu: int | None = Field(gt=0)


class VerlKlCtrlValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    kl_coef: float | None = Field(default=0.001, gt=0)
    target_kl: float | None = Field(default=0.1, gt=0)


class VerlAlgorithmValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    kl_ctrl: VerlKlCtrlValidator | None = None
    adv_estimator: str | None = None  # gae or grpo

    @model_validator(mode="after")
    def validate_adv_estimator(self):
        if self.adv_estimator and self.adv_estimator not in ["gae", "grpo"]:
            raise ValueError("adv_estimator must be either 'gae' or 'grpo'")
        return self


class VerlRecipeValidator(BaseModel):
    """Top-level validator for VERL recipes."""

    model_config = ConfigDict(extra="allow")

    critic: VerlCriticValidator | dict | None = None
    algorithm: VerlAlgorithmValidator | dict | None = None
    ray_init: dict | None = None

    @model_validator(mode="after")
    def validate_nested_fields(self):
        # Validate algorithm if present
        if self.algorithm and isinstance(self.algorithm, dict):
            try:
                VerlAlgorithmValidator(**self.algorithm)
            except Exception as e:
                raise ValueError(f"Error validating algorithm configuration: {str(e)}")
        if self.critic and isinstance(self.critic, dict):
            try:
                VerlCriticValidator(**self.critic)
            except Exception as e:
                raise ValueError(f"Error validating critic configuration: {str(e)}")

        return self
