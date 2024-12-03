# Core NeMo launching implementations
This folder contains the core launching framework for NeMo based implementations. We use the same design as the [NeMo-Framework-Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher/tree/main). Bsaically there are 2 steps:
- A stage defined in `stages.py` will prepare for the training script launching command and the cluster configs, passing these configs into the actual launcher
- A launcher defined in `launchers.py` will take the configs from the stage and generate the real launching script. Then launcher will kick off the run using corresponding cluster methods, i.e. slurm or k8s.

## Stages
We support different use cases, and each will be corresponding to a stage:
- `SMTraining`: Stage to run native NeMo workload
- `SMTrainingGPURecipe`: Stage used to run our GPU recipes
- `SMTrainingTrainiumRecipe`: Stage to run our Trainium recipes
- `SMCustomTrainingGPU`: Stage for training with custom script on GPU
- `SMCustomTrainingTrainium`: Stage for training with custom script on Trainium

## Launchers
Currently we only need our own launchers for custom jobs, because we need to manage the `torchrun` command
- `SMSlurmLauncher`: Launcher for custom jobs using slurm
