# Config for running with custom scripts
Custom config allows user to use launcher to run some custom jobs that does not use our recipe. We use hydra format for the configs, same as our recipes. Please refer to the `config.yaml` as the template, which also aligns with the `config.yaml` in the recipe folder with some extra configs on cluster and custom script.
## Config fields
Here are some essential fields that user might want to override during for custom training
- training_cfg: This field contains most configs about the training runs
    - entry_script: Path to the entry script of training/fine-tuning. This path can be one in the container mounts.
    - script_args: The args that will be used to run this script
    - run: All runtime configs
        - name: Current run name
        - nodes: Number of nodes to use
        - ntasks_per_node: Number of devices to use per node
        - results_dir: Directories to store the result. It is recommended to keep it as `${base_results_dir}/${.name}` so everything will be in `base_results_dir`
- cluster: All cluster based configs
    - cluster_type: Type of the cluster, can be slrum(bcm) or k8s
    - instance_type: Instance type to use, if null will use default instance type in cluster.
    - cluster_config: The detailed cluster config, will be different between slrum and k8s. For details please refer to recipe's doc about cluster setup.
      - namespace: Namespace to launch jobs
      - custom_labels: k8s labels applied to job and also each pod running the job, see more details about labels in https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/
      - annotations: k8s annotations added to the job, see more details in https://kubernetes.io/docs/concepts/overview/working-with-objects/annotations/
      - priority_class_name: Kueue scheduler priority class name, see more details in https://kueue.sigs.k8s.io/
      - label_selector: k8s NodeAffinity functionality. To allow node selection based on required labels or priority scheduling based on preferred labels.
      - service_account_name: aws eks service account name. To give pods credentials to call aws services.
      - persistent_volume_claims: specify multiple persistent volume claims to mount job pod.
The rest of the configs are similar to the recipe configs.
## Launch
To launch the job, simply run inside the `SagemakerTrainingLauncher/launcher folder` with command `python main.py --config-path examples/custom_script/ --config-name config` or use your own config folder.
