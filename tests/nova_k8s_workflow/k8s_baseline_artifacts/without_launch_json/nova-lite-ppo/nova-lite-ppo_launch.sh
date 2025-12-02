#!/bin/bash
helm install --timeout=15m  --namespace default nova-lite-ppo {$results_dir}/nova-lite-ppo/k8s_templates
