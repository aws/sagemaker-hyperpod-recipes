#!/bin/bash
helm install --timeout=15m  --namespace kubeflow nova-lite-rft {$results_dir}/nova-lite-rft/k8s_templates
