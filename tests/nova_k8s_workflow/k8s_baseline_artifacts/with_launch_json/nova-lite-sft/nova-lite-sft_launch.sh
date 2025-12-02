#!/bin/bash
helm install --timeout=15m  --namespace default nova-lite-sft {$results_dir}/nova-lite-sft/k8s_templates
