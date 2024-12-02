#!/bin/bash
helm install --timeout=15m --wait  --namespace default llama-8b {$results_dir}/llama-8b/k8s_template
