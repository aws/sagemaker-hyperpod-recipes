#!/bin/bash
helm install --timeout=15m  --namespace default nova-lite-eval {$results_dir}/nova-lite-eval/k8s_templates
