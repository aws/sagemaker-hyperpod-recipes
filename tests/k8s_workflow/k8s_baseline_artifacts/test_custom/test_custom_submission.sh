#!/bin/bash
helm install --timeout=15m --wait  --namespace default test-custom {$results_dir}/test_custom/k8s_template
