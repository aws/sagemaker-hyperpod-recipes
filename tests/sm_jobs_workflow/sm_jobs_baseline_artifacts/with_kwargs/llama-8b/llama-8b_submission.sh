#!/bin/bash

pushd $(dirname -- $0)
python launch.py --job_name llama-8b --instance_type p5.48xlarge
popd