#!/bin/bash

pushd $(dirname -- $0)
python launch.py --job_name llama3.2-11b --instance_type p5.48xlarge
popd