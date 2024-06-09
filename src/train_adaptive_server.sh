#!/bin/bash

source activate jepa
which python

python -u train_classifier_adaptive_ssfl.py --output_root_dir output_adaptive/ --config_file config_adaptive.yml --data_name CIFAR10 --model_name wresnet28x2 --control_name 250_fix@0.95_5_1_non-iid-d-0.3_5-5_0.5_0_1 --resume_mode=1
