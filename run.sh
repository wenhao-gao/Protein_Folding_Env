#!/bin/bash

TASK=`date +"%Y%m%d_%T"`_1l2y_mc_pre
CUDA_VISIBLE_DEVICES=5 nohup python -u run_ppo.py \
    -c 'agents/configs/ppo.json' \
    --task ${TASK} \
    --log_frequency 1 \
    --parameters 'model_parameters/model.pth' \
    --ref_pdb 'data/protein_folding/1l2y.pdb' &> ${TASK}.out&
