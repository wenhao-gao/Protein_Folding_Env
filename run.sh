#!/bin/bash

TASK=`date +"%Y%m%d_%T"`_1l2y_nmc
CUDA_VISIBLE_DEVICES=2 nohup python -u run_ppo_nmc.py \
    -c 'agents/configs/ppo.json' \
    --task ${TASK} \
    --log_frequency 1 \
    --ref_pdb 'data/protein_folding/1l2y.pdb' &> ${TASK}.out&
