#!/bin/bash

TASK=`date +"%Y%m%d_%T"`_1qgm_nmc
CUDA_VISIBLE_DEVICES=0 nohup python -u run_ppo_nmc.py \
    -c 'agents/configs/ppo.json' \
    --task ${TASK} \
    --log_frequency 1 \
    --ref_pdb 'data/protein_folding/1qgm.pdb' &> ${TASK}.out&
