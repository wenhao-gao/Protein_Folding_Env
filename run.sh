#!/bin/bash


CUDA_VISIBLE_DEVICES= nohup python -u run_ppo.py \
    -c 'agents/configs/ppo.json' \
    --task '1qgm_test' \
    --log_frequency 1 \
    --ref_pdb 'data/protein_folding/1qgm.pdb' &> temp.out&
