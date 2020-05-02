#!/bin/bash


CUDA_VISIBLE_DEVICES= nohup python -u run_ppo.py \
    -c 'agents/configs/ppo.json' \
    --task '1qgm_test' \
    --ref_pdb 'data/folding/1qgm.pdb' &> temp.out&
