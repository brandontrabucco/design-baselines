#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash $HOME/design-baselines/scripts/coms-ant.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash $HOME/design-baselines/scripts/coms-dkitty.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash $HOME/design-baselines/scripts/coms-hopper.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash $HOME/design-baselines/scripts/coms-superconductor.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash $HOME/design-baselines/scripts/coms-gfp.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash $HOME/design-baselines/scripts/coms-utr.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash $HOME/design-baselines/scripts/coms-tf-bind-8.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash $HOME/design-baselines/scripts/coms-chembl.sh
