#!/bin/bash
#PBS -N GBs_SimpleTemplate_FT_123_caption
#PBS -S /bin/bash
#PBS -l hostlist=^trick,nodes=1:ppn=16,gpus=2,mem=40gb,walltime=24:00:00
#PBS -M sharmaa@informatik.uni-freiburg.de
#PBS -j oe
source ~/.bashrc
conda activate ngroupvit
cd /misc/student/sharmaa/groupvit/GroupViT 
./tools/dist_launch.sh main_group_vit.py configs/entropy_group_vit_gcc_yfcc_30e.yml 2 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping=true data.text_aug.template_set='simple'  --output /misc/lmbraid21/sharmaa/outputs/fT_123_11_simpletemp_onlycaption