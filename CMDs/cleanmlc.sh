#!/bin/bash
#PBS -N CC_newmcl_256
#PBS -S /bin/bash
#PBS -l hostlist=^quack,nodes=2:ppn=16,gpus=2,mem=40gb,walltime=24:00:00
#PBS -M sharmaa@informatik.uni-freiburg.de
#PBS -j oe
source ~/.bashrc
conda activate ngroupvit
cd /misc/student/sharmaa/groupvit/GroupViT 
#./tools/dist_launch.sh new_run_gvit.py configs/dataloader_group_vit_gcc_yfcc_30e.yml 2 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping=true --output /misc/lmbraid21/sharmaa/outputs/gs2_newpipeline2_64*4
./tools/dist_launch.sh new_run_gvit.py configs/newdataloader_group_vit_gcc_yfcc_30e.yml 2 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping=true --output /misc/lmbraid21/sharmaa/outputs/gs2_refinedmcl_256
#./tools/dist_launch.sh new_run_gvit.py configs/newdataloader_group_vit_gcc_yfcc_30e.yml 2 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping=true --output /misc/lmbraid21/sharmaa/outputs/gs2_newmcl_newmlc2

####PBS -l hostlist=^chip,nodes=1:ppn=48,gpus=4,mem=80gb,walltime=24:00:00
#PBS -l hostlist=^dicky,nodes=1:ppn=32,gpus=2,mem=50gb,walltime=24:00:00