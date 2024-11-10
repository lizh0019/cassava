#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

#if [ "$#" -lt 3 ]; then
#    echo "Usage: $0 PATH_TO_IMAGES RESNET_CHECKPOINT_FILE EXPERIMENT_ROOT ..."
#    echo "See the README for more info"
#    echo "Download ResNet-50 checkpoint from https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models"
#    exit 1
#fi

# Shift the arguments so that we can just forward the remainder.
#IMAGE_ROOT=$1 ; shift
#INIT_CHECKPT=$1 ; shift
#EXP_ROOT=$1 ; shift

IMAGE_ROOT=/home/haria/triplet-reid/Huilue ; 
INIT_CHECKPT=/home/haria/triplet-reid/data/resnet_v1_50.ckpt ; 
EXP_ROOT=/home/haria/triplet-reid/experiments ; 

python3 train.py \
    --train_set data/huilue_train.csv \
    --model_name resnet_v1_50 \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --embedding_dim 128 \
    --loading_threads 1\
    --batch_p 18 \
    --batch_k 8 \
    --pre_crop_height 144 --pre_crop_width 288 \
    --net_input_height 128 --net_input_width 256 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 3e-4 \
    --train_iterations 500000 \
    --decay_start_iteration 15000 \
    "$@"
