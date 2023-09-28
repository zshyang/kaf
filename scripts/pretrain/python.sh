#!/bin/bash

cd /workspace/src/gcl_0

export num_layer=8

CUDA_VISIBLE_DEVICES=0,1,2,3                                                    \
python                                                                      \
train_contrast.py                                                           \
--batch_size             160                                                \
--classes                512                                                \
--contrastive_method     pcl                                                \
--data_dir               /workspace/data/chd/contrastive                    \
--dataset                chd                                                \
--device                 cuda:0                                             \
--do_contrast                                                               \
--epochs                 301                                                \
--experiment_name        contrast_chd_pcl_                                  \
--graph_contrast_weight  1.0                                                \
--initial_filter_size    32                                                 \
--lr                     0.1                                                \
--num_layer              $num_layer                                         \
--num_works              48                                                 \
--patch_size             512 512                                            \
--pick_y_methods         a,a,a,a,a,a,a,avg_pool                             \
--pick_y_numbers         0,0,0,0,0,0,0,1024                                 \
--ratios                 0,0,0,0,8,8,4,4                                    \
--slice_threshold        0.1                                                \
--temp                   0.1                                                \
--use_graph_flags        0,0,0,0,0,0,0,0
# --debug_mode
# -m debugpy --listen 0.0.0.0:5569 --wait-for-client          \
