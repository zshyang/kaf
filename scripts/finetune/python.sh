#!/bin/bash

cd /workspace/src/gcl_1

sample_size=$1
fold=$2
export num_layer=8

CUDA_VISIBLE_DEVICES=0                                                      \
python                                                                      \
sup.py                                                                      \
--batch_size             10                                                 \
--classes                8                                                  \
--data_dir               /workspace/data/chd/supervised                     \
--dataset                chd                                                \
--device                 cuda:0                                             \
--enable_few_data                                                           \
--epochs                 101                                                \
--experiment_name        sup_chd_pcl_sample_$sample_size\_                  \
--fold                   $fold                                              \
--initial_filter_size    32                                                 \
--lr                     5e-5                                               \
--min_lr                 5e-6                                               \
--num_works              12                                                 \
--patch_size             512 512                                            \
--pretrained_model_path  /workspace/src/gcl_1/results/graph_pretrain/contrast_chd_pcl_full_contrast/model/latest.pth \
--results_dir            ./results/finetune                                 \
--runs_dir               ./runs/finetune                                    \
--restart                                                                   \
--sampling_k             $sample_size                                       \
--save                   full_finetune