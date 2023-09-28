#!/bin/bash

cd /workspace/src/gcl_1

sample_size=$1
fold=$2


CUDA_VISIBLE_DEVICES=0                                                      \
python                                                                      \
sup.py                                                       \
--batch_size             10                                                  \
--classes                4                                                  \
--data_dir               /workspace/data/acdc/downloaded/supervised                     \
--dataset                acdc                                                \
--device                 cuda:0                                             \
--enable_few_data                                                           \
--epochs                 201                                                \
--experiment_name        pcl_sample_$sample_size\_fold_$fold\_      \
--fold                   $fold                                              \
--initial_filter_size    32                                                 \
--lr                     5e-4                                               \
--min_lr                 5e-5                                               \
--num_works              12                                                 \
--patch_size             352 352                                            \
--results_dir            ./results/sup_acdc                                      \
--runs_dir               ./runs/sup_acdc                                         \
--sampling_k             $sample_size                                       \
--save                   all_graph_features                                 

# remove debug
# change batch size to 10
