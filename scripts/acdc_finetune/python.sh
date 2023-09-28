#!/bin/bash

cd /workspace/src/gcl_1

sample_size=$1
fold=$2

CUDA_VISIBLE_DEVICES=0                                                      \
python                                                                      \
sup.py                                                                      \
--batch_size             10                                                 \
--classes                4                                                  \
--data_dir               /workspace/data/acdc/downloaded/supervised         \
--dataset                acdc                                               \
--device                 cuda:0                                             \
--enable_few_data                                                           \
--epochs                 201                                                \
--experiment_name        sample_$sample_size\_                              \
--fold                   $fold                                              \
--initial_filter_size    48                                                 \
--lr                     5e-4                                               \
--min_lr                 5e-5                                               \
--num_works              12                                                 \
--patch_size             352 352                                            \
--pretrained_model_path  /workspace/src/gcl_1/results/acdc_graph_pretrain/contrast_default/model/epoch_050.pth \
--results_dir            ./results/acdc_finetune                            \
--runs_dir               ./runs/acdc_finetune                               \
--restart                                                                   \
--sampling_k             $sample_size                                       \
--save                   default

# remove debug
# chagne batch size to 10