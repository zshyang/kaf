#!/bin/bash
cd /workspace/src/gcl_1

WEIGHT_CNN_CONTRAST=$1
WEIGHT_GRAPH_CONTRAST=$2
WEIGHT_CORR=$3

CUDA_VISIBLE_DEVICES=0,1,2,3                                                \
python                                                                      \
pretrain1.py                                                                \
--batch_size             32                                                 \
--classes                512                                                \
--contrastive_method     pcl                                                \
--data_dir               /workspace/data/acdc/downloaded/supervised         \
--dataset                acdc                                               \
--device                 cuda:0                                             \
--do_contrast                                                               \
--epochs                 51                                                 \
--experiment_name        contrast_                                          \
--initial_filter_size    48                                                 \
--lr                     0.002                                              \
--num_layers             6                                                  \
--num_works              48                                                 \
--patch_size             352 352                                            \
--results_dir            ./results/acdc_graph_pretrain                      \
--runs_dir               ./runs/acdc_graph_pretrain                         \
--slice_threshold        0.35                                               \
--save                   wcn_$WEIGHT_CNN_CONTRAST\_wg_$WEIGHT_GRAPH_CONTRAST\_wco_$WEIGHT_CORR \
--temp                   0.1                                                \
--weight_cnn_contrast    $WEIGHT_CNN_CONTRAST                               \
--weight_graph_contrast  $WEIGHT_GRAPH_CONTRAST                             \
--weight_corr            $WEIGHT_CORR                                       \
--weight_local_contrast  0.0

# i need to change batch size to 32
# i need to remove debug option
# i need to change num_works
