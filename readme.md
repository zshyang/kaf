To pretrain:
```
#!/bin/bash

cd /workspace/src/

CUDA_VISIBLE_DEVICES=0,1,2,3                                                \
python                                                                      \
pretrain1.py                                                                \
--batch_size             32                                                 \
--classes                512                                                \
--contrastive_method     pcl                                                \
--data_dir               /workspace/data/chd/contrastive                    \
--dataset                chd                                                \
--device                 cuda:0                                             \
--do_contrast                                                               \
--epochs                 51                                                 \
--experiment_name        contrast_chd_pcl_                                  \
--initial_filter_size    32                                                 \
--lr                     0.002                                              \
--num_works              48                                                 \
--patch_size             512 512                                            \
--results_dir            ./results/graph_pretrain                           \
--slice_threshold        0.1                                                \
--save                   full_contrast                                      \
--temp                   0.1                                                \
--weight_cnn_contrast    1.0                                                \
--weight_graph_contrast  1.0                                                \
--weight_corr            1.0
```

To finetune:

```
#!/bin/bash

cd /workspace/src/

sample_size=$1
fold=$2

WEIGHT_CNN=$3
WEIGHT_GRAPH=$4
WEIGHT_CORR=$5

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
--pretrained_model_path  model/latest.pth \
--results_dir            ./results/finetune                                 \
--runs_dir               ./runs/finetune                                    \
--restart                                                                   \
--sampling_k             $sample_size                                       \
--save                   finetune_c_$WEIGHT_CNN\_g_$WEIGHT_GRAPH\_co_$WEIGHT_CORR
```

To train from scratch:

```
#!/bin/bash

cd /workspace/src/

sample_size=$1
fold=$2

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
--experiment_name        sup_chd_pcl_sample_$sample_size\_fold_$fold\_      \
--fold                   $fold                                              \
--initial_filter_size    32                                                 \
--lr                     0.05                                               \
--lr                     5e-5                                               \
--min_lr                 5e-6                                               \
--num_works              12                                                 \
--patch_size             512 512                                            \
--results_dir            ./results/sup                                      \
--runs_dir               ./runs/sup                                         \
--sampling_k             $sample_size                                       \
--save                   all_graph_features
```
