#!/bin/bash
export META_DATASET_ROOT=/home/guests/lbt/meta-dataset
export META_RECORDS_ROOT=/home/guests/lbt/data/medical_records
export CHECKPOINTS=/home/guests/lbt/few-shot/model-checkpoints/ProtoNet-Meta-PretrainedImagenetResnet.pt
# export TMPDIR=/home/guests/lbt/few-shot/wandb/tmp
export CUDA_VISIBLE_DEVICES=1

# activate environment
workon few-shot

ulimit -n 50000

lr=0.0001
Q=15

for i in {1..3}
do      
        echo "Iteration: $i"
        for K in 1 5 15 30 50
        do      
                echo "Num_support: ${K}"    
                python3 run_proto_medical.py \
                        --experiment_name "test_protonet_meta_full_lr-${lr}_support_$i" \
                        --mode "test" \
                        --matching_fn 'l2' \
                        --shuffle \
                        --test_model_path $CHECKPOINTS \
                        --num_support $K \
                        --num_query $Q
        done
done