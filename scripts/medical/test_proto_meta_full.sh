#!/bin/bash
export META_DATASET_ROOT=/home/guests/lbt/meta-dataset
export META_RECORDS_ROOT=/home/guests/lbt/data/medical_records
export CHECKPOINTS=/home/guests/lbt/few-shot/model-checkpoints/ProtoNet-Meta-PretrainedImagenetResnet.pt
export CUDA_VISIBLE_DEVICES=0

# activate environment
workon few-shot

ulimit -n 50000

K=50
Q=15

python3 run_proto_medical.py \
        --experiment_name "test_protonet_meta_full_${lr}" \
        --mode "test" \
        --matching_fn 'l2' \
        --shuffle \
        --test_model_path $CHECKPOINTS \
        --num_support $K \
        --num_query $Q
        
