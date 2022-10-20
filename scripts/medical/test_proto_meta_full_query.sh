#!/bin/bash
export META_DATASET_ROOT=/home/guests/lbt/meta-dataset
export META_RECORDS_ROOT=/home/guests/lbt/data/medical_records
export CHECKPOINTS=/home/guests/lbt/few-shot/model-checkpoints/ProtoNet-Meta-PretrainedImagenetResnet.pt
export CUDA_VISIBLE_DEVICES=3  

# activate environment
workon few-shot

ulimit -n 50000

lr=0.0001
K=5

for Q in 10 20 30 40 50
do      
        echo "Num_query: ${Q}"    
        python3 run_proto_medical.py \
                --experiment_name "test_protonet_meta_full_lr-${lr}_query" \
                --mode "test" \
                --matching_fn 'l2' \
                --shuffle \
                --test_model_path $CHECKPOINTS \
                --num_support $K \
                --num_query $Q
done
                
