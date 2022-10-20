export CUDA_VISIBLE_DEVICES=3  
export CHECKPOINTS=/home/guests/lbt/few-shot/model-checkpoints/ProtoNet-Meta-PretrainedImagenetResnet.pt

ulimit -n 50000

lr=0.0001

python3 run_proto.py \
        --experiment_name "test_${lr}" \
        --test_model_path $CHECKPOINTS \
        --mode "test" \
        --matching_fn 'l2' \
        --shuffle 
        
        
        