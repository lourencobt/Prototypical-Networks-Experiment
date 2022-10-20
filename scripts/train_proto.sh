export CUDA_VISIBLE_DEVICES=0

ulimit -n 50000

lr=0.0001

python3 run_proto.py \
        --experiment_name "train_${lr}" \
        --mode "train" \
        --optimizer "adam" \
        --learning_rate "${lr}" \
        --momentum "0" \
        --weight_decay "0.00001" \
        --matching_fn 'l2' 