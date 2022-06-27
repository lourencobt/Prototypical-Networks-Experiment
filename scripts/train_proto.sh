export CUDA_VISIBLE_DEVICES=0

ulimit -n 50000

python3 run_proto.py \
        --mode "train" \
        --learning_rate 0.001 \
        --momentum 0 \
        --weight_decay 0 \
        --matching_fn 'l2' 
