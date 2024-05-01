set -x

mkdir -p ./ckpt/mpa/7b_mistral_66k_rs
mkdir -p ./log/mpa

export CUDA_VISIBLE_DEVICES="0,1,2,3"

GENERATE_OUTPUT=./ckpt/mpa/7b_mistral_66k_rs/generate.jsonl
RM_OUTPUT=./ckpt/mpa/7b_mistral_66k_rs/rm.jsonl
MODEL_OUTPUT_PATH=./ckpt/mpa/7b_mistral_66k_rs
ITER_LOG_PATH=./log/mpa/7b_mistral_66k_rs_iter.txt

TRAINING_ITERS=20
ROLLOUT_BATCH_SIZE=2048

POLICY_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
REWARD_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-rm-66k"
DATASET_PATH="/mnt/nas/suehyun/MPA/data/train/preferences_v1_responses_for_orpo_64k_v2_ppo_mistral_input.jsonl"

BEST_OF=4

WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
WANDB_ENTITY="suehyun"
WANDB_PROJECT="mpa-rm"
WANDB_RUN_NAME="mpa-Mistral-7b-v0.2-hf-rs-66k"


checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "FAILED $1"
        exit 1
    fi
}

export PATH=$HOME/.local/bin/:$PATH

iter=0
if [ -f $ITER_LOG_PATH ]; then
    iter=$(cat $ITER_LOG_PATH)
fi

while (($iter < $TRAINING_ITERS)); do
    echo "Iter: $iter"
    # Use latest model if past first iteration
    if ((iter > 0)); then
        POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
    fi

    read -r -d '' generate_commands <<EOF
../../batch_inference.py
    --eval_task generate_vllm \
    --pretrain $POLICY_MODEL_PATH \
    --max_new_tokens 1024 \
    --dataset $DATASET_PATH  \
    --input_key prompt \
    --dataset_probs 0.5,0.5 \
    --temperature 0.9 \
    --flash_attn \
    --tp_size 8 \
    --best_of_n $BEST_OF \
    --iter $iter \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --output_path $GENERATE_OUTPUT
EOF
    echo $generate_commands
    python $generate_commands
    checkSuccess "GENERATE"

    read -r -d '' get_rewards_commands <<EOF
../../batch_inference.py
    --eval_task rm \
    --pretrain $REWARD_MODEL_PATH \
    --bf16 \
    --max_len 2048 \
    --dataset $GENERATE_OUTPUT  \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --flash_attn \
    --post_processor rs \
    --micro_batch_size 4 \
    --output_path $RM_OUTPUT
EOF
    echo $get_rewards_commands
    deepspeed $get_rewards_commands
    checkSuccess "RM"

    read -r -d '' sft_commands <<EOF
../../train_sft.py \
    --max_len 2048 \
    --dataset $RM_OUTPUT \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain $POLICY_MODEL_PATH \
    --save_path $MODEL_OUTPUT_PATH \
    --lr_scheduler cosine \
    --zero_stage 2 \
    --max_epochs 2 \
    --bf16 \
    --learning_rate 2e-6 \
    --gradient_checkpointing \
    --flash_attn \
    --use_wandb $WANDB_API_KEY \
    --wandb_org $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME
EOF
    echo $sft_commands
    deepspeed $sft_commands
    checkSuccess "SFT"

    iter=$((iter + 1))
    if [[ "$ITER_LOG_PATH" != "null" ]]; then
        echo $iter >$ITER_LOG_PATH
    fi
done