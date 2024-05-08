set -x

mkdir -p ./ckpt/7b_mistral_66k_rs
mkdir -p ./log

# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG=WARN

GENERATE_OUTPUT=./ckpt/7b_mistral_66k_rs/generate.jsonl
RM_OUTPUT=./ckpt/7b_mistral_66k_rs/rm.jsonl
MODEL_OUTPUT_PATH=./ckpt/7b_mistral_66k_rs
ITER_LOG_PATH=./log/7b_mistral_66k_rs_iter.txt

TRAINING_ITERS=480
ROLLOUT_BATCH_SIZE=128

POLICY_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
REWARD_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-rm-66k-openrlhf"
DATASET_PATH="kaist-ai/mpa-pairwise-merged-66k"

BEST_OF=4

WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
WANDB_ENTITY="suehyun"
WANDB_PROJECT="mpa-rm"
WANDB_RUN_NAME="mpa-Mistral-7b-v0.2-hf-rs-66k"

# INPUT_TEMPLATE='[INST] {} [/INST] '  # cannot pass string with curly brackets to python as argument
    # --input_template "$INPUT_TEMPLATE" \  # only need to pass this during first generation of rollouts
# do not use flash attention 2 with mistral-based reward model because of padding side problem

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
    --input_key input \
    --dataset_probs 1.0 \
    --temperature 0.9 \
    --flash_attn \
    --tp_size 4 \
    --best_of_n $BEST_OF \
    --iter $iter \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
# #     --output_path $GENERATE_OUTPUT
# EOF
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
    --tp_size 4 \
    --post_processor rs \
    --micro_batch_size 32 \
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
    --micro_train_batch_size 8 \
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