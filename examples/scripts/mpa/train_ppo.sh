set -x 

mkdir -p ./ckpt/mpa/7b_mistral_66k_ppo

export CUDA_VISIBLE_DEVICES="5,6,4,7"

SFT_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
REWARD_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-rm-66k"
DATASET_PATH="kaist-ai/mpa-train-pairwise-merged-66k"

SAVE_PATH="./ckpt/mpa/7b_mistral_66k_ppo"

WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
WANDB_ENTITY="suehyun"Ω
WANDB_PROJECT="mpa-rm"
WANDB_RUN_NAME="mpa-Mistral-7b-v0.2-hf-ppo-66k"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

read -r -d '' training_commands <<EOF
../../train_ppo.py \
    --pretrain $SFT_MODEL_PATH \
    --reward_pretrain $REWARD_MODEL_PATH \
    --save_path $SAVE_PATH \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data $DATASET_PATH \
    --prompt_data_probs 1.0 \
    --input_key input \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_org $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
