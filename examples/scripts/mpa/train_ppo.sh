set -x 

mkdir -p ./ckpt/7b_mistral_66k_ppo_helpful
export CUDA_VISIBLE_DEVICES="0,1,2,3"

SFT_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-sft-66k"
REWARD_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-rm-66k-helpful"
DATASET_PATH="kaist-ai/mpa-dpo-ppo-rs-66k"

SAVE_PATH="./ckpt/7b_mistral_66k_ppo_helpful"

WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
WANDB_ENTITY="suehyun"
WANDB_PROJECT="mpa-ppo"
WANDB_RUN_NAME="use-helpful-rm"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

read -r -d '' training_commands <<EOF
../../train_ppo.py \
    --pretrain $SFT_MODEL_PATH \
    --reward_pretrain $REWARD_MODEL_PATH \
    --save_path $SAVE_PATH \
    --micro_train_batch_size 4 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 256 \
    --max_epochs 1 \
    --save_steps 400 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.1 \
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
