set -x 
mkdir -p ./ckpt/mpa/7b_mistral_66k_ppo

export PATH=$HOME/.local/bin/:$PATH

# export CUDA_VISIBLE_DEVICES="5,4,6,7"

SFT_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
REWARD_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-rm-66k"
DATASET_PATH="kaist-ai/mpa-train-pairwise-merged-66k"

SAVE_PATH="./ckpt/mpa/7b_mistral_66k_ppo"

WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
WANDB_ENTITY="suehyun"
WANDB_PROJECT="mpa-rm"
WANDB_RUN_NAME="mpa-Mistral-7b-v0.2-hf-ppo-66k"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/openrlhf", "pip": "/openrlhf/requirements.txt", "excludes": ["examples/scripts/mpa/ckpt/mpa/7b_mistral_66k_rs"]}' \
    -- python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --pretrain $SFT_MODEL_PATH \
    --reward_pretrain $REWARD_MODEL_PATH \
    --save_path $SAVE_PATH \
    --save_steps 800 \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
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
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_org $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --flash_attn \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
        # --colocate_critic_reward \

    #     --flash_attn \
      # OOM when rollout micro batch size is 16 (1 vllm engine, 1/2 tensor parallel)
      # OOM when train micro batch size is 8
      # vllm engine 2 stalls
