#!/bin/bash

#SBATCH --partition=LocalQ         # Partition name
#SBATCH -J mpa-Mistral-7b-v0.2-hf-ppo
#SBATCH -o /mnt/nas/suehyun/OpenRLHF/examples/scripts/mpa/logs/out_%j.txt              # Path to output log file (%j expands to job name)
#SBATCH -e /mnt/nas/suehyun/OpenRLHF/examples/scripts/mpa/logs/err_%j.err              # Path to error log file (%j expands to job name)
#SBATCH --time=24:00:00            # Time limit
#SBATCH --ntasks-per-node=1        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --nodes=1                  # Request one node
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch
#SBATCH --output=out.log
#SBATCH --gres=gpu:4               # Number of GPUs to be allocated
#SBATCH --partition=LocalQ         # Partition name


# project settings
PROJECT_PATH=$(cd ../../; realpath .)
IMAGE_NAME="nvcr.io/nvidia/pytorch:23.12-py3"
MOUNT="$PROJECT_PATH:/openrlhf,$HOME/.cache:/root/.cache,/dev/null:/root/.bashrc"

JOBLOG="$(realpath .)/logs/train_ppo_llama_ray-$SLURM_JOB_ID.log"
mkdir logs
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# launch ray daemon
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"  &>> ${JOBLOG}

echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
  && pip install ray[default] \
  && /root/.local/bin/ray start --head --node-ip-address=$ip --port=$port --block" &>> ${JOBLOG} &
sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
for ((i = 1; i < worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
  srun --nodes=1 --ntasks=1 -w "$node_i" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    && pip install ray[default] \
    && /root/.local/bin/ray start --address "$ip_head" --block" &>> ${JOBLOG} &
  sleep 1s;
done

sleep 30s

# variables
SFT_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
REWARD_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-hf-rm-66k"
DATASET_PATH="kaist-ai/mpa-train-pairwise-merged-66k"

mkdir -p ./ckpt/mpa/7b_mistral_66k_ppo
SAVE_PATH="/openrlhf/examples/scripts/ckpt/mpa/7b_mistral_66k_ppo"

WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
WANDB_ENTITY="suehyun"
WANDB_PROJECT="mpa-rm"
WANDB_RUN_NAME="mpa-Mistral-7b-v0.2-hf-ppo-66k"

# ===== submit ray job =====
# Job start
srun --overlap --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
  "pip install ray[default] \
  && /root/.local/bin/ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{\"working_dir\": \"/openrlhf\", \"pip\": \"/openrlhf/requirements.txt\"}' \
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
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_org $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME" &>> ${JOBLOG}


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}