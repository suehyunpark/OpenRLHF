set -x 


export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG=WARN


PRETRAIN_MODEL_PATH="kaist-ai/mpa-Mistral-7b-v0.2-rm-66k-helpful"
DATASET_PATH="kaist-ai/mpa-pairwise-merged-66k"


read -r -d '' training_commands <<EOF
../../evaluate_rm.py \
     --pretrain $PRETRAIN_MODEL_PATH \
     --bf16 \
     --max_len 2048 \
     --zero_stage 3 \
     --dataset $DATASET_PATH \
     --dataset_probs 1.0 \
     --prompt_key input \
     --chosen_key chosen \
     --rejected_key rejected \
     --flash_attn
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # micro_train_batch_size 16 OOMs on 4xA100 80GB gpu


if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port 29501 $training_commands
fi
