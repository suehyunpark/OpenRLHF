echo "${1:-"my default"}"

WORLD_SIZE=${1:-4}
REWARD_BATCH_SIZE=${2:-32}
SFT_BATCH_SIZE=${3:-8}

ROLLOUT_BATCH_SIZE=$((REWARD_BATCH_SIZE * WORLD_SIZE))
TRAINING_ITERS=$((66000 / ROLLOUT_BATCH_SIZE))

echo "WORLD_SIZE: $WORLD_SIZE"
echo "REWARD_BATCH_SIZE: $REWARD_BATCH_SIZE"
echo "SFT_BATCH_SIZE: $SFT_BATCH_SIZE"
echo "ROLLOUT_BATCH_SIZE: $ROLLOUT_BATCH_SIZE"
echo "TRAINING_ITERS: $TRAINING_ITERS"