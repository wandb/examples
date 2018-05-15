#!/bin/sh

# Runs a single experiment, which is a group of two training workers, and one validation worker
#
# To run:
# sh run_experiment.sh <group_id> <training_learning_rate> <training_momentum> <validation_phase_shift>
#
# ex:
# sh run_experiment.sh akj172 0.91 0.82 13

GROUP_ID="$1"
TRAIN_LEARNING_RATE="$2"
TRAIN_MOMENTUM="$3"
VALIDATE_PHASE_SHIFT="$4"

python train.py --group_id=experiment-"$GROUP_ID" --worker_index=0 \
    --learning_rate="$TRAIN_LEARNING_RATE" --momentum="$TRAIN_MOMENTUM"
python train.py --group_id=experiment-"$GROUP_ID" --worker_index=1 \
    --learning_rate="$TRAIN_LEARNING_RATE" --momentum="$TRAIN_MOMENTUM"

WANDB_RESUME=allow WANDB_RUN_ID="$GROUP_ID"-validation \
    python validate.py --group_id=experiment-"$GROUP_ID" --epoch=10 \
    --phase_shift="$VALIDATE_PHASE_SHIFT"
WANDB_RESUME=allow WANDB_RUN_ID="$GROUP_ID"-validation \
    python validate.py --group_id=experiment-"$GROUP_ID" --epoch=20 \
    --phase_shift="$VALIDATE_PHASE_SHIFT"
WANDB_RESUME=allow WANDB_RUN_ID="$GROUP_ID"-validation \
    python validate.py --group_id=experiment-"$GROUP_ID" --epoch=30 \
    --phase_shift="$VALIDATE_PHASE_SHIFT"
WANDB_RESUME=allow WANDB_RUN_ID="$GROUP_ID"-validation \
    python validate.py --group_id=experiment-"$GROUP_ID" --epoch=40 \
    --phase_shift="$VALIDATE_PHASE_SHIFT"
WANDB_RESUME=allow WANDB_RUN_ID="$GROUP_ID"-validation \
    python validate.py --group_id=experiment-"$GROUP_ID" --epoch=50 \
    --phase_shift="$VALIDATE_PHASE_SHIFT"