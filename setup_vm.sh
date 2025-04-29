#!/bin/bash
VM_ID=$1
STORAGE_ACCOUNT=$2
CONTAINER_NAME=$3
RESULT_CONTAINER=$4

echo "Setting up VM $VM_ID with storage account $STORAGE_ACCOUNT"

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git jq

PROJECT_DIR="$HOME/rl-project"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

export AZURE_STORAGE_ACCOUNT="$STORAGE_ACCOUNT"

echo "Downloading project files from $CONTAINER_NAME..."
az storage blob download-batch \
  --source "$CONTAINER_NAME" \
  --destination . \
  --account-name "$STORAGE_ACCOUNT" \
  --auth-mode login

if [ $? -ne 0 ]; then
  echo "Error: Failed to download project files."
  exit 1
fi

CONFIG_FILE="azure_vm_config.json"
echo "Downloading configuration file $CONFIG_FILE..."
az storage blob download \
  --container-name "$CONTAINER_NAME" \
  --file "$CONFIG_FILE" \
  --name "$CONFIG_FILE" \
  --account-name "$STORAGE_ACCOUNT" \
  --auth-mode login

if [ $? -ne 0 ]; then
  echo "Error: Failed to download config file."
  exit 1
fi

mkdir -p logs models

echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
  echo "Error: Failed to install dependencies."
  exit 1
fi

chmod +x *.py

VM_INDEX=$((VM_ID - 1))
TRAINING_CONFIG=$(jq ".vms[$VM_INDEX].training" "$CONFIG_FILE")

if [ $? -ne 0 ] || [ -z "$TRAINING_CONFIG" ] || [ "$TRAINING_CONFIG" = "null" ]; then
  echo "Error: Failed to parse training configuration."
  exit 1
fi

TOTAL_TIMESTEPS=$(echo "$TRAINING_CONFIG" | jq -r ".total_timesteps")
LEARNING_RATE=$(echo "$TRAINING_CONFIG" | jq -r ".learning_rate")
BUFFER_SIZE=$(echo "$TRAINING_CONFIG" | jq -r ".buffer_size")
SEED=$(echo "$TRAINING_CONFIG" | jq -r ".seed")

if [ -z "$TOTAL_TIMESTEPS" ] || [ -z "$LEARNING_RATE" ] || [ -z "$BUFFER_SIZE" ] || [ -z "$SEED" ] || \
   [ "$TOTAL_TIMESTEPS" = "null" ] || [ "$LEARNING_RATE" = "null" ] || [ "$BUFFER_SIZE" = "null" ] || [ "$SEED" = "null" ]; then
  echo "Error: Incomplete training parameters."
  echo "Timesteps=$TOTAL_TIMESTEPS, LR=$LEARNING_RATE, BufferSize=$BUFFER_SIZE, Seed=$SEED"
  exit 1
fi

RESULTS_CONTAINER_NAME="$RESULT_CONTAINER"

echo "Starting training with VM_ID=$VM_ID (Index $VM_INDEX)"
echo "Parameters: Seed=$SEED, Timesteps=$TOTAL_TIMESTEPS, LR=$LEARNING_RATE, BufferSize=$BUFFER_SIZE, ResultsContainer=$RESULTS_CONTAINER_NAME"

source venv/bin/activate

python cloud_train.py \
  --vm-id "$VM_ID" \
  --seed "$SEED" \
  --timesteps "$TOTAL_TIMESTEPS" \
  --lr "$LEARNING_RATE" \
  --buffer-size "$BUFFER_SIZE" \
  --container "$RESULTS_CONTAINER_NAME"

if [ $? -ne 0 ]; then
  echo "Error: Training script exited with an error."
  exit 1
fi

echo "Training script finished successfully on VM $VM_ID."
