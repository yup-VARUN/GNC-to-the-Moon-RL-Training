#!/bin/bash

# CLI parameters
VM_ID=$1
STORAGE_ACCOUNT=$2
CONTAINER_NAME=$3
RESULT_CONTAINER=$4

echo "Setting up VM $VM_ID with storage account $STORAGE_ACCOUNT"

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Create a directory for the project
mkdir -p /home/Varun/rl-project
cd /home/Varun/rl-project

# Environment variables
echo "export AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT" >> ~/.bashrc
source ~/.bashrc

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Azure CLI
pip install azure-cli

# Login with managed identity
az login --identity

# Download project files from blob storage
az storage blob download-batch --source $CONTAINER_NAME --destination . --account-name $STORAGE_ACCOUNT

# Create directories for logs and models
mkdir -p logs models

# Install dependencies
pip install -r requirements.txt

# Set execution permissions
chmod +x *.py

# Start training with VM-specific parameters
# Use a different seed for each VM
SEED=$((1000 + $VM_ID))

echo "Starting training with VM_ID=$VM_ID and SEED=$SEED"

python cloud_train.py --vm-id $VM_ID --seed $SEED --timesteps 500000 