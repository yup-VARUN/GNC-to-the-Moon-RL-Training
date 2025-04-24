import os
import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

def upload_to_blob(local_file_path, blob_path, container_name):
    """Upload a file to Azure Blob Storage"""
    try:
        # Connect to Azure using default credentials
        credential = DefaultAzureCredential()
        
        # Get the connection string from environment variables
        account_name = os.environ.get("AZURE_STORAGE_ACCOUNT")
        if not account_name:
            raise ValueError("AZURE_STORAGE_ACCOUNT environment variable not set")
        
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=credential
        )
        
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Upload the file
        blob_client = container_client.get_blob_client(blob_path)
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded {local_file_path} to {blob_path} in container {container_name}")
        
    except Exception as e:
        print(f"Error uploading file: {str(e)}")

def train_sac(vm_id, seed, total_timesteps, learning_rate, buffer_size, 
              storage_container="rl-training-results"):
    """Train SAC model with the given parameters and upload results to Azure"""
    
    # Create unique run ID
    run_id = f"vm{vm_id}_seed{seed}_{int(time.time())}"
    
    # Configure logging and directories
    log_dir = os.path.join("logs", f"SAC_lunar_lander_{run_id}")
    model_dir = os.path.join("models", run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create and wrap the environment
    env = gym.make("LunarLander-v2", render_mode=None)
    env = Monitor(env, log_dir)
    
    # Create evaluation environment
    eval_env = gym.make("LunarLander-v2", render_mode=None)
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
    
    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix=f"sac_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create the SAC model with custom parameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed
    )
    
    # Train the model
    print(f"Starting training on VM {vm_id} with seed {seed}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "sac_lunar_lander_final")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save evaluation results
    eval_results = {
        "vm_id": vm_id,
        "seed": seed,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "timestamp": time.time()
    }
    
    results_file = os.path.join(model_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        import json
        json.dump(eval_results, f, indent=2)
    
    # Upload results to Azure Blob Storage
    upload_to_blob(
        results_file, 
        f"{run_id}/evaluation_results.json", 
        storage_container
    )
    
    # Upload final model
    upload_to_blob(
        f"{final_model_path}.zip", 
        f"{run_id}/sac_lunar_lander_final.zip", 
        storage_container
    )
    
    # Upload best model if it exists
    best_model_path = os.path.join(model_dir, "best", "best_model")
    if os.path.exists(f"{best_model_path}.zip"):
        upload_to_blob(
            f"{best_model_path}.zip", 
            f"{run_id}/best_model.zip", 
            storage_container
        )
    
    return mean_reward, run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC for LunarLander on Azure VM")
    parser.add_argument("--vm-id", type=int, required=True, help="VM identifier")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps to train")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--container", type=str, default="rl-training-results", 
                         help="Azure storage container for results")
    
    args = parser.parse_args()
    
    mean_reward, run_id = train_sac(
        vm_id=args.vm_id,
        seed=args.seed,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        storage_container=args.container
    )
    
    print(f"Training completed. Run ID: {run_id}, Mean reward: {mean_reward:.2f}")