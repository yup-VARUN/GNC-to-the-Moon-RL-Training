import torch
import torch.multiprocessing as mp
from stable_baselines3 import SAC
import gymnasium as gym

def setup_shared_model():
    # Create a model with continuous action space
    env = gym.make("Pendulum-v1")
    model = SAC("MlpPolicy", env)
    
    # Move model to shared memory
    model.policy.share_memory()
    model.critic.share_memory()
    
    return model, env

def worker_process(rank, shared_model):
    # Each worker uses the shared model (policy and Q networks)
    # But has its own copy of the environment
    env = gym.make("Pendulum-v1")
    
    # Run episodes using the shared model
    for episode in range(5):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use shared model for prediction
            action, _ = shared_model.predict(obs, deterministic=False)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            obs = next_obs
            
        print(f"Worker {rank}, Episode {episode}, Reward: {total_reward}")

if __name__ == "__main__":
    # Setup multiprocessing method (important for CUDA)
    mp.set_start_method('spawn')
    
    # Create shared model
    shared_model, _ = setup_shared_model()
    
    # Create and start worker processes
    processes = []
    num_processes = 4
    
    for rank in range(num_processes):
        p = mp.Process(target=worker_process, args=(rank, shared_model))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()