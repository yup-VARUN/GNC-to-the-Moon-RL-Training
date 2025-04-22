'''
Stores prime SAC model in main program to hold and overwrite worker models

'''
import torch
import torch.multiprocessing as mp
from stable_baselines3 import SAC

class prime_SAC_model(SAC):
    def __init__(self, 
                 policy, 
                 env, 
                 learning_rate = 0.0003, 
                 buffer_size = 1000000, 
                 learning_starts = 100, 
                 batch_size = 256, 
                 tau = 0.005, 
                 gamma = 0.99, 
                 train_freq = 1, 
                 gradient_steps = 1, 
                 action_noise = None, 
                 replay_buffer_class = None, 
                 replay_buffer_kwargs = None, 
                 optimize_memory_usage = False, 
                 ent_coef = "auto", 
                 target_update_interval = 1, 
                 target_entropy = "auto", 
                 use_sde = False, 
                 sde_sample_freq = -1, 
                 use_sde_at_warmup = False, 
                 stats_window_size = 100, 
                 tensorboard_log = None, 
                 policy_kwargs = None, 
                 verbose = 0, 
                 seed = None, 
                 device = "auto", 
                 _init_setup_model = True):
        
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, ent_coef, target_update_interval, target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)

    def share_model_across_memory(self):
        '''
        Ensures that model's networks is being shared across memory
        '''
        self.policy.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()
        if hasattr(self, "log_ent_coef") and self.log_ent_coef is not None:
            self.log_ent_coef.share_memory_()

    def run_episode(self, is_determinsitic = False):
        experiences = []
        # rewards = []
        obs = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            # Use shared model for prediction
            action, _ = self.predict(obs, deterministic = is_determinsitic)
            next_obs, reward, done, info = self.env.step(action)
            experiences.append((obs, next_obs, action, reward, done, info))
            total_reward += reward
            obs = next_obs

        return experiences, total_reward
    
    def push_replay_buff(self, experience: tuple):
        '''
        Pushes a single experience to the replay buff
        
        '''
        obs, next_obs, action, reward, done, info = experience
        self.replay_buffer.add(obs, next_obs, action, reward, done, info)

    def mass_push_to_replay_buff(self, experiences: list[tuple]):
        '''
        Accepts a list of experiences to push to replay buff; often from a single episode
        
        '''
        for experience in experiences:
            self.push_replay_buff(experience=experience)

            
    def fetch_networks(self, source_policy, source_critic, source_critic_target, log_ent_coef = None ):
        '''
        For workers to get copies of source model after done updating
        '''

        self.policy.load_state_dict(source_policy)
        self.critic.load_state_dict(source_critic)
        self.critic_target.load_state_dict(source_critic_target)
        # If using automatic entropy tuning, transfer log_ent_coef
        if log_ent_coef and hasattr(self, "log_ent_coef"):
            self.log_ent_coef.data = log_ent_coef

        

