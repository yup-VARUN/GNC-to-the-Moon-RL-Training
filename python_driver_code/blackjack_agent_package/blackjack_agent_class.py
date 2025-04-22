import random
from .Q_value_table_class import QValueTable
class BlackjackAgent:
    """Agent that interacts with the Blackjack environment"""
    def __init__(self, env, q_values, epsilon=0.1, exploration_function = random.random):
        self.env = env
        self.q_values = q_values  # Shared Q-value object
        self.epsilon = epsilon    # Exploration rate
        # Set the distribution to determination the exploration 
        self.exploration_function =  exploration_function
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        Returns selected action
        """
        if training and self.exploration_function() < self.epsilon:
            # Exploration: random action
            return self.env.action_space.sample()
        else:
            # Exploitation: best action according to Q-values
            return self.q_values.get_best_action(state)
    
    def collect_experience(self, num_episodes=1):
        """
        Collect experience by running multiple episodes
        Returns list of experiences and total rewards
        """
        all_experiences = []
        episode_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_experience = []
            total_reward = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                episode_experience.append((state, action, reward, next_state, done))
                
                state = next_state
                total_reward += reward
            
            all_experiences.extend(episode_experience)
            episode_rewards.append(total_reward)
        
        return all_experiences, episode_rewards
    
    def run_episode(self):
        # Runs a single episode
        state, _ = self.env.reset()
        done = False
        episode_experience = []
        total_reward = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            episode_experience.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
        
        return episode_experience, total_reward

    def set_epsilon(self, epsilon):
        """Update exploration rate"""
        self.epsilon = epsilon

    
    def evaluate(self, num_episodes=1000):
        """Evaluate agent performance"""
        rewards = []
        wins, draws, losses = 0, 0, 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.select_action(state, training=False)  # No exploration during evaluation
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
            
            # Track outcome
            if total_reward > 0:
                wins += 1
            elif total_reward == 0:
                draws += 1
            else:
                losses += 1
        
        avg_reward = sum(rewards) / len(rewards)
        win_rate = wins / num_episodes
        draw_rate = draws / num_episodes
        loss_rate = losses / num_episodes
        
        return {
            'avg_reward': avg_reward,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate
        }