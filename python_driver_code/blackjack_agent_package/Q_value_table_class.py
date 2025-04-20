import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import torch.multiprocessing as mp

from torch.multiprocessing import Manager

# Create a SINGLE shared manager at the module level
global_manager = None

def get_global_manager():
    global global_manager
    if global_manager is None:
        global_manager = Manager()  # Using torch's manager
    return global_manager

class QValueTable:
    """
    Q-value table implemented with PyTorch tensors for shared memory access.
    Manages the state-action values and provides update functionality.
    """
    def __init__(self, action_size,learning_rate=0.01, gamma=0.99):
        # self.env = env
        self.alpha = learning_rate
        self.gamma = gamma
        
        # Initialize Q-table dictionary that maps state tuples to tensors
        # For blackjack: (player_sum, dealer_card, usable_ace) -> actions tensor
        # self.q_dict = {}

        # Create manager to handle shared dictionary
        # self.manager = get_global_manager()
        # self.q_dict = self.manager.dict()
        self.q_dict = {}
        
        # Action space size
        self.action_size = action_size

        # Add pickling methods
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove the manager as it can't be pickled
    #     state['manager'] = None
    #     # Keep the q_dict as it's a managed dict that can be shared
    #     return state
    
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # Use the global manager
    #     self.manager = get_global_manager()
    
    def get_q_value(self, state):
        """Get Q-values for a given state"""
        state_tuple = tuple(state)
        if state_tuple not in self.q_dict:
            # Initialize with zeros if state not seen before
            tensor = torch.zeros(self.action_size, requires_grad=False)
            shared_tensor = tensor.share_memory_()
            self.q_dict[state_tuple] = shared_tensor
        return self.q_dict[state_tuple]
    
    def get_max_q(self, state):
        """Get maximum Q-value for a state"""
        return torch.max(self.get_q_value(state)).item()
    
    def get_best_action(self, state):
        """Get action with highest Q-value"""
        return torch.argmax(self.get_q_value(state)).item()
    
    def update(self, experiences):
        """
        Update Q-values based on collected experiences
        experiences: list of (state, action, reward, next_state, done) tuples
        """
        for state, action, reward, next_state, done in experiences:
            # Convert state to tuple for dictionary lookup
            state_tuple = tuple(state)
            next_state_tuple = tuple(next_state)
            
            # Get current Q value
            if state_tuple not in self.q_dict:
                self.q_dict[state_tuple] = torch.zeros(self.action_size, requires_grad=False)
            
            # Get next state's max Q value
            if done:
                target = reward
            else:
                if next_state_tuple not in self.q_dict:
                    self.q_dict[next_state_tuple] = torch.zeros(self.action_size, requires_grad=False)
                target = reward + self.gamma * torch.max(self.q_dict[next_state_tuple]).item()
            
            # Update Q value for (state, action)
            self.q_dict[state_tuple][action] = (1 - self.alpha) * self.q_dict[state_tuple][action] + \
                                               self.alpha * target
    
    def save(self, filename="q_values.pt"):
        """Save Q-values to file"""
        try:
            # Convert dictionary of tensors to serializable format
            serializable_dict = {state: values.tolist() for state, values in self.q_dict.items()}
            torch.save(serializable_dict, filename)
            return 0 #Successfully
        except Exception as e:
            print(e)
            return -1
    

        
        
    def load(self, filename="q_values.pt"):
        """Load Q-values from file"""
        loaded_dict = torch.load(filename)
        self.q_dict = {state: torch.tensor(values, requires_grad=False) 
                       for state, values in loaded_dict.items()}