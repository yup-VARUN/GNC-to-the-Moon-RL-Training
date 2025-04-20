import multiprocessing as mp
from multiprocessing import synchronize
import gymnasium as gym
from torch import multiprocessing as tmp
import time
from collections import deque
from functools import partial
from blackjack_agent_package import blackjack_agent_class as BJA
import os

class Worker:
    '''
    Object class that will store target sim, target sim's , events, outputs of sim
    '''
    def __init__(self, 
                 target_sim_func: partial,
                 episode_count: int, 
                 update_event: synchronize.Event,
                 completion_barrier: synchronize.Barrier, 
                 extinction_event: synchronize.Event,
                 shared_output_dump: dict
                 ):
        
        self.target_sim_func = target_sim_func #how to call/store? Partial function?
        self.episode_count = episode_count

        self.update_event = update_event
        self.completion_barrier = completion_barrier
        self.extinction_event = extinction_event

        self.status = 'New' #Have different states to control worker perhaps?
        self.shared_output_dump = shared_output_dump #Where process will dump it's output after running episodes
        # Store the each episode output's
        self.outputs = {}

    # Make this is a Pipe obj for active monitoring in the future
    # def current_status(self):
    #     return self.status


    def sim_execution(self, shared_policy_net = None, shared_q_nets = None):
        '''
        Pass these onto the retained sim function
        shared_policy_net := Policy network
        shared_q_nets := List of q-nets
        '''

        if not (shared_policy_net and shared_q_nets):
            print("Missing nets for updating")
            raise 
        # Blackjack example:
        env = gym.make("Blackjack-v1", sab=False)
        
        blackjack_agent = BJA.BlackjackAgent(env, shared_q_nets)
        
        # Worker Loop
        while not self.extinction_event.is_set():
            # Future improvements: Room for utilizing self.status for optimizations and readability/monitoring/debugging
                # New: build new env for sim
                # Ready: after P&Q net update and env has been reset (can avoid fully recreating RL env each time!)
                # Running: actively running sim/an episode
                # Ended: output episode and their respective results
                # Waiting: P&Q nets are in process of being updated and env is being cleaned up
                # Terminated: terminate worker after program ends/extinction_event is set
            self.status = 'Running'
            # self.target_sim_func
            # Execute sim/episode
            # for episode in range(self.episode_count):

                # run an episode
                    # Get SAR and store as an experience
                # Get episode's SAR results and store inside worker


            for episode in range(self.episode_count):
                # Run Episode
                experience, reward = blackjack_agent.run_episode()
                
                self.outputs.update({episode: (experience, reward)})
            pass
        
            
            # Exit sim/episode
            # dump class obj onto Shared dictionary
            self.shared_output_dump.update({f"Worker {os.getpid()}": self.outputs})

            self.status = 'Waiting'
            self.completion_barrier.wait()
            
            time.wait(0.5) #Wait a bit for the update event to trigger properly
            
            # wait while updating, should also wait until end of epoch
            self.update_event.wait()

                

    def sim_env_cleanup(self,):
        '''
        Clean up environment and restart the sim; might have to not worry about if abstracted else where
        '''
        pass


def load_worker(
        target_sim_func,
        EPISODE_CNT,
        update_completion,
        sim_barrier_completion,
        extinction_event,
        processes_outputs,
        shared_q_nets):
    '''
    Params:
    '''
    worker = Worker(
        target_sim_func= None,
        episode_count = EPISODE_CNT,
        update_event = update_completion,
        completion_barrier = sim_barrier_completion,
        extinction_event = extinction_event,
        shared_output_dump = processes_outputs

        )
    
    worker.sim_execution(shared_q_nets=shared_q_nets)