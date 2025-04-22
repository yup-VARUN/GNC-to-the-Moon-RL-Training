import multiprocessing as mp
from multiprocessing import synchronize

from torch import multiprocessing as tmp
from collections import deque
from functools import partial

import time
import os
from tqdm import tqdm

import gymnasium as gym
from SAC_agent_package.prime_model import prime_SAC_model

class Worker:
    '''
    Object class that will store target sim, target sim's , events, outputs of sim
    '''
    def __init__(self, 
                 episode_count: int, 
                 update_event: synchronize.Event,
                 completion_barrier: synchronize.Barrier, 
                 extinction_event: synchronize.Event,
                 shared_output_dump: dict,
                 target_sim_func: partial = None,
                 target_sim_func_params: dict = {}
                 ):
        
        self.target_sim_func = target_sim_func #how to call/store? Partial function?
        # self.target_sim_func_params = target_sim_func_params
        self.episode_count = episode_count

        self.update_event = update_event
        self.completion_barrier = completion_barrier
        self.extinction_event = extinction_event

        self.status = 'New' #Have different states to control worker perhaps?
        # Store the each episode output's
        self.outputs = {}
        self.shared_output_dump = shared_output_dump #Where process will dump it's output after running episodes

    # Make this is a Pipe obj for active monitoring in the future
    # def current_status(self):
    #     return self.status


    def sim_execution(self, shared_nets:dict, shared_policy_net = None, shared_q_nets = None):
        '''
        Pass these onto the retained sim function
        

        shared_policy_net := Policy network
        shared_q_nets := List of q-nets
        '''

        if not (shared_policy_net or shared_q_nets or shared_nets):
            print("Missing nets for updating")
            raise 
        
        SAC_agent:prime_SAC_model = self.target_sim_func()


        
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
            # print(f"At runtime Worker {os.getpid()} has shared q-nets: ",shared_q_nets)
            # Debug print: Check if q-net is growing/properly being referenced
            # print("Learning with Q-net key size",len(shared_q_nets.q_dict.keys()))
            # for episode in tqdm(range(self.episode_count)):
            for episode in range(self.episode_count):
                # Run Episode
                experience, reward = SAC_agent.run_episode()
                
                self.outputs.update({episode: (experience, reward)})
            pass
        
            
            # Exit sim/episode
            # dump class obj onto Shared dictionary
            self.shared_output_dump.update({f"Worker {os.getpid()}": self.outputs})

            self.status = 'Waiting'
            # print(f"Worker {os.getpid()} now {self.status}")
            # print("From Worker",self.completion_barrier)
            self.completion_barrier.wait()
            
            # if self.completion_barrier == 0:
            #     print(f"Worker {os.getpid()}",self.completion_barrier.n_waiting)
            # time.sleep(0.5) #Wait a bit for the update event to trigger properly
            # while not self.update_event.is_set():
            #     print("worker Waiting on update", self.update_event.is_set())
            #     time.sleep(10)
            
            # wait while updating, should also wait until end of epoch
            self.update_event.wait()
            
            # Refetch q_table after updating
            # print(f"After updating Worker {os.getpid()} has q-net: ",shared_q_nets[0])
            SAC_agent.fetch_networks(source_policy= shared_nets["Policy"], source_critic= shared_nets["Critic"], source_critic_target=shared_nets["Critic_Target"])

                

    def sim_env_cleanup(self,):
        '''
        Clean up environment and restart the sim; might have to not worry about if abstracted else where
        '''
        pass


def load_worker(
        episode_count,
        update_event,
        completion_barrier,
        extinction_event,
        shared_output_dump,
        target_sim_func,
        shared_nets,
        shared_q_nets= None,
        shared_policy_net =None,
        ):
    '''
    A function that can load a worker instance
    Params:
    '''
    # print(shared_q_nets)
    worker = Worker(
        episode_count = episode_count,
        update_event = update_event,
        completion_barrier = completion_barrier,
        extinction_event = extinction_event,
        shared_output_dump = shared_output_dump,
        target_sim_func = target_sim_func,
        )
    
    worker.sim_execution(shared_nets=shared_nets,shared_policy_net = shared_policy_net, shared_q_nets=shared_q_nets)