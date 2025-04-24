import multiprocessing as mp
from multiprocessing import synchronize
from torch import multiprocessing as tmp

from collections import deque
from functools import partial
from custom_utils import log_dump_to_json

import time
# import pandas as pd #Will be used to track a worker's metrics
import os
import sys
from tqdm import tqdm
import psutil

import gymnasium as gym
from SAC_agent_package.prime_model import prime_SAC_model

# OS_DIR_SEPARATOR = os.sep
LOG_DIR = "worker_logs" #where workers log their work upon execution

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
                 parent_process_id: int,
                 target_sim_func: partial = None,
                 target_sim_func_params: dict = {},

                 log_usage = False #Writes to a pandas DF for tracking RAM and episode completion
                 ):
        
        self.target_sim_func = target_sim_func #how to call/store? Partial function?
        # self.target_sim_func_params = target_sim_func_params
        
        

        self.episode_count = episode_count
        self.parent_process_id = parent_process_id
        self.update_event = update_event
        self.completion_barrier = completion_barrier
        self.extinction_event = extinction_event

        self.status = 'New' #Have different states to control worker perhaps?
        # Store the each episode output's
        self.outputs = {}
        self.log_usage = log_usage
        self.shared_output_dump = shared_output_dump #Where process will dump it's output after running episodes

    # Make this is a Pipe obj for active monitoring in the future
    # def current_status(self):
    #     return self.status
    def run(self, shared_nets:dict, shared_policy_net = None, shared_q_nets = None):

        args = {
            "shared_nets": shared_nets,
            "shared_policy_net" : shared_policy_net,
            "shared_q_nets" : shared_q_nets
        }

        if self.log_usage:
            print("Running with log")
            return self.sim_execution_logged(**args)

        else:
            return self.sim_execution(**args)


    def sim_execution_logged(self, shared_nets:dict, shared_policy_net = None, shared_q_nets = None):
        '''
        Same as running a regular sim, just gathering metrics for a worker
        In future, have a separate thread for active logging to an open file
        
        Pass these onto the retained sim function
        
        shared_nets := shared dict where networks are stored as <"network_type": network/network values obj> Make sure obj is picklable
        shared_policy_net := Policy network
        shared_q_nets := List of q-nets
        '''
        
        # Inits for metrics
        '''
        # Have a dict for tracking metrics of a worker, will be saved to a json file at the end of execution
        log schema:
            total_execution_time : float
            initial_mem_usage : in MB
            episodes_per_epoch : int
            Epoches:
                Epoch:
                    total_compute_time_for_episodes_cnt/spent_in_curr_epoch: float
                    total_num_SAR_collected: int
                    time_spent_waiting_on_update: float
                    
                    episodes:
                        [Episode:
                                Memory_usage_at_episode
                            ...
                        Episode_n]
                ...
                Epoch:
                    ...
        '''
        self.log = {}
        self.log["Epoches"] = {}

        #For tracking what epoch worker is on, ONLY for LOGGING Purposes
        internal_epoch = 0 
        curr_process = psutil.Process(os.getpid())

        if not (shared_policy_net or shared_q_nets or shared_nets):
            print("Missing nets for updating")
            raise 
        
        SAC_agent: prime_SAC_model = self.target_sim_func()
        
        init_mem_usage = curr_process.memory_info().rss / (1024 * 1024)
        self.log["init_mem_usage"] = init_mem_usage
        self.log["episodes_per_epoch"] = self.episode_count

        total_execution_time_start = time.perf_counter()
        # Worker Loop
        while not self.extinction_event.is_set():
            # Future improvements: Room for utilizing self.status for optimizations and readability/monitoring/debugging
                # New: build new env for sim
                # Ready: after P&Q net update and env has been reset (can avoid fully recreating RL env each time!)
                # Running: actively running sim/an episode
                # Ended: output episode and their respective results
                # Waiting: P&Q nets are in process of being updated and env is being cleaned up
                # Terminated: terminate worker after program ends/extinction_event is set
            if self.extinction_event.is_set():
                break
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
            total_SAR_collection = 0
            episode_batch_timer_start = time.perf_counter()
            self.log["Epoches"][internal_epoch] = {}
            self.log["Epoches"][internal_epoch]["Episodes"] = {}

            # for episode in tqdm(range(self.episode_count)):
            for episode in range(self.episode_count):
                # Run Episode and collect SARs/Experience
                experiences, reward = SAC_agent.run_episode()

                # Track memory usage when collecting SAR by episode
                run_time_usage__mem = (curr_process.memory_info().rss - sys.getsizeof(self.log)) / (1024 * 1024)  # in MB and subtracting size of the log itself
                self.log["Epoches"][internal_epoch]["Episodes"][episode] = run_time_usage__mem
                # print(f"[PID {os.getpid()}] Memory usage: {mem:.2f} MB")
                # Add episode's SAR/experience to worker's output
                self.outputs.update({episode: (experiences, reward)})
                
                # Count SARs
                total_SAR_collection += len(experiences)
            # Exit sim/episode
            # dump class obj onto Shared dictionary
            self.shared_output_dump.update({f"Worker {os.getpid()}": self.outputs})
            
            episode_batch_timer_end = time.perf_counter()

            time_to_complete_epis_batch = episode_batch_timer_end - episode_batch_timer_start
            self.log["Epoches"][internal_epoch]["total_compute_time_for_episodes_cnt"] = time_to_complete_epis_batch
            # Episode metrics
                # SAR collection Count from batch of episode
            
                # Average time spent processing episodes; nvm post process this 
            # average_p_episode_in_epoch = time_to_complete_epis_batch/self.episode_count

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
            print("Worker waiting for update")
            time_spent_waiting_update_start = time.perf_counter()
            print(self.extinction_event.is_set())
            
            self.update_event.wait()
            time_spent_waiting_update_end = time.perf_counter()
            time_spent_waiting_update = time_spent_waiting_update_end - time_spent_waiting_update_start
            
            # Metric Collection
            self.log["Epoches"][internal_epoch]["total_num_SAR_collected"] = total_SAR_collection
            self.log["Epoches"][internal_epoch]["time_spent_waiting_on_update"] = time_spent_waiting_update
            
            if not self.extinction_event.is_set():
                # Refetch q_table after updating
                if "log_ent_coef" in shared_nets.keys() and hasattr(SAC_agent, "log_ent_coef"):
                    SAC_agent.fetch_networks(source_policy= shared_nets["Policy"], source_critic= shared_nets["Critic"], source_critic_target=shared_nets["Critic_Target"], log_ent_coef= shared_nets["log_ent_coef"])
                else:
                    SAC_agent.fetch_networks(source_policy= shared_nets["Policy"], source_critic= shared_nets["Critic"], source_critic_target=shared_nets["Critic_Target"])
            
            
            internal_epoch += 1

        total_execution_time_end = time.perf_counter()
        total_execution_time = total_execution_time_end - total_execution_time_start
        
        self.log["total_execution_time"] = total_execution_time
        # Write log to a csv file
        os.makedirs(os.path.join(LOG_DIR, f"Process_{self.parent_process_id}_job"), exist_ok=True)
        file_out_dir = os.path.join(LOG_DIR, f"Process_{self.parent_process_id}_job", f"Worker_{os.getpid()}_log") #Do not put .json here, that's done in the help func
        
        log_dump_to_json(file_out_dir, self.log)

        print(f"Worker {os.getpid()} ending")
        
    def sim_execution(self, shared_nets:dict, shared_policy_net = None, shared_q_nets = None):
        '''
        Pass these onto the retained sim function
        
        shared_nets := shared dict where networks are stored as <"network_type": network/network values obj> Make sure obj is picklable
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
                
                # Add episode's SAR/experience to worker's output
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
            if not self.extinction_event.is_set():
                if "log_ent_coef" in shared_nets.keys() and hasattr(SAC_agent, "log_ent_coef"):
                    SAC_agent.fetch_networks(source_policy= shared_nets["Policy"], source_critic= shared_nets["Critic"], source_critic_target=shared_nets["Critic_Target"], log_ent_coef= shared_nets["log_ent_coef"])
                else:
                    SAC_agent.fetch_networks(source_policy= shared_nets["Policy"], source_critic= shared_nets["Critic"], source_critic_target=shared_nets["Critic_Target"])

                

    def sim_env_cleanup(self,):
        '''
        Clean up environment and restart the sim; might have to not worry about if abstracted else where
        '''
        pass


def load_worker(
        parent_process_id,
        episode_count,
        update_event,
        completion_barrier,
        extinction_event,
        shared_output_dump,
        target_sim_func,
        shared_nets,
        shared_q_nets= None,
        shared_policy_net =None,
        log_usage = False
        ):
    '''
    A function that can load a worker instance; 
    essentially bootstrap function to load the worker
    Params:
    '''
    # worker_args = {
    #     parent_process_id,
    #     episode_count,
    #     update_event,
    #     completion_barrier,
    #     extinction_event,
    #     shared_output_dump,
    #     target_sim_func,
    # }
    # print(shared_q_nets)
    worker = Worker(
        episode_count = episode_count,
        update_event = update_event,
        completion_barrier = completion_barrier,
        extinction_event = extinction_event,
        shared_output_dump = shared_output_dump,
        target_sim_func = target_sim_func,
        parent_process_id= parent_process_id,
        log_usage = log_usage
        )
    
    worker.run(shared_nets=shared_nets,shared_policy_net = shared_policy_net, shared_q_nets=shared_q_nets)