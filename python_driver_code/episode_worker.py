import multiprocessing as mp
from multiprocessing import synchronize

from torch import multiprocessing as tmp
import time
from collections import deque
from functools import partial


class Worker:
    '''
    Object class that will store target sim, target sim's , events, outputs of sim
    '''
    def __init__(self, 
                 target_sim_func,
                 episode_count, 
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

        self.status = 'new' #Have different states to control worker perhaps?
        self.shared_output_dump = shared_output_dump #Where process will dump it's output after running episodes
        # Store the each episode output's
        self.outputs = {}

    def current_status(self):
        return self.status


    def sim_execution(self, shared_policy_net, shared_q_nets):
        '''
        Pass these onto the retained sim function
        shared_policy_net := 
        shared_q_nets :=
        '''
        while not self.extinction_event.is_set():
            # Future improvements: Room for utilizing self.status for optimizations and readability/monitoring/debugging
                # New: build new env for sim
                # Ready: after P&Q net update and env has been reset (can avoid fully recreating RL env each time!)
                # Running: actively running sim/an episode
                # Ended: output episode and their respective results
                # Waiting: P&Q nets are in process of being updated and env is being cleaned up
                # Terminated: terminate worker after program ends/extinction_event is set
            self.status = 'Running'
            # Execute sim/episode
            for episode in range(self.episode_count):
                # run an episode

                # Get episode's results and store inside worker

                self.outputs.update({episode: output})
                pass

            self.status = 'Waiting'
            # Exit sim/episode
            self.shared_output_dump.update({})
            self.completion_barrier.wait()
            
            time.wait(0.5) #Wait a bit for the update event to trigger properly
            # wait while updating 
            self.update_event.wait()
                

    def sim_env_cleanup(self,):
        '''
        Clean up environment and restart the sim
        '''
        pass


