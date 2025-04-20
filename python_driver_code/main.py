
from stable_baselines3 import SAC
import gymnasium as gym

import torch
from torch import multiprocessing as tmp

import multiprocessing as mp
from multiprocessing.synchronize import Event as mpEventType
from tqdm import tqdm
import time

from blackjack_agent_package import Q_value_table_class as QVT
from episode_worker import Worker, load_worker
# from multiprocessing.
# from multiprocessing.managers import SyncManager

# Hyperparameters
EPISODE_CNT = 10 #Total Episodes per process
NUM_PROCESSES = 4 #Later dictate this to be dependent on GPU capbility, number of workers at a time processing each episode
EPOCH_CNT = 5
EPLISON = 0.1 #For eplison decay of each process/worker across all episode loops; 
EXPLORATION_SELECTION = None #Random number generation method for epsilon greedy policy


def update_nets(update_event: mpEventType, outputs_of_processes: dict[str : dict[int: tuple]], pnet= None, qnets=None):
    '''
    outputs_of_processes := dict contained as 
    #<process_id: end_results_of_episodes_of_that_process>
        # end_results_of_episodes_of_that_process : episode
            # <episode: SAR outputs ~(experience, reward), ...>
                # episode: (experience, reward)
    
    '''

    if not (pnet and qnets):
        print("No nets given!")
        raise 

    # Pause workers for update
    update_event.set()
    # In future: Scrap this nested dict crap for redis :/
    print("Performing update to networks:")
    for proc_out in tqdm(outputs_of_processes.values()):
        for episode in proc_out.values():
            experiences = episode[0]
            qnets.update(experiences)
            
    # Use Blackjack Q-net/shared_network class value obj's method to update here to simplify things

    update_event.clear()


def pass_nets_to_workers(update_event: mpEventType, new_nets, worker_processes):
    update_event.clear()
    pass
    # May not need if directly overwriting nets by reference

def setting_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
    else:
        device = torch.device('cpu')

    


def main():
    
    # Inits
    extinction_event = mp.Event() #Kill everything after program exits
    
    #Force any processes created to utilize spawn method within program
    tmp.set_start_method('spawn',force=True) 


    # Have manager for syncing updates to nets
    update_sync_manager = tmp.Manager()

    sim_barrier_completion = update_sync_manager.Barrier(parties=NUM_PROCESSES, ) 
    # processes_running_state = update_sync_manager.dict() #{<process_obj: curr_status>}
    processes = [] #Where The actual process objs
    
    #<process_id: end_results_of_episodes_of_that_process>
    # end_results_of_episodes_of_that_process : <episode: SAR outputs, ...>
    # episode: [exp1, exp2,...]
    processes_outputs = update_sync_manager.dict() 

    # <Epoche: [<process_id: end_results_of_episodes_of_that_process>]>
    playback_queue = {}
    
    update_completion = tmp.Event() #To allow processes to rerun after updating P&Q nets is FULLY completed
    # print(type(update_completion))
    # print(type(episode_barrier_completion))

    # Create shared Q-nets and Policy Nets 
        # Dependent on sim's requirements, regardless will be shared tensors;
        # To avoid consuming vram memory and reserve for sim execution:
        # consider storing main copy as cpu tensor then mov copies to workers' gpu tensors
    # Blackjack example:

    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1000)
    BJ_Q_net = QVT.QValueTable(env)


    # Initialize workers
    
    target_sim_func: function

    worker = Worker(
        target_sim_func= None,
        episode_count = EPISODE_CNT,
        update_event = update_completion,
        completion_barrier = sim_barrier_completion,
        extinction_event = extinction_event,
        shared_output_dump = processes_outputs

    )
    
    # Init Processes
    print("Initializing Workers")
    for i in tqdm(range(NUM_PROCESSES)):
        

        # Run worker's
        process = tmp.Process(
            target= load_worker,
            kwargs={
            "shared_q_nets": BJ_Q_net,
            "episode_count": EPISODE_CNT,
            "update_event": update_completion,
            "completion_barrier": sim_barrier_completion, 
            "extinction_event": extinction_event,
            "shared_output_dump": processes_outputs
        }

        )

        process.daemon = True #Make sure that child dies with the parent if being terminated
        processes.append(process)
        process.start()
        # processes_running_state.update({process.ident: worker.status})

    


    # Main Training Loop

    for epoch in range(EPOCH_CNT):

        # Workers will wait for each other to complete
        while sim_barrier_completion.parties != sim_barrier_completion.n_waiting:
            print(f"Workers done: {sim_barrier_completion.n_waiting} / {sim_barrier_completion.parties}")
            
            time.sleep(5)
        # <Epoche: [<process_id: end_results_of_episodes_of_that_process>]>
        playback_queue.update({epoch: processes_outputs})
        

        # Update nets
        update_nets(update_completion, processes_outputs, qnets = BJ_Q_net)
        sim_barrier_completion.reset()


    
    pass
    # Kill any left over processes
    extinction_event.set()
    process: tmp.Process
    for process in processes:
        process.join()
    
    # Export the model

    export = BJ_Q_net.save()
    if export != 0:
        print("Unable to save model")


if __name__ == "__main__":
    main()
