
from stable_baselines3 import SAC
# import gymnasium

import torch
from torch import multiprocessing as tmp

import multiprocessing as mp
from multiprocessing.synchronize import Event as mpEventType
from episode_worker import Worker
# from multiprocessing.
# from multiprocessing.managers import SyncManager

# Hyperparameters
EPISODE_CNT = 10 #Total Episodes per process
NUM_PROCESSES = 4 #Later dictate this to be dependent on GPU capbility, number of workers at a time processing each episode
EPOCH_CNT = 5

def update_nets(update_event: mpEventType, outputs_of_processes: dict, pnet, qnets,):

    # Pause workers for update
    update_event.set()

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
    
    extinction_event = mp.Event() #Kill everything after program exits
    
    
    #Force any processes created to utilize spawn method within program
    tmp.set_start_method('spawn',force=True) 

    # Have manager for syncing updates to nets
    update_sync_manager = tmp.Manager()

    sim_barrier_completion = update_sync_manager.Barrier(parties=NUM_PROCESSES, ) 
    processes_running_state = update_sync_manager.dict() #{<process_obj: curr_status>}
    processes = [] #The actual process objs
    
    #<process_id: end_results_of_episodes_of_that_process>
    # end_results_of_episodes_of_that_process : <episode: output, ...>
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



    # Initialize workers
    
    target_sim_func: function
    for i in range(NUM_PROCESSES):
        
        worker = Worker(
            target_sim_func,
            EPISODE_CNT,
            update_completion,
            sim_barrier_completion,
            extinction_event,
            processes_outputs

        )

        # Run worker's
        process = tmp.Process(
            # target=
            # shared_nets
        )
        processes.append((worker, process))
        processes_running_state.update({process.ident: worker.status})



    # Main Training Loop
    for epoch in range(EPOCH_CNT):
        
        
        


    #     pass
        # Workers will wait for each other to complete
        

        playback_queue.update({epoch: processes_outputs})
        # Update nets
        update_nets(update_completion, )
        sim_barrier_completion.reset()


    
    pass
    # Kill any left over processes
    extinction_event.set()


if __name__ == "__main__":
    main()
