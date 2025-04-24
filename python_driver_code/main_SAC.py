
from stable_baselines3 import SAC
from stable_baselines3.common.logger import Logger
import gymnasium as gym

import torch
from torch import multiprocessing as tmp
import multiprocessing as mp
from multiprocessing.synchronize import Event as mpEventType

from tqdm import tqdm
import time
import os
from functools import partial

from custom_utils import prepare_agent


from episode_worker_SAC import load_worker
from SAC_agent_package.prime_model import prime_SAC_model

# Hyperparameters
EPISODE_CNT = 10 #Total Episodes per process
NUM_PROCESSES = 2 #Later dictate this to be dependent on GPU capbility, number of workers at a time processing each episode
EPOCH_CNT = 3
EPLISON = 0.1 #For eplison decay of each process/worker across all episode loops; 
EXPLORATION_SELECTION = None #Random number generation method for epsilon greedy policy, must be 0-1
MODEL_OUTPUT_DIR = "output_models"


def update_nets(update_event: mpEventType, outputs_of_processes: dict[str : dict[int: tuple]], model: prime_SAC_model, shared_nets: dict):
    '''
    outputs_of_processes := dict contained as 
        <process_id: end_results_of_episodes_of_that_process>
            end_results_of_episodes_of_that_process : episode
                <episode: SAR outputs ~(experience, reward), ...>
                    episode: (experience, reward)
    
    '''

    # if not (pnet or qnets):
    #     print("No nets given!")
    #     raise 
 
    # In future: maybe Scrap this nested dict crap for redis :/
    if not hasattr(model, "_logger"):
        
        model._logger = Logger(folder=None, output_formats=[])
    print("Performing update to networks:")
    # Extract from output each worker
    for proc_out in tqdm(outputs_of_processes.values()):
        # Extract by episode
        for episode in proc_out.values():
            experiences = episode[0]
            model.mass_push_to_replay_buff(experiences)
            # May have to tune this or implement env- step by episode loop
            # Current issue: 'prime_SAC_model' object has no attribute '_logger', 
            # the prime model may not update effectively without it
            # Future plan: pull log from workers and port it to prime model to optimize update with lr schedule

            model.train(gradient_steps=1)
    
    # print("Qnets Objs after updating", qnets)
    shared_nets["Policy"] = model.policy.state_dict()

    # Critic/Q nets
    shared_nets["Critic"] = model.critic.state_dict()
    shared_nets["Critic_Target"] = model.critic_target.state_dict()
    if "log_ent_coef" in shared_nets.keys() and hasattr(model, "log_ent_coef"):
        shared_nets["log_ent_coef"] = model.log_ent_coef.data
    # Set event to True and allow workers to run again
    update_event.set()
    print("Update status:", update_event.is_set())
    
    

# Deappreciated
# def pass_nets_to_workers(update_event: mpEventType, new_nets, worker_processes):
#     update_event.clear()
#     pass
#     # May not need if directly overwriting nets by reference

def setting_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        return device, gpu_count
    else:
        device = torch.device('cpu')
        return device
    
def celebrate():
    print("All workers reached barrier!")


def main():
    
    # Inits
    
    extinction_event = mp.Event() #Kill everything after program exits
    
    print("Device detected:", setting_device())
    input("Hit enter to proceed")
    #Force any processes created to utilize spawn method within program
    tmp.set_start_method('spawn',force=True) 

    # global_manager = QVT.get_global_manager()

    # Have manager for syncing updates to nets
    update_sync_manager = tmp.Manager()

    sim_barrier_completion = update_sync_manager.Barrier(parties=NUM_PROCESSES+1, action=celebrate ) 
    # processes_running_state = update_sync_manager.dict() #{<process_obj: curr_status>}
    processes : list[tmp.Process]
    processes = [] #Where The actual process objs
    
    #<process_id: end_results_of_episodes_of_that_process>
    # end_results_of_episodes_of_that_process : <episode: SAR outputs, ...>
    # episode: [exp1, exp2,...]
    processes_outputs = update_sync_manager.dict() 

    # <Epoche: [<process_id: end_results_of_episodes_of_that_process>]>
    playback_queue = {}
    
    update_completion = tmp.Event() #To allow processes to rerun after updating P&Q nets is FULLY completed

    # Create shared Q-nets and Policy Nets 
        # Dependent on sim's requirements, regardless will be shared tensors;
        # To avoid consuming vram memory and reserve for sim execution:
        # consider storing main copy as cpu tensor then mov copies to workers' gpu tensors
    env = gym.make("Pendulum-v1")  # This has a continuous action space
    
    target_sim_func: partial
    # Add Model's 
    target_sim_func_params = {
        'policy' : "MlpPolicy",
        'env' : env,
        'verbose': 1
    }

    env = gym.make("Pendulum-v1")
    target_sim_func = prepare_agent(prime_SAC_model, target_sim_func_params)
    prime_model: prime_SAC_model = target_sim_func()
    prime_model.share_model_across_memory()
    shared_networks_dict = update_sync_manager.dict()
    # Policy
    shared_networks_dict["Policy"] = prime_model.policy.state_dict()

    # Critic/Q nets
    shared_networks_dict["Critic"] = prime_model.critic.state_dict()
    shared_networks_dict["Critic_Target"] = prime_model.critic_target.state_dict()
    
    # log_ent_coef
    if hasattr(prime_model, "log_ent_coef"):
        shared_networks_dict["log_ent_coef"] = prime_model.log_ent_coef.data
    
    # Pass parameters to target sim function
    
    
    # Initialize workers
    # Init Processes
    print("Initializing Workers")
    for i in tqdm(range(NUM_PROCESSES)):
        

        # Run worker's
            # Note: in future replace kwargs with a env file or web interface for interaction
        process = tmp.Process(
            target= load_worker,
            kwargs={
            "parent_process_id": os.getpid(),
            "shared_nets": shared_networks_dict,
            "episode_count": EPISODE_CNT,
            "update_event": update_completion,
            "completion_barrier": sim_barrier_completion, 
            "extinction_event": extinction_event,
            "shared_output_dump": processes_outputs,
            "target_sim_func" : target_sim_func,
            'log_usage' : True
        }

        )
        try:
            process.daemon = True #Make sure that child dies with the parent if being terminated
            processes.append(process)
            time.sleep(5)
            process.start()
        except Exception as exception:
            print(exception)
            exit()
        # processes_running_state.update({process.ident: worker.status})

    env.close()


    # Main Training Loop

    for epoch in tqdm(range(EPOCH_CNT)):

        # Workers will wait for each other to complete
        # while sim_barrier_completion.parties != sim_barrier_completion.n_waiting:
        #     if False in [process.is_alive() for process in processes]:
        #         print("Failed worker found exiting")
        #         exit()
        #     print(f"Reporting from parent {os.getpid()}")
        #     print(f"Workers done: {sim_barrier_completion.n_waiting} / {sim_barrier_completion.parties}")
        #     print(sim_barrier_completion)
        #     time.sleep(5)
        print("Main Process waiting:")
        sim_barrier_completion.wait()
        print("Resetting Sim Completion Barrier")
        sim_barrier_completion.reset()
        print("Main Process Proceeding to update")
        # <Epoche: [<process_id: end_results_of_episodes_of_that_process>]>
        playback_queue.update({epoch: processes_outputs})
        
        # Update nets
        print("Now updating P&Q-nets")
        print(epoch)
        if epoch == EPOCH_CNT - 1:
            extinction_event.set()
            update_completion.clear()
        update_nets(update_completion, processes_outputs, prime_model, shared_networks_dict)
        time.sleep(3)
        # sim_barrier_completion.reset()
        update_completion.clear()


    
    pass
    # End of all Epoches
    

    # Kill any left over processes
    extinction_event.set()
    process: tmp.Process
    for process in processes:
        process.join(timeout=10) #Wait for Workers to potentially write logs 
        if process.is_alive():
            process.terminate()
    print("ALL EPOCHES Finished!")
    # print("Final Q-net key size: ",len(shared_Q_nets[0].q_dict.keys()))

    print("Saving model")
    # Export the model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export = prime_model.save(os.path.join(MODEL_OUTPUT_DIR, f"{os.getpid()}_model_{timestamp}"))
    # if export != 0:
    #     print("Unable to save model")


if __name__ == "__main__":
    tmp.freeze_support()
    main()
