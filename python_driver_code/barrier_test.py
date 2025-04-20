# from torch import multiprocessing as tmp
# import multiprocessing as mp
# from multiprocessing.synchronize import Event as mpEventType
from threading import BrokenBarrierError
import torch.multiprocessing as tmp
# from torch.multiprocessing import Barrier, Process, Event


import time

def worker(barrier, 
           worker_id, 
           cycles, 
           exit_event,
           update_event):
    
    while not exit_event.is_set():
        for cycle in range(cycles):
            print(f"Worker {worker_id} in cycle {cycle}")
            time.sleep(worker_id)  # Different work times
        
        print(f"Worker {worker_id} reached barrier in cycle {cycle}")
            

        try:
            barrier.wait()
            print(f"Worker {worker_id} passed barrier in cycle {cycle}")
        except BrokenBarrierError:
            print(f"Worker {worker_id}: barrier was broken!")
            # Handle the broken barrier scenario


def dummy_update(update_trigger_event):
    print("Now updating")
    time.sleep(5)

    update_trigger_event.set()


def celebrate():
    print("All workers reach barrier!")

def main():
    num_workers = 3
    num_epoch = 3
    worker_cycles = 3

    program_manager = tmp.Manager()
    execution_barrier = program_manager.Barrier(num_workers + 1, celebrate)
    update_event = program_manager.Event()
    exit_event = program_manager.Event()
    processes = []
    for i in range(num_workers):
        p = tmp.Process(target=worker, 
                    kwargs={

                    })
        p.daemon = True  # Optional: makes processes exit when main exits
        processes.append(p)
        p.start()
    # Example of manual reset between cycles from main process
    for epoch in range(num_epoch):
        
        # Wait for workers
        execution_barrier.wait()
        dummy_update(update_event)



    # for cycle in range(cycles):
    #     # Start worker processes for this cycle
    #     processes = []
    #     for i in range(num_workers):
    #         p = tmp.Process(target=worker, args=(barrier, i, 1))  # Just one cycle
    #         processes.append(p)
    #         p.start()
        
    #     # Wait for processes to finish
    #     for p in processes:
    #         p.join()
            
    #     print(f"Cycle {cycle} complete, resetting barrier")
    #     barrier.reset()  # Reset for next cycle
    for p in processes:
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()
        
    # print("All cycles complete")

if __name__ == "__main__":
    main()