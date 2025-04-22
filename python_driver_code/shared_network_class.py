from torch import multiprocessing as tmp
import pickle
import copy
'''
This is a object class that can store a policy or Q network

'''
# Force Spawn method
tmp.set_start_method('spawn')

class Shared_Network:
    def __init__(self, name: str, ):
        # self.update_event
        self.name = name #Name of network
        
    # 
    def share_as_shared_tensor(self): 
        pass

        
    def update_network(self):
        pass


    def export_network(self, output_name = None):
        if not output_name:
            output_name = self.name
        with open(f'{output_name}.pkl', 'wb') as file:
            pickle.dump
