from functools import partial
import gymnasium as gym
from blackjack_agent_package import blackjack_agent_class as BJA
from blackjack_agent_package import Q_value_table_class as QVT
import torch.multiprocessing as tmp
import inspect

def prepare_agent(agent: object, keyargs:dict)-> partial:
    '''
    Prepares agent for multiprocess process to avoid passing direct args to child process
    agent := object where agent can run episodes, collect experiences with provided env
    '''
    signature = inspect.signature(agent.__init__)
    bounded_args = {}

    # Process provided user args
    for arg in keyargs:
        if arg in signature.parameters:
            bounded_args[arg] = keyargs[arg]

    # test user provided args to if missing args
    try:
        signature.bind_partial(**bounded_args)
    except Exception as e:
        print("Unexpected binding error:",e)
        # print("Unknown User provided arg for agent:", arg)
        raise
    

    target_agent = partial(agent, **bounded_args)
    # for arg, value in signature.parameters.items():

    #     if arg in keyargs:
    #         target_agent = partial(target_agent, keyargs[arg])
    #     if isinstance(value, inspect.Parameter) and arg not in keyargs and arg != 'self':
    #         print("Missing necessary parameter in user provided keyargs!", arg)
    #         raise
        
    return target_agent
    

def dummy_process(target_agent_func: partial):
    agent: BJA.BlackjackAgent = target_agent_func()
    # print(agent.q_values)
    exp, rewards = agent.collect_experience()
    print(exp)
def main():
    env = gym.make("Blackjack-v1", sab=False)
    Q_value_table = QVT.QValueTable(action_size = env.action_space.n)
    keyword_args = {
        "env" : env, 
        "q_values" : Q_value_table

    }

    print(Q_value_table)
    # print(BJA.BlackjackAgent.__init__)
    # my_agent_target = partial(BJA.BlackjackAgent,)
    # signature = inspect.signature(BJA.BlackjackAgent.__init__)
    # print(signature.parameters['epsilon'])
    
    # my_agent_target = partial(BJA.BlackjackAgent, env = keyword_args["env"], q_values = keyword_args['q_values'])
    my_agent_target = prepare_agent(BJA.BlackjackAgent, keyword_args)
    # agent.keywords = keyword_args
    # print(my_agent_target.args)
    tmp.set_start_method('spawn',force=True) 
    child_process = tmp.Process(target=dummy_process, kwargs={"target_agent_func": my_agent_target})
    

    child_process.start()

    child_process.join(10)

if __name__ == '__main__':
    tmp.freeze_support()
    main()
