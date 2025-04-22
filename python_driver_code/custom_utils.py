import inspect
from functools import partial

def prepare_agent(agent: object, keyargs:dict)-> partial:
    '''
    Prepares agent for multiprocess process to avoid passing direct args/class to child process
    agent := object where agent can run episodes, collect experiences with provided env
    keyargs := key arguments for 
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