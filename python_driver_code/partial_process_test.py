from functools import partial
import gymnasium as gym
from blackjack_agent_package import blackjack_agent_class as BJA
from blackjack_agent_package import Q_value_table_class as QVT
import torch.multiprocessing as tmp


def dummy_process(target_agent_func: partial):
    agent: BJA.BlackjackAgent = target_agent_func()
    print(agent.q_values)
    exp = agent.collect_experience()
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
    
    my_agent_target = partial(BJA.BlackjackAgent, env = env, q_values = Q_value_table)
    # agent.keywords = keyword_args
    print(type(my_agent_target))
    tmp.set_start_method('spawn',force=True) 
    child_process = tmp.Process(target=dummy_process, kwargs={"target_agent_func": my_agent_target})
    

    child_process.start()

    child_process.join(10)

if __name__ == '__main__':
    tmp.freeze_support()
    main()
