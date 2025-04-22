from SAC_agent_package.prime_model import prime_SAC_model
from custom_utils import prepare_agent
import torch.multiprocessing as tmp
import gymnasium as gym


# def dummy_process(shared_array):
#     critic_net = shared_array
#     print(type(critic_net))
#     # env = gym.make("Pendulum-v1")  # This has a continuous action space
#     print("Critic's addr in child process", id(critic_net))
    # target_agent.run_episode()


def dummy_process(target_agent, shared_dict:dict):
    # print("Critic's",id(target_agent.critic))
    print("Address in ",id(shared_dict["Policy net"]))
    model: prime_SAC_model = target_agent()
    exp, reward = model.run_episode()
    print(reward)

if __name__ == '__main__':
    env = gym.make("Pendulum-v1")  # This has a continuous action space
    tmp.set_start_method('spawn',force=True) 

    # model = prime_SAC_model("MlpPolicy", env, verbose=1)
    keyargs = {
        'policy' : "MlpPolicy",
        'env' : env,
        'verbose': 1
    }

    target_agent = prepare_agent(prime_SAC_model, keyargs)
    prime_model: prime_SAC_model = target_agent()
    prime_model.share_model_across_memory()

    test_manager = tmp.Manager()
    shared_dict = test_manager.dict({"Policy net": prime_model.policy.state_dict()})
    print(id(shared_dict["Policy net"]))


    # print("Critic's addr in parent process", id(model.critic))
    child_process = tmp.Process(target=dummy_process, kwargs={"target_agent": target_agent, "shared_dict":shared_dict})
    child_process.start()
    child_process.join(10)
    # print(model.run_episode())