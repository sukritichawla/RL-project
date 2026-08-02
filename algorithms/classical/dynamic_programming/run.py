from environment.water_env import WaterEnvironment
import matplotlib.pyplot as plt
from algorithms.classical.dynamic_programming.policy_iteration import PolicyIterationAgent
from algorithms.classical.dynamic_programming.value_iteration import ValueIterationAgent


def test_agent(agent, env):

    print(f"\n{'='*50}")
    print(f"Testing {agent.name}")
    print(f"{'='*50}\n")

    policy = agent.train()

    print("Optimal Policy:")
    print(policy)

    

    print("\nTesting Policy (10 steps)\n")

    state = env.reset()

    for step in range(10):

        action = agent.select_action(state)

        state, reward, done, _ = env.step(action)

        print(f"Step {step + 1}")
        print(f"Action : {action}")
        print(f"Reward : {reward}")
        print(f"State  : {state}\n")


def main():

    env = WaterEnvironment()

    # ----------------------------
    # Policy Iteration
    # ----------------------------
    policy_agent = PolicyIterationAgent(env)
    test_agent(policy_agent, env)
    import matplotlib.pyplot as plt



    # ----------------------------
    # Value Iteration
    # ----------------------------
    value_agent = ValueIterationAgent(env)
    test_agent(value_agent, env)


if __name__ == "__main__":
    main()