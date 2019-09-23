import os
import gym
from tensorforce.agents import PPOAgent


def main():
    env = gym.make('CartPole-v0')

    # (4,)
    print(env.observation_space.shape)
    # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
    print(env.observation_space.high)
    # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
    print(env.observation_space.low)
    # 2
    print(env.action_space.n)

    agent = PPOAgent(
        states=dict(type='float', shape=env.observation_space.shape),
        network=[
            dict(type='dense', size=32, activation='relu'),
            dict(type='dense', size=32, activation='relu'),
        ],
        actions=dict(type='int', num_actions=env.action_space.n),
        step_optimizer=dict(type='adam', learning_rate=1e-4)
    )

    model_dir = 'models/cartpole'

    if os.path.exists(f'{model_dir}/checkpoint'):
        agent.restore_model(directory=model_dir)

    try:
        for ep in range(2000):
            observation = env.reset()
            done = False
            ep_reward = 0
            while not done:
                # env.render()

                states = observation / 4

                action = agent.act(states=states)

                observation, reward, done, info = env.step(action)

                agent.observe(reward=reward, terminal=done)

                ep_reward += reward

                if done:
                    print(f'ep = {ep}, ep_reward = {ep_reward}')
    except Exception as e:
        raise e
    finally:
        agent.save_model(directory=f'{model_dir}/agent')


if __name__ == '__main__':
    main()
