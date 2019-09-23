import os
import gym
from tensorforce.agents import PPOAgent


def main():
    env = gym.make('Breakout-v0')

    # (210, 160, 3)
    print(env.observation_space.shape)
    # [[[255...]]]
    print(env.observation_space.high)
    # [[[0...]]]
    print(env.observation_space.low)
    # 4
    print(env.action_space.n)

    agent = PPOAgent(
        # (210, 160, 3)
        states=dict(type='float', shape=env.observation_space.shape),
        network=[
            # (51, 29, 32)
            dict(type='conv2d', size=32, window=8, stride=4, activation='relu'),
            # (24, 18, 64)
            dict(type='conv2d', size=64, window=4, stride=2, activation='relu'),
            # (22, 16, 64)
            dict(type='conv2d', size=64, window=3, stride=1, activation='relu'),
            # 22528
            dict(type='flatten'),
            dict(type='dense', size=512, activation='relu'),
            dict(type='dense', size=32, activation='relu'),
        ],
        # batching_capacity=10,
        memory=dict(
            type='latest',
            include_next_states=False,
            capacity=1000,
        ),
        # update=dict(unit='timesteps', batch_size=64),
        actions=dict(type='int', num_actions=env.action_space.n),
        step_optimizer=dict(type='adam', learning_rate=1e-4)
    )

    model_dir = 'models/breakout'

    # load model
    if os.path.exists(f'{model_dir}/checkpoint'):
        agent.restore_model(directory=model_dir)

    try:
        for step in range(100000):
            observation = env.reset()

            done = False
            step_reward = 0
            while not done:
                # env.render()

                # from PIL import Image
                # pil_img = Image.fromarray(observation)
                # pil_img.save('./observation.png')

                states = observation / 256

                action = agent.act(states=states)

                observation, reward, done, info = env.step(action)

                reward = reward / 10

                agent.observe(reward=reward, terminal=done)

                step_reward += reward

                if done:
                    print(f'step = {step}, reward = {step_reward}')
    except Exception as e:
        raise e
    finally:
        agent.save_model(directory=f'{model_dir}/agent')


if __name__ == '__main__':
    main()
