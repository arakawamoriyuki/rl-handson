import os

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def create_model(input_shape, output):
    model = Sequential()

    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(output))
    model.add(Activation('linear'))

    return model


def main():
    weight_path = 'models/breakout/keras_weights.h5'

    env = gym.make('BreakoutDeterministic-v4')
    nb_actions = env.action_space.n
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    model = create_model(input_shape, nb_actions)

    print(model.summary())

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
        nb_steps=1000000)

    dqn = DQNAgent(
        model=model, nb_actions=nb_actions, policy=policy, memory=memory,
        processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
        train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if os.path.exists(weight_path):
        dqn.load_weights(weight_path)

    try:
        dqn.fit(
            env,
            nb_steps=1750000,  # 8h
            visualize=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        dqn.save_weights(weight_path, overwrite=True)

    # dqn.test(env, nb_episodes=10, visualize=True)


if __name__ == '__main__':
    main()
