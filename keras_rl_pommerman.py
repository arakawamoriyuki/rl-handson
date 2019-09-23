import os

from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam

from rl.core import Env, Processor
from rl.agents.dqn import DQNAgent
from rl.policy import MaxBoltzmannQPolicy
from rl.memory import SequentialMemory


# 1. トレーニングエージェントのobsを加工して返す `EnvWrapper.featurize`
#   - obsにはボードの状態やアイテムの取得状況など必要な状態は全て揃っているので、そのデータを加工してネットワークのinput_shapeを変更する
#   - 全部の情報を利用すると情報過剰で収束にかなり時間がかかる。最初は周りの9マスのみに変更するなど情報量を削って試してみる
#   - 1と2は連続値ではなくidなのでカテゴリカルデータに変更する
#     - [[1, 2], [3, 4]] => [[[1,0,0,0]], [0,1,0,0]], [[0,0,1,0], [0,0,0,1]]]
#   - 自分(10)、敵(11,12,13)なので自分以外は敵としてデータを調整する
#     board = obs['board']
#     board = np.where(13 == board, 10, board)
#     board = np.where(10 < board, 11, board)
#     nb_classes = 12
#     board = to_categorical(board, nb_classes).astype(np.float32)
# 2. リワードの調整を行う `Rewards.get_rewards`
#   - 最初に敵倒せるなど不可能なのでよく起こりうることに対して報酬を与えれないか考える
#     - 壁壊したら報酬与える
#     - アイテムとったら報酬与える
#   - ボム置かなくなる対策
#     - ボムを置いたら報酬を与える
#     - 時間経過(step)ごとにマイナス報酬を増大/減少させる
#       - 生存のために減少させると動かなくなりうる
#       - 挑戦させるためにマイナス報酬を増大させる
# 3. パラメータ調整やネットワークを変えてみる
#   - DQNのパラメータ、policyや割引率など変えてみる
#   - DQN以外にもNAFAgent,DDPGAgent,SARSAAgent,CEMAgent,PPOAgentなどネットワークいっぱいある
#   - 教師あり学習でも良いしニューラルネットワークでなくても良い。別フレームワークでも良い
# 4. その他のアイディアを考える
#   - SimpleAgentには全く勝てないのでStoppingAgentを実装して動かない相手と対戦して知識をつける
#   - BaseLineAgentの動作を教師あり学習させたあと、蒸留してネットワークを縮める
#   - ActionFiterのような知識(半ロジック)を埋め込む
#   - PlayerAgent(キーボードで人間が操作可能)なエージェントのログを教師あり学習で学習する
#   - 学習にかなり時間かかるのでtry and error難しい。GPUなど高火力で攻める

# class StoppingAgent(BaseAgent):
#     def act(self, obs, action_space):
#         # 0 stop, 1 up, 2 down, 3 left, 4 right, 5 bom
#         return 0


class Rewards:
    def __init__(self):
        pass

    def get_rewards(self, obs_now, action_now, reward):

        # TODO: リワードの調整

        # print(f'Rewards.get_rewards obs_now = {obs_now}')
        # print(f'Rewards.get_rewards action_now = {action_now}')
        # print(f'Rewards.get_rewards reward = {reward}')

        # 0  床
        # 1  壁 破壊不可
        # 2  壁 破壊可
        # 3  爆弾
        # 4  爆風?
        # 5  ない?
        # 6  ボム
        # 7  火力
        # 8  キック
        # 9  ない?
        # 10 1P左上
        # 11 2P左下
        # 12 3P右下
        # 13 4P右上 自エージェント

        # 爆弾獲得
        if self.ammo < obs_now['ammo']:
            reward += 0.2
        self.ammo = obs_now['ammo']

        # 火力獲得
        if self.blast_strength < obs_now['blast_strength']:
            reward += 0.2
        self.blast_strength = obs_now['blast_strength']

        # キック獲得
        if self.can_kick is False and obs_now['can_kick'] is True:
            reward += 0.2
        self.can_kick = obs_now['can_kick']

        # 生存step数ペナルティ
        reward += obs_now['step_count'] * -0.0001

        # print(reward)

        return reward

    def reset(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False

        # print(f'Rewards.reset')
        pass


class EnvWrapper(Env):
    def __init__(self, gym):
        self.gym = gym
        self.rewardShaping = Rewards()

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    def render(self, mode='human', close=False):
        self.gym.render(mode=mode, close=close)

    def close(self):
        self.gym.close()

    def seed(self, seed=None):
        raise self.gym.seed(seed)

    def configure(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        # print('EnvWrapper.reset')
        self.rewardShaping.reset()
        obs = self.gym.reset()
        agent_obs = self.featurize(obs[self.gym.training_agent])
        return agent_obs

    def step(self, action):
        # print(f'EnvWrapper.step action = {action}')
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, info = self.gym.step(all_actions)
        action = all_actions[self.gym.training_agent]
        agent_state = self.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]

        agent_reward = self.rewardShaping.get_rewards(
            obs[self.gym.training_agent],
            action,
            agent_reward,
        )

        return agent_state, agent_reward, terminal, info

    def featurize(self, obs):

        # TODO: トレーニングエージェントのobsを加工して返す

        # print(f'EnvWrapper.featurize obs = {obs}')
        # from tensorflow.keras.utils import to_categorical
        # 自分は10、敵は11に変更 (11, 11, 12)
        # board = obs['board']
        # board = np.where(13 == board, 10, board)
        # board = np.where(10 < board, 11, board)
        # nb_classes = 12
        # board = to_categorical(board, nb_classes).astype(np.float32)
        # print(board.shape)

        return obs['board']


class CustomProcessor(Processor):
    def process_state_batch(self, batch):
        # print(f'CustomProcessor.process_state_batch batch = {batch.shape}')
        return batch

    def process_info(self, info):
        info['result'] = info['result'].value
        return info


class DQN(BaseAgent):
    def act(self, obs, action_space):
        pass


def get_env():
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])
    # env.seed(0)

    agent_id = 0

    agents = [
        DQN(config["agent"](0, config["game_type"])),
        SimpleAgent(config["agent"](1, config["game_type"])),
        SimpleAgent(config["agent"](2, config["game_type"])),
        SimpleAgent(config["agent"](3, config["game_type"])),
    ]

    env.set_agents(agents)

    env.set_training_agent(agents[agent_id].agent_id)
    env.set_init_game_state(None)

    return env


def create_model(input_shape, output_units, history_length):
    model = Sequential()

    model.add(Conv2D(
        32,
        kernel_size=2,
        strides=(1, 1),
        input_shape=(history_length, input_shape[0], input_shape[1]),
        activation='relu',
    ))
    model.add(Conv2D(
        64,
        kernel_size=2,
        strides=(1, 1),
        activation='relu'),
    )
    model.add(Conv2D(
        64,
        kernel_size=2,
        strides=(1, 1),
        activation='relu',
    ))
    model.add(Flatten())
    model.add(Dense(
        units=128,
        activation='relu',
    ))
    model.add(Dense(
        units=128,
        activation='relu',
    ))
    model.add(Dense(
        units=output_units,
        activation='linear',
    ))

    # print(model.input_shape)
    model.summary()

    return model


def create_dqn(model, history_length):
    memory = SequentialMemory(limit=500000, window_length=history_length)
    policy = MaxBoltzmannQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=model.output_shape[1],
        memory=memory,
        policy=policy,
        processor=CustomProcessor(),
        nb_steps_warmup=512,
        enable_dueling_network=True,
        dueling_type='avg',
        target_model_update=5e2,
        batch_size=32,
    )
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    return dqn


def main():
    weight_path = 'models/pommerman/keras_weights.h5'
    history_length = 4
    input_shape = (11, 11)
    output_units = 6

    env = get_env()
    env_wrapper = EnvWrapper(env)

    model = create_model(input_shape, output_units, history_length)
    dqn = create_dqn(model, history_length)

    if os.path.exists(weight_path):
        dqn.load_weights(weight_path)

    try:
        dqn.fit(
            env_wrapper,
            nb_steps=2000000,  # 8h
            visualize=False,
            nb_max_episode_steps=env._max_steps,
        )
    except KeyboardInterrupt:
        pass
    finally:
        dqn.save_weights(weight_path, overwrite=True)

    # dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()
