import os

import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def create_model(input_shape, output):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(output))
    model.add(Activation('linear'))
    return model


def main():
    weight_path = 'models/cartpole/keras_weights.h5'

    env = gym.make('CartPole-v0')

    env = gym.wrappers.Monitor(env, "./gym-results", force=True)

    input_shape = (1,) + env.observation_space.shape
    output = env.action_space.n
    model = create_model(input_shape, output)

    model.summary()

    # https://qiita.com/goodclues/items/9b2b618ac5ba4c3be1c5
    dqn = DQNAgent(
        model=model,
        # 出力 分類数 action数
        nb_actions=output,
        # 割引率 https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py#L307
        gamma=0.99,
        # experience replay
        # メモリにaction、reward、observationsなどのデータを経験（Experience）として保管しておいて、
        # 後でランダムにデータを再生（Replay）して学習を行う
        memory=SequentialMemory(
            # メモリの上限サイズ
            limit=5000,
            # 観測を何個連結して処理するか。例えば時系列の複数の観測をまとめて1つの状態とする場合に利用。
            window_length=1,
        ),
        # ウォームアップステップ数。学習の初期は安定しないため、学習率を徐々に上げていく期間。
        nb_steps_warmup=10,
        # bellman equation
        # 1未満の値の場合はSoft update
        # 1以上の値の場合はHard update = ステップごとに重みが完全に更新
        target_model_update=1e-2,
        # 環境において行動を選択する基準
        # GreedyQPolicy デフォルト 探索か活用か、学習が進むにつれて探索率を下げていく
        # BoltzmannQPolicy ボルツマン分布を利用したソフトマックス手法による方策
        policy=BoltzmannQPolicy(),
    )
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    if os.path.exists(weight_path):
        dqn.load_weights(weight_path)

    try:
        dqn.fit(
            env,
            nb_steps=5000,  # 3min
            visualize=False,
            log_interval=1000,
        )
    except KeyboardInterrupt:
        pass
    finally:
        dqn.save_weights(weight_path, overwrite=True)

    # dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()
