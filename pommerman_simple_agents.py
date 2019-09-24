import pommerman
from pommerman import agents


class StoppingAgent(agents.BaseAgent):
    def act(self, obs, action_space):
        # 0 stop, 1 up, 2 down, 3 left, 4 right, 5 bom
        return 0


def main():
    env = pommerman.make('PommeFFACompetition-v0', [
        agents.PlayerAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
        StoppingAgent(),
    ])

    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if done:
                win_player = info['winners'][0] + 1
                print(f'win {win_player}P')
        print(f'Episode {i_episode} finished')
    env.close()


if __name__ == '__main__':
    main()
