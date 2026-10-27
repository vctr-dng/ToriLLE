#  two_player_example.py
#  Both fighters are controlled from outside, here with random actions.
#  Each agent gets its own observation ("self" and "opponent" in its own frame)
#  and its own (zero-sum) reward. ToriLLE has no policies: replace the
#  `sample()` calls with your own.
from torille.envs import TwoPlayerToriEnv

NUM_EPISODES = 2
DRAW_GAME = True

env = TwoPlayerToriEnv(matchframes=200, turnframes=5)
env.set_draw_game(DRAW_GAME)

try:
    for episode in range(NUM_EPISODES):
        observations, infos = env.reset(seed=episode)
        returns = dict.fromkeys(env.agents, 0.0)
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent, reward in rewards.items():
                returns[agent] += reward
        info = infos["player_0"]
        print(
            f"Episode {episode}: winner={info['winner_agent'] or 'tie'} "
            f"frame={info['match_frame']} returns={returns}"
        )
finally:
    env.close()
