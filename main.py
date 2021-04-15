import agent
import environment
import time
from collections import deque
import numpy as np

name = 'PongDeterministic-v4'
agent = agent.Agent(actions=[
    0, 2, 3], starting_mem_len=50000, max_mem_len=750000, starting_epsilon=1, learn_rate=.00025)
env = environment.make_env(name, agent)
scores = []
max_scores = []
steps = []
max_score = -2
env.reset()

for i in range(10):
    timesteps = agent.total_timesteps

    cur_time = time.time()
    score = environment.play_episode(name, env, agent, debug=False)

    scores.append(score)
    steps.append(agent.total_timesteps - timesteps)

    if score > max_score:

        max_score = score

    print('\nEpisode: ' + str(i))

    print('Steps: ' + str(agent.total_timesteps - timesteps))

    print('Duration: ' + str(time.time() - cur_time))

    print('Score: ' + str(score))

    print('Max Score: ' + str(max_score))

    np.save('Epochs_gamma_99.npy', scores)
    np.save('Steps_gamma_99.npy', steps)
