import gym
import frame_preprocessing as resize
import numpy as np


def new_game(name, env, agent):
    env.reset()

    starting_frame = resize.resize_frame(env.step(0)[0], 1)

    dummy_action = 0

    dummy_reward = 0

    dummy_done = False
    temp = 3

    for i in range(temp):
        agent.memory.add_memory(
            starting_frame, dummy_reward, dummy_action, dummy_done)


def make_env(name, agent):
    env = gym.make(name)
    return env


def steps(name, env, agent, score, debug):
    if True:
        agent.total_timesteps += 1
        if agent.total_timesteps % 5000 == 0:

            agent.model.save('model_gamma_99.h5')
            print('\nWeights saved!')

        frame_2, frame_2_reward, frame_2_terminal, info = env.step(
            agent.memory.actions[-1])

        frame_2 = resize.resize_frame(frame_2, 1)
        next_state = [agent.memory.frames[-3], agent.memory.frames[-2],

                      agent.memory.frames[-1], frame_2]

        next_state = np.moveaxis(next_state, 0, 2)/(2.55*100)
        next_state = np.expand_dims(next_state, 0)

        next_action = agent.get_action(next_state)

        if frame_2_terminal:
            for i in range(1):
                agent.memory.add_memory(
                    frame_2, frame_2_reward, next_action,
                    frame_2_terminal)

            return (score + frame_2_reward), True
        else:
            temp_length = 1
            while True:
                agent.memory.add_memory(
                    frame_2, frame_2_reward, next_action, frame_2_terminal)

                if len(agent.memory.frames) > agent.starting_mem_len:
                    agent.learn()

                return (score + frame_2_reward), False
                break


def play_episode(name, env, agent, debug=False):
    new_game(name, env, agent)
    done = False
    score = 0

    while True:
        score, done = steps(name, env, agent, score, debug)
        if done:
            break
    return score
