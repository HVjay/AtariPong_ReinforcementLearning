from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from agentMemory import Memory
import numpy as np
import random
from tensorflow import keras


class Agent():
    def __init__(self, actions, starting_mem_len, max_mem_len, starting_epsilon, learn_rate, epsilon_decay=.9/100000, gamma=0.95, directory=None, debug=False):
        self.gamma = gamma
        self.actions = actions

        self.eps = starting_epsilon
        self.epsilon_decay = epsilon_decay

        self.epsilon_min = .05
        self.memory = Memory(max_mem_len)
        self.lr = learn_rate

        if directory:
            self.model = self._load_model(directory)
        else:
            self.model = self._build_model()

        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.starting_mem_len = starting_mem_len/100
        self.starting_mem_len = self.starting_mem_len*100
        self.learn_steps = 0

    def _load_model(self, directory):
        print('LOADED MODEL')
        model = keras.models.load_model(str(directory))
        return model

    def _build_model(self):
        model = Sequential()
        model.add(Input((84, 84, 4)))

        model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=4, data_format="channels_last",
                  activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))

        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, data_format="channels_last",
                  activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, data_format="channels_last",
                  activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu',
                  kernel_initializer=keras.initializers.VarianceScaling(scale=2)))

        model.add(Dense(len(self.actions), activation='linear'))

        optimizer = Adam(self.lr)
        model.compile(optimizer, loss=keras.losses.Huber())
        print("BUILT MODEL")
        return model

    def get_action(self, state):

        if np.random.rand() < self.eps:
            return random.sample(self.actions, 1)[0]

        best_action = np.argmax(self.model.predict(state))
        return self.actions[best_action]

    def _index_valid(self, index):
        if self.memory.terminal_flags[index-3] or self.memory.terminal_flags[index-2] or self.memory.terminal_flags[index-1] or self.memory.terminal_flags[index]:
            return False
        else:
            return True

    def learn(self):
        actions_taken = []
        cur_states = []
        rewards = []
        next_cur_states = []
        next_done_flags = []

        while len(cur_states) < 32:
            ind = np.random.randint(4, len(self.memory.frames) - 1)
            if self._index_valid(ind):
                state = [self.memory.frames[ind-3], self.memory.frames[ind-2],
                         self.memory.frames[ind-1], self.memory.frames[ind]]

                state = np.moveaxis(state, 0, 2)/(2.55*100)

                next_cur_states = [self.memory.frames[ind-2], self.memory.frames[ind-1],
                                   self.memory.frames[ind], self.memory.frames[ind+1]]

                next_cur_states = np.moveaxis(next_cur_states, 0, 2)/(2.55*100)

                cur_states.append(state)
                next_cur_states.append(next_cur_states)
                actions_taken.append(self.memory.actions[ind])
                rewards.append(self.memory.rewards[ind+1])
                next_done_flags.append(self.memory.terminal_flags[ind+1])

        output = self.model.predict(np.array(cur_states))

        next_state_values = self.model_target.predict(
            np.array(next_cur_states))

        for i in range(32):
            action = self.actions.index(actions_taken[i])
            output[i][action] = rewards[i] + \
                (not next_done_flags[i]) * \
                self.gamma * max(next_state_values[i])

        self.model.fit(np.array(cur_states), output,
                       batch_size=32, epochs=1, verbose=0)

        if self.eps > self.epsilon_min:
            self.eps -= self.epsilon_decay
        self.learn_steps += 1

        if self.learn_steps % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')
