from collections import deque


class Memory():
    def __init__(self, max_length):
        self.max_length = max_length
        self.frames = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.terminal_flags = deque(maxlen=max_length)

    def add_memory(self, next_frame, next_frames_reward, next_action, next_frame_terminal):
        self.frames.append(next_frame)
        self.actions.append(next_action)
        self.rewards.append(next_frames_reward)
        self.terminal_flags.append(next_frame_terminal)
