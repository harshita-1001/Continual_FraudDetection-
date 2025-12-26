import random

class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.buffer = []
        self.max_size = max_size

    def add(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.max_size:
            self.buffer = random.sample(self.buffer, self.max_size)

    def sample(self, size):
        return random.sample(self.buffer, min(size, len(self.buffer)))
