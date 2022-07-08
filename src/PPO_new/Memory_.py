import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states.get_boolean_map = []
        self.states.get_float_map = []
        self.states.get_scalars = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states.get_boolean_map)
        batch_start = np.arrange(0, n_states, self.batch_size)
        indices = np.arrange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states.get_boolean_map),\
            np.array(self.states.get_float_map),\
            np.array(self.states.get_scalars),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.actions),\
            np.array(self.dones),\
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.get_boolean_map.append(state.get_boolean_map)
        self.states.get_float_map.append(state.get_float_map)
        self.states.get_scalars.append(state.get_scalars)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states.get_boolean_map = []
        self.states.get_float_map = []
        self.states.get_scalars = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

