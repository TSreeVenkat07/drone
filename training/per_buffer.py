import numpy as np
import torch
from typing import Tuple, Optional


class SumTree:
    """Binary sum tree for O(log n) PER sampling."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.size = 0
        self.ptr = 0

    def update(self, idx: int, priority: float):
        tree_idx = idx + self.capacity - 1
        diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff

    def add(self, priority: float, data):
        self.data[self.ptr] = data
        self.update(self.ptr, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, value: float) -> Tuple[int, float, object]:
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (Schaul et al. 2016).
    Stores (obs_n, action_n, reward_n, next_obs_n, done, mask_n, next_mask_n, global_state).
    """
    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_steps: int = 100000, epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_increment = (beta_end - beta_start) / beta_steps
        self.epsilon = epsilon
        self.max_priority = 1.0

    def push(self, transition):
        """Push a transition with max priority (will be corrected at first sample)."""
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size: int) -> Tuple:
        indices, priorities, transitions = [], [], []
        segment = self.tree.total / batch_size
        self.beta = min(self.beta_end, self.beta + self.beta_increment)

        for i in range(batch_size):
            low, high = segment * i, segment * (i + 1)
            val = np.random.uniform(low, high)
            idx, priority, trans = self.tree.get(val)
            if trans is None:
                continue
            indices.append(idx)
            priorities.append(priority)
            transitions.append(trans)

        if not transitions:
            return None

        # IS weights
        probs = np.array(priorities) / (self.tree.total + 1e-8)
        n = self.tree.size
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()

        return transitions, np.array(indices), torch.FloatTensor(weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            # Guard against NaN/Inf TD errors to protect the SumTree
            if not np.isfinite(err):
                err = 0.0
            priority = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def trim_to_newest(self, keep_fraction: float = 0.30):
        """Trim the buffer to keep only the newest fraction of transitions."""
        current_size = self.tree.size
        keep_count = int(current_size * keep_fraction)
        if keep_count <= 0:
            return
        
        # Identify the indices of the newest elements
        newest_indices = set()
        for i in range(1, keep_count + 1):
            idx = (self.tree.ptr - i) % self.tree.capacity
            newest_indices.add(idx)
            
        # Clear all other elements
        for idx in range(self.tree.capacity):
            if idx not in newest_indices:
                self.tree.data[idx] = None
                self.tree.update(idx, 0.0) # Set priority to 0 so it's not sampled
                
        # Update sizes
        self.tree.size = keep_count
        print(f"Trimmed replay buffer from {current_size} to newest {keep_count} elements.")

    def __len__(self):
        return self.tree.size

