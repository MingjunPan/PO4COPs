import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, num_node, node_dim, device):
        self.buffer_size = buffer_size
        self.num_node = num_node
        self.node_dim = node_dim
        self.device = device

        self.buffer = {
            'node': torch.zeros((buffer_size, num_node, node_dim), dtype=torch.float32).to(device),
            'old_logit': torch.zeros((buffer_size, num_node, node_dim), dtype=torch.float32).to(device),
            'value': torch.zeros((buffer_size, num_node), dtype=torch.float32).to(device),
            'reward': torch.zeros((buffer_size, num_node), dtype=torch.float32).to(device),
        }
