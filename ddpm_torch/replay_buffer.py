import torch

class SharedMemoryManager:
    def __init__(self, capacity, world_size):
        self.capacity = capacity
        self.world_size = world_size
        self.x_size = 1000
        self.y_size = 1

        self.x_div_size = self.x_size//self.world_size

        self.store_kl_diff = torch.zeros(self.world_size, self.x_div_size).share_memory_()
        self.store_kl_diff_sum = torch.zeros(self.world_size, self.y_size).share_memory_()

        self.buffer_X = torch.zeros(self.capacity, self.x_size).share_memory_()
        self.buffer_y = torch.zeros(self.capacity, self.y_size).share_memory_()

        self.position = torch.zeros(1, dtype=torch.int).share_memory_()
        self.size = torch.zeros(1, dtype=torch.int).share_memory_()

        self.x_sampled = torch.zeros(1, 3, 32, 32).share_memory_()

    def sum_gpus(self, kl_diff_lasso, kl_diff_sum_lasso, rank):
        self.store_kl_diff[rank] = kl_diff_lasso
        self.store_kl_diff_sum[rank] = kl_diff_sum_lasso

    def add(self, X, y):
        X = self.store_kl_diff.flatten()
        y = self.store_kl_diff_sum.sum()

        index = self.position.item()

        self.buffer_X[index] = X
        self.buffer_y[index] = y

        self.position[0] = (self.position.item() + 1) % self.capacity
        self.size[0] = min(self.size.item() + 1, self.capacity)
    
    def add_x_sampled(self, x_sampled):
        self.x_sampled[0] = x_sampled