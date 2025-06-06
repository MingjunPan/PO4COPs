import numpy as np
import numba as nb
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import concurrent.futures
from functools import partial

@dataclass
class Route_Info:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    reward: torch.Tensor
    # shape: (batch, pomo)
    route: torch.Tensor = None
    # shape: (batch, pomo, node)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node - 1, node)

class TSPLocalSearch:
    def __init__(self, **search_param):
        self.search_param = search_param
        self.search_proportion = search_param['search_proportion']
        # the proportion of search samples in one batch
        self.proportion_type = search_param['proportion_type']
        self.number_of_cpu = search_param['number_of_cpu']

    def search(self, route, reward, distmat, problems):
        # shape: (batch, pomo, problem)
        batch_size = route.size(0)
        pomo_size = route.size(1)
        problem_size = route.size(2)
        batch_index = torch.arange(batch_size).view(-1, 1)
        search_pomo_size = int(pomo_size * self.search_proportion)
        
        if self.proportion_type == 'random':
            search_pomo_idx = torch.randint(pomo_size, size=(batch_size, search_pomo_size))
            search_route = route[batch_index, search_pomo_idx]
            search_reward = reward[batch_index, search_pomo_idx]
        elif self.proportion_type == 'maximum':
            search_pomo_idx = torch.argsort(reward, dim=1, descending=True)[:, :search_pomo_size]
            search_route = route[batch_index, search_pomo_idx]
            search_reward = reward[batch_index, search_pomo_idx]
        else:
            raise NotImplementedError

        search_route = search_route.reshape(-1, problem_size).cpu().numpy()
        search_reward = search_reward.cpu().numpy()
        distmat = distmat.unsqueeze(1). \
            expand(batch_size, search_pomo_size, problem_size, problem_size). \
            reshape(-1, problem_size, problem_size).cpu().numpy()
        new_route = self.two_opt(search_route, distmat)
        
        # numpy to torch
        new_route = torch.from_numpy(new_route).view(batch_size, search_pomo_size, problem_size).to(route.device)
        return self.pack_route(new_route, problems)
        
        
    def pack_route(self, route, problems):     
        batch_size = route.size(0)
        search_pomo_size = route.size(1)
        problem_size = route.size(2)
        
        BATCH_IDX = torch.arange(batch_size)[:, None].expand(batch_size, search_pomo_size)
        POMO_IDX = torch.arange(search_pomo_size)[None, :].expand(batch_size, search_pomo_size)

        one_hot = F.one_hot(route, problem_size).to(torch.float)
        # shape: (batch, pomo, problem, problem)
        till_mat = torch.tril(torch.ones(batch_size, search_pomo_size, problem_size - 1, problem_size))
        ninf_mask = torch.zeros_like(till_mat)
        ninf_mask = torch.where((till_mat @ one_hot) == 0, ninf_mask, float('-inf'))
        ninf_mask = ninf_mask.reshape(batch_size, search_pomo_size * (problem_size - 1), problem_size)
        
        gathering_index = route.unsqueeze(3).expand(batch_size, -1, problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, search_pomo_size, problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        return Route_Info(BATCH_IDX, POMO_IDX, -travel_distances, route, ninf_mask)
 
    
    def two_opt(self, route, dist):
        new_route = batched_two_opt_python(dist, route, 10, self.number_of_cpu)
        new_route = new_route.astype(np.int64)
        return new_route

@nb.njit(nb.float32(nb.float32[:,:], nb.int16[:], nb.int16), nogil=True)
def two_opt_once(distmat, tour, fixed_i = 0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i==0 else range(fixed_i, fixed_i+1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i-1], tour[(j+1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (  distmat[node_prev, node_j] 
                        + distmat[node_i, node_next]
                        - distmat[node_prev, node_i] 
                        - distmat[node_j, node_next])                    
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q+1] = np.flip(tour[p: q+1])
        return delta
    else:
        return 0.0


@nb.njit(nb.int16[:](nb.float32[:,:], nb.int16[:], nb.int64), nogil=True)
def _two_opt_python(distmat, tour, max_iterations=1000):
    iterations = 0
    tour = tour.copy()
    min_change = -1.0
    while min_change < -1e-6 and iterations < max_iterations:
        min_change = two_opt_once(distmat, tour, 0)
        iterations += 1
        assert min_change <= 0
    return tour

def batched_two_opt_python(dist: np.ndarray, tours: np.ndarray, max_iterations=1000, n_cpu=32):
    dist = dist.astype(np.float32)
    tours = tours.astype(np.int16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpu) as executor:
        futures = []
        for tour, d in zip(tours, dist):
            future = executor.submit(partial(_two_opt_python, distmat=d, max_iterations=max_iterations), tour = tour)
            futures.append(future)
        results = [f.result() for f in futures]
        return np.stack(results)
