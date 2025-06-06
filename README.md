<div align="center">

# Preference Optimization for Combinatorial Optimization Problems

</div>

This repository contains PyTorch implementation of [POMO](https://github.com/yd-kwon/POMO) version of *ICML 2025 poster -- "[Preference Optimization for Combinatorial Optimization Problems](https://arxiv.org/abs/2505.08735)"*.

<p align="center"><img src="./Figs/PO_framework.png" width=95%></p>

**TL;DR**: We theoretically transform numerical rewards into pairwise preference signals and integrate local search during fine-tuning, empirically enabling faster convergence and higher-quality solutions for COPs like TSP, CVRP, and scheduling.

**Abstract**: Reinforcement Learning (RL) has emerged as a powerful tool for neural combinatorial optimization, enabling models to learn heuristics that solve complex problems without requiring expert knowledge. Despite significant progress, existing RL approaches face challenges such as diminishing reward signals and inefficient exploration in vast combinatorial action spaces, leading to inefficiency. In this paper, we propose Preference Optimization, a novel method that transforms quantitative reward signals into qualitative preference signals via statistical comparison modeling, emphasizing the superiority among sampled solutions. Methodologically, by reparameterizing the reward function in terms of policy and utilizing preference models, we formulate an entropy-regularized RL objective that aligns the policy directly with preferences while avoiding intractable computations. Furthermore, we integrate local search techniques into the fine-tuning rather than post-process to generate high-quality preference pairs, helping the policy escape local optima. Empirical results on various benchmarks, such as the Traveling Salesman Problem (TSP), the Capacitated Vehicle Routing Problem (CVRP) and the Flexible Flow Shop Problem (FFSP), demonstrate that our method significantly outperforms existing RL algorithms, achieving superior convergence efficiency and solution quality.

### PO vs. REINFORCE-based algorithms on identical models

<p align="center"><img src="./Figs/learning_curve_tsp.png" width=95%></p>

### Setup

We recommend using conda to set up the environment and install the dependencies:

```bash
conda create -n po_cops python=3.9
conda activate po_cops
cd POMO
pip install -r requirements.txt
```

### Usage

```bash
cd POMO/TSP/POMO
# testing
python test_n100.py
# training
python train_n100_po.py
```

### Acknowledgments

We would like to express our sincere gratitude to the anonymous reviewers and (S)ACs of ICML 2025 for their thoughtful feedback and insightful comments, which have been instrumental in enhancing the quality of this work.

Our code is originally implemented upon [POMO](https://github.com/yd-kwon/POMO), implementations on ELG, AM, Pointerformer etc. will be released soon. We also thank the authors of the following repositories for their valuable insights and code:
- [POMO] https://github.com/yd-kwon/POMO
- [ELG] https://github.com/gaocrr/ELG
- [AM] https://github.com/wouterkool/attention-learn-to-route
- [Sym-NCO] https://github.com/alstn12088/Sym-NCO
- [Pointerformer] https://github.com/Pointerformer/Pointerformer
- [AMDKD] https://github.com/jieyibi/AMDKD/tree/main/AMDKD-POMO
- [Omni-VRP] https://github.com/RoyalSkye/Omni-VRP
- [MatNet] https://github.com/yd-kwon/MatNet
- [Poppy] https://github.com/instadeepai/poppy
- [COMPASS] https://github.com/instadeepai/compass


### Citation

If you find our paper/blog or our code useful, we would appreciate it if you could cite our work:

```bibtex
@article{pan2025preference,
  title={Preference Optimization for Combinatorial Optimization Problems},
  author={Pan, Mingjun and Lin, Guanquan and Luo, You-Wei and Zhu, Bin and Dai, Zhien and Sun, Lijun and Yuan, Chun},
  journal={arXiv preprint arXiv:2505.08735},
  year={2025}
}
```