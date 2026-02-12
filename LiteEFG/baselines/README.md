# Baselines

Currently, the following algorithms are implemented.

- Counterfactual Regret Minimization (CFR) [[1]](#1)
- Monte-Carlo Counterfactual Regret Minimization (MCCFR) [[2]](#2)
- Counterfactual Regret Minimization+ (CFR+) [[3]](#3)
- Dilated Optimistic Mirror Descent (DOMD) [[4]](#4)
- Magnetic Mirror Descent (MMD) [[5]](#5)
- Clairvoyant Mirror Descent (CMD) [[6]](#6)
- Discounted Counterfactual Regret Minimization (DCFR) [[7]](#7)
- Predictive Counterfactual Regret Minimization (PCFR) [[8]](#8)
- Q-Function based Regret Minimization (QFR) [[9]](#9)
- Regularized Dilated Optimistic Mirror Descent (Reg-DOMD) [[10]](#10)
- Regularized Counterfactual Regret Minimization (Reg-CFR) [[10]](#10)
- Implicit Exploration Online Mirror Descent (IXOMD) [[11]](#11)
- Balanced Online Mirror Descent (Balanced OMD) [[12]](#12)
- Balanced Follow the Regularized Leader (Balanced FTRL) [[13]](#13)
- Follow the Perturbed Leader (FTPL) [[14]](#14)

## References
<a id="1">[1]</a> 
Zinkevich, Martin, et al. "Regret minimization in games with incomplete information." Advances in neural information processing systems 20 (2007).

<a id="2">[2]</a>
Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive games." Advances in neural information processing systems 22 (2009).

<a id="3">[3]</a> 
Tammelin, Oskari. "Solving large imperfect information games using CFR+." arXiv preprint arXiv:1407.5042 (2014).

<a id="4">[4]</a> 
Lee, Chung-Wei, Christian Kroer, and Haipeng Luo. "Last-iterate convergence in extensive-form games. Advances in Neural Information Processing Systems (2021).

<a id="5">[5]</a> 
Samuel Sokota, Ryan D'Orazio, J. Zico Kolter, Nicolas Loizou, Marc Lanctot, Ioannis Mitliagkas, Noam Brown, and Christian Kroer. "A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games." International Conference on Learning Representations (2023)

<a id="6">[6]</a> 
Piliouras, Georgios, Ryann Sim, and Stratis Skoulakis. "Beyond time-average convergence: Near-optimal uncoupled online learning via clairvoyant multiplicative weights update." Advances in Neural Information Processing Systems (2022).

<a id="7">[7]</a> 
Brown, Noam, and Tuomas Sandholm. "Solving imperfect-information games via discounted regret minimization." Proceedings of the AAAI Conference on Artificial Intelligence. 2019.

<a id="8">[8]</a> 
Farina, Gabriele, Christian Kroer, and Tuomas Sandholm. "Faster game solving via predictive blackwell approachability: Connecting regret matching and mirror descent." Proceedings of the AAAI Conference on Artificial Intelligence. 2021.

<a id="9">[9]</a> 
Liu, Mingyang, Gabriele Farina, and Asuman Ozdaglar. "A policy-gradient approach to solving imperfect-information games with iterate convergence." International Conference on Learning Representations (2025)

<a id="10">[10]</a> 
Liu, Mingyang, Asuman Ozdaglar, Tiancheng Yu, and Kaiqing Zhang. "The power of regularization in solving extensive-form games." International Conference on Learning Representations (2023)

<a id="11">[11]</a> 
Kozuno, Tadashi, Pierre Ménard, Rémi Munos, and Michal Valko. "Learning in two-player zero-sum partially observable Markov games with perfect recall." Advances in Neural Information Processing Systems (2021)

<a id="12">[12]</a> 
Bai, Yu, Chi Jin, Song Mei, and Tiancheng Yu. "Near-optimal learning of extensive-form games with imperfect information." International Conference on Machine Learning (2022)

<a id="13">[13]</a> 
Fiegel, Côme, Pierre Ménard, Tadashi Kozuno, Rémi Munos, Vianney Perchet, and Michal Valko. "Adapting to game trees in zero-sum imperfect information games." International Conference on Machine Learning (2023).

<a id="14">[14]</a> 
Hazan, Elad. "Introduction to online convex optimization." Foundations and Trends® in Optimization 2.3-4 (2016): 157-325.
