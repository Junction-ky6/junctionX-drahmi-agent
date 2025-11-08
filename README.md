# Reinforcement Learning for Financial Management

**1st Prize — JunctionX Algiers Hackathon**
**CELEC — Université des Sciences et de la Technologie Houari Boumediene (USTHB), Algiers, Algeria**

## Overview

This project presents a **financial management application** powered by **reinforcement learning (RL)**.
It formulates personal finance optimization as a **sequential decision-making problem**, where an intelligent agent learns to balance **spending, saving, investing, and goal achievement** dynamically.

The system is implemented using **Proximal Policy Optimization (PPO)** — a modern policy-gradient method for stable and efficient training.

## Core Idea

Financial management is modeled as a **Markov Decision Process (MDP)**:

| Element    | Description                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------ |
| **State**  | Current money balance, spending ratios, user satisfaction, goal progress, investment performance |
| **Action** | Allocation ratios across expense categories, savings, goals, and investments                     |
| **Reward** | Weighted combination of financial benefit, satisfaction, and achievement metrics                 |
| **Policy** | Learned neural network (Actor-Critic) that outputs optimal allocation strategies                 |

The system continuously learns how to best allocate income between categories to **maximize long-term reward and satisfaction**.

## Architecture

* **Environment (`Env`)** — Simulates a financial ecosystem with:

  * Categories (`Category`) — e.g., Billing, Debt, Purchase, Entertainment, Savings
  * Goal (`Goal`) — Measures progress toward a financial objective
  * Investment (`Investment`) — Tracks ongoing returns and profits

* **Policy Model (`ActorCriticNetwork`)** — Neural network producing mean & variance of the action distribution.

* **Trainer (`PPOTrainer`)** — Implements PPO algorithm for stable updates of the policy and value networks.

* **Simulation Rollouts (`collect_trajectories`)** — Gathers experience batches from synthetic or recorded simulations.

## Training Flow

1. Load simulation data from JSON files (realistic or synthetic).
2. Initialize the RL environment and PPO agent.
3. Collect trajectories over multiple epochs.
4. Update policy and value networks using **PPO clipping** and **Generalized Advantage Estimation (GAE)**.
5. Save the trained model as `ppo_actor_critic.pth`.