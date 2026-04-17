# DQN Taxi Agent (Deep Reinforcement Learning)

A Deep Reinforcement Learning project using a Deep Q-Network (DQN) to solve the Taxi-v3 environment from Gymnasium.

---

## Features

- Deep Q-Network (DQN) implementation
- Neural network-based policy
- Experience-based learning
- Training and evaluation phases
- Uses stable-baselines3

---

## How It Works

1. Environment (Taxi-v3) is created  
2. DQN model is initialized with:
   - MLP policy  
   - 2 hidden layers (64, 64)  
3. Model is trained for 500,000 timesteps  
4. After training, the agent interacts with the environment  
5. The model predicts the best action at each step  
6. Total reward is calculated for each episode  

---

## Model Configuration

- Policy: MLP (Multi-Layer Perceptron)  
- Network architecture: [64, 64]  
- Learning rate: 0.001  
- Discount factor (gamma): 0.95  
- Exploration fraction: 0.1  
- Final epsilon: 0.02  
- Training steps: 500,000  
