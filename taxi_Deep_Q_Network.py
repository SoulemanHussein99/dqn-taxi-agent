import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("Taxi-v3")

# model 
model = DQN(policy="MlpPolicy",
            env = env, 
            learning_rate=0.001,
            gamma=0.95,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            verbose=1,
            policy_kwargs={
                "net_arch": [64,64]}
            )
# training the model
model.learn(total_timesteps=500000)

# choosing env with render
env = gym.make("Taxi-v3", render_mode="human")

# testing the model
# for changing the poistion in the env and try it again
for _ in range (5):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print("total_raward: ",total_reward)

env.close()
