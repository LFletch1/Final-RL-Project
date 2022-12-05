from missile_defense_env import *
from stable_baselines3 import A2C
import time


target_type = "all"
env = MissileEnv(display=True, target_type=target_type)
model = A2C("MlpPolicy", env, learning_rate=0.0007, ent_coef=0.01, n_steps=1000, verbose=1)
file_string = f"A2C_{target_type}-5000000"
model = model.load(f"models/{target_type}/{file_string}")

input("Hit Enter to Continue:")
time.sleep(2)
obs = env.reset()
for _ in range(5000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        env.reset()

