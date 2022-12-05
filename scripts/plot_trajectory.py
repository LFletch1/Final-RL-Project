import matplotlib.pyplot as plt
from missile_defense_env import *
from stable_baselines3 import A2C

def main():
    target_type = "noisy"
    env = MissileEnv(display=False, target_type=target_type)
    model = A2C("MlpPolicy", env, learning_rate=0.0007, ent_coef=0.01, n_steps=1000, verbose=1)
    file_string = f"A2C_{target_type}-1000000"
    model = model.load(f"models/{target_type}/{file_string}")
    obs = env.reset()
    agent = {}
    agent["x"] = []
    agent["y"] = []
    target = {}
    target["x"] = []
    target["y"] = []

    for _ in range(5000):
        agent["x"].append(env.agent_missile.x_t)
        agent["y"].append(-(env.agent_missile.y_t-800))
        target["x"].append(env.target_missile.x_t)
        target["y"].append(-(env.target_missile.y_t-800))
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            plt.plot(agent["x"], agent["y"])
            plt.plot(target["x"], target["y"], color="red")
            plt.xlabel("x", fontsize=15)
            plt.ylabel("y", fontsize=15)
            plt.legend(["Agent", "Target"])
            plt.show()
            agent["x"] = []
            agent["y"] = []
            target["x"] = []
            target["y"] = []
            env.reset()

if __name__ == "__main__":
    main()