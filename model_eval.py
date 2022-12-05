from missile_defense_env import *
from stable_baselines3 import A2C
import matplotlib.pyplot as plt

def main():

    target_types = ["static", "dynamic", "v_dynamic", "noisy", "all"]
    out_file = open("model_results.txt", "w")
    out_file.write("environment,successes\n")
    results = {}
    for target_type in target_types:
        print(target_type)
        env = MissileEnv(display=False, target_type=target_type)
        model = A2C("MlpPolicy", env, learning_rate=0.0007, ent_coef=0.01, n_steps=1000, verbose=1)
        file_string = f"A2C_{target_type}-1000000"
        model = model.load(f"models/{target_type}/{file_string}")
        obs = env.reset()
        number_of_episodes = 0
        successes = 0
        while number_of_episodes <= 100:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                if info["result"] == "success":
                    successes += 1
                number_of_episodes += 1
                env.reset()
        results[target_type] = successes
        out_file.write(f"{target_type},{successes}\n")

    y_pos = np.arange(len(list(results.values())))
    plt.bar(y_pos, list(results.values()))
    plt.xticks(y_pos, list(results.keys()), fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Successful Epidsodes with Various Target Types", fontsize=)
    plt.ylabel("Successful Episodes", fontsize=16)
    plt.xlabel("Target Type", fontsize=16)

    plt.show()



if __name__ == "__main__":
    main()