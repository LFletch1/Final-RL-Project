import matplotlib.pyplot as plt

def main():
    file = open("data/noisy_model_stats.txt")

    episode_lengths = []   
    episode_rewards = []
    for line in file:
        values = line.strip().split(" ")
        episode_lengths.append(float(values[0]))
        episode_rewards.append(float(values[1]))

    time = [t for t in range(len(episode_rewards))]

    plt.plot(time, episode_rewards)
    plt.xlabel("Episode", fontsize=15)
    plt.ylabel("Reward", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    plt.clf()

    plt.plot(time, episode_lengths)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Episode", fontsize=15)
    plt.ylabel("Episode Length", fontsize=15)
    plt.show()



if __name__ == "__main__":
    main()