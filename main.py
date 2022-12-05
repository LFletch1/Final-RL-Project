from missile_defense_env import *
from stable_baselines3 import A2C

def main():
    env = MissileEnv(display=False, target_type="noisy")
    model = A2C("MlpPolicy", env, learning_rate=0.0007, ent_coef=0.01, n_steps=1000, verbose=1)

    MODEL_TYPE = "A2C_noisy"
    STEPS = 10000
    for i in range(1,51):
        model.learn(total_timesteps=STEPS, reset_num_timesteps=False)
        if i % 10 == 0:
            print(f"Trained {i*STEPS} time steps")

    model.save(f"models/{MODEL_TYPE}-{STEPS*i}")
	
    print("Training Done!")
    input("Hit Enter to Continue:")
    env.set_display(True)
    obs = env.reset()
    for _ in range(5000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()


if __name__ == "__main__":
    main()