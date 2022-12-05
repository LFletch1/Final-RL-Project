import numpy as np
import gym
from gym import spaces
import pygame
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_checker import check_env
import math
import time
import random as rd

X_DIM = 800
Y_DIM = 800

class Missile(pygame.sprite.Sprite):
    def __init__(self, type, target_type="All"):
        super().__init__()

        self.type = type
        self.target_type = target_type
        
        if type == "agent":
            self.x_t = 100
            self.y_t = Y_DIM -150
            self.theta = 45
            self.v = 5
        
        else:
            if target_type == "all":
                # Generate random trajectory in the environment for the target missile to take
                # Also generate random velocity for target missile to have. Additionally, noise
                # will be added to the trajectory during updates of positions
                self.x_t = rd.randint(50, X_DIM-50)
                self.y_t = 0
                destination_x = rd.randint(50, X_DIM-50)
                if destination_x == self.x_t:
                    self.theta = -90
                elif destination_x > self.x_t:
                    self.theta = -(180/math.pi) * math.atan(Y_DIM/(destination_x - self.x_t))
                else:
                    self.theta = -90 + (180/math.pi) * math.atan((destination_x - self.x_t)/Y_DIM)
                if self.theta < 0:
                    self.theta += 360 
                self.v = rd.randint(2,4)

            elif target_type == "v_dynamic":
                # Generate random trajectory in the environment for the target missile to take
                # Also generate random velocity for target missile to have
                self.x_t = rd.randint(50, X_DIM-50)
                self.y_t = 0
                destination_x = rd.randint(50, X_DIM-50)
                if destination_x == self.x_t:
                    self.theta = -90
                elif destination_x > self.x_t:
                    self.theta = -(180/math.pi) * math.atan(Y_DIM/(destination_x - self.x_t))
                else:
                    self.theta = -90 + (180/math.pi) * math.atan((destination_x - self.x_t)/Y_DIM)
                if self.theta < 0:
                    self.theta += 360 
                self.v = rd.randint(2,4)

            elif target_type == "dynamic" or target_type == "noisy":
                # Generate random trajectory in the environment for the target missile to take
                self.x_t = rd.randint(50, X_DIM-50)
                self.y_t = 0
                destination_x = rd.randint(50, X_DIM-50)
                if destination_x == self.x_t:
                    self.theta = -90
                elif destination_x > self.x_t:
                    self.theta = -(180/math.pi) * math.atan(Y_DIM/(destination_x - self.x_t))
                else:
                    self.theta = -90 + (180/math.pi) * math.atan((destination_x - self.x_t)/Y_DIM)
                if self.theta < 0:
                    self.theta += 360 
                self.v = 2

            elif target_type == "static":
                # Generate random coordinates for target to be placed at
                self.x_t = rd.randint(50, X_DIM-50)
                self.y_t = rd.randint(50, X_DIM-300)
                self.theta = 270
                self.v = 0
            else:
                print("Not an acceptable target missile type")
                exit()


        self.image = pygame.image.load("sprites/missile3.png")
        self.small_image = pygame.transform.scale(self.image, (int(self.image.get_size()[0]/7),int(self.image.get_size()[1]/7)))
        self.rotate_image = pygame.transform.rotate(self.small_image, self.theta)
        self.rect = self.rotate_image.get_rect()
        # Set collision box coordinates
        self.rect.x = self.x_t
        self.rect.y = self.y_t

    def update_pos(self):
        if self.type == "target":
            if self.target_type == "noisy" or self.target_type == "all":
                if rd.random() < 0.05: # With small probablitly change trajectory of target missile
                    if rd.random() < 0.5:
                        self.theta += 5
                    else:
                        self.theta -= 5
                
        if self.theta < 0:
            self.theta += 360
        elif self.theta > 360:
            self.theta = self.theta % 360
        rad_theta = (math.pi / 180) * self.theta
        delta_x = self.v * math.cos(rad_theta)
        delta_y = self.v * math.sin(rad_theta)
        self.x_t += delta_x
        self.y_t -= delta_y
        # Set collision box coordinates
        self.rect.x = self.x_t
        self.rect.y = self.y_t
        self.rotate_image = pygame.transform.rotate(self.small_image, self.theta)

    def draw(self, screen):
        screen.blit(self.rotate_image, (self.x_t, self.y_t))


class Ground(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("sprites/grass.png")
        self.bigger_image = pygame.transform.scale(self.image, (int(self.image.get_size()[0]*2),int(self.image.get_size()[1])))
        self.rect = self.bigger_image.get_rect()

    def draw(self, screen):
        screen.blit(self.bigger_image, (0,Y_DIM - self.bigger_image.get_size()[1]))


class MissileEnv(gym.Env): 

    def __init__(self, display=True, target_type="all"):
        super().__init__()

        self.target_type = target_type
        self.display = display
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.init_r = 0
        self.total_episode_reward = 0
        self.t = 0

        if display:
            pygame.init()
            self.screen = pygame.display.set_mode([X_DIM, Y_DIM])
            self.background_color = (199, 250, 252)
            self.ground = Ground()

        self.t = 0
        self.agent_missile = Missile("agent", target_type=None)
        self.target_missile = Missile("target", target_type=self.target_type)
        explosion_img = pygame.image.load("sprites/explosion.png")
        self.explosion_img = pygame.transform.scale(explosion_img, (int(explosion_img.get_size()[0]/10),int(explosion_img.get_size()[1]/10)))
        pygame.font.init()

        # Action Space
        self.action_space = spaces.Box(np.array([-15]),np.array([15]))
        
        # State Space
        self.observation_space = spaces.Box(low=np.array([
                                                            0,
                                                            0,
                                                            -50,
                                                            -360,
                                                            0
                                                        ]).astype(np.float32),
                                            high=np.array([
                                                            3000,
                                                            360,
                                                            50,
                                                            360,
                                                            100
                                                        ]).astype(np.float32)
                                                    )


    def step(self, action):
        # Update rocket based on action
        self.agent_missile.theta += action[0] 
        # if self.agent_missile.v + action[1] >= 3 and self.agent_missile.v + action[1] <= 8:
        #     self.agent_missile.v += action[1] # Change v
        self.agent_missile.update_pos()
        self.target_missile.update_pos()

        # Calculate new observation space variables
        x_dif = self.target_missile.x_t - self.agent_missile.x_t
        y_dif = -(self.target_missile.y_t - self.agent_missile.y_t)
        r = math.sqrt((x_dif)**2 + (y_dif)**2)
        lamb = np.arctan2(y_dif, x_dif)
        if lamb < 0:
            lamb += 2 * math.pi
        lamb_deg = (180/math.pi) * lamb
        if lamb_deg < 0:
            lamb_deg += 360
        agent_rad_theta = self.agent_missile.theta * (math.pi/180)
        target_rad_theta = self.target_missile.theta * (math.pi/180)
        r_dot = self.target_missile.v * math.cos(target_rad_theta - lamb) - self.agent_missile.v * math.cos(agent_rad_theta - lamb)
        if r == 0:
            lambda_dot = 0
        else:
            lambda_dot = (self.target_missile.v * math.sin(target_rad_theta - lamb) - self.agent_missile.v * math.sin(agent_rad_theta - lamb)) / (r + 0.000001)


        if self.display:
            self.agent_missile.draw(self.screen)
            self.target_missile.draw(self.screen)
            self.ground.draw(self.screen)
            pygame.display.flip()
            self.screen.fill(self.background_color)
            time.sleep(0.009)

        # Target missile hit ground
        if self.target_missile.y_t > Y_DIM-124:
            if self.display:
                self.screen.blit(self.explosion_img, (self.target_missile.x_t-15, self.target_missile.y_t))
                pygame.display.flip()
                time.sleep(0.1)
            reward = -100
            info = "fail"
            done = True

        elif self.agent_missile.y_t > Y_DIM-124:
            if self.display:
                self.screen.blit(self.explosion_img, (self.agent_missile.x_t-15, self.agent_missile.y_t))
                pygame.display.flip()
                time.sleep(0.1)
            reward = -100
            info = "fail"
            done = True

        # If there is a missile to missile collision
        elif r < 20:
            if self.display:
                self.screen.blit(self.explosion_img, (self.target_missile.x_t, self.target_missile.y_t))
                pygame.display.flip()
                time.sleep(0.1)
            reward = 100
            info = "success"
            # print("BOOM")
            done = True

        else: # No collisions occured
            # Reward based on current distance between agent and target missile (1 / r) and
            # based on change in distance between agent and target missile and
            # a -1 everytime to motivate missile to hit target quicker
            reward = (1 / r) - r_dot - 1
            info = ""
            done = False

        # Time limit of an episode, only neccessary when target missile is stationary
        if self.t == 1000:
            info = "fail"
            done = True

        self.t += 1
        self.total_episode_reward += reward
        # if done:
        #     print(self.t,self.total_episode_reward)

        # observation = np.array([(r / self.init_r), lamb, r_dot, lambda_dot], dtype=np.float32)
        observation = np.array([r/self.init_r, lamb, r_dot, lambda_dot, (self.agent_missile.y_t/Y_DIM)], dtype=np.float32)
        return observation, reward, done, {"result": info}

    def reset(self):
        self.t = 0
        self.total_episode_reward = 0
        self.agent_missile = Missile(type="agent", target_type=None)
        self.target_missile = Missile(type="target", target_type=self.target_type)
        x_dif = self.target_missile.x_t - self.agent_missile.x_t
        y_dif = -(self.target_missile.y_t - self.agent_missile.y_t)
        r = math.sqrt((x_dif)**2 + (y_dif)**2)
        self.init_r = abs(r)
        lamb = np.arctan2(y_dif, x_dif)
        agent_rad_theta = self.agent_missile.theta * (math.pi/180)
        target_rad_theta = self.target_missile.theta * (math.pi/180)
        r_dot = self.target_missile.v * math.cos(target_rad_theta - lamb) - self.agent_missile.v * math.cos(agent_rad_theta - lamb)
        if r == 0:
            lambda_dot = 0
        else:
            lambda_dot = (self.target_missile.v * math.sin(target_rad_theta - lamb) - self.agent_missile.v * math.sin(agent_rad_theta - lamb)) / r

        observation = np.array([r/self.init_r, lamb, r_dot, lambda_dot, (self.agent_missile.y_t/Y_DIM)], dtype=np.float32)
        return observation 

    def set_display(self, show):
        if show:
            if self.display:
                return
            else:
                self.display = True
                pygame.init()
                self.screen = pygame.display.set_mode([X_DIM, Y_DIM])
                self.background_color = (199, 250, 252)
                self.ground = Ground()
        else:
            if self.display:
                self.display = False
                pygame.display.quit()
                pygame.quit()
            else:
                return

def main():
    env = MissileEnv(display=False, target_type="noisy")
    model = A2C("MlpPolicy", env, learning_rate=0.0007, ent_coef=0.01, n_steps=1000, verbose=1)

    MODEL_TYPE = "A2C_all"
    STEPS = 10000
    for i in range(1,101):
        model.learn(total_timesteps=STEPS, reset_num_timesteps=False)
        if i % 10 == 0:
            print(i*STEPS, "timesteps")

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
