# import gym
# import gym_pull
#
# gym_pull.pull('github.com/ppaquette/gym-super-breakout')
#
# env = gym.make('SuperMarioBros-1-1-v0')
# observation = env.reset()
# done = False
# t = 0
# while not done:
#     action = env.action_space.sample()  # choose random action
#     observation, reward, done, info = env.step(action)  # feedback from environment
#     t += 1
#     if not t % 100:
#         print(t, info)
import numpy as np

from neat_core.activation_function import modified_sigmoid_activation

rnd = np.random.RandomState()

print(modified_sigmoid_activation(-1))

for _ in range(100):
    print(rnd.normal(loc=0, scale=0.5))
