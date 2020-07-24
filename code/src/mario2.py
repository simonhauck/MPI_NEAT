import gym
import gym_pull

gym_pull.pull('github.com/ppaquette/gym-super-breakout')

env = gym.make('SuperMarioBros-1-1-v0')
observation = env.reset()
done = False
t = 0
while not done:
    action = env.action_space.sample()  # choose random action
    observation, reward, done, info = env.step(action)  # feedback from environment
    t += 1
    if not t % 100:
        print(t, info)
