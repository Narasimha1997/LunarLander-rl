import gym
import numpy as np
import random
from rl.agents import DQNAgent
import NeuralNetwork
from rl.callbacks import Callback
from gym import wrappers
from keras.optimizers import Adam

env = gym.make('LunarLander-v2')
sd = 16
np.random.seed(sd)
random.seed(sd)
env.seed(sd)

wrappers.Monitor(env = env, directory = './Monitor', force = True)

#env.monitor.start('./monitor')


#use this callback to decay eps value:
class EpsDecayCallback(Callback) :

    def __init__(self, eps_policy, decay_rate = 0.95):
        self.eps_policy = eps_policy
        self.decay_rate = decay_rate
    
    def on_episode_begin(self, episode, logs ={}):
        self.eps_policy.eps*=self.decay_rate

model = NeuralNetwork.Network(space = env.observation_space.shape, output = env.action_space.n).model()
#model.load_weights('World.hdf5')
memory, policy = NeuralNetwork.RLInit(memory = 500000, eps = 1.0, window_size = 1).getEpsPolicyAndMemory()

dqn = NeuralNetwork.DQNAgentInitializer(
    model = model,
    memory = memory,
    policy = policy,
    action = env.action_space.n
).getAgent()


dqn.compile(Adam(lr=0.002, decay=2.25e-05), metrics=['mse'])



dqn.fit(env = env, nb_steps = 300000, visualize = False, callbacks = [
    EpsDecayCallback(eps_policy = policy, decay_rate = 0.975)], verbose=2)


dqn.save_weights(filepath = 'weights.hdf5', overwrite = True)

dqn.test(env, nb_episodes=500, action_repetition=1)
