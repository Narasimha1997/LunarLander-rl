from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential
from rl.policy import EpsGreedyQPolicy
from keras.optimizers import Adam

import gym

#Create neural network:

class Network():

    def __init__(self, space, output):
        self.input_space = space
        self.output_space = output

    def model(self):
        model = Sequential()
        print("Input shape : "+str((1, )+self.input_space))
        model = Sequential()
        model.add(Flatten(input_shape=(1,) +self.input_space))
        model.add(Dense(40))
        model.add(Activation('relu'))
        model.add(Dense(40))
        model.add(Activation('relu'))

        model.add(Dense(self.output_space))
        model.add(Activation('linear'))
        print(model.summary())
        return model

#create environment:
env = gym.make("LunarLander-v2")
model = Network(space = env.observation_space.shape, output = env.action_space.n).model()


#create and define policies:
class RLInit():

    def __init__(self, memory, eps, window_size = 1):
        self.memory = memory
        self.eps = eps
        self.window_size = window_size
    
    def getEpsPolicyAndMemory(self):

        return (
          SequentialMemory(limit = self.memory, window_length = self.window_size),
          EpsGreedyQPolicy(eps = self.eps)
        )

class DQNAgentInitializer():

    def __init__(self, model, memory, policy, action):

        self.model = model
        self.memory = memory
        self.policy = policy
        self.action = action

    def getAgent(self):
        agent = DQNAgent(
            model = self.model, 
            policy = self.policy,
            nb_steps_warmup = 10,
            nb_actions = self.action,
            memory = self.memory,
            target_model_update = 1e-2,
            enable_double_dqn=False
        )
        return agent


memory, policy = RLInit(memory = 500000, eps = 1.0, window_size = 1).getEpsPolicyAndMemory()

<<<<<<< HEAD
model.load_weights('./Weights/Episode840.hdf5')
model.save('Eps840.h5')
=======
model.load_weights('./Weights/weights.hdf5')
>>>>>>> 29bdbdfe2117d45f7316cda3de21e1dfaf76fc66

dqn = DQNAgentInitializer(
    model = model,
    memory = memory,
    policy = policy,
    action = env.action_space.n
).getAgent()
dqn.compile(Adam(lr=0.002, decay=2.25e-05), metrics=['mse'])

eps = input('Enter the number of episodes to test: ')

dqn.test(env, nb_episodes = int(eps), visualize=True)

<<<<<<< HEAD
 
=======
 
>>>>>>> 29bdbdfe2117d45f7316cda3de21e1dfaf76fc66
