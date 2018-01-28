from keras.models import Sequential
from keras.layers import Dense, Flatten
from rl.policy import EpsGreedyQPolicy
from rl.agents import SARSAAgent, DQNAgent
from rl.memory import SequentialMemory
from keras.layers import Activation 

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
        return model

#use Network2 for higher accuracy, low performance
class Network2():

    def __init__(self, space, output):
        self.input_space = space
        self.output_space = output

    def model(self):
        model = Sequential()
        print("Input shape : "+str((1, )+self.input_space))
        model = Sequential()
        model.add(Flatten(input_shape=(1,) +self.input_space))
        model.add(Dense(80))
        model.add(Activation('relu'))
        model.add(Dense(80))
        model.add(Activation('relu'))
        model.add(Dense(80))
        model.add(Activation('relu'))
        model.add(Dense(80))
        model.add(Activation('relu'))
        model.add(Dense(80))
        model.add(Activation('relu'))

        model.add(Dense(self.output_space))
        model.add(Activation('linear'))
        return model

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
            target_model_update = 1e-2,
            nb_actions = self.action,
            memory = self.memory,
            enable_double_dqn=False
        )
        return agent

        
