import gym
import numpy
from keras.models import load_model


env = gym.make('LunarLander-v2')
model = load_model(filepath = 'Pretrained-model/Eps840.h5')

while True:

    observation = env.reset()
    while True:
        env.render()
        observation = observation.reshape(1, 1, 8)
        observation , reward, done, info = env.step(
            numpy.argmax(model.predict(observation))
        )
        if done: 
            print('Reward : '+str(reward))
            break
