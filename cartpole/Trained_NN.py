# Game:           CartPole-v0
# Summary:        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
#                 The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts
#                 upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every
#                 timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees
#                 from vertical, or the cart moves more than 2.4 units from the center.
# Criteria:       CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
# Date:           March 16, 2017
# Note:           To view the tensorboard, paste  'tensorboard --logdir=path/to/log-directory' in terminal
#                 follow the local host

import gym
import random
import numpy as np
import ssl
import tflearn
from tflearn.data_utils import *
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean

#   Choose which game to play
game = 'CartPole-v0'
#   Create environment
env = gym.make(game)
num_games = 5000

def print_score_info(top_scores):

    print 'Number of games: ', len(top_scores)
    print 'Average (mean) top score: ', mean(top_scores)
    print 'Average (median) top score: ', median(top_scores)
    print 'Highest score: ', max(top_scores)
    print 'Lowest score: ', min(top_scores)
#    print 'All top scores: ', top_scores


def run_episode(model=False):

    game_memory = []
    env.reset()
    done = False
    score = 0
    action = env.action_space.sample()
    prev_obs = []
    while done == False:

        observation, reward, done, info = env.step(action)
        score += reward
        #   convert to one-hot (this is the output layer for our neural network)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        if len(prev_obs) > 0 :
            game_memory.append([prev_obs, actionblank])
        prev_obs = observation
        action = env.action_space.sample() if not model else np.argmax(model.predict(observation.reshape(-1, len(observation), 1))[0])

    return game_memory, score


def neural_network(input_size):

    #   input layer
    network = input_data(shape=[None, input_size, 1], name='input')

    #   hidden layer 1
    network = fully_connected(network, 200, activation='relu')
    network = dropout(network, 0.8)

    #   output layer
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return  model


def train_neural_network(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': Y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaiCartPole')

    return model


#   Initially use random actions for training data
#   Source:
#           0, random action
#           1, trained neural network

training_data, scores = run_episode(src, num_games)

'''
print('---Training neural network---')
print_score_info(scores)
print(np.shape(training_data))

model = train_neural_network(training_data)
#model.save('cartpole_nn_model.model')

scores = []
choices = []
for each_game in range(100):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(500):
        #env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))


'''