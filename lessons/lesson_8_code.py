import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium as gym
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import pandas as pd


def createDNN_keras(nInputs, nOutputs, nLayer, nNodes):
    """
    Function that generates a neural network with the given requirements.

    Args:
        nInputs: number of input nodes
        nOutputs: number of output nodes
        nLayer: number of hidden layers
        nNodes: number nodes in the hidden layers
        
    Returns:
        model: the generated tensorflow model

    """
    
    # Initialize the neural network
    model = Sequential()
    # Input layer
    model.add(Dense(nNodes, input_dim=nInputs, activation='relu'))
    # Hidden layers
    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation='relu'))
    # Output layer
    model.add(Dense(nOutputs, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    
    return model

class TorchModel(nn.Module):
    """
    Class that generates a neural network with PyTorch and specific parameters.

    Args:
        nInputs: number of input nodes
        nOutputs: number of output nodes
        nLayer: number of hidden layers
        nNodes: number nodes in the hidden layers
        
    """

    # Initialize the neural network
    def __init__(self, nInputs, nOutputs, nLayer, nNodes):
        super(TorchModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(nInputs, nNodes))
        for _ in range(nLayer - 1):
            self.layers.append(nn.Linear(nNodes, nNodes))
        self.output = nn.Linear(nNodes, nOutputs)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output(x)
        return x


def mse(network, dataset_input, target):
    """
    Compute the MSE loss function

    """
    
    # Compute the predicted value, over time this value should
    # looks more like to the expected output (i.e., target)
    predicted_value = network(dataset_input)
    
    # Compute MSE between the predicted value and the expected labels
    mse = tf.math.square(predicted_value - target)
    mse = tf.math.reduce_mean(mse)
    
    # Return the averaged values for computational optimization
    return mse


def training_loop(env, neural_net, updateRule, nInputs, nOutputs, nLayer, nNodes, keras=True, eps=1.0, updates=1, episodes=100):
    """
    Main loop of the reinforcement learning algorithm. Execute the actions and interact
    with the environment to collect the experience for the training.

    Args:
        env: gymnasium environment for the training
        neural_net: the model to train 
        updateRule: external function for the training of the neural network
        nInputs: number of input nodes
        nOutputs: number of output nodes
        nLayer: number of hidden layers
        nNodes: number nodes in the hidden layers
        keras: boolean flag for using keras or torch
        eps: epsilon value for epsilon-greedy strategy
        updates: number of updates per episode
        episodes: number of training episodes
        
    Returns:
        averaged_rewards: array with the averaged rewards obtained

    """

    # Initialize the optimizer 
    if keras:
        optimizer = tf.keras.optimizers.Adam()
    else:
        optimizer = optim.Adam(neural_net.parameters())

     
    rewards_list, memory_buffer = [], collections.deque(maxlen=1000)
    averaged_rewards = []
    for ep in range(episodes):

        # Reset the environment and obtain the initial state
        state = env.reset()[0]
        state = np.reshape(state, [1, nInputs])
        ep_reward = 0
        while True:

            # Select the action to perform exploiting an epsilon-greedy strategy
            if keras:
                if np.random.rand() <= eps:
                    action = np.random.choice(nOutputs)
                else:
                    q_values = neural_net.predict(state)
                    action = np.argmax(q_values[0])
            else:
                state_tensor = torch.FloatTensor(state)
                if np.random.rand() <= eps:
                    action = np.random.choice(nOutputs)
                else:
                    q_values = neural_net(state_tensor)
                    action = torch.argmax(q_values).item()

            # Update epsilon value
            eps *= 0.995
            eps = max(0.01, eps)

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, nInputs])
            memory_buffer.append((state, action, reward, next_state, done))
            ep_reward += reward

            # Perform the actual training
            for _ in range(updates):
                DQNupdate(neural_net, keras, memory_buffer, optimizer)

            # Modify the exit condition for the episode
            if done: break

            # Update the current state
            state = next_state

        # Update the reward list to return
        rewards_list.append(ep_reward)
        averaged_rewards.append(np.mean(rewards_list))
        print(f"episode {ep:2d}: mean reward: {averaged_rewards[-1]:3.2f}, eps: {eps:3.2f}")

    # Close the environment and return the rewards list
    env.close()
    return averaged_rewards


def DQNupdate(neural_net, keras, memory_buffer, optimizer, batch_size=32, gamma=0.99):
    """
    Main update rule for the DQN process. Extract data from the memory buffer and update 
    the network computing the gradient.

    """

    if len(memory_buffer) < batch_size: return

    indices = np.random.randint(len(memory_buffer), size=batch_size)
    for idx in indices: 

        # Extract data from the buffer 
        state, action, reward, next_state, done = memory_buffer[idx]

        # Compute the target for the training
        if keras:
            target = neural_net.predict(state)
            if done:
                target[0][action] = reward
            else:
                next_q_values = neural_net.predict(next_state)
                target[0][action] = reward + gamma * np.amax(next_q_values[0])
        else:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            reward_tensor = torch.FloatTensor([reward])
            action_tensor = torch.LongTensor([action])
            if done:
                target = reward_tensor
            else:
                next_q_values = neural_net(next_state_tensor)
                target = reward_tensor + gamma * torch.max(next_q_values)

        # Compute the gradient and perform the backpropagation step using the selected framework
        if keras:
            with tf.GradientTape() as tape:
                q_values = neural_net(state)
                loss = tf.reduce_mean(tf.square(target - q_values))
            grads = tape.gradient(loss, neural_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, neural_net.trainable_variables))
        else:
            optimizer.zero_grad()
            q_values = neural_net(state_tensor)
            loss = nn.functional.mse_loss(q_values[action_tensor], target)
            loss.backward()
            optimizer.step()


def main():
    print("\n************************************************")
    print("*  Welcome to the eighth lesson of the RL-Lab!   *")
    print("*               (Deep Q-Network)                 *")
    print("**************************************************\n")

    training_steps = 50
    
    # setting DNN configuration
    nInputs = 4
    nOutputs = 2
    nLayer = 2
    nNodes = 32

    print("\nTraining torch model...\n")
    rewards_torch = []
    for _ in range(10):
        env = gym.make("CartPole-v1") # , render_mode="human")
        neural_net_torch = TorchModel(nInputs, nOutputs, nLayer, nNodes)
        rewards_torch.append(training_loop(env, neural_net_torch, DQNupdate, nInputs, nOutputs, nLayer, nNodes, keras=False, episodes=training_steps))

    print("\nTraining keras model...\n")
    rewards_keras = []
    for _ in range(10):
        env = gym.make("CartPole-v1") # , render_mode="human")
        neural_net_keras = createDNN_keras(nInputs, nOutputs, nLayer, nNodes)
        rewards_keras.append(training_loop(env, neural_net_keras, DQNupdate, nInputs, nOutputs, nLayer, nNodes, keras=True, episodes=training_steps))

    # plotting the results
    t = list(range(0, training_steps))

    data = {'Environment Step': [], 'Mean Reward': []}
    for _, rewards in enumerate(rewards_torch):
        for step, reward in zip(t, rewards):
            data['Environment Step'].append(step)
            data['Mean Reward'].append(reward)
    df_torch = pd.DataFrame(data)

    data_keras = {'Environment Step': [], 'Mean Reward': []}
    for _, rewards in enumerate(rewards_keras):
        for step, reward in zip(t, rewards):
            data_keras['Environment Step'].append(step)
            data_keras['Mean Reward'].append(reward)
    df_keras = pd.DataFrame(data_keras)

    # Plotting
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.lineplot(data=df_torch, x='Environment Step', y='Mean Reward', label='torch', errorbar='se')
    sns.lineplot(data=df_keras, x='Environment Step', y='Mean Reward', label='keras', errorbar='se')

    # Add title and labels
    plt.title('Comparison Keras and PyTorch on CartPole-v1')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')

    # Show legend
    plt.legend()

    # Show plot
    plt.show()
    plt.savefig('comparison.pdf')


if __name__ == "__main__":
    main()
