import sys
import gym
import pylab
import random
import os
import numpy as np
import copy
from keras.models import load_model
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm

load = False

EPISODES = 1000  # Maximum number of episodes


# DQN Agent for the Cartpole
# Q function approximation with NN, experience replay, and target network
class DQNAgent:
    # Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_size, action_size, n_nodes_l1=16, n_nodes_l2=0,
                 discount_factor=0.95, learning_rate=0.005,
                 memory_size=1000, target_update_frequency=1):

        self.check_solve = False  # If True, stop if you satisfy solution confition
        self.render = False  # If you want to see Cartpole learning, then change to True

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        ################################################################################
        ################################################################################
        # Set hyper parameters for the DQN. Do not adjust those labeled as Fixed.
        self.discount_factor = discount_factor  # 0.95
        self.learning_rate = learning_rate  # 0.005
        self.epsilon = 0.02  # Fixed
        self.batch_size = 32  # Fixed
        self.memory_size = memory_size  # 1000
        self.train_start = 1000  # Fixed
        self.target_update_frequency = target_update_frequency  # 1
        ################################################################################
        ################################################################################

        # Number of test states for Q value plots
        self.test_state_no = 10000

        # Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        # Create main network and target network (using build_model defined below)
        self.model = self.build_model(n_nodes_l1, n_nodes_l2)
        self.target_model = self.build_model(n_nodes_l1,n_nodes_l2)

        # Initialize target network
        self.update_target_model()

    # Approximate Q function using Neural Network
    # State is the input and the Q Values are the output.
    ###############################################################################
    ###############################################################################
    # Edit the Neural Network model here
    # Tip: Consult https://keras.io/getting-started/sequential-model-guide/
    def build_model(self, n_nodes_l1=16, n_nodes_l2=0):
        model = Sequential()
        model.add(Dense(n_nodes_l1, input_dim=self.state_size, activation='relu',  # 16
                        kernel_initializer='he_uniform'))
        if n_nodes_l2 > 0:
            print("second layer dim: ",n_nodes_l2)
            model.add(Dense(n_nodes_l2, input_dim=self.state_size, activation='relu',
                            kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    ###############################################################################
    ###############################################################################

    # After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, state):
        ###############################################################################
        ###############################################################################
        # Insert your e-greedy policy code here
        # Tip 1: Use the random package to generate a random action.
        # Tip 2: Use keras.model.predict() to compute Q-values from the state.
        if random.uniform(0, 1) < self.epsilon:
            action = random.randrange(self.action_size)  # given
        else:
            action = np.argmax(self.model.predict(state)[0])

        return action
        ###############################################################################
        ###############################################################################

    # Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Add sample to the end of the list

    # Sample <s,a,r,s'> from replay memory
    def train_model(self):
        if len(self.memory) < self.train_start:  # Do not train if not enough memory
            return
        batch_size = min(self.batch_size, len(self.memory))  # Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.memory, batch_size)  # Uniformly sample the memory buffer
        # Preallocate network and target network input matrices.
        update_input = np.zeros(
            (batch_size, self.state_size))  # batch_size by state_size two-dimensional array (not matrix!)
        update_target = np.zeros((batch_size, self.state_size))  # Same as above, but used for the target network
        action, reward, done = [], [], []  # Empty arrays that will grow dynamically

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]  # Allocate s(i) to the network input array from iteration i in the batch
            action.append(mini_batch[i][1])  # Store a(i)
            reward.append(mini_batch[i][2])  # Store r(i)
            update_target[i] = mini_batch[i][
                3]  # Allocate s'(i) for the target network array from iteration i in the batch
            done.append(mini_batch[i][4])  # Store done(i)

        target = self.model.predict(
            update_input)  # Generate target values for training the inner loop network using the network model
        target_val = self.target_model.predict(
            update_target)  # Generate the target values for training the outer loop target network

        # Q Learning: get maximum Q value at s' from target network
        ###############################################################################
        ###############################################################################
        # Insert your Q-learning code here
        # Tip 1: Observe that the Q-values are stored in the variable target
        # # Tip 2: What is the Q-value of the action taken at the last state of the episode?
        target[np.arange(batch_size), np.asarray(action)] = reward + np.logical_not(np.asarray(done)) * self.discount_factor * np.max(target_val, axis=1)

        ###############################################################################
        ###############################################################################

        # Train the inner loop network
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        return

    # Plots the score per episode as well as the maximum q value per episode, averaged over precollected states.
    def plot_data(self, episodes, scores, max_q_mean, name, my_path):
        pylab.figure(0)
        pylab.plot(episodes, max_q_mean, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Average Q Value")
        pylab.savefig(os.path.join(my_path, "qvalues_"+name+".png"))


        pylab.figure(1)
        pylab.plot(episodes, scores[1:], 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Score")
        pylab.savefig(os.path.join(my_path, "scores_"+name+".png"))

        pylab.close('all')

    def save_model(self, model, name, my_path):
        model.save(os.path.join(my_path, name))  # creates a HDF5 file 'my_model.h5'
        return


def main():
    # For CartPole-v0, maximum episode length is 200
    env = gym.make('CartPole-v0')  # Generate Cartpole-v0 environment object from the gym library
    # Get state and action sizes from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    n_node_l1_list = [8, 16, 32, 64]
    n_node_l2_list = [0, 4, 8, 16, 32]
    discount_factor_list = [0.9, 0.95, 0.99]
    learning_rate_list = [0.01, 0.05, 0.1]
    memory_size_list = [1000, 2000, 4000]
    target_update_frequency_list = [1, 10, 100]

    n_node_l1 = n_node_l1_list[0]
    n_node_l2 = n_node_l2_list[0]
    discount_factor = discount_factor_list[1]
    learning_rate = learning_rate_list[1]
    memory_size = memory_size_list[0]
    target_update_frequency = target_update_frequency_list[0]

    for n_node_l1 in n_node_l1_list:
        for n_node_l2 in n_node_l2_list:
            if n_node_l2 >= n_node_l1:
                break

            # Create agent, see the DQNAgent __init__ method for details
            agent = DQNAgent(state_size, action_size, n_node_l1, n_node_l2,
                             discount_factor, learning_rate,
                             memory_size, target_update_frequency)

            # Create agent, see the DQNAgent __init__ method for details
            # agent = DQNAgent(state_size, action_size)

            if load:
                agent.model = load_model('my_model.h5')

            # Collect test states for plotting Q values using uniform random policy
            test_states = np.zeros((agent.test_state_no, state_size))
            max_q = np.zeros((EPISODES, agent.test_state_no))
            max_q_mean = np.zeros((EPISODES, 1))

            done = True
            for i in range(agent.test_state_no):
                if done:
                    done = False
                    state = env.reset()
                    state = np.reshape(state, [1, state_size])
                    test_states[i] = state
                else:
                    action = random.randrange(action_size)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    test_states[i] = state
                    state = next_state

            scores, episodes = [0], []  # Create dynamically growing score and episode counters
            for e in tqdm(range(EPISODES)):
                done = False
                score = 0
                state = env.reset()  # Initialize/reset the environment
                state = np.reshape(state, [1, state_size])  # Reshape state so that to a 1 by state_size
                                                            # two-dimensional array ie. [x_1,x_2] to [[x_1,x_2]]
                # Compute Q values for plotting
                tmp = agent.model.predict(test_states)
                max_q[e][:] = np.max(tmp, axis=1)
                max_q_mean[e] = np.mean(max_q[e][:])

                while not done:
                    if agent.render:
                        env.render()  # Show cartpole animation

                    # Get action for the current state and go one step in environment
                    action = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])  # Reshape next_state similarly to state

                    # Save sample <s, a, r, s'> to the replay memory
                    agent.append_sample(state, action, reward, next_state, done)
                    # Training step
                    if not load:
                        agent.train_model()
                    score += reward  # Store episodic reward
                    state = next_state  # Propagate state

                    if done:
                        # At the end of very episode, update the target network
                        if e % agent.target_update_frequency == 0:
                            agent.update_target_model()
                        # Plot the play time for every episode
                        # scores.append(score)
                        scores.append(scores[-1] * 0.99 + score * 0.01)
                        episodes.append(e)

                        # print("episode:", e, "  score:", score, " q_value:", max_q_mean[e], "  memory length:",
                        #      len(agent.memory))

                        # if the mean of scores of last 100 episodes is bigger than 195
                        # stop training
                        if agent.check_solve:
                            if np.mean(scores[-min(100, len(scores)):]) >= 195:
                                print("solved after", e - 100, "episodes")
                                agent.plot_data(episodes, scores, max_q_mean[:e + 1])
                                sys.exit()

            name = 'model_ep'+str(EPISODES)+'_nl1_'+str(n_node_l1)+'_nl2_'+str(n_node_l2)+'_df_'+str(discount_factor)+'_eta_' + \
                   str(learning_rate)+'_mem_'+str(memory_size)+'_tuf_'+str(target_update_frequency)

            my_path = "./out"
            agent.plot_data(episodes, scores, max_q_mean, name, my_path)
            agent.save_model(agent.model, name, my_path)


if __name__ == "__main__":
    main()

