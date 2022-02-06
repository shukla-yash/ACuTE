import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
# hyperparameters
hidden_size = 10
learning_rate = 1e-3
random_seed = 1
# Constants
GAMMA = 0.99
num_steps = 600
max_episodes = 50000
total_episodes_arr = []
total_timesteps_array = []
total_reward_array = []
avg_reward_array = []
final_timesteps_array = []
final_reward_array = []
final_avg_reward_array = []
task_completion_array = []
def CheckTrainingDoneCallback(reward_array, done_array, env):

    done_cond = False
    reward_cond = False
    if len(done_array) > 30:
        if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
            if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
                done_cond = True

        if done_cond == True:
            if env < 3:
                if np.mean(reward_array[-10:]) > 950:
                    reward_cond = True
            else:
                if np.mean(reward_array[-10:]) > 950:
                    reward_cond = True

        if done_cond == True and reward_cond == True:
            return 1
        else:
            return 0
    else:
        return 0

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=1e-3):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

def a2c(env, i):
    num_inputs = 83
    num_outputs = 5

    if i == 0:
        actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
        ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    else:
        path = '_c' + str(0) + '_b' + str(0) + '_e' + str(i-1)
        path_to_load = 'results' + os.sep + 'NovelGridworld-v0' + path + '.pt'
        actor_critic = torch.load(path_to_load)
        ac_optimizer = optim.Adam(actor_critic.parameters(), lr = learning_rate)
        print("loaded model")
        time.sleep(1.0)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0
    episode = 0

    reward_sum = 0
    reward_arr = []
    avg_reward = []
    done_arr = []
    env_flag = 0

    while True:
        log_probs = []
        values = []
        rewards = []
        episode += 1

        state = env.reset()
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            
            if done or steps == num_steps-1:
                if done == True:
                    done_arr.append(1)
                    task_completion_array.append(1)
                elif steps == num_steps-1:
                    done_arr.append(0)
                    task_completion_array.append(0)
                
                print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")

                reward_sum = np.sum(rewards)
                reward_arr.append(reward_sum)
                avg_reward.append(np.mean(reward_arr[-40:]))

                total_reward_array.append(reward_sum)
                avg_reward_array.append(np.mean(reward_arr[-40:]))
                total_timesteps_array.append(steps)
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break
        
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
        env_flag = 0
        if i < 3:
            env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)

        if episode == max_episodes or env_flag == 1:
            path = '_c' + str(0) + '_b' + str(0) + '_e' + str(i)
            path_to_save = 'results' + os.sep + 'NovelGridworld-v0' + path + '.pt'
            torch.save(actor_critic, path_to_save)
            print("saved model")
            time.sleep(2.0)
            total_episodes_arr.append(episode)

            if i == 3:
                log_dir = 'logs_' + str(random_seed)
                os.makedirs(log_dir, exist_ok = True)
                total_timesteps_array_2 = np.asarray(total_timesteps_array)
                print("size total_timesteps_array: ", total_timesteps_array_2.shape)

                total_reward_array_2 = np.asarray(total_reward_array)
                print("size total_reward_array: ", total_reward_array_2.shape)

                avg_reward_array_2 = np.asarray(avg_reward_array)
                print("size avg_reward_array: ", avg_reward_array_2.shape)

                total_episodes_arr_2 = np.asarray(total_episodes_arr)
                print("size total_episodes_arr: ", total_episodes_arr_2.shape)

                task_completion_arr_2 = np.asarray(task_completion_array)

                experiment_file_name_total_timesteps = 'randomseed_' + str(random_seed) + '_total_timesteps'
                path_to_save_total_timesteps = log_dir + os.sep + experiment_file_name_total_timesteps + '.npz'

                experiment_file_name_total_reward = 'randomseed_' + str(random_seed) + '_total_reward'
                path_to_save_total_reward = log_dir + os.sep + experiment_file_name_total_reward + '.npz'

                experiment_file_name_avg_reward = 'randomseed_' + str(random_seed) + '_avg_reward'
                path_to_save_avg_reward = log_dir + os.sep + experiment_file_name_avg_reward + '.npz'

                experiment_file_name_total_episodes = 'randomseed_' + str(random_seed) + '_total_episodes'
                path_to_save_total_episodes = log_dir + os.sep + experiment_file_name_total_episodes + '.npz'

                experiment_file_name_task_completion = 'randomseed_' + str(random_seed) + '_task_completion_curr'
                path_to_save_task_completion = log_dir + os.sep + experiment_file_name_task_completion + '.npz'

                np.savez_compressed(path_to_save_total_timesteps, curriculum_timesteps = total_timesteps_array_2)
                np.savez_compressed(path_to_save_total_reward, curriculum_reward = total_reward_array_2)
                np.savez_compressed(path_to_save_avg_reward, curriculum_avg_reward = avg_reward_array_2)
                np.savez_compressed(path_to_save_total_episodes, curriculum_episodes = total_episodes_arr_2)
                np.savez_compressed(path_to_save_task_completion, task_completion_curr = task_completion_arr_2)

            break