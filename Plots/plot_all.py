import numpy as np
import matplotlib.pyplot as plt
import os

num_random_seeds = 9
# seed_arr = ['HF_wo_Fire_HC', 'HF_wo_Fire_AC']

seed_arr = ['LF_wo_Fire_HC', 'LF_wo_Fire_AC', 'LF_Fire_HC', 'LF_Fire_AC', 'HF_wo_Fire_HC', 'HF_wo_Fire_AC', 'HF_wo_Fire_A2C', 'HF_Fire_HC', 'HF_Fire_AC']
data_total_timesteps = [[] for _ in range(num_random_seeds)]
data_total_reward = [[] for _ in range(num_random_seeds)]
data_std_dev = [[] for _ in range(num_random_seeds)]
data_total_episodes = [[] for _ in range(num_random_seeds)]
data_final_timesteps = [[] for _ in range(num_random_seeds)]
data_final_reward = [[] for _ in range(num_random_seeds)]
data_final_std_dev = [[] for _ in range(num_random_seeds)]

for i in range(num_random_seeds):

	experiment_file_name_curriculum_shifted_timestep_arr = 'curriculum_shifted_timestep_arr'
	path_to_save_curriculum_shifted_timestep_arr = seed_arr[i] + os.sep + experiment_file_name_curriculum_shifted_timestep_arr + '.npz'
	temp = np.load(path_to_save_curriculum_shifted_timestep_arr)
	data_total_timesteps[i] = temp['curriculum_timesteps']

	experiment_file_name_curriculum_shifted_reward_arr = 'curriculum_shifted_reward_arr'
	path_to_save_curriculum_shifted_reward_arr = seed_arr[i] + os.sep + experiment_file_name_curriculum_shifted_reward_arr + '.npz'
	temp = np.load(path_to_save_curriculum_shifted_reward_arr)
	data_total_reward[i] = temp['curriculum_reward']

	experiment_file_name_std_dev_curr = 'std_dev_curr'
	path_to_save_std_dev_curr = seed_arr[i] + os.sep + experiment_file_name_std_dev_curr + '.npz'
	temp = np.load(path_to_save_std_dev_curr)
	data_std_dev[i] = temp['curriculum_std_dev']

	experiment_file_name_final_timestep_arr = 'final_timestep_arr'
	path_to_save_final_timestep_arr = seed_arr[i] + os.sep + experiment_file_name_final_timestep_arr + '.npz'
	temp = np.load(path_to_save_final_timestep_arr)
	data_final_timesteps[i] = temp['final_timesteps']

	experiment_file_name_final_reward_arr = 'final_reward_arr'
	path_to_save_final_reward_arr = seed_arr[i] + os.sep + experiment_file_name_final_reward_arr + '.npz'
	temp = np.load(path_to_save_final_reward_arr)
	data_final_reward[i] = temp['final_reward']

	experiment_file_name_std_dev_final = 'std_dev_final'
	path_to_save_std_dev_final = seed_arr[i] + os.sep + experiment_file_name_std_dev_final + '.npz'
	temp = np.load(path_to_save_std_dev_final)
	data_final_std_dev[i] = temp['final_std_dev']

	SMALL_SIZE = 16
	MEDIUM_SIZE = 16
	BIGGER_SIZE = 16

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	plt.figure(figsize=(7.8,4.8))

	# plt.rcParams.update({'font.size': 16})
	if i<=6:
		plt.rcParams.update({'legend.loc':'lower right'})
	else:
		plt.rcParams.update({'legend.loc':'upper left'})

	plt.rcParams.update({'lines.markersize': 8})
	# plt.figure(figsize=(8.5,6))
	# print('marker size: ',  plt.rcParams['lines.markersize'] ** 2)
	fig, ax = plt.subplots(figsize=(12,7.5))

	ax.scatter(data_total_timesteps[i], data_total_reward[i], antialiased=True, color='b')
	# ax.fill_between(data_total_timesteps[i], data_total_reward[i]-data_std_dev[i], data_total_reward[i]+data_std_dev[i], alpha=0.2, antialiased=True, color='b')
	ax.errorbar(data_total_timesteps[i], data_total_reward[i], yerr = data_std_dev[i])
	# ax.plot(curriculum_shifted_timestep_arr[:min_x-19], curriculum_shifted_reward_arr[:min_x-19], label = 'learning through curriculum')

	ax.scatter(data_final_timesteps[i], data_final_reward[i], antialiased=True, color='r')
	ax.errorbar(data_final_timesteps[i], data_final_reward[i], yerr = data_final_std_dev[i])

	# ax.fill_between(data_final_timesteps[i], data_final_reward[i]-data_final_std_dev[i], data_final_reward[i]+data_final_std_dev[i], alpha=0.2, antialiased=True, color='r')
	plt.ylim([-900,900])
	plt.xlim([0,1e8])

	# ax.plot(curriculum_shifted_timestep_arr, curriculum_shifted_reward_arr, label = 'learning through curriculum')
	plt.xlabel('Timesteps')
	plt.ylabel('Average Reward')
	ax.legend()
	# plt.show()
	plt.savefig(seed_arr[i])
	# ax.clf()