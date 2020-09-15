import gym
import gym_novel_gridworlds
import numpy as np
import time
import copy
import os

from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
import matplotlib.pyplot as plt

global_timesteps = 0
time_array_0 = []
time_array_1 = []
time_array_2 = []

time_array_final = []


time_array = []


class ReturnTimestepsCallback(BaseCallback):
    """
    Callback for returning timesteps of a model.

    """

    def __init__(self, env, no_of_envs, beam_search_width):
        super(ReturnTimestepsCallback, self).__init__()

        self.env = env
        self.times_called = 0
        self.step_no = 0
        self.no_of_envs = no_of_envs
        self.beam_search_width = beam_search_width

    def _on_step(self):

    	self.step_no += 1
    	return True

    def _on_training_end(self):
    	self.step_no += 1
    	self.times_called += 1
    	global_timesteps = self.step_no
    	print("timesteps required: ", global_timesteps)

    	time_array.append(global_timesteps)

def parameterizing_function(navigation, breaking, crafting, prev_width, prev_height):

	if navigation == 1 and breaking == 1:
		type_of_env = 2 
	if breaking == 1 and crafting == 1:
		type_of_env = 0
	if navigation == 1 and crafting == 1:
		type_of_env = 1
	if navigation == 1 and breaking < 1 and crafting < 1:
		type_of_env = np.random.randint(1,3)
	if breaking == 1 and navigation < 1 and crafting < 1:
		temp = np.random.randint(0,2)
		if temp == 0:
			type_of_env = 0
		if temp == 1:
			type_of_env = 2
	if crafting == 1 and breaking < 1 and navigation < 1:
		type_of_env = np.random.randint(0,2)
	if navigation < 1 and breaking < 1 and crafting < 1:
		type_of_env = np.random.randint(0,3)



	while True:
		width = np.random.randint(7,21)
		if width > prev_width:
			break
		elif prev_width == 20:
			width = 20
			break

	while True:
		height = np.random.randint(7,21)
		if height > prev_height:
			break
		elif prev_height == 20:
			height = 20
			break

	if type_of_env == 0:
		object_present = np.random.randint(0,3) # 0 -> Tree, 1 -> Rock, 2 -> Crafting Table
		if object_present == 0:
			total_trees =1 
			starting_trees = 0
			no_trees = 1
			total_rocks = 0
			no_rocks = 0
			starting_rocks = 0
			crafting_table = 0
		if object_present == 1:
			total_trees = 0 
			starting_trees = 0
			no_trees = 0
			total_rocks = 1
			no_rocks = 1
			starting_rocks = 0
			crafting_table = 0
		if object_present == 2:
			total_trees = 2 
			starting_trees = 2
			no_trees = 0
			total_rocks = 1
			no_rocks = 0
			starting_rocks = 1
			crafting_table = 1

	if type_of_env == 1:

		total_trees = np.random.randint(0,3)

		while True:
			total_trees = np.random.randint(0,3)
			total_rocks = np.random.randint(0,2)
			if total_trees > 0 or total_rocks > 0:
				if total_trees < 2 or total_rocks < 1:
					break

		while True:
			starting_trees = np.random.randint(0,total_trees + 1)
			starting_rocks = np.random.randint(0,total_rocks + 1)
			if starting_trees == 0:
				if starting_rocks is not total_rocks:
					break
			if starting_rocks == 0:
				if starting_trees is not total_trees:
					break

		no_trees = total_trees - starting_trees
		no_rocks = total_rocks - starting_rocks
		crafting_table = 0

	if type_of_env == 2:
		total_trees = np.random.randint(2,6)
		starting_trees = np.random.randint(0, 2)
		no_trees = total_trees - starting_trees

		total_rocks = np.random.randint(1,3)
		if total_rocks == 1:
			starting_rocks = 0
		else:
			starting_rocks = np.random.randint(0, total_rocks - 1)
		no_rocks = total_rocks - starting_rocks
		crafting_table = 1

	return width, height, no_trees, no_rocks, crafting_table, starting_trees, starting_rocks, type_of_env

if __name__ == '__main__':

	no_of_environmets = 3
	beam_search_width = 2
	curriculum_breadth = 3

	width_array = []
	height_array = []
	no_trees_array = []
	no_rocks_array = []
	crafting_table_array = []
	starting_trees_array = []
	starting_rocks_array = []
	type_of_env_array = []
	total_timesteps_array = []

	curriculum_params = [[{'width0': 0, 'height0': 0, 'no_trees': 0, 'no_rocks': 0, 'starting_trees': 0, 'starting_rocks': 0, 'crafting_table': 0, 'navigation': 0, 'breaking': 0, 'crafting': 0, 'type_of_env0': 0}] for _ in range(beam_search_width)]

	env = 'NovelGridworld-v7'
	log_dir = 'results'

	for i in range(no_of_environmets):
		print("Curriculum 0")
		print("Environment: ", i)

		width, height, no_trees, no_rocks, crafting_table, starting_trees, starting_rocks, type_of_env = parameterizing_function(navigation = 0, breaking = 0, crafting = 0, prev_width = 0, prev_height = 0)
		width_array.append(width)
		height_array.append(height)
		no_trees_array.append(no_trees)
		no_rocks_array.append(no_rocks)
		crafting_table_array.append(crafting_table)
		starting_trees_array.append(starting_trees)
		starting_rocks_array.append(starting_rocks)
		type_of_env_array.append(type_of_env)


		source_env = gym.make(env)
		source_env.map_width = width
		source_env.map_height = height
		source_env.goal_env = type_of_env
		source_env.items_quantity = {'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table, 'pogo_stick':0}
		source_env.initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'pogo_stick':0}

		source_env  = DummyVecEnv([lambda: source_env])
		source_env = VecNormalize(source_env, norm_obs=True, norm_reward=False,clip_obs=5.)

		model = PPO2('MlpPolicy', source_env, learning_rate=1e-3, verbose=0)

		callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=800, verbose=1)
		eval_callback = EvalCallback(source_env, callback_on_new_best=callback_on_best, eval_freq = 10000, verbose=1, n_eval_episodes = 20) # Callback functions
		timestep_callback = ReturnTimestepsCallback(source_env, no_of_environmets, beam_search_width)
		model.learn(total_timesteps = 1000, callback = [eval_callback, timestep_callback]) # Learn that model 1000000
		model.save('initial_env_' + str(i))
		total_timesteps_array.append(time_array[0])
		time_array.clear()
		del source_env, model

	temp_list = total_timesteps_array.copy() # Create a temp array
	temp_list.sort() # Sort temp array
	elements_list = [total_timesteps_array.index(temp_list[i]) for i in range(beam_search_width)] # Find indices of the best 'beam_search_width' envs

	for beam_no in range(beam_search_width):
		curriculum_params[beam_no][0]['width0'] = width_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['height0'] = height_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['no_trees'] = no_trees_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['no_rocks'] = no_rocks_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['starting_trees'] = starting_trees_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['starting_rocks'] = starting_rocks_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['crafting_table'] = crafting_table_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['type_of_env0'] = type_of_env_array[elements_list[beam_no]]

		if type_of_env_array[elements_list[beam_no]] == 0:
			curriculum_params[beam_no][0]['navigation'] += 1
		if type_of_env_array[elements_list[beam_no]] == 1:
			curriculum_params[beam_no][0]['breaking'] += 1
		if type_of_env_array[elements_list[beam_no]] == 2:
			curriculum_params[beam_no][0]['crafting'] += 1

		model = PPO2.load('initial_env_' + str(elements_list[beam_no]))
		for env_no in range(no_of_environmets):
			model.save(str(beam_no)+"_"+str(env_no) + "_v0")
		del model

	# for i in range(beam_search_width): # Run them again in order to save them
	# 	curriculum_params[i][0]['width0'] = width_array[elements_list[i]]
	# 	curriculum_params[i][0]['height0'] = height_array[elements_list[i]]
	# 	curriculum_params[i][0]['no_trees'] = no_trees_array[elements_list[i]]
	# 	curriculum_params[i][0]['no_rocks'] = no_rocks_array[elements_list[i]]
	# 	curriculum_params[i][0]['starting_trees'] = starting_trees_array[elements_list[i]]
	# 	curriculum_params[i][0]['starting_rocks'] = starting_rocks_array[elements_list[i]]
	# 	curriculum_params[i][0]['crafting_table'] = crafting_table_array[elements_list[i]]
	# 	curriculum_params[i][0]['type_of_env0'] = type_of_env_array[elements_list[i]]

	# 	if type_of_env_array[elements_list[i]] == 0:
	# 		curriculum_params[i][0]['navigation'] += 1
	# 	if type_of_env_array[elements_list[i]] == 1:
	# 		curriculum_params[i][0]['breaking'] += 1
	# 	if type_of_env_array[elements_list[i]] == 2:
	# 		curriculum_params[i][0]['crafting'] += 1

	# 	source_env = gym.make(env)
	# 	source_env.map_width = width
	# 	source_env.map_height = height
	# 	source_env.goal_env = type_of_env
	# 	source_env.items_quantity = {'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table, 'pogo_stick':0}
	# 	source_env.initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'pogo_stick':0}

	# 	source_env  = DummyVecEnv([lambda: source_env])
	# 	source_env = VecNormalize(source_env, norm_obs=True, norm_reward=False,clip_obs=5.)
	# 	model = PPO2('MlpPolicy', source_env, learning_rate=1e-3, verbose=0)

	# 	callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=800, verbose=1)
	# 	eval_callback = EvalCallback(source_env, callback_on_new_best=callback_on_best, eval_freq = 10000, verbose=1, n_eval_episodes = 20) # Callback functions
	# 	model.learn(total_timesteps = 1000, callback = eval_callback) # Learn that model 1000000
	# 	for environment_number in range(no_of_environmets):
	# 		model.save(str(i)+"_"+str(environment_number) + "_v0")
	# 	del source_env, model

	width_array.clear()
	height_array.clear()
	no_trees_array.clear()
	no_rocks_array.clear()
	starting_trees_array.clear()
	starting_rocks_array.clear()
	crafting_table_array.clear()

	width_after = []
	height_after = []
	no_trees_after = []
	no_rocks_after = []
	starting_trees_after = []
	starting_rocks_after = []
	crafting_table_after = []
	type_of_env_after = []

	curriculum_params_1 = [[{'width0': 0, 'height0': 0, 'no_trees': 0, 'no_rocks': 0, 'starting_trees': 0, 'starting_rocks': 0, 'crafting_table': 0, 'navigation': 0, 'breaking': 0, 'crafting': 0, 'type_of_env0' : 0,\
	'width1': 0, 'height1': 0, 'no_trees1': 0, 'no_rocks1': 0, 'starting_trees1': 0, 'starting_rocks1': 0, 'crafting_table1': 0, 'type_of_env1' : 0}] for _ in range(beam_search_width)]

	curriculum_params_2 = [[{'width0': 0, 'height0': 0, 'no_trees': 0, 'no_rocks': 0, 'starting_trees': 0, 'starting_rocks': 0, 'crafting_table': 0, 'navigation': 0, 'breaking': 0, 'crafting': 0, 'type_of_env0' : 0, \
	'width1': 0, 'height1': 0, 'no_trees1': 0, 'no_rocks1': 0, 'starting_trees1': 0, 'starting_rocks1': 0, 'crafting_table1': 0, 'type_of_env1' : 0, \
	'width2': 0, 'height2': 0, 'no_trees2': 0, 'no_rocks2': 0, 'starting_trees2': 0, 'starting_rocks2': 0, 'crafting_table2': 0, 'type_of_env2' : 0}] for _ in range(beam_search_width)]


	for curriculum_number in range(1,curriculum_breadth):
		width_after.clear()
		height_after.clear()
		no_trees_after.clear()
		no_rocks_after.clear()
		starting_trees_after.clear()
		starting_rocks_after.clear()
		crafting_table_after.clear()
		type_of_env_after.clear()

		for beam_number in range(beam_search_width):
			for environment_number in range(no_of_environmets):

				print("Curriculum number: ", curriculum_number)
				print("Beam number: ", beam_number)
				print("Environment number: ", environment_number)

				if curriculum_number == 1:
					no_navigation = curriculum_params[beam_number][0]['navigation']
					no_breaking = curriculum_params[beam_number][0]['breaking']
					no_crafting = curriculum_params[beam_number][0]['crafting']
					prev_width = curriculum_params[beam_number][0]['width' + str(curriculum_number - 1)]
					prev_height = curriculum_params[beam_number][0]['height' + str(curriculum_number - 1)]
				if curriculum_number == 2:
					no_navigation = curriculum_params_1[beam_number][0]['navigation']
					no_breaking = curriculum_params_1[beam_number][0]['breaking']
					no_crafting = curriculum_params_1[beam_number][0]['crafting']
					prev_width = curriculum_params_1[beam_number][0]['width' + str(curriculum_number - 1)]
					prev_height = curriculum_params_1[beam_number][0]['height' + str(curriculum_number - 1)]

				width, height, no_trees, no_rocks, crafting_table, starting_trees, starting_rocks, type_of_env = parameterizing_function(navigation = no_navigation, breaking = no_breaking, crafting = no_crafting, prev_width = prev_width, prev_height = prev_height)
				width_after.append(width)
				height_after.append(height)
				no_trees_after.append(no_trees)
				no_rocks_after.append(no_rocks)
				crafting_table_after.append(crafting_table)
				starting_trees_after.append(starting_trees)
				starting_rocks_after.append(starting_rocks)
				type_of_env_after.append(type_of_env)

				source_env = gym.make(env)
				source_env.map_width = width
				source_env.map_height = height
				source_env.goal_env = type_of_env
				source_env.items_quantity = {'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table, 'pogo_stick':0}
				source_env.initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'pogo_stick':0}

				source_env  = DummyVecEnv([lambda: source_env])
				source_env = VecNormalize(source_env, norm_obs=True, norm_reward=False,clip_obs=5.)

				model = PPO2.load(str(beam_number)+"_"+str(environment_number) + "_v" + str(curriculum_number - 1))
				model.set_env(source_env)

				callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=800, verbose=1)
				eval_callback = EvalCallback(source_env, callback_on_new_best=callback_on_best, eval_freq = 10000, verbose=1, n_eval_episodes = 20) # Callback functions
				timestep_callback = ReturnTimestepsCallback(source_env, no_of_environmets, beam_search_width)
				model.learn(total_timesteps = 1000, callback = [eval_callback, timestep_callback]) # Learn that model 1000000
				model.save(str(beam_number)+ "_" + str(environment_number) + "_v" + str(curriculum_number - 1))
				del source_env, model
				if curriculum_number == 1:
					time_array_1.append(time_array[0])
					time_array.clear()
				elif curriculum_number == 2:
					time_array_2.append(time_array[0])
					time_array.clear()
		
		elements_list.clear()
		if curriculum_number == 1:
			temp_list = time_array_1.copy()
			temp_list.sort()
			elements_list = [time_array_1.index(temp_list[i]) for i in range(beam_search_width)]
		elif curriculum_number == 2:
			temp_list = time_array_2.copy()
			temp_list.sort()
			elements_list = [time_array_2.index(temp_list[i]) for i in range(beam_search_width)]

		for i in range(beam_search_width):
			quotient = elements_list[i] // no_of_environmets
			remainder = elements_list[i] % no_of_environmets

			if curriculum_number == 1:
				curriculum_params_1[i] = copy.deepcopy(curriculum_params[quotient])
				curriculum_params_1[i][0].update({'width'+str(curriculum_number) : width_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'height'+str(curriculum_number) : height_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'no_trees'+str(curriculum_number) : no_trees_after[elements_list[i]]}) 
				curriculum_params_1[i][0].update({'no_rocks'+str(curriculum_number) : no_rocks_after[elements_list[i]]}) 
				curriculum_params_1[i][0].update({'starting_trees'+str(curriculum_number) : starting_trees_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'starting_rocks'+str(curriculum_number) : starting_rocks_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'crafting_table'+str(curriculum_number) : crafting_table_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'type_of_env'+str(curriculum_number) : type_of_env_after[elements_list[i]]})

				if type_of_env_after[elements_list[i]] == 0:
					curriculum_params_1[i][0]['navigation'] += 1
				elif type_of_env_after[elements_list[i]] == 1:
					curriculum_params_1[i][0]['breaking'] += 1
				elif type_of_env_after[elements_list[i]] == 2:
					curriculum_params_1[i][0]['crafting'] += 1

			if curriculum_number == 2:
				curriculum_params_2[i] = copy.deepcopy(curriculum_params_1[quotient])
				curriculum_params_2[i][0].update({'width'+str(curriculum_number) : width_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'height'+str(curriculum_number) : height_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'no_trees'+str(curriculum_number) : no_trees_after[elements_list[i]]}) 
				curriculum_params_2[i][0].update({'no_rocks'+str(curriculum_number) : no_rocks_after[elements_list[i]]}) 
				curriculum_params_2[i][0].update({'starting_trees'+str(curriculum_number) : starting_trees_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'starting_rocks'+str(curriculum_number) : starting_rocks_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'crafting_table'+str(curriculum_number) : crafting_table_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'type_of_env'+str(curriculum_number) : type_of_env_after[elements_list[i]]})

				if type_of_env_after[elements_list[i]] == 0:
					curriculum_params_2[i][0]['navigation'] += 1
				elif type_of_env_after[elements_list[i]] == 1:
					curriculum_params_2[i][0]['breaking'] += 1
				elif type_of_env_after[elements_list[i]] == 2:
					curriculum_params_2[i][0]['crafting'] += 1

			model = PPO2.load(str(quotient) + "_" + str(remainder) + "_v" + str(curriculum_number - 1))

			if curriculum_number is not curriculum_breadth-1:
				for j in range(no_of_environmets):
					model.save(str(i) + "_" + str(j) + "_v" + str(curriculum_number))
			if curriculum_number == curriculum_breadth-1:
				model.save(str(i) + "_final_v" + str(curriculum_number))
			del model

	for final_env in range(beam_search_width):

		print("Final Env!")
		print("Beam no: ", final_env)

		source_env = gym.make(env)
		source_env.map_width = 20
		source_env.map_height = 20
		source_env.goal_env = 2
		source_env.items_quantity = {'tree': 5, 'rock': 2, 'crafting_table': 1, 'pogo_stick':0}
		source_env.initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'pogo_stick':0}

		source_env  = DummyVecEnv([lambda: source_env])
		source_env = VecNormalize(source_env, norm_obs=True, norm_reward=False,clip_obs=5.)

		model = PPO2.load(str(final_env)+"_final_v" + str(curriculum_breadth-1))
		model.set_env(source_env)

		callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=800, verbose=1)
		eval_callback = EvalCallback(source_env, callback_on_new_best=callback_on_best, eval_freq = 10000, verbose=1, n_eval_episodes = 20) # Callback functions
		timestep_callback = ReturnTimestepsCallback(source_env, no_of_environmets, beam_search_width)
		model.learn(total_timesteps = 1000, callback = [eval_callback, timestep_callback]) # Learn that model 10000000
		model.save(str(final_env)+"_final_v" + str(curriculum_breadth))
		time_array_final.append(time_array[0])
		print("time array length:", len(time_array))
		time_array.clear()
		del source_env, model

	print("Curriculum params: ", curriculum_params_2)
	print("\n")
	print("Time taken for the final environment is: ", time_array_final)