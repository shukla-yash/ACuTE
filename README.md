Readme file for the Source Code of 'ACuTE: Automatic Curriculum Transfer from Simple to Complex Environments'

Overview of the paper:

Despite recent advances in Reinforcement Learning (RL), many problems, especially real-world tasks, remain prohibitively expensive to learn. To address this issue, several  lines  of  research  have  explored  how  tasks,  or  data  samples themselves, can be sequenced into a curriculum to learn a problem that may otherwise be too difficult to learn from scratch. However, generating and optimizing a curriculum in a realistic scenario still requires extensive interactions with the environment. To address this challenge, we formulate the {\it curriculum transfer} problem, in which the schema of a curriculum optimized in a simpler, easy-to-solve environment (e.g., a grid world) is transferred to a complex, realistic scenario (e.g., a physics-based robotics simulation or the real world). We present "ACuTE", Automatic Curriculum Transfer from Simple to Complex Environments, a novel framework to solve this problem, and evaluate our proposed method by comparing it to other baseline approaches (e.g., domain adaptation) designed to speed up learning. We observe that our approach produces improved jumpstart and time-to-threshold performance even when adding task elements that further increase the difficulty of the realistic scenario. Finally, we demonstrate our approach is independent of the learning algorithm used for curriculum generation, and is Sim2Real transferrable to a real world scenario using a physical robot.

The requirements are listed in the file: requirements.txt

To install: pip install -r requirements.txt


The experiments were conducted using a 64-bit Linux Machine, having Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz processor and 126GB RAM memory. 
The maximum duration for running the experiments was set at 24 hours.

Terminology for Source Code:
LF - Low-Fidelity
HF - High-Fidelity
HC - Handcrafted Curriculum
AC - Automated Curriculum
PG - Policy Gradient
AC - Actor Critic (PPO) algorithm

LF_wo_Fire refers to the code written for LF curriculum experiments without 'fire' element
LF_Fire refers to the code written for LF curriculum experiments with 'fire' element

HF_wo_Fire refers to the code written for HF curriculum experiments without 'fire' element
HF_Fire refers to the code written for HF curriculum experiments with 'fire' element

Before running the HF experiments, please copy and paste the 'data' folder in the folder alongside test_curr.py

To generate the automated curriculum in the LF environment without fire, run:
$ python LF_wo_Fire/curr.py

To generate the automated curriculum in the LF environment with fire, run:
$ python LF_Fire/currinsurance on international license.py

To test the handcrafted curriculum in the LF environment without fire, run:
$ python LF_wo_Fire/test_curr.py

To test the handcrafted curriculum in the LF environment with fire, run:
$ python LF_Fire/test_curr.py

To test the handcrafted curriculum in the HF environment without fire, run:
$ python HF_wo_Fire/PG/test_curr.py

To test the handcrafted curriculum in the HF environment with fire, run:
$ python HF_Fire/test_curr.py

To test the automated curriculum transfer method with noisy mapping (HF without Fire):
$ python Noisy_exp/HF_wo_Fire/test_curr.py

To test the automated curriculum transfer method with noisy mapping (HF with Fire):
$ python Noisy_exp/HF_w_Fire/test_curr.py

Running the above programs would generate a log file that stores the number of timesteps, rewards, episodes and other information from the curriculum run and the learning from scratch run. After conducting 10 trials, the results can be plotted using the plot_episode.py file. 

The paper demonstrates results from 10 trails. The experiments are conducted on seeds 1-10


To test the automated params, generate the params using LF_wo_Fire/curr.py or LF_Fire/curr.py and change the params on #47-54 of python LF_wo_Fire/test_curr.py 

To test the automated curriculum in the LF environment without fire, run:
$ python LF_wo_Fire/test_curr.py

To test the automated curriculum in the LF environment with fire, run:
$ python LF_Fire/test_curr.py

To test the automated curriculum in the HF environment without fire (in Policy Gradient), run:
$ python HF_wo_Fire/test_curr.py

To test the automated curriculum in the HF environment without fire (in Advantage Actor Critic), run:
$ python HF_wo_Fire/A2C/test_curr.py

To test the automated curriculum in the HF environment with fire, run:
$ python HF_Fire/test_curr.py