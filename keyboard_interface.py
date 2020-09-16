import time
import sys

import keyboard

sys.path.append('gym_novel_gridworlds/envs')
from novel_gridworld_v7_env import NovelGridworldV7Env


def print_play_keys(action_str):
    print("Press a key to play: ")
    for key, key_id in KEY_ACTION_DICT.items():
        print(key, ": ", action_str[key_id])


def get_action_from_keyboard():
    while True:
        key_pressed = keyboard.read_key()
        # return index of action if valid key is pressed
        if key_pressed:
            if key_pressed in KEY_ACTION_DICT:
                return KEY_ACTION_DICT[key_pressed]
            elif key_pressed == "esc":
                print("You pressed esc, exiting!!")
                break
            else:
                print("You pressed wrong key. Press Esc key to exit, OR:")
                print_play_keys(env.action_str)


env = NovelGridworldV7Env()

KEY_ACTION_DICT = {
    "w": 0,  # Forward
    "a": 1,  # Left
    "d": 2,  # Right
    "e": 3,  # Break
    "1": 4,  # Crafting
}

obs = env.reset()
env.render()
for i in range(100):
    print_play_keys(env.action_str)
    action = get_action_from_keyboard()  # take action from keyboard
    observation, reward, done, info = env.step(action)

    print("action: ", action, env.action_str[action])
    print("Step: " + str(i) + ", reward: ", reward)
    print("observation: ", len(observation), observation)

    print("inventory_items_quantity: ", len(env.inventory_items_quantity), env.inventory_items_quantity)
    print("items_id: ", len(env.items_id), env.items_id)

    time.sleep(0.2)
    print("")

    env.render()
    if done:
        print("Finished after " + str(i) + " timesteps\n")
        time.sleep(2)
        obs = env.reset()
        env.render()

env.close()
