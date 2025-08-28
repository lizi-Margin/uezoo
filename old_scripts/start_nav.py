import gym
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE,agents
import time
import cv2

from kbd_input import get_key_action_drone, get_key_action_player



if __name__ == '__main__':
    env = gym.make('UnrealCombatAirPatrol-SuburbNeighborhood_Day-MixedColor-v0')
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(240, 240))
    # env.unwrapped.agents_category=['player']
    env.unwrapped.agents_category=['drone']
    

    count_step = 0
    env.seed(1)
    obs = env.reset()
    t0 = time.time()
    while True:
        action = get_key_action_drone()
        obs, rewards, done, info = env.step([action])
        cv2.imshow('obs',obs[0])
        cv2.waitKey(1)
        count_step += 1
        if done:
            if info['Success']:
                print('Success')
            else:
                print('Failed')
            fps = count_step / (time.time() - t0)
            print('Fps:' + str(fps))
            break
    env.close()