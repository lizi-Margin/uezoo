import time
import gym
import cv2
import time
import os, sys
import subprocess
import numpy as np
from gym_unrealcv.envs.tracking.baseline import PoseTracker, Nav2GoalAgent
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE,agents


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        print("action_space", action_space)

    def act(self, observation, reward, done):
        return self.action_space.sample()


def handle_obs(obs, episode_dir, count_steps):
    rgb = obs[0][..., :3]
    mask = obs[0][..., 3:]

    white_zone = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)
    black_zone = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
    # mask_filtered = np.ones_like(mask) * 255
    mask_filtered = np.ones_like(mask) * 50
    mask_filtered[white_zone] = 0
    mask_filtered[black_zone] = 50

    cv2.imwrite(f"{episode_dir}/{count_steps}_rgb.png", rgb)
    cv2.imwrite(f"{episode_dir}/{count_steps}_mask.png", mask)
    cv2.imwrite(f"{episode_dir}/{count_steps}_mask_filtered.png", mask_filtered)
    cv2.imshow('show', mask_filtered)
    cv2.waitKey(1)



def run_bg_genvid(input_dir):
    cmd = ["python", f"{os.path.dirname(__file__)}/genvid.py"]
    cmd.extend(["--input-dir", input_dir])
    

    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stdout,
        shell=False,
        start_new_session=True
    )
    
    print(f"已启动后台进程，进程ID: {process.pid}")
    return process



if __name__ == '__main__':
    data_dir = f"./data_dir_{time.asctime()}"
    os.makedirs(data_dir, exist_ok=True)
    
    # render = True
    env = gym.make('UnrealTrack-Demo_Roof-ContinuousColorMask-v0')
    # env = gym.make('UnrealTrack-Old_Town-ContinuousColorMask-v0')
    # env = gym.make('UnrealTrack-ContainerYard_Night-ContinuousColorMask-v0')
    env_unwrapped = env.unwrapped
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(240,240))
    env.unwrapped.agents_category=['player'] #choose the agent type in the scene
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.reset()
    
    episode_count = 100
    rewards = 0
    done = False
    for i in range(episode_count):
        env.seed(i)
        env_unwrapped.direction = 2*np.pi/8.0 * env_unwrapped.count_eps
        # env_unwrapped.unrealcv.config_ue(resolution=(240, 240), low_quality=True, disable_all_screen_messages=False)
        obs = env.reset()
        agent_0 = PoseTracker(env.action_space[0], env.unwrapped.reward_params['exp_distance'])
        agent_1 = Nav2GoalAgent(env.action_space[1], env.unwrapped.reset_area, max_len=100)
        print(obs.shape)
        episode_dir = f"{data_dir}/{env_unwrapped.count_eps:d}"
        os.makedirs(episode_dir, exist_ok=True)
        handle_obs(obs, episode_dir, env_unwrapped.count_steps)
        count_step = 0
        t0 = time.time()
        while True:
            obj_poses = env.unwrapped.obj_poses
            # action_0 = agent_0.act(obs, rewards, done)
            # action_1 = agent_1.act(obs, rewards, done)
            action_0 = agent_0.act(obj_poses[0], obj_poses[1])
            action_1 = agent_1.act(obj_poses[1])
            obs, rewards, done, info = env.step([action_0, action_1])
            handle_obs(obs, episode_dir, env_unwrapped.count_steps) 

            # print("info['Done']", info['Done'])
            # print("done", done)
            if info['Done'] or done:
                import json
                with open(f"{episode_dir}/info.json", 'w') as f:
                    json.dump(env_unwrapped.trajectory, f)
                fps = count_step / (time.time() - t0)
                print ('Fps:' + str(fps))
                print("start replay")

                replay_obss = env_unwrapped.replay([np.array([0., 0.]), np.array([0., 0.])], handle_obs, episode_dir)
                run_bg_genvid(episode_dir)
                break
            count_step += 1

    env.close()