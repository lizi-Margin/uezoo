import gym
import cv2
import json
import time
import random
import os, sys
import datetime
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


def handle_obs(episode_dir, count_steps, bmp_path):
    # rgb = cv2.imread(
    #     bmp_path + '_rgb.bmp',
    #     cv2.IMREAD_COLOR
    # )
    mask = cv2.imread(
        bmp_path + '_mask.bmp',
        cv2.IMREAD_COLOR
    )
    # rgb_no_target = cv2.imread(
    #     bmp_path + '_rgb_no_target.bmp',
    #     cv2.IMREAD_COLOR
    # )


    white_zone = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)
    # black_zone = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
    green_zone = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)
    # mask_filtered = np.ones_like(mask) * 255
    mask_filtered = np.ones_like(mask) * 50
    mask_filtered[white_zone] = 0
    # mask_filtered[black_zone] = 50
    mask_filtered[green_zone] = 255

    # cv2.imwrite(f"{episode_dir}/{count_steps}_rgb.png", rgb)
    cv2.imwrite(f"{episode_dir}/{count_steps}_mask_filtered.png", mask_filtered)
    # cv2.imshow('mask_filtered', mask_filtered)
    cv2.imshow('mask', mask)
    # cv2.imshow('rgb', rgb)
    # cv2.imshow('rgb_no_target', rgb_no_target)
    cv2.waitKey(1)



def run_bg_genvid(input_dir, fps):
    cmd = ["python", f"{os.path.dirname(__file__)}/genvid.py"]
    cmd.extend(["--input-dir", input_dir])
    cmd.extend(["--fps", str(fps)])

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
    N_npc = 0
    fps = 30
    episode_time = 10
    data_dir = f"./data_dir_{datetime.datetime.now().strftime("%y-%m-%d-%H-%S")}"
    os.makedirs(data_dir, exist_ok=True)
    
    # render = True
    env = gym.make('UnrealTrack-HUAWEI_Project-ContinuousColorMask-v0')
    # env = gym.make('UnrealTrack-Old_Town-ContinuousColorMask-v0')
    # env = gym.make('UnrealTrack-ContainerYard_Night-ContinuousColorMask-v0')
    env_unwrapped = env.unwrapped
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(480,480))
    # env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(240,240))
    env.unwrapped.agents_category=['player'] #choose the agent type in the scene
    env = augmentation.RandomPopulationWrapper(env, 2 + N_npc, 2 + N_npc, random_target=False)
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
        agent_other_npc = [PoseTracker(env.action_space[idx], expected_distance=random.uniform(100, 200), expected_angle=random.uniform(-100, 100)) for idx in range(2, 2 + N_npc)]

        print(obs.shape)
        action_traj = {'agent_0': [], 'agent_1': []}
        episode_dir = f"{data_dir}/{env_unwrapped.count_eps:d}"
        os.makedirs(episode_dir, exist_ok=True)
        # handle_obs(obs, episode_dir, env_unwrapped.count_steps)
        count_step = 0
        t0 = time.time()

        env.unwrapped.unrealcv.start_record(
            env.unwrapped.cam_list[env.unwrapped.tracker_id],
            f'{episode_dir}/.png',
            episode_time, 
            fps=fps,
            target_to_hide=env.unwrapped.player_list[env.unwrapped.target_id]
        )
        while env.unwrapped.unrealcv.get_record_status(env.unwrapped.cam_list[env.unwrapped.tracker_id]):
            obj_poses = env.unwrapped.obj_poses
            action_0 = agent_0.act(obj_poses[0], obj_poses[1])
            action_1 = agent_1.act(obj_poses[1])
            action_other_npc = [agent_other_npc[idx].act(obj_poses[idx + 2], obj_poses[1]) for idx in range(len(agent_other_npc))]
            env.unwrapped.step_action([action_0, action_1] + action_other_npc)
            action_traj['agent_0'].append(action_0)
            action_traj['agent_1'].append(action_1)

            count_step += 1
            time.sleep(0.1)

        fps_ = (fps * episode_time) / (time.time() - t0)
        print ('推算渲染速度相对真实时间Fps (仅参考):' + str(fps_))
        with open(f"{episode_dir}/info.json", 'w') as f:
            json.dump({'pos':env_unwrapped.trajectory, 'act':action_traj}, f)
        run_bg_genvid(episode_dir, fps)


    env.close()