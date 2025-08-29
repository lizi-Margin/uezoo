import re
import gym
import cv2
import json
import time
import random
import os, sys
import datetime
import subprocess
import numpy as np
from uhtk.imitation.traj import trajectory
from uhtk.imitation.utils import safe_dump
from uhtk.siri.utils.sleeper import Sleeper
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


def get_latest_obs(input_dir):
    max_index = -1
    max_path = None
    pattern_png = re.compile(r'^(\d+)_rgb\.png$')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            match = pattern_png.match(filename)
            if match:
                frame_index = int(match.group(1))
                if frame_index > max_index:
                    max_path = os.path.join(input_dir, filename)
                    max_index = frame_index
    
    if max_index == -1:
        assert False, "pattern not match"
    
    obs = cv2.imread(max_path)
    return obs


if __name__ == '__main__':
    N_npc = 0
    # fps = 30; dilation = 0.2
    fps = 15; dilation = 0.4
    episode_time = 10
    SAVE_TRAJ = True
    data_dir = f"./data_dir_{datetime.datetime.now().strftime("%y-%m-%d-%H-%S")}"
    os.makedirs(data_dir, exist_ok=True)
    
    env = gym.make('UnrealTrack-HUAWEI_Project-ContinuousColorMask-v0')
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(480,480))
    # env_unwrapped.unrealcv.config_ue(resolution=(240, 240), low_quality=True, disable_all_screen_messages=False)
    env.unwrapped.agents_category=['player'] #choose the agent type in the scene
    env = augmentation.RandomPopulationWrapper(env, 2 + N_npc, 2 + N_npc, random_target=False)
    env.reset()
    
    episode_count = 100
    rewards = 0
    done = False
    for i in range(episode_count):
        if SAVE_TRAJ:
            uhtk_traj = trajectory(traj_limit=500, env_id=0)
            dict_traj = {}; dict_traj['posrot_agent_0'] = []; dict_traj['real_time'] = []
            for i in range(2 + N_npc): dict_traj[f'action_agent_{i}'] = []

        env.seed(i)
        # env.unwrapped.direction = 2*np.pi/8.0 * env.unwrapped.count_eps
        env.unwrapped.unrealcv.set_global_time_dilation(dilation)
        obs = env.reset()
        agent_0 = PoseTracker(env.action_space[0], env.unwrapped.reward_params['exp_distance'])
        agent_1 = Nav2GoalAgent(env.action_space[1], env.unwrapped.reset_area, max_len=100)
        agent_other_npc = [PoseTracker(env.action_space[idx], expected_distance=random.uniform(100, 200), expected_angle=random.uniform(-100, 100)) for idx in range(2, 2 + N_npc)]
        
        episode_dir = f"{data_dir}/{env.unwrapped.count_eps:d}"
        os.makedirs(episode_dir, exist_ok=True)
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
            sleeper = Sleeper(tick=1.0)
            obj_poses = env.unwrapped.obj_poses
            action_0 = agent_0.act(obj_poses[0], obj_poses[1])
            action_1 = agent_1.act(obj_poses[1])
            action_other_npc = [agent_other_npc[idx].act(obj_poses[idx + 2], obj_poses[1]) for idx in range(len(agent_other_npc))]
            actions = [action_0, action_1] + action_other_npc
            env.unwrapped.step_action(actions)

            if SAVE_TRAJ:
                for i, action in enumerate(actions):
                    dict_traj[f'action_agent_{i}'].append(action)
                    uhtk_traj.remember(f'action_agent_{i}', np.array(action))
                
                [tracker_pos, tracker_rot] = env.unwrapped.unrealcv.batch_cmd(
                    [env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0], return_cmd=True), env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0], return_cmd=True)],
                    [env.unwrapped.unrealcv.decoder.string2floats, env.unwrapped.unrealcv.decoder.string2floats]
                )
                Pose = tracker_pos + tracker_rot
                tracker_FRAME = get_latest_obs(episode_dir)
                uhtk_traj.remember('FRAME_obs_agent_0', tracker_FRAME)
                
                dict_traj['posrot_agent_0'].append(Pose)
                uhtk_traj.remember('posrot_agent_0', np.array(Pose))

                real_time = time.time() - t0
                dict_traj['real_time'] = real_time
                uhtk_traj.remember('real_time', np.float32(real_time))

                uhtk_traj.time_shift()

            count_step += 1
            sleeper.sleep()

        fps_ = (fps * episode_time) / (time.time() - t0)
        print ('推算渲染速度相对真实时间Fps (仅参考):' + str(fps_))
        if SAVE_TRAJ:
            with open(f"{episode_dir}/dict_traj.json", 'w') as f:
                json.dump(dict_traj, f, indent=2)
            uhtk_traj.cut_tail()
            safe_dump(uhtk_traj, f"{episode_dir}/traj-hw-0.d")
            
        run_bg_genvid(episode_dir, fps)


    env.close()