import time
import gym_unrealcv
from gym_unrealcv.envs.base_env import UnrealCv_base
import numpy as np
import cv2
import random
import os
'''
Tasks: The task is to make the agents find the injured person and rescue him. 
The agents are allowed to communicate with others to plan.
The agents observe the environment with their own camera.
The agents are rewarded based on the distance to the other agents.
The episode ends when the agents meet or the maximum steps are reached.
'''

class Track(UnrealCv_base):
    def __init__(self,
                 env_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task TODO: use this file to config task specific parameters
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160),
                 reset_type = 0
                 ):
        super(Track, self).__init__(setting_file=env_file,  # the setting file to define the task
                                         action_type=action_type,  # 'discrete', 'continuous'
                                         observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                         resolution=resolution,
                                    reset_type=reset_type)
        self.count_lost = 0
        self.max_lost_steps = 20
        self.agents_category = ['player']
        self.reward_type = 'dense'  # 'dense', 'sparse'
        self.reward_params = {
            "min_distance": 100,
            "max_direction": 60,
            "max_distance": 750,
            "exp_distance": 250,
            "exp_angle": 0
        }
        self.distance_threshold = self.reward_params["min_distance"]  # distance threshold for collision
        self.tracker_id = self.protagonist_id
        self.target_id = self.protagonist_id+1

    def get_obs(self):
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, self.cam_flag)
        self.obj_poses = obj_poses
        observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)
        return observations

    def step(self, action):
        obs, rewards, done, info = super(Track, self).step(action)
        return obs, rewards, done, info
    
    def step_action(self, actions):
        actions2move, actions2turn, actions2animate = self.action_mapping(actions, self.player_list)
        move_cmds = [self.unrealcv.set_move_bp(obj, actions2move[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2move[i] is not None]
        head_cmds = [self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], actions2turn[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2turn[i] is not None]
        # head_cmds = []
        anim_cmds = [self.unrealcv.set_animation(obj, actions2animate[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2animate[i] is not None]
        # anim_cmds = []
        # hide_cmds = [self.unrealcv.set_hide_obj(obj, return_cmd=True) for i, obj in enumerate(self.player_list)]
        # hide_cmds = [self.unrealcv.set_hide_obj(self.player_list[1], return_cmd=True)]
        self.unrealcv.batch_cmd(move_cmds+head_cmds+anim_cmds, None)
        self.count_steps += 1

        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, [True, False, False, False])
        self.obj_poses = obj_poses
        return None

    def reset(self):
        hide_cmds = [self.unrealcv.set_show_obj(obj, return_cmd=True) for i, obj in enumerate(self.player_list)]
        self.unrealcv.batch_cmd(hide_cmds, None)
        # self.trajectory = []
        # initialize the environment
        observations = super(Track, self).reset()
        target_pos = self.unrealcv.get_obj_location(self.player_list[self.target_id])
        # target_rot = self.unrealcv.get_obj_rotation(self.player_list[self.target_id])
        # tracker_pos = self.unrealcv.get_obj_location(self.player_list[self.tracker_id])
        # tracker_rot = self.unrealcv.get_obj_rotation(self.player_list[self.tracker_id])
        print(target_pos)
        self.unrealcv.nav_to_goal(self.player_list[self.target_id], target_pos)
        time.sleep(1)
        super(Track, self).random_app()
        object_list = self.unrealcv.get_objects()
        ############
        for obj in object_list: #binary mask configure
            if obj == self.player_list[self.target_id]:
                self.unrealcv.set_obj_color(obj, (255, 255, 255))
            else:
                if obj in self.player_list:
                    self.unrealcv.set_obj_color(obj, (0, 255, 0))
                else:
                    random_color= np.random.randint(60, 200, 3)
                    if obj not in self.objects_list and obj not in self.player_list:
                        try:
                            self.unrealcv.set_obj_color(obj, random_color)
                        except:
                            pass
                # if random.random() < 0.9:
                #     if obj not in self.objects_list and obj not in self.player_list:
                #         try:
                #             self.unrealcv.set_obj_color(obj, (0, 0, 0))
                #         except:
                #             pass
                # else:
                #     random_color=color = np.random.randint(100, 255, 3)
                #     if obj not in self.objects_list and obj not in self.player_list:
                #         try:
                #             self.unrealcv.set_obj_color(obj, random_color)
                #         except:
                #             pass
        ##############
        time.sleep(1)
        target_pos = self.unrealcv.get_obj_location(self.player_list[self.target_id])
        # initialize the tracker
        # cam_pos_exp, yaw_exp= self.get_tracker_init_point(target_pos, self.reward_params["exp_distance"])
        cam_pos_exp, yaw_exp= self.get_tracker_init_point(target_pos, random.uniform(400,1200))
        # set tracker location
        tracker_name = self.player_list[self.tracker_id]
        self.unrealcv.set_obj_location(tracker_name, cam_pos_exp)
        self.unrealcv.set_obj_rotation(tracker_name, [0, yaw_exp, 0])

        # reset if cannot see the target at initial frame
        try:
            while self.unwrapped.unrealcv.check_visibility(self.cam_list[self.tracker_id],self.player_list[self.target_id]) == 0:
                print("retrying born spot")
                target_locations = self.sample_init_pose()
                self.unrealcv.set_obj_location(self.player_list[self.target_id], target_locations[0])
                self.unrealcv.set_cam(self.player_list[self.target_id],
                                      self.agents[self.player_list[self.target_id]]['relative_location'],
                                      self.agents[self.player_list[self.target_id]]['relative_rotation'])
                target_pos = self.unrealcv.get_obj_location(self.player_list[self.target_id])
                # initialize the tracker
                cam_pos_exp, yaw_exp = self.get_tracker_init_point(target_pos, self.reward_params["exp_distance"])
                # set tracker location
                tracker_name = self.player_list[self.tracker_id]
                self.unrealcv.set_obj_location(tracker_name, cam_pos_exp)
                self.unrealcv.set_obj_rotation(tracker_name, [0, yaw_exp, 0])
                time.sleep(1)
        except:
            pass

        # update the observation
        observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)
        self.count_lost = 0



        # set npc location
        for i in range(2, len(self.player_list)):
            try:
                self.unrealcv.set_obj_location(
                    self.player_list[i],
                    np.array(target_pos) + random.uniform(50, 250) * np.array([1, 0, 0]) + random.uniform(50, 250) * np.array([0, 1, 0])
                )
            except KeyError:
                pass

        # assert isinstance(Pose, list) and len(Pose) == 6, f"{Pose}"

        return observations
    


    def get_tracker_init_point(self, target_pos, distance, direction=None):
        if direction is None:
            direction = 2 * np.pi * np.random.sample(1)
        else:
            direction = direction % (2 * np.pi)

        dx = float(distance * np.cos(direction))
        dy = float(distance * np.sin(direction))
        x = dx + target_pos[0]
        y = dy + target_pos[1]
        z = target_pos[2]
        cam_pos_exp = [x, y, z]
        yaw = float(direction / np.pi * 180 - 180)

        return [cam_pos_exp, yaw]

    def check_visibility(self, cam_id):
        mask = self.unrealcv.get_image(cam_id, 'object_mask', 'bmp')
        mask, bbox = self.unrealcv.get_bbox(mask, self.player_list[self.target_id], normalize=False)
        mask_percent = mask.sum()/(self.resolution[0] * self.resolution[1])
        return mask_percent

    # def environment_augmentation(self, player_mesh=False, player_texture=False,
    #                              light=False, background_texture=False,
    #                              layout=False, layout_texture=False):
    #     if player_mesh:  # random human mesh
    #         for obj in self.player_list:
    #             if self.agents[obj]['agent_type'] == 'player':
    #                 if self.env_name == 'MPRoom':
    #                     map_id = [2, 3, 6, 7, 9]
    #                     spline = False
    #                     app_id = np.random.choice(map_id)
    #                 else:
    #                     map_id = [1, 2, 3, 4]
    #                     spline = True
    #                     app_id = np.random.choice(map_id)
    #                 self.unrealcv.set_appearance(obj, app_id)
    #             if self.agents[obj]['agent_type'] == 'animal':
    #                 map_id = [2, 5, 6, 7, 11, 12, 16]
    #                 spline = True
    #                 app_id = np.random.choice(map_id)
    #                 self.unrealcv.set_appearance(obj, app_id, spline)
    #     # random light and texture of the agents
    #     if player_texture:
    #         if self.env_name == 'MPRoom':  # random target texture
    #             for obj in self.player_list:
    #                 if self.agents[obj]['agent_type'] == 'player':
    #                     self.unrealcv.random_player_texture(obj, self.textures_list, 3)
    #     if light:
    #         self.unrealcv.random_lit(self.env_configs["lights"])
    #
    #     # random the texture of the background
    #     if background_texture:
    #         self.unrealcv.random_texture(self.env_configs["backgrounds"], self.textures_list, 3)
    #
    #     # random place the obstacle
    #     if layout:
    #         self.unrealcv.clean_obstacles()
    #         self.unrealcv.random_obstacles(self.objects_list, self.textures_list,
    #                                        20, self.reset_area, self.start_area, layout_texture)