import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os


class sallyReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.initializing = True
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        self.right_target = 0
        self.left_target = 0
        self.right_grasp = 0
        self.left_grasp = 0
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/reach_v0.xml', 20)
        self.right_target = self.sim.model.site_name2id('right_target')
        self.left_target = self.sim.model.site_name2id('left_target')
        self.right_grasp = self.sim.model.site_name2id('right_grasp')
        self.left_grasp = self.sim.model.site_name2id('left_grasp')
        self.initializing = False

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()

        score, reward_dict, solved, done = self._get_score_reward_solved_done(self.obs_dict)

        # finalize step
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'rewards': reward_dict,
            'score': score,
            'solved': solved
        }
        return obs, reward_dict['total'], done, env_info

    def get_obs(self):
        self.obs_dict = {}
        self.obs_dict['t'] = self.sim.data.time
        self.obs_dict['qp'] = self.sim.data.qpos.copy()
        self.obs_dict['qv'] = self.sim.data.qvel.copy()
        self.obs_dict['right_err'] = self.sim.data.site_xpos[self.right_target]-self.sim.data.site_xpos[self.right_grasp]
        self.obs_dict['left_err'] = self.sim.data.site_xpos[self.left_target]-self.sim.data.site_xpos[self.left_grasp]

        return np.concatenate([
            self.obs_dict['qp'],
            self.obs_dict['qv'],
            self.obs_dict['left_err'],
            self.obs_dict['right_err']])

    def _get_score_reward_solved_done(self, obs, act=None):
        left_dist = np.linalg.norm(obs['left_err'])
        right_dist = np.linalg.norm(obs['right_err'])

        # print(right_dist, left_dist)
        done = (bool( left_dist > 1.0) or bool(right_dist>1.0)) \
            if not self.initializing else False

        reward_dict = {}
        avg_dist = (left_dist+right_dist)/2.0
        score = -1.* avg_dist
        reward_dict["avg_dist"] = score
        reward_dict["small_bonus"] = 2.0*(left_dist<.1) + 2.0*(right_dist<.1)
        reward_dict["big_bonus"] = 2.0*(left_dist<.1) * 2.0*(right_dist<.1)
        reward_dict["total"] = reward_dict["avg_dist"] + reward_dict["small_bonus"] + reward_dict["big_bonus"] - 50.0 * int(done) 
        
        solved = bool(avg_dist<0.100)
        return score, reward_dict, solved, done


    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        score, rewards, done = self._get_score_reward_solved_done(obs)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    def reset_model(self):
        raise NotImplementedError # for child class to define 

    def evaluate_success(self, paths, logger=None):
        success = 0.0
        for p in paths:
            if np.mean(p['env_infos']['solved'][-4:]) > 0.0:
                success += 1.0
        success_rate = 100.0*success/len(paths)
        if logger is None:
            # nowhere to log so return the value
            return success_rate
        else:
            # log the success
            # can log multiple statistics here if needed
            logger.log_kv('success_rate', success_rate)
            return None

    # --------------------------------
    # get and set states
    # --------------------------------
    def get_env_state(self):
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy())

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.set_state(qp, qv)
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------
    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -90
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30

        self.sim.forward()

    def close_env(self):
        pass


class sallyReachEnvFixed(sallyReachEnv):
    def __init__(self):
        super().__init__()

    def reset_model(self):
        self.sim.model.site_pos[self.right_target] = np.array([0.15, 0.2, .6])
        self.sim.model.site_pos[self.left_target] = np.array([-0.15, 0.2, .3])
        self.set_state(self.init_qpos, self.init_qvel)
        self.sim.forward()
        return self.get_obs()


class sallyReachEnvRandom(sallyReachEnv):
    def __init__(self):
        super().__init__()

    def reset_model(self):
        self.sim.model.site_pos[self.right_target] = self.np_random.uniform(high=[0.5, .5, .6], low=[0, .1, .3])
        self.sim.model.site_pos[self.left_target] = self.np_random.uniform(high=[0, .5, .6], low=[-.5, .1, .3])
        self.set_state(self.init_qpos, self.init_qvel)
        self.sim.forward()
        return self.get_obs()
