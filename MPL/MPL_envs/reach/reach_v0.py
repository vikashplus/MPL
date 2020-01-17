import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from MPL.MPL_robot.robot import Robot
import os

# TODO: Action normalization is missing

class sallyReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, noise_scale=0.0):

        # prep
        utils.EzPickle.__init__(self)
        self._noise_scale = noise_scale
        self.initializing = True
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.Rtarget = 0
        self.Ltarget = 0
        self.Rgrasp = 0
        self.Lgrasp = 0
        
        # acquire robot
        self.mpl = Robot(name='sallyReach', model_file=curr_dir+'/reach_v0.xml', config_file=curr_dir+'/reach_v0.config')

        # acquire env
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/reach_v0.xml', 20)
        self.Rtarget = self.sim.model.site_name2id('Rtarget')
        self.Ltarget = self.sim.model.site_name2id('Ltarget')
        self.Rgrasp = self.sim.model.site_name2id('Rgrasp')
        self.Lgrasp = self.sim.model.site_name2id('Lgrasp')
        
        # env ready
        self.initializing = False


    def step(self, a):

        self.mpl.step(self, a, self.frame_skip*self.sim.model.opt.timestep)
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


    # query robot and populate observations
    def get_obs(self):

        # ask robot for sensor data
        sen = self.mpl.get_sensors(self, noise_scale=self._noise_scale)

        # parse sensor data into obs dict
        self.obs_dict = {}
        self.obs_dict['t'] = sen['time']
        self.obs_dict['Tmpl_pos'] = sen['Tmpl_pos']
        self.obs_dict['Rmpl_pos'] = sen['Rmpl_pos']
        self.obs_dict['Lmpl_pos'] = sen['Lmpl_pos']
        self.obs_dict['Tmpl_vel'] = sen['Tmpl_vel']
        self.obs_dict['Rmpl_vel'] = sen['Rmpl_vel']
        self.obs_dict['Lmpl_vel'] = sen['Lmpl_vel']
        self.obs_dict['Rerr'] = self.sim.data.site_xpos[self.Rtarget]-self.sim.data.site_xpos[self.Rgrasp]
        self.obs_dict['Lerr'] = self.sim.data.site_xpos[self.Ltarget]-self.sim.data.site_xpos[self.Lgrasp]

        # vectorize observations
        return np.concatenate([
            self.obs_dict['Tmpl_pos'],
            self.obs_dict['Rmpl_pos'],
            self.obs_dict['Lmpl_pos'],
            self.obs_dict['Tmpl_vel'],
            self.obs_dict['Rmpl_vel'],
            self.obs_dict['Lmpl_vel'],
            self.obs_dict['Lerr'],
            self.obs_dict['Rerr']])


    # evaluate observations
    def _get_score_reward_solved_done(self, obs, act=None):
        Ldist = np.linalg.norm(obs['Lerr'])
        Rdist = np.linalg.norm(obs['Rerr'])

        # print(Rdist, Ldist)
        done = (bool( Ldist > 1.0) or bool(Rdist>1.0)) \
            if not self.initializing else False

        reward_dict = {}
        avg_dist = (Ldist+Rdist)/2.0
        score = -1.* avg_dist
        reward_dict["avg_dist"] = score
        reward_dict["small_bonus"] = 2.0*(Ldist<.1) + 2.0*(Rdist<.1)
        reward_dict["big_bonus"] = 2.0*(Ldist<.1) * 2.0*(Rdist<.1)
        reward_dict["total"] = reward_dict["avg_dist"] + reward_dict["small_bonus"] + reward_dict["big_bonus"] - 50.0 * int(done) 
        
        solved = bool(avg_dist<0.100)
        return score, reward_dict, solved, done


    # reset model
    def reset_model(self):
        raise NotImplementedError # for child class to define 


    # evaluate a path
    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        score, rewards, done = self._get_score_reward_solved_done(obs)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()


    # evaluate policy's success from a collection of paths
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


# Reach at fixed targets
class sallyReachEnvFixed(sallyReachEnv):
    def __init__(self):
        super().__init__()

    def reset_model(self):
        self.sim.model.site_pos[self.Rtarget] = np.array([0.15, 0.2, .6])
        self.sim.model.site_pos[self.Ltarget] = np.array([-0.15, 0.2, .3])
        self.set_state(self.init_qpos, self.init_qvel)
        self.sim.forward()
        return self.get_obs()

# Reach at random targets
class sallyReachEnvRandom(sallyReachEnv):
    def __init__(self):
        super().__init__()

    def reset_model(self):
        self.sim.model.site_pos[self.Rtarget] = self.np_random.uniform(high=[0.5, .5, .6], low=[0, .1, .3])
        self.sim.model.site_pos[self.Ltarget] = self.np_random.uniform(high=[0, .5, .6], low=[-.5, .1, .3])
        self.set_state(self.init_qpos, self.init_qvel)
        self.sim.forward()
        return self.get_obs()
