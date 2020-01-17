import time
from termcolor import cprint
import numpy as np
from collections import deque
from mujoco_py import load_model_from_path, MjSim

# TODO: populate sim_ids from name

init_robot = '' # saves the persistent connection to the robot
class Robot():
    """
    A unified viewpoint of robot between simulation(sim) and hardware(hdr)
    """

    def __init__(self,
                name:str = 'default_robot',
                model_file = None,
                config_file: str = None,
                device_id = None,
                sensor_cache_maxsize = 5,
                noise_scale = 0.0,
                *args, **kwargs):

        global init_robot
        self.name = name+'(sim)' if device_id is None else name+'(hdr)'
        self.device_id = device_id
        self._sensor_cache_maxsize = sensor_cache_maxsize
        self._noise_scale = noise_scale
        
        # sensor cache
        self._sensor_cache = deque([], maxlen=self._sensor_cache_maxsize)

        # Read and parse configurations
        self.configure_robot(model_file, config_file)
        
        # connect to the hardware
        cprint("Initializing {}".format(self.name), 'white', 'on_grey')
        if self.device_id is not None:
            if init_robot is '':
                # connect to a new robot and save connection 
                # self.init_robot = robot
                raise NotImplementedError
                cprint("Initialization Status:%d" % self.okay(), 'white', 'on_grey')
            else:
                self.robot = init_robot
                cprint("Reusing previours hardware session", 'white', 'on_grey')
            if (not self.okay()):
                cprint("Initialization failed", 'white', 'on_red')

        # Robot's time
        self.time_start = time.time()
        self.time = time.time() - self.time_start
        self.time_render = -1  # time of rendering

    # configure robot
    def configure_robot(self, model_file, config_file):

        print("Reading robot configurations from %s" % config_file)
        with open(config_file, 'r') as f:
            self.robot_config = eval(f.read())

        print("Configuring {}".format(self.name))
        model = load_model_from_path(model_file)
        for name, device in self.robot_config.items():
            for sensor in device['sensor']:
                # print(sensor['sim_id'], model.joint_name2id(sensor['name']))
                sensor['sim_id'] = model.joint_name2id(sensor['name'])
            for actuator in device['actuator']:
                # print(actuator['sim_id'], model.actuator_name2id(actuator['name']))
                actuator['sim_id'] = model.actuator_name2id(actuator['name'])

    # refresh the sensor cache
    def _sensor_cache_refresh(self, env):
        for _ in range(self._sensor_cache_maxsize):
            self.get_sensors(env)

    # get past sensor
    def get_obs_from_cache(self, index=-1):
        assert (index>=0 and index<self._sensor_cache_maxsize) or \
                (index<0 and index>=-self._sensor_cache_maxsize), \
                "cache index out of bound. (cache size is %2d)"%self._sensor_cache_maxsize
        return self._sensor_cache[index]

    def get_sensors(
            self,
            env,
            noise_scale=None, # override the default
    ):
        """
        Get sensor data
        """
        current_sen={}
        noise_scale = self._noise_scale if noise_scale is None else noise_scale

        if self.device_id:
            # record sensor*device['scale']+device['offset']
            # update the sim (qpos, qvel) as per the hardware observations
            raise NotImplementedError
            env.sim.forward() # propagate effects through the sim
        else:
            self.time = env.sim.data.time # update robot time
            current_sen['time']= self.time # record data at this time
            for name, device in self.robot_config.items():
                pos=[]; vel=[]
                for sensor in device['sensor']:
                    p = env.sim.data.qpos[sensor['sim_id']]
                    v = env.sim.data.qvel[sensor['sim_id']]
                    # ensure range
                    p = np.clip(p, sensor['range'][0], sensor['range'][1])
                    v = np.clip(v, sensor['range'][0], sensor['range'][1])
                    # add noise
                    if noise_scale!=0:
                        p += noise_scale*sensor['noise']*env.np_random.uniform(low=-1.0, high=1.0) 
                        v += noise_scale*sensor['noise']*env.np_random.uniform(low=-1.0, high=1.0)
                    # create sensor reading
                    pos.append(p)
                    vel.append(v)
                current_sen[name+'_pos'] = np.array(pos)
                current_sen[name+'_vel'] = np.array(vel)
                
        # cache sensors
        self._sensor_cache.append(current_sen)

        return current_sen

    # enforce velocity limits.
    def ctrl_position_limits(self, ctrl_position, out_space='sim'):
        ctrl_feasible_position = ctrl_position.copy()
        for name, device in self.robot_config.items():
            for actuator in device['actuator']:
                in_id = actuator['sim_id']
                out_id = actuator[out_space+'_id']
                ctrl_feasible_position[out_id] = np.clip(ctrl_position[in_id], 
                    actuator['pos_range'][0], actuator['pos_range'][1])
        return ctrl_feasible_position

    # enforce velocity limits.
    # ALERT: This depends on previous sensor. This is not ideal as it breaks MDP addumptions. Be careful
    def ctrl_velocity_limits(self, ctrl_position, step_duration, out_space='sim'):
        last_obs = self.get_obs_from_cache(-1)
        ctrl_feasible_position = ctrl_position.copy()

        for name, device in self.robot_config.items():
            for act_id, actuator in enumerate(device['actuator']):
                in_id = actuator['sim_id']
                out_id = actuator[out_space+'_id']
                ctrl_desired_vel = (ctrl_position[in_id] - last_obs[name+'_pos'][act_id]) / step_duration
                ctrl_feasible_vel = np.clip(ctrl_desired_vel, actuator['vel_range'][0], actuator['vel_range'][1])
                ctrl_feasible_position[out_id] = last_obs[name+'_pos'][act_id] + ctrl_feasible_vel*step_duration

        return ctrl_feasible_position

    # step the robot env
    def step(self, env, ctrl_desired, step_duration):
        """
        Apply controls and step forward in time
        INPUTS:
            env:            Gym env
            ctrl_desired:   Dessired control to be applied(sim_space)
            step_duration:  Step duration (seconds)
        """

        # Populate sensor cache during startup
        if env.initializing:
            self._sensor_cache_refresh(env)

        # pick output space
        robot_type = 'hdr' if self.device_id else 'sim'
        # enforce velocity limits
        ctrl_feasible = self.ctrl_velocity_limits(ctrl_desired, step_duration, out_space=robot_type)
        # enforce position limits
        ctrl_feasible = self.ctrl_position_limits(ctrl_feasible, out_space=robot_type)

        # Send controls to the robot
        if self.device_id:
            # apply (ctrl_desired-actuator['offfset'])*actuator['scale'] to hardware
            raise NotImplementedError
        else:
            env.do_simulation(ctrl_feasible,
                n_frames=int(step_duration/env.sim.model.opt.timestep))  # render is folded in here

        # synchronize time to maintain step_duration
        if self.device_id:
            time_now = (time.time() - self.time_start)
            time_left_in_step = step_duration - (time_now - self.time)
            if (time_left_in_step > 0.0001):
                time.sleep(time_left_in_step)
            # print("Step took %0.4fs, time left in step %0.4f"% ((time_now-self.time),  time_left_in_step))
        return 1

    def reset(self,
              env,
              reset_pos,
              reset_vel,
              overlay_mimic_reset_pose=False,
              sim_override=False):

        cprint("Resetting {}".format(self.name), 'white', 'on_grey', flush=True)

        # pick output space
        in_id = 'sim_id'
        out_id = 'hdr_id' if self.device_id else 'sim_id'

        # Enforce specs on the request
        #   for actuated dofs => actoator specs
        #   for passive dofs => sensor specs
        feasibe_pos = reset_pos.copy()
        feasibe_vel = reset_vel.copy()
        for name, device in self.robot_config.items():
            if device['actuator']: # actuated dofs
                for actuator in device['actuator']:
                    feasibe_pos[actuator[out_id]] = np.clip(reset_pos[actuator[in_id]], actuator['pos_range'][0], actuator['pos_range'][0])
                    feasibe_vel[actuator[out_id]] = np.clip(reset_vel[actuator[in_id]], actuator['vel_range'][0], actuator['vel_range'][0])
            else: # passive dofs
                for sensor in device['sensor']:
                    feasibe_pos[actuator[out_id]] = np.clip(reset_pos[actuator[in_id]], sensor['range'][0], sensor['range'][0])

        if self.device_id:
            print("Rollout took:", time.time() - self.time_start)
            cprint("Reset> Started", 'white', 'on_grey', flush=True)
            # send request to the actuated dofs 
            # engage other reset mechanisms for passive dofs
            # asset reset as per specs
            raise NotImplementedError
        else:
            # Ideally we should use actuator/ reset mechanism as in the real world 
            # but choosing to directly resetting sim for efficiency
            env.sim.reset()
            self.time = env.sim.data.time # update robot time
            env.sim.data.qpos[:] = feasibe_pos
            env.sim.data.qvel[:] = feasibe_vel
            env.sim.forward()

        # refresh sensor cache before exit
        self._sensor_cache_refresh(env)

    def close(self):
        cprint("Closing {}".format(self.name), 'white', 'on_grey', flush=True)
        if self.device_id:
            raise NotImplementedError
            cprint("Closed (Status: {})".format(status), 'white', 'on_grey', flush=True)

def main():
    import gym
    import MPL
    import pprint

    
    print("Starting Robot===================")
    env = gym.make('SallyReachRandom-v0')
    rob = env.env.mpl

    print("Getting sensor data==============")
    sen = rob.get_sensors(env.env)
    pprint.pprint(sen)

    print("stepping forward=================")
    ctrl = env.env.np_random.uniform(size=env.env.sim.model.nu)
    rob.step(env.env, ctrl, 1.0)

    print("Resetting Robot==================")
    pos = env.env.np_random.uniform(size=env.env.sim.model.nq)
    vel = env.env.np_random.uniform(size=env.env.sim.model.nv)
    rob.reset(env.env, pos, vel)

    print("Closing Robot====================")
    rob.close()

if __name__ == '__main__':
    main()