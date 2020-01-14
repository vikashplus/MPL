# MPL

`MPL` is a collection of environments/tasks using MPL system from APL, simulated with the [Mujoco](http://www.mujoco.org/) physics engine and wrapped in the OpenAI `gym` API.

## Getting Started
`MPL` uses git submodules to resolve dependencies. Please follow steps exactly as below to install correctly.

0. Ensure you have access these two repositories - [MPL](https://github.com/vikashplus/MPL) and [MPL_sim](https://github.com/vikashplus/MPL_sim.git).

1. Clone this repo with pre-populated submodule dependencies
```
$ git clone --recursive https://github.com/vikashplus/MPL.git
```
2. Update submodules
```
$ cd MPL  
$ git submodule update --remote
```
3. Add repo to pythonpath by updating `~/.bashrc` or `~/.bash_profile`
```
export PYTHONPATH="<path/to/MPL>:$PYTHONPATH"
```
4. Follow install instructions from [mjrl](https://github.com/aravindr93/mjrl) to get model free agents for `MPL'
5. To visualize an env using a random policy
```
python MPL_agents/mjrl/examine_policy.py -e SallyReachRandom-v0
```
5. To visualize an trained policy
```
python MPL_agents/mjrl/examine_policy.py -e SallyReachRandom-v0 -p MPL_agents/mjrl/sallyReachRandom-v0/best_policy.pickle
```