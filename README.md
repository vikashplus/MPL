# MPL

`MPL` is a collection of environments/tasks using MPL system from APL, simulated with the [Mujoco](http://www.mujoco.org/) physics engine and wrapped in the OpenAI `gym` API.

## Getting Started
`MPL` uses git submodules to resolve dependencies. Please follow steps exactly as below to install correctly.

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
