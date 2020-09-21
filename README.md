# Deep Reinforcement Learning for Mobile Robot Path Planning #
This repository contains the semester project for ROB962.

TODO: Add more description about the repository and how to get started.

## Installation ##

### System requirements ###

* Webots R2020a or newer: https://cyberbotics.com
* Python 3.7
  * IMPORTANT! you must use the distribution from python.org if using `macOS` or `Windows`, only this distribution is supported in Webots R2020a, else the Webots controllers must be recompiled. This also makes it difficult to use virtual environments e.g. with Anaconda.
  * This may not be an issue in Ubuntu where the system-wide Python distribution can be used
* Stable Baselines - Deep Reinforcement Learning algorithms: https://stable-baselines.readthedocs.io/en/master/guide/install.html

### How to install the gym environment ###
The repository contains two gym environments:

* gym-webots-RLenv (Not working atm)
* P9-RL-env (Use this one)

To install the environment, go to the `P9-RL-env` folder and run the pip install command:

```bash
$ cd P9/P9-RL-env/

$ pip install -e .
```

The environment was created using the following guide: (https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)

## How to run the Webots example ##
Assuming you already have WeBots set up with an existing project, go to the project library, and

* copy the `master` folder to the controller folder
* copy the contents of the `world` folder into your project's world folder (P9_v0.0.1.wbt is used currently)

That should be all. You should be able to open the world from WeBots.

## How to modify the example ##
If you want to modify something (e.g. reward function, action space, etc) you need to modify the gym environment, which is at `P9/P9-RL-env/P9_RL_env_v01/envs/P9RLEnv.py`

The master controller uses and manages this environment, therefore the robot needs to be set as having the master controller.

Alternatively, you can set the Webots robot controller to `extern` and run `master.py` from e.g. vscode.

## Authors ##
* Mark Adamik
* Asger Printz Madsen
