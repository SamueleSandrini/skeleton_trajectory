# 🦾 Skeleton Trajectory Repository

Welcome to the **Skeleton Trajectory Repository** project of the **CARI with Vis4Mechs Lab**

<p align="center">
  <img height="200" src="http://schoolcommunity.altervista.org/uni/immagini/logo.PNG">
</p>

This repository develops a ROS-based framework to identify human keypoints and reconstruct them in 3D within a collaborative robotic cell, enabling effective human movement tracking.

## 🚀 Start the Skeleton Tracking Node
The entire framework can be run in a simplified way via a launcher file that automatically handles the different nodes:

```bash
roslaunch skeleton_trajectory skeleton_trajectory.launch
```

## ⚙️ Launch Parameters
The following parameters are available:
- **kalmanBase**: specifies the filter type.
  - `kalmanBase:=True`: the default condition, filters the points independently (assuming constant acceleration in Cartesian space).
  - `kalmanBase:=False`: applies filtering based on a model of the person’s limb kinematics (assuming constant acceleration in joint space).

## 📋 Requirements
- Create a [venv](https://docs.python.org/3/library/venv.html):
  ```bash
  python -m venv /path/to/new/virtual/environment
  ```
- [Install requirements](https://stackoverflow.com/questions/7225900/how-can-i-install-packages-using-pip-according-to-the-requirements-txt-file-from) (pip or pip3) (check path to requirements): 
  ```bash
  pip3 install -r requirements.txt
  ```
- Activate the virtual environment: 
  ```bash
  source path_to_venv/bin/activate
  ```
  
## 👥 Maintainers
- Samuele Sandrini, [SamueleSandrini](https://github.com/SamueleSandrini)
- Manuel Beschi, [ManuelBeschi](https://github.com/ManuelBeschi)
