# Skeleton Trajectory Repository
Welcome to the repository Project of CARI and Vis4Mechs Laboratory, University of Brescia, Italy !<br>

<p align="center">
  <img height="350" src="http://schoolcommunity.altervista.org/uni/immagini/logo.PNG">
</p>

This repository is a thesis project and aims at developing a ROS-based framework to identify human keypoints and to make the 3d-reconstruction of them inside a collaborative robotic cell, usefull for human movements tracking.

## Start the skeleton node
We summarise here how the framework can be launched in a simplified manner. 
The entire framework can be executed in a simplified manner for a user by means of a launcher file, which takes care of calling the various nodes:

```bash
roslaunch skeleton_trajectory skeleton_trajectory.launch
```

The launch files made available are detailed below.

## Launch files
There are two available launch file one for single camera acquisition and tracking (single_camera.launch) and another one for managing multiple cameras acquisition and tracking (multipla.launch) that incorporates the single camera launch by specifing the proper camera namespace and eventually differente parameters.

## Single Camera Launch File
```bash
roslaunch skeleton_trajectory single_camera.launch
```
The `single_camera.launch` launch launch file is designed to handle the acquisition and skeleton tracking of a single camera. It includes parameters for specifying camera namespaces, topics for color and depth images, camera info, and synchronization options. Additionally, it allows toggling the usage of a Kalman filter in a base version (each keypoints filtered indipendently) or with an advanced option (Kalman Filter with a kinematic model to the limbs) for tracking. Users can specify parameters that are explained in the following table:

### Parameters

| Parameter                | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| `kalman_base`            | Enables or disables the Kalman filter for tracking.              |
| `camera_ns`              | Namespace for the camera.                                        |
| `camera_window`          | Specifies whether to display a GUI window for camera visualization.|
| `camera_color_topic_name` | Topic name for color images.                                     |
| `camera_depth_topic_name` | Topic name for depth images.                                     |
| `camera_info_topic_name`  | Topic name for camera information.                               |
| `approx_sync`             | Enables or disables the approximate synchronization of color and depth topics(*). |

(*) By default it is set to False and this means that the colour and depth frame must be synchronised (best case). However, sometimes the frames may be slightly out of synchronisation and therefore approximate synchronisation may have to be used.

### Multi-Camera Launch File
```bash
roslaunch skeleton_trajectory skeleton_trajectory.launch
```
The `skeleton_trajectory.launch` launch file extends functionality to manage multiple cameras for acquisition and tracking. It includes the `single_camera.launch` file multiple times, each with different namespaces for individual cameras. The users parameters are the same of the `single_camera.launch`

In more details:
- **kalmanBase**: it specifies the filter typology.
  * <code>kalmanBase:=True</code> is the default condition and performs filtering with the points managed independently of each other (with assumption of constant acceleration in cartesian space).
  * <code>kalmanBase:=False</code> performed filtering with the model of the limb kinematics of person.

## Requirements
- Create a [https://docs.python.org/3/library/venv.html](venv): 
```bash
python -m venv /path/to/new/virtual/environment
```
- [Install requirements](https://stackoverflow.com/questions/7225900/how-can-i-install-packages-using-pip-according-to-the-requirements-txt-file-from) (pip or pip3) (check path to requirements): 
```bash
pip3 install -r requirements.txt
```
- Source virtual environment: 
```bash
source path_to_venv/bin/activate
```

## Maintainers
- Samuele Sandrini, [SamueleSandrini](https://github.com/SamueleSandrini)
- Manuel Beschi, [ManuelBeschi](https://github.com/ManuelBeschi)
