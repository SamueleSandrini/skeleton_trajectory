# Skeleton Trajectory Repository
Welcome to the repository Project of CARI and Vis4Mechs Laboratory, University of Brescia, Italy !<br>

<p align="center">
  <img height="350" src="http://schoolcommunity.altervista.org/uni/immagini/logo.PNG">
</p>

This repository is a thesis project and aims at developing a ROS-based framework to identify human keypoints and to make the 3d-reconstruction of them inside a collaborative robotic cell, usefull for human muvements tracking.

## Start the skeleton node
The entire framework can be executed in a simplified manner for a user by means of a launcher file, which takes care of calling the various nodes:

```bash
roslaunch skeleton_trajectory skeleton_trajectory.launch
```
## Launch parameters
The following parameter is available:
- **KalmanBase**: it specifies the filter typology. 
  * <code>KalmanBase:=True</code> is the default condition and performs filtering with the points managed independently of each other (with assumption of constant acceleration in cartesian space). 
  * <code>KalmanBase:=False</code> performed filtering with the model of the limb kinematics of person.

## Maintainers
- Samuele Sandrini, [SamueleSandrini](https://github.com/SamueleSandrini)
- Manuel Beschi, [ManuelBeschi](https://github.com/ManuelBeschi)
