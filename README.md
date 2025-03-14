# Skeleton tracking 

ROS 2 Package for human tracking.

## Installation

To install and build the Skeleton Tracking package, follow these steps:

1. **Clone the Repository into the 'src' Folder**:

   Open a terminal and execute:

   ```bash
   git clone https://github.com/SamueleSandrini/skeleton_trajectory.git -b ros2_wip src/skeleton_trajectory
   cd src/skeleton_trajectory
   pip install -r requirements.txt
   ```
2. **Import Dependencies at the Workspace Level**:
   Navigate back to the workspace root and run:
   ```bash
   vcs import src < src/skeleton_trajectory/dependencies.repos
   ```
3. **Build the Workspace:**
   Use colcon to build the workspace with symbolic links:
   ```bash
   colcon build --symlink-install
   ```

## Usage

To launch the skeleton tracking system:

1. **Configure Camera Topics**:
  Adjust the `config.yaml` file to match your camera's topic names. For example:

  ```yaml
  camera_node:
    ros__parameters:
      color_image_topic: "/head_front_camera/rgb/image_raw"
      depth_image_topic: "/head_front_camera/depth_registered/image_raw"
      camera_info_topic: "/head_front_camera/rgb/camera_info"
      frames_approx_sync: false
  ```
2. **Run the Launch File**:
  ```bash
  ros2 launch skeleton_tracking skeleton_tracking_bringup.launch.py
  ```
  This launch file initializes the necessary nodes for skeleton tracking.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

## Issues

We welcome your feedback and contributions to enhance this project. Feel free to open an issue. 
Your participation is vital in improving this project, and we appreciate your contributions.

## Maintainer

- [Samuele Sandrini](https://github.com/SamueleSandrini) - samuele.sandrini@polito.it
