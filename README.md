# Skeleton tracking 
[![GitHub Action
Status](https://github.com/SamueleSandrini/skeleton_trajectory/workflows/humble/badge.svg)](https://github.com/SamueleSandrini/skeleton_trajectory)

ROS 2 Package for human tracking.

## Installation

To install and build the Skeleton Tracking package, follow these steps:

1. **Clone the Repository into the 'src' Folder**:

   Open a terminal and execute:

   ```bash
   git clone https://github.com/SamueleSandrini/skeleton_trajectory.git -b ros2
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
2. **Configure the  `config.yaml` file to set the parameters of the `skeleton_filter_node`**. 
  ```yaml
  skeleton_filter_node:
  ros__parameters:
    q_noise: [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
    r_noise: [0.05, 0.05, 0.1]

    custom_publishers: ["pose_array", "centroid"]
    pose_array:
      module: "skeleton_tracking.custom_publishers.pose_array_publisher.PoseArrayPublisher"
    centroid:
      module: "skeleton_tracking.custom_publishers.centroid_publisher.CentroidPublisher"
  ```
  Note that you can write you own custom "publisher" to elaborate the skeleton as you prefer and publish the message that you prefer. You just need to inherit from `CustomPublisherBase` and put add the `package.subpackage.module.classname`.

3. **Run the Launch File**:
  ```bash
  ros2 launch skeleton_tracking skeleton_tracking_bringup.launch.py
  ```
  This launch file initializes the necessary nodes for skeleton tracking.

## Integrating sam-3d-body

If you plan to integrate the sam-3d-body module into this package, you may find the following environment setup useful.
1. Create a Dedicated Conda Environment
  ```bash
	conda create -n sam_3d_body python=3.10 -y
	conda activate sam_3d_body
  ```
2. Automatically Add sam-3d-body to PYTHONPATH
```bash
	mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
	mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
```
Add the repository to your environment’s PYTHONPATH:
```bash
	cat > "$CONDA_PREFIX/etc/conda/activate.d/sam_3d_body.sh" << 'EOF'
	# Add the sam-3d-body repository to PYTHONPATH: 
	export SAM_3D_BODY_ROOT="$HOME/path_to/sam-3d-body"
	export PYTHONPATH="$SAM_3D_BODY_ROOT:$PYTHONPATH"
	EOF
```
3. Install Colcon Inside the Conda Environment
```bash
	pip install colcon-common-extensions
```
4. Useful Debug Utility
To quickly inspect the interpreter used by a node installed in another workspace:
```bash
	head -n 1 ~/projects/perception_ws/install/vision_system/lib/vision_system/vision_system_node
```
## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

## Issues

We welcome your feedback and contributions to enhance this project. Feel free to open an issue. 
Your participation is vital in improving this project, and we appreciate your contributions.

## Maintainer

- [Samuele Sandrini](https://github.com/SamueleSandrini) - samuele.sandrini@polito.it
