# 6-DOF-Robot-Arm-in-ROS2-Gazebo-controlled-by-ASL-recognition-with-MediaPipe
Inside is a ROS2 workspace for controlling a robot arm with hand gestures. Simulation environment is Gazebo Classics with ros2_control plugins. Gesture recognition is done through Google's MediaPipe HandLandmarker &amp; a custom Keras MLP

You would need to install gazebo_ros2_control instead of the new gz_ros2_control (for Gazebo Sims). This is running on ROS2 Humble and Ubuntu 22.4 LTS. 

For inverse kinematics, I am using Newton Raphson method with Products of Exponential (POE) representaion. 

You can download everything and colcon build, it should work if you have downloaded all the required modules. The launch file for everything is inside the node_bringup folder, named robot_launch.launch.xml. Launching that file will:
1. Opens Gazebo (there will be an arm and a chessboard).
2. Opens your webcam and display it on a window.
3. Opens the kinematics node and control node.
4. Your left hand is letters and right is numbers. Use ASL to signal a-h and 1-8, if you're holding the combination correctly, after 3 seconds the robot will move.
