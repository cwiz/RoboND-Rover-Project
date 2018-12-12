## Project: Search and Sample Return

This is Udacity Robotics Software Engineer Nanodegree project. It involves two basic stages of robot's operations: perception and actutation. In this project we are given a simulator of Mars rover with task of mapping terrain and collecting yellow stones. Project is graded by performance metrics which are % of mapped terrain and fidelity (accuracy of mapping).

### Steps to complete a project  

* [Download](https://github.com/udacity/RoboND-Rover-Project) the simulator and take data in "Training Mode". Links are available at 
* Draft workflow on recorded test scenario on a [Jupyter Notebook](https://github.com/cwiz/RoboND-Rover-Project/blob/master/code/Rover_Project_Test_Notebook.ipynb)
* Modify code/perception.py and code/decision.py 

### Autonomous Navigation and Mapping

* We recieve a camera image, and x, y coordinates along with global yaw angle from a rover
* We use vision segmentation and perspective transform to build a navigable map of environment
* We use decision tree to generate router commands

### Results

This project requires 75% accuracy and 60% fidelity metrics.

![results](https://github.com/cwiz/RoboND-Rover-Project/blob/master/output/minimum_requirements.PNG?raw=true "Results")