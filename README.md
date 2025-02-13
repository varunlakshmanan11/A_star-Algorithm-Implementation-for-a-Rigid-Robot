# A* Algorithm Implementation for a Rigid Robot

This package contains the implementation of the A* Algorithm for a point robot navigating a 2D space with obstacles. It's part of our coursework for ENPM661.

## Installation of Packages

Ensure you have Python installed on your system. Then, run the following command to install the necessary libraries:

* pip install numpy opencv-python

## Libraries Used

* OpenCV: Utilized for visualizing the map and the path exploration process.

* NumPy: It's used here for algebraic operations, defining the map's layout, and identifying free spaces within it and also used for defining the action set.

* Time: This library is used to calculate and display the runtime of the pathfinding process.

* PriorityQueue from queue: Employs a priority queue to effectively manage nodes based on their overall cost (distance from the start node plus the heuristic estimate), ensuring the algorithm always proceeds along the most optimal path.


## Input Start and Goal Nodes

* After initiating the script, you'll be prompted to input the start and goal nodes' coordinates:

* Coordinates Adjustment: Y-coordinates are inverted to align with the given map in the document since using OpenCv.

* Coordinate Range: Enter X coordinates subracting the clearance + robot radius user given and same goes for Y coordinates, considering a User given clearance and robot radius from the canvas edges and near obstacles, to avoid input error.

* Error Handling: Anyway the script validates the inputs and prompts for re-entry if the specified nodes are inside an obstacle, outside canvas bounds, or not valid integers.

* we have used the below inputs for the visualization of the path:

Enter the radius of the robot: 5

Enter the clearance of the robot: 5

Enter the step size between 1 and 10(inclusive): 10

Enter the x-coordinate of the start node(Xs): 11

Enter the y-coordinate of the start node(Ys): 11

Enter the angle of the start_node: 0

Enter the x-coordinate of the goal node(Xg): 1189

Enter the y-coordinate of the goal node(Yg): 489

Enter the angle of the goal node: 0

* Output and run time
 - a video file named A_star_Varun_Lakshmanan_Sai_Jagadeesh_Muralikrishnan.mp4 will be generated with optimal path and explored nodes
 - Runtime : 0.77 Minutes

 * Feel free to use different orientations and nodes to test the script.

## View the Visualization

* The exploration process and the resulting path are visually presented in an OpenCV window.

* The final path from the start to the goal node is highlighted upon the algorithm's completion.

* The entire visualization, including exploration and pathfinding, is saved to a video file named A_star_Varun_Lakshmanan_Sai_Jagadeesh_Muralikrishnan.mp4.

## How It Works

* The program initializes a canvas to represent obstacles and free spaces using algebraic equations and half planes.

* The program prompts users to input the coordinates and orientations for both the start and goal nodes, ensuring they are within the map's bounds and not within any obstacles.

* We have employed the "second method" to mark nodes as visited, utilizing a three-dimensional array to track node states across x, y coordinates and orientation.

* A crucial part of A*'s efficiency, the heuristic function estimates the cost from any node to the goal. Our implementation uses Euclidean distance, optimized with a caching mechanism to reduce computation.

* The robot can move in five different directions determined by the action set. Each action considers the robot's current orientation and updates its position and orientation accordingly.

* For every node, possible actions are evaluated to find new nodes to explore. Each node's cost is calculated, and it is added to the priority queue if it presents a potential path to the goal.

* At the core of the program is the A* algorithm function block, which navigates from the start node to the goal node. It employs a priority queue to process nodes in order of their total estimated cost (current path cost plus heuristic estimate).

* Once the goal node is reached, the algorithm backtracks from the goal to the start node, using a parent tracking system to reconstruct the path taken.

## GitHub Repository Link

Find the same submission on github using the link below

https://github.com/varunlakshmanan11/ENPM-661-Project-3 

## Contributors

| Team Members                 | UID       | Email ID             | GitHUB                              |
|------------------------------|-----------|----------------------|-------------------------------------|
| Varun Lakshmanan             | 120169595 | varunl11@umd.edu     | https://github.com/varunlakshmanan11|
| Sai Jagadeesh Muralikrishnan | 120172243 | jagkrish@umd.edu     | https://github.com/cravotics        |

