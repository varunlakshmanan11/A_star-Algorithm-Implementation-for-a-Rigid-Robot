# Importing Necessary libraries.
import numpy as np
import matplotlib.pyplot as plt 
from queue import PriorityQueue
import cv2
import time
import math

# Creating a empty space for drawing graph.
Graph_map = np.ones((500, 1200, 3), dtype=np.uint8)*255
G = np.zeros((500, 1200, 12), dtype=np.uint8)

# Center of the hexagon.
center_h = (650,250)
# Side of hexagon.
side = 150
# radius from thhe center.
r = np.cos(np.pi/6) * side
# Center Coordinates of hexagon.
c_x,c_y = center_h

angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, 7)[:-1]
v_x = c_x + r * np.cos(angles) # x_coordinate_vertices.
v_y = c_y + r * np.sin(angles) # y_coordinate_vertices.
radius_clearance = r + 5 # Clearance from radius.
v_x_c = c_x + radius_clearance * np.cos(angles) # x_coordinate_clearance_vertices.
v_y_c= c_y + radius_clearance * np.sin(angles) # y_coordinate_clearance_vertices.
vertices = np.vstack((v_x, v_y)).T # storing x and y vertices in a tuple.
clearance_verticies = np.vstack((v_x_c, v_y_c)).T # storing clearance x and y vertices.

# Drawaing objects on the empty_space by iterating in for loop using half plane equations.
for x in range(1200):
    for y in range(500):
        y_transform = 500 - y

        # Wall clearance.
        if (x <= 5 or x >= 1195 or y_transform <= 5 or y_transform >= 495):
            Graph_map[y,x] = [0,255,0]
        
        # object 1(rectangle)
        if (x >= 100 and x <= 175  and y_transform >= 100 and y_transform <= 500 ):
            Graph_map[y,x] = [0,0,0]
        elif (x >= 100 - 5  and x <= 175 + 5 and y_transform >= 100 - 5 and y_transform <= 500 + 5):
            Graph_map[y,x] = [0, 255, 0]
        
        # object 2(rectangle)
        if (x >= 275 and x <= 350 and y_transform >= 0 and y_transform <= 400):
            Graph_map[y,x] = [0,0,0]
        elif(x >= 275 - 5 and x <= 350 + 5 and y_transform >= 0 - 5 and y_transform <= 400 + 5):
             Graph_map[y,x] = [0, 255, 0] 

        # object 3 (combination of 3 rectangles)
        if (x >= 1020 - 5 and x <= 1100 + 5 and y_transform>= 50 - 5  and y_transform <= 450 + 5):
            Graph_map[y,x] = [0,255,0]
        elif (x >= 900 - 5  and x <= 1100 + 5  and y_transform >= 50 - 5 and y_transform <= 125 + 5):
            Graph_map[y,x] = [0, 255, 0]
        elif (x >= 900 - 5 and x <= 1100 + 5 and y_transform >= 375 - 5 and y_transform <= 450 + 5):
            Graph_map[y,x] = [0,255,0]
        
        if (x >= 1020 and x <= 1100 and y_transform>= 50  and y_transform <= 450 ):
            Graph_map[y,x] = [0,0,0]
        elif (x >= 900 and x <= 1100  and y_transform >= 50 and y_transform <= 125):
            Graph_map[y,x] = [0,0,0]
        elif (x >= 900 and x <= 1100 and y_transform >= 375 and y_transform <= 450):
            Graph_map[y,x] = [0,0,0]

# object 4 (hexagon)
def hexagon(x, y, vertices): # Defining a function to calucalate cross product of vertices inside hexagon.
    result = np.zeros(x.shape, dtype=bool)
    num_vertices = len(vertices)
    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        cross_product = (vertices[j, 1] - vertices[i, 1]) * (x - vertices[i, 0]) - (vertices[j, 0] - vertices[i, 0]) * (y - vertices[i, 1])
        result |= cross_product > 0
    return ~result


x, y = np.meshgrid(np.arange(1200), np.arange(500))

hexagon_original = hexagon(x, y, vertices)
hexagon_clearance = hexagon(x, y,clearance_verticies) & ~hexagon_original

# Drawing hexagon and its clearance on the graph_map.
Graph_map[hexagon_clearance] = [0, 255, 0]
Graph_map[hexagon_original] = [0, 0, 0]

output = cv2.VideoWriter('A_star.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (1200, 500))

# Creating Action sets.
def movement_1(node):
    x, y, theta = node
    new_node = (x + np.cos(np.radians(theta)), y + np.sin(np.radians(theta)), theta)
    x, y, theta = new_node
    return x,y,theta

def movement_2(node):
    x, y, theta = node
    new_node = (x + np.cos(np.radians(theta + 30)), y + np.sin(np.radians(theta + 30)), theta + 30)
    x, y, theta = new_node
    return x,y,theta

def movement_3(node):
    x, y, theta = node
    new_node = (x + np.cos(np.radians(theta + 60)), y + np.sin(np.radians(theta + 60)), theta + 60)
    x, y, theta = new_node
    return x,y,theta

def movement_4(node):
    x, y, theta = node
    new_node = (x + np.cos(np.radians(theta - 30)), y + np.sin(np.radians(theta - 30)), theta - 30)
    x, y, theta = new_node
    return x,y,theta

def movement_5(node):
    x, y, theta = node
    new_node = (x + np.cos(np.radians(theta - 60)), y + np.sin(np.radians(theta - 60)), theta - 60)
    x, y, theta = new_node
    return x, y, theta
    
def possible_node(node):
    x, y, theta = node
    new_node = []
    action_set = {movement_1:1,
                  movement_2:1,
                  movement_3:1,
                  movement_4:1,
                  movement_5:1}

    new_nodes = []
    for action, cost in action_set.items():
        new_node = action(node)
        next_x, next_y, new_theta = new_node
        if next_x >= 0 and next_x < 1200 and next_y >= 0 and next_y < 500 and np.all(Graph_map[int(next_y), int(next_x)] == [255, 255, 255]):
            if new_node not in new_nodes:
                new_nodes.append((cost, new_node))

    return new_nodes

def heuristc(node, goal):
    x1, y1, _ = node
    x2, y2, _ = goal
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def A_star(start_node, goal_node):
    parent = {}
    cost_list = {start_node:0}
    closed_list = set()
    open_list = PriorityQueue()
    open_list.put((0, start_node))
    map_visualization = np.copy(Graph_map)
    step_count = 0 
    visted_nodes(start_node)
    
    while not open_list.empty():
        current = open_list.get()
        current_cost, current_node = current
        closed_list.add(current_node)
        if heuristc(current_node, goal_node) < 1.5:
            path = A_star_Backtracting(parent, start_node, current_node, map_visualization, step_count)
            for _ in range(30):
                output.write(map_visualization)
            break
        
        for cost, new_node in possible_node(current_node):
            cost_to_come = current_cost + cost
            if new_node in closed_list and visited(new_node):
                continue
            if new_node not in cost_list or cost_to_come < cost_list[new_node]:
                cost_list[new_node] = cost_to_come
                parent[new_node] = current_node
                cost_total = cost_to_come + heuristc(new_node, goal_node) 
                open_list.put((cost_total, new_node))
                cv2.arrowedLine(map_visualization, (int(current_node[0]), int(current_node[1])), (int(new_node[0]), int(new_node[1])), (255, 0, 0), 1)
                if step_count % 100000 == 0:
                    output.write(map_visualization)
                step_count += 1
    
    output.release()
    return None

def visted_nodes(node):
    x, y, theta = node
    a = int(x/0.5)
    b = int(y/0.5)
    c = int(theta/30) % 12
    G[a][b][c] = 1

def visited(node):
    x, y, theta = node
    a = int(x/0.5)
    b = int(y/0.5)
    c = int(theta/30) % 12
    return G[a][b][c] == 1


def A_star_Backtracting(parent, start_node, end_node, map_visualization, step_count):
    path = [end_node] # Adding end node to the path
    while end_node != start_node: # If the end node is not equal to start_node, parent of the end_node is added to path and continues.
        path.append(parent[end_node])
        end_node = parent[end_node] # The parent of end node becomes the current node.
    path.reverse()
    for j in range(1, len(path)):
        start_point = (int(path[j - 1][0]), int(500 - path[j - 1][1]))
        end_point = (int(path[j][0]), int(500 - path[j][1]))
        cv2.line(map_visualization, start_point, end_point, (0, 0, 255), thickness=2)  # Drawing lines to explore the path.
        if step_count % 15 == 0:
            output.write(map_visualization)
        step_count += 1
    return path

start_node = (200,50,0)
goal_node = (1150, 450, 0)

path = A_star(start_node, goal_node)
