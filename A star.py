# Importing Necessary libraries.
import numpy as np
import matplotlib.pyplot as plt 
from queue import PriorityQueue
import cv2
import time
import math

# Creating a empty space for drawing graph.
Graph_map = np.ones((500, 1200, 3), dtype=np.uint8)*255

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

def draw_hexagon_using_half_planes(x,y):
    
    cx, cy = 650, 250
    side = 150
    r = side * np.cos(np.radians(30))  # radius of the hexagon
    y = abs(y - 500)  # inverted to match the map given
    # Horizontal boundary points of the hexagon are calculated using the radius of the hexagon
    x_boundary_left , x_boundary_right  = cx - r, cx + r
    y_top = 325
    y_bottom = 175
    # diagoonal boundary line eqaution
    y_top_left = ((np.radians(30))*(x - x_boundary_left))+ y_top
    y_top_right = - ((np.radians(30))*(x - x_boundary_right))+ y_top
    y_bottom_right =  ((np.radians(30))*(x - x_boundary_right))+ y_bottom
    y_bottom_left = - ((np.radians(30))*(x - x_boundary_left))+ y_bottom

    # Check if the point is inside the hexagon
    if (x >= x_boundary_left and x <= x_boundary_right and y<= y_top_left and y>= y_bottom_left and y<= y_top_right and y>= y_bottom_right):
        return True
    else:
        return False

def hex_clearance(x,y):
    
    cx, cy = 650, 250
    side = 150
    clearance = 5
    r = side * np.cos(np.radians(30)) + clearance  # radius of the hexagon
    y = abs(y - 500)  # inverted to match the map given
    # Horizontal boundary points of the hexagon are calculated using the radius of the hexagon
    x_boundary_left , x_boundary_right  = cx - r, cx + r
    y_top = 325
    y_bottom = 175
    # Diagonal boundary line eqaution
    y_top_left = ((np.radians(30))*(x - x_boundary_left))+ y_top + clearance
    y_top_right = - ((np.radians(30))*(x - x_boundary_right))+ y_top + clearance
    y_bottom_right =  ((np.radians(30))*(x - x_boundary_right))+ y_bottom - clearance
    y_bottom_left = - ((np.radians(30))*(x - x_boundary_left))+ y_bottom - clearance

    # Check if the point is inside the hexagon
    if (x >= x_boundary_left and x <= x_boundary_right and y<= y_top_left and y>= y_bottom_left and y<= y_top_right and y>= y_bottom_right):
        return True
    else:
        return False
    
for x in range(1200):
    for y in range(500):
        if hex_clearance(x, y):
            Graph_map[y, x] = (0, 0, 255)
        if draw_hexagon_using_half_planes(x, y):
            Graph_map[y, x] = (0, 0, 0)    


def is_free_space(x, y):
    return np.array_equal(Graph_map[y, x], [255, 255, 255])


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
    
