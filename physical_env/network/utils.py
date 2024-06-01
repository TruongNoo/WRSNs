import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry import Polygon, Point
import numpy as np
from math import sqrt

def find_receiver(node):
    if not (node.status == 1):
        return -1
    candidates = [other_node for other_node in node.neighbors
                  if other_node.level < node.level and other_node.status == 1]

    if len(candidates) > 0:
        distances = [euclidean(candidate.location, node.location) for candidate in candidates]
        return candidates[np.argmin(distances)].id
    else:
        return -1

def request_function(node, optimizer, t):
    optimizer.list_request.append(
        {"id": node.id, "energy": node.energy, "energyCS": node.energyCS, "energyRR": node.energyRR,
         "time": t})

def network_cluster_id_node(network = None):
    nodes = network.listNodes
    location_nodes = []
    radius_nodes = []
    for node in nodes:
        radius_nodes.append(node.radius)
        location_nodes.append(node.location)

    set_arr_interecting_circles = find_set_of_interecting_circles(location_nodes, radius_nodes)
    set_arr_interecting_circles = remove_arr_of_set(set_arr_interecting_circles)
    set_arr_interecting_circles = remove_common_elements2(set_arr_interecting_circles, location_nodes)
    arr_name_nodes = set_arr_interecting_circles
    return arr_name_nodes

def network_clustering(network=None):
    nodes = network.listNodes
    location_nodes = []
    radius_nodes = []
    for node in nodes:
        radius_nodes.append(node.radius)
        location_nodes.append(node.location)

    set_arr_interecting_circles = find_set_of_interecting_circles(location_nodes, radius_nodes)
    set_arr_interecting_circles = remove_arr_of_set(set_arr_interecting_circles)
    set_arr_interecting_circles = remove_common_elements2(set_arr_interecting_circles, location_nodes)
    arr_name_nodes = set_arr_interecting_circles

    all_intersections = []
    charging_pos = []
    for centers in arr_name_nodes:
        circles = []
        for i in centers:
            circle = Point(location_nodes[i]).buffer(radius_nodes[i])
            circles.append(circle)

        intersections = circles[0]
        for circle in circles[1:]:
            intersections = intersections.intersection(circle)

        if isinstance(intersections, Polygon) and not intersections.is_empty:
            centroid = intersections.centroid
            charging_pos.append((centroid.x, centroid.y))
        else:
            charging_pos.append((location_nodes[centers[0]][0], location_nodes[centers[0]][1]))
        all_intersections.append(intersections)

    charging_pos.append(network.baseStation.location)
    return charging_pos

def node_distribution_plot(network, charging_pos):
    x_node = []
    y_node = []
    c_node = []
    for node in network.listNodes:
        x_node.append(node.location[0])
        y_node.append(node.location[1])
        c_node.append(node.avg_energy)
    x_centroid = []
    y_centroid = []
    plt.hist(c_node, bins=100)
    plt.savefig('fig/node_distribution.png')


def network_plot(network, charging_pos):
    x_node = []
    y_node = []
    c_node = []
    for node in network.listNodes:
        x_node.append(node.location[0])
        y_node.append(node.location[1])
        c_node.append(node.avg_energy)
    x_centroid = []
    y_centroid = []
    for centroid in charging_pos:
        x_centroid.append(centroid[0])
        y_centroid.append(centroid[1])
    c_node = np.array(c_node)
    d = np.linalg.norm(c_node)
    c_node = c_node / d * 80
    plt.scatter(x_node, y_node, s=c_node)
    plt.scatter(x_centroid, y_centroid, c='red', marker='^')
    plt.savefig('fig/network_plot.png')

def remove_element_in_arr(arr, key):
  for i in range(len(arr)):
    if arr[i] == key:
      arr.pop(i)
      break

def find_set_of_interecting_circles(centers, radius):
  intersections = set()
  for i in range(len(centers)):
    for j in range(len(centers)):
      if (j != i):
        d = np.sqrt((centers[j][0] - centers[i][0])**2 + (centers[j][1] - centers[i][1])**2)

        a = (radius[i]**2 - radius[j]**2 + d**2) / (2 * d)
        h = np.sqrt(radius[i]**2 - a**2)
        
        x_intersect1 = centers[i][0] + a * (centers[j][1] - centers[i][0]) / d
        y_intersect1 = centers[i][1] + a * (centers[j][1] - centers[i][1]) / d
        x_intersect2 = x_intersect1 + h * (centers[j][1] - centers[i][1]) / d
        y_intersect2 = y_intersect1 - h * (centers[j][1] - centers[i][0]) / d
        intersections.add((x_intersect1, y_intersect1))
        intersections.add((x_intersect2, y_intersect2))


  set_points_in_circles = []
  for point in intersections:
    point_in_circles = []
    for i in range(len(centers)):
      if np.sqrt((point[0] - centers[i][0])**2 + (point[1] - centers[i][1])**2) <= radius[i] + 0.01:
        point_in_circles.append(i)
    set_points_in_circles.append(point_in_circles)

  set_points_in_circles = sorted(set_points_in_circles, key=lambda x: len(x), reverse=True)

  return set_points_in_circles

def remove_arr_of_set(set):
  different = False
  new_arr = []
  new_arr.append(set[0])
  for i in range(1, len(set)):
    different = False
    for arr in new_arr:
      if set[i] == arr:
        different = True
    if not different:
      new_arr.append(set[i])
  return new_arr

def circle_intersection(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    
    circle1 = Point(x1, y1).buffer(r1)
    circle2 = Point(x2, y2).buffer(r2)

    intersection = circle1.intersection(circle2)
    return intersection

def find_intersecting_circles(centers, arr_radii):
  count_circle = []
  count_circle_intersection = []
  for i in range(len(centers)-1):
      count_circle = []
      for j in range(i+1, len(centers)):
          intersection = circle_intersection((centers[i][0], centers[i][1], arr_radii[i]),
                                            (centers[j][0], centers[j][1], arr_radii[j]))
          if intersection.area > 0:
              count_circle.append(j)
      count_circle.append(i)
      count_circle_intersection.append(count_circle)
  return count_circle_intersection

def find_intersecting_circles2(centers, radii):
  count_circle = []
  count_circle_intersection = []
  for i in range(len(centers)-1):
      count_circle = []
      for j in range(i+1, len(centers)):
          intersection = circle_intersection((centers[i][0], centers[i][1], radii),
                                            (centers[j][0], centers[j][1], radii))
          if intersection.area > 0:
              count_circle.append(j)
      count_circle.append(i)
      count_circle_intersection.append(count_circle)
  return count_circle_intersection

def remove_common_elements2(arr, nodes):
    for i in range(len(arr)-2):
      arr = sorted(arr, key=lambda x: len(x), reverse=True)
      first_subarray = arr[i]
      for num in first_subarray:
          for subarray in arr[(i+1):]: 
              if num in subarray:
                  subarray.remove(num) 
      arr = sorted(arr, key=lambda x: len(x), reverse=True)
    arr = [subarray for subarray in arr if subarray]

    find_and_add_alone_circle(arr, nodes)
    return arr

def find_nearest_point(d, points):
  nearest_points = []
  for point in points:
    nearest_point = None
    min_distance = float('inf')
    for other_point in points:
      if point == other_point:
        continue
      distance = sqrt(((point[0] - other_point[0]) ** 2) + ((point[1] - other_point[1]) ** 2))
      if distance < min_distance and distance <= d:
        min_distance = distance
        nearest_point = other_point
    nearest_points.append(nearest_point)
  return nearest_points

def find_and_add_alone_circle(arr_nodes, nodes):
  merge_arr_node = []
  for arr_node in arr_nodes:
    for node in arr_node:
      merge_arr_node.append(node)

  arr_node_alone = []
  for i in range(len(nodes)):
    arr_node_alone = []
    if i not in merge_arr_node:
      arr_node_alone.append(i)
    if (len(arr_node_alone) == 1):
      arr_nodes.append(arr_node_alone)
