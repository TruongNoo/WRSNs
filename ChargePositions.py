import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from math import sqrt
from itertools import combinations
import yaml

# Đọc dữ liệu từ tệp YAML
with open('Data_WRSN/network_scenarios/hanoiEditbyUser.yaml', 'r') as file:
    data = yaml.safe_load(file)

with open('Data_WRSN/default.yaml', 'r') as file:
    defData = yaml.safe_load(file)

node_phy_spe = data['node_phy_spe']
efs = node_phy_spe['efs']
emp = node_phy_spe['emp']
elec = node_phy_spe['er'] 
l = node_phy_spe['package_size']
Rc = node_phy_spe['com_range']
d0 = np.sqrt(efs / emp)
base_station = data['base_station']

nodes = data['nodes']
alpha = defData['alpha']
beta = defData['beta']

def find_circle_intersections(centers, radius):
    """
    Tìm và trả về tập hợp các điểm giao nhau giữa các hình tròn.

    Parameters:
        centers (list of tuples): Tọa độ các trung tâm hình tròn.
        radius (list of floats): Bán kính của từng hình tròn.

    Returns:
        set: Tập hợp các điểm giao nhau.
    """
    intersections = set()
    for i, j in combinations(range(len(centers)), 2):
        dist = np.linalg.norm(np.array(centers[j]) - np.array(centers[i]))
        if dist <= radius[i] + radius[j]:
            d = (dist**2) / (2 * dist)
            h = np.sqrt(radius[i] * radius[j] - d**2)
            x = centers[i][0] + d * (centers[j][0] - centers[i][0]) / dist
            y = centers[i][1] + d * (centers[j][1] - centers[i][1]) / dist
            x1 = x + h * (centers[j][1] - centers[i][1]) / dist
            y1 = y - h * (centers[j][0] - centers[i][0]) / dist
            x2 = x - h * (centers[j][1] - centers[i][1]) / dist
            y2 = y + h * (centers[j][0] - centers[i][0]) / dist
            intersections.add((x1, y1))
            intersections.add((x2, y2))
    return intersections

def find_nearest_points_within_distance(d, points):
    """
    Tìm và trả về tập hợp các điểm gần nhất với từng điểm trong tập hợp.

    Parameters:
        d (float): Khoảng cách tối đa cho phép.
        points (list of tuples): Tập hợp các điểm cần kiểm tra.

    Returns:
        list: Tập hợp các điểm gần nhất tương ứng với từng điểm đầu vào.
    """
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

def compute_node_radius(centers):
    """
    Tính và trả về bán kính của từng hình tròn.

    Parameters:
        centers (list of tuples): Tập hợp các trung tâm hình tròn.

    Returns:
        list: Bán kính của từng hình tròn.
    """
    arr_nearest_point = find_nearest_points_within_distance(Rc, centers)
    arr_radius = []
    for i in range(len(centers)):
        dist = np.sqrt((centers[i][0] - arr_nearest_point[i][0])**2 + (centers[i][1] - arr_nearest_point[i][1])**2)
        er = l * elec
        if (dist < d0):
            et = er + l * efs * dist**2
        else:
            et = er + l * efs * dist**4
        ej = et + er
        radius = np.sqrt(alpha / ej) - beta
        print(radius)
        arr_radius.append(radius)
    return arr_radius

def find_non_intersecting_circles(centers, radius):
    """
    Tìm và trả về tập hợp các hình tròn giao nhau.

    Parameters:
        centers (list of tuples): Tập hợp các trung tâm hình tròn.
        radius (list of floats): Bán kính của từng hình tròn.

    Returns:
        list: Tập hợp các hình tròn giao nhau.
    """
    intersections = set()
    for i in range(len(centers)):
        for j in range(len(centers)):
            if (j != i):
                dist = np.sqrt((centers[j][0] - centers[i][0])**2 + (centers[j][1] - centers[i][1])**2)
                if dist <= radius[i] + radius[j]:
                    d = (dist**2) / (2 * dist)
                    h = np.sqrt(radius[i] * radius[j] - d**2)
                    x = centers[i][0] + d * (centers[j][0] - centers[i][0]) / dist
                    y = centers[i][1] + d * (centers[j][1] - centers[i][1]) / dist
                    x1 = x + h * (centers[j][1] - centers[i][1]) / dist
                    y1 = y - h * (centers[j][0] - centers[i][0]) / dist
                    x2 = x - h * (centers[j][1] - centers[i][1]) / dist
                    y2 = y + h * (centers[j][0] - centers[i][0]) / dist
                    intersections.add((x1, y1))
                    intersections.add((x2, y2))

    set_points_in_circles = []
    for point in intersections:
        point_in_circles = []
        for i in range(len(centers)):
            if np.sqrt((point[0] - centers[i][0])**2 + (point[1] - centers[i][1])**2) <= radius[i] + 0.01:
                point_in_circles.append(i)
        set_points_in_circles.append(point_in_circles)

    set_points_in_circles = sorted(set_points_in_circles, key=lambda x: len(x), reverse=True)

    return set_points_in_circles

def remove_duplicate_sets(set):
    """
    Loại bỏ các tập con trùng lặp trong tập hợp.

    Parameters:
        set (list of lists): Tập hợp các tập con.

    Returns:
        list: Tập hợp các tập con không trùng lặp.
    """
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

def find_and_add_isolated_circle(arr_nodes, nodes):
    """
    Tìm và thêm các hình tròn không giao với bất kỳ hình tròn nào khác.

    Parameters:
        arr_nodes (list): Tập hợp các hình tròn giao nhau.
        nodes (list): Tập hợp các điểm trung tâm hình tròn.

    Returns:
        None
    """
    merge_arr_node = []
    for arr_node in arr_nodes:
        for node in arr_node:
            merge_arr_node.append(node)

    arr_node_isolated = []
    for i in range(len(nodes)):
        arr_node_isolated = []
        if i not in merge_arr_node:
            arr_node_isolated.append(i)
        if (len(arr_node_isolated) == 1):
            arr_nodes.append(arr_node_isolated)

def remove_common_elements(arr, nodes):
    """
    Loại bỏ các phần tử chung giữa các tập hợp con.

    Parameters:
        arr (list): Tập hợp các tập con.
        nodes (list): Tập hợp các điểm trung tâm hình tròn.

    Returns:
        list: Tập hợp các tập con không chứa phần tử chung.
    """
    for i in range(len(arr)-2):
        arr = sorted(arr, key=lambda x: len(x), reverse=True)
        first_subarray = arr[i]
        for num in first_subarray:
            for subarray in arr[(i+1):]:
                if num in subarray:
                    subarray.remove(num)
        arr = sorted(arr, key=lambda x: len(x), reverse=True)
    arr = [subarray for subarray in arr if subarray]

    find_and_add_isolated_circle(arr, nodes)
    return arr

def plot_circles(nodes, arr_name_nodes, radiuses):
    """
    Vẽ các hình tròn và các điểm giao nhau trên đồ thị.

    Parameters:
        nodes (list): Tập hợp các điểm trung tâm hình tròn.
        arr_ten_nodes (list): Tập hợp các tên hình tròn.
        radiuses (list of floats): Bán kính của từng hình tròn.

    Returns:
        list: Tập hợp các điểm giao nhau.
    """
    all_intersections = []
    for centers in arr_name_nodes:
        circles = []
        for i in centers:
            circle = Point(nodes[i]).buffer(radiuses[i])
            circles.append(circle)
            plt.plot(nodes[i][0], nodes[i][1], marker='o', color='black', markersize=1)

        for circle in circles:
            plt.plot(circle.exterior.xy[0], circle.exterior.xy[1], color='b', linewidth=0.5)

        intersections = circles[0]
        for circle in circles[1:]:
            intersections = intersections.intersection(circle)

        if isinstance(intersections, Polygon):
            centroid = intersections.centroid
            plt.plot(centroid.x, centroid.y, marker='o', color='r', markersize=3)
        else:
            plt.fill(circles[0].exterior.xy[0], circles[0].exterior.xy[1], color='orange', alpha=0.3)
            plt.plot(nodes[centers[0]][0], nodes[centers[0]][1], marker='o', color='r', markersize=3)
        all_intersections.append(intersections)
    return all_intersections

radius_node = compute_node_radius(nodes)

set_arr_non_intersecting_circles = find_non_intersecting_circles(nodes, radius_node)
set_arr_non_intersecting_circles = remove_duplicate_sets(set_arr_non_intersecting_circles)
set_arr_non_intersecting_circles = remove_common_elements(set_arr_non_intersecting_circles, nodes)

# Plotting
plt.figure(figsize=(10, 10))
plot_circles(nodes, set_arr_non_intersecting_circles, radius_node)
plt.scatter(*base_station, color='green', label='Base Station')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Circles and Intersections in Hanoi')
plt.grid(True)
plt.show()