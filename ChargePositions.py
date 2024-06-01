import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from itertools import combinations
import yaml
from matplotlib.patches import Rectangle
from shapely.geometry import Point, Polygon, MultiPolygon

# Đọc dữ liệu từ tệp YAML
with open('D:\Code\Python\BaoCaoNghienCuu\WRSN\physical_env\\network\\network_scenarios\hanoi1000n50.yaml', 'r') as file:
    data = yaml.safe_load(file)

with open('D:\Code\Python\BaoCaoNghienCuu\WRSN\physical_env\mc\mc_types\default.yaml', 'r') as file:
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

def compute_node_radius(nodes_info):
    """
    Tính và trả về bán kính của từng hình tròn dựa trên năng lượng tiêu hao của mỗi nút cảm biến.

    Parameters:
        nodes_info (dict): Dữ liệu về năng lượng tiêu hao của mỗi nút cảm biến.

    Returns:
        list: Bán kính của từng hình tròn dựa trên năng lượng tiêu hao của mỗi nút cảm biến.
    """
    arrRadius = []
    for node_id, energy_consumption in nodes_info.items():
        radius = np.sqrt(alpha / energy_consumption) - beta
        print(radius)
        arrRadius.append(radius)
    return arrRadius

def find_non_intersecting_circles(centers, radius):
    """
    Tìm và trả về tập hợp các hình tròn không giao nhau.

    Parameters:
        centers (list of tuples): Tập hợp các trung tâm hình tròn.
        radius (list of floats): Bán kính của từng hình tròn.

    Returns:
        list: Tập hợp các hình tròn không giao nhau.
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
    Loại bỏ các tập hợp con trùng lặp trong danh sách.

    Parameters:
        set (list): Danh sách các tập hợp con.

    Returns:
        list: Danh sách các tập hợp con không trùng lặp.
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

def draw_sensor_icon(ax, x, y, icon_path, icon_size):
    """
    Vẽ biểu tượng cảm biến tại tọa độ (x, y).

    Parameters:
        ax (matplotlib.axes.Axes): Đối tượng trục của đồ thị.
        x (float): Tọa độ x của trung tâm.
        y (float): Tọa độ y của trung tâm.
        icon_path (str): Đường dẫn đến hình ảnh biểu tượng cảm biến.
        icon_size (float): Kích thước của biểu tượng.

    Returns:
        None
    """
    icon = plt.imread(icon_path)
    ax.imshow(icon, extent=[x - icon_size / 2, x + icon_size / 2, y - icon_size / 2, y + icon_size / 2])

def draw_mc_icon(ax, x, y, icon_path, icon_size):
    """
    Vẽ biểu tượng của thiết bị sạc tại tọa độ (x, y).

    Parameters:
        ax (matplotlib.axes.Axes): Đối tượng trục của đồ thị.
        x (float): Tọa độ x của trung tâm.
        y (float): Tọa độ y của trung tâm.
        icon_path (str): Đường dẫn đến hình ảnh biểu tượng của thiết bị sạc.
        icon_size (float): Kích thước của biểu tượng.

    Returns:
        None
    """
    icon = plt.imread(icon_path)
    ax.imshow(icon, extent=[x - icon_size / 2, x + icon_size / 2, y - icon_size / 2, y + icon_size / 2])

def draw_battery(ax, x, y, width, height, charge_percentage):
    """
    Vẽ biểu tượng pin tại tọa độ (x, y).

    Parameters:
        ax (matplotlib.axes.Axes): Đối tượng trục của đồ thị.
        x (float): Tọa độ x của góc dưới bên trái của biểu tượng pin.
        y (float): Tọa độ y của góc dưới bên trái của biểu tượng pin.
        width (float): Chiều rộng của biểu tượng pin.
        height (float): Chiều cao của biểu tượng pin.
        charge_percentage (float): Phần trăm pin còn lại.

    Returns:
        None
    """
    ax.add_patch(Rectangle((x, y), width, height, edgecolor='black', facecolor='none'))
    charge_height = height * charge_percentage / 100

    if charge_percentage <= 20:
        color = 'red'
    elif charge_percentage > 20 and charge_percentage <= 75:
        color = 'yellow'
    else:
        color = 'lime'

    ax.add_patch(Rectangle((x, y), width, charge_height, edgecolor='none', facecolor=color))

    bolt_x = x + width / 2
    bolt_y = y + charge_height / 2
    ax.text(bolt_x, bolt_y, '⚡', fontsize=20, color='blue', va='center', ha='center')

def plot_circles(net, nodes, arr_name_nodes, radiuses, base_station_icon_path="images/bs.png"):
    """
    Vẽ các hình tròn và các điểm giao nhau trên đồ thị, bao gồm bán kính sạc của các hình tròn.

    Parameters:
        net (object): Đối tượng mạng.
        nodes (list): Tập hợp các điểm trung tâm hình tròn.
        arr_name_nodes (list): Tập hợp các tên hình tròn.
        radiuses (list of floats): Bán kính sạc của từng hình tròn.
        base_station_icon_path (str): Đường dẫn đến hình ảnh biểu tượng trạm cơ sở.

    Returns:
        list: Tập hợp các điểm giao nhau.
    """
    all_intersections = []
    for centers in arr_name_nodes:
        circles = []
        for i in centers:
            circle = Point(nodes[i]).buffer(radiuses[i])
            circles.append(circle)
            draw_sensor_icon(plt.gca(), nodes[i][0], nodes[i][1], 'images/sensor.png', 35)

            # Vẽ bán kính sạc của hình tròn
            charging_circle = plt.Circle((nodes[i][0], nodes[i][1]), radiuses[i], color='green', fill=False, linestyle='--', linewidth=0.5)
            plt.gca().add_patch(charging_circle)

        # Vẽ biểu tượng trạm cơ sở
        base_station_x, base_station_y = base_station
        draw_sensor_icon(plt.gca(), base_station_x, base_station_y, base_station_icon_path, 75)

        for circle in circles:
            plt.plot(circle.exterior.xy[0], circle.exterior.xy[1], color='b', linewidth=0.5)

        intersections = circles[0]
        for circle in circles[1:]:
            intersections = intersections.intersection(circle)

        if isinstance(intersections, (Polygon, MultiPolygon)):
            centroid = intersections.centroid
            if not centroid.is_empty:
                plt.plot(centroid.x, centroid.y, marker='*', color='blue', markersize=7.5)
        else:
            plt.fill(circles[0].exterior.xy[0], circles[0].exterior.xy[1], color='orange', alpha=0.3)
            plt.plot(nodes[centers[0]][0], nodes[centers[0]][1], marker='*', color='blue', markersize=7.5)
        
        all_intersections.append(intersections)
    return all_intersections
