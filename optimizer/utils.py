from timeit import repeat
from scipy.spatial import distance
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry import Polygon, Point
import numpy as np
from math import sqrt
from itertools import combinations
from matplotlib.patches import Rectangle
import math
import pickle
from optimizer import parameter as para
from physical_env.network.utils import find_receiver
from physical_env.network import Node

def q_max_function(q_table, state):
    """
    Tính toán giá trị Q_max cho một trạng thái cụ thể.

    Args:
        q_table (numpy.ndarray): Bảng Q.
        state (int): Trạng thái cần tính giá trị Q_max.

    Returns:
        numpy.ndarray: Mảng chứa giá trị Q_max cho mỗi trạng thái, ngoại trừ trạng thái hiện tại.
    """
    temp = [max(row) if index != state else -float("inf") for index, row in enumerate(q_table)]
    return np.asarray(temp)

def reward_function(network, mc, q_learning, state, time_stem):
    """
    Tính toán các phần thưởng cho một hành động cụ thể trong mạng.

    Args:
        network (object): Đối tượng mạng.
        mc (object): Đối tượng MC.
        q_learning (object): Q-learning.
        state (int): Trạng thái của hành động cần tính phần thưởng.
        time_stem (int): Thời gian.

    Returns:
        tuple: Bao gồm các phần thưởng đầu tiên, thứ hai và thứ ba, cùng với thời gian sạc.
    """
    alpha = q_learning.alpha
    charging_time = get_charging_time(network, mc, q_learning, time_stem=time_stem, state=state, alpha=alpha)
    w, nb_target_alive = get_weight(network, mc, q_learning, state, charging_time)
    p = get_charge_per_sec(network, q_learning, state)
    p_hat = p / np.sum(p)
    E = np.asarray([network.listNodes[request["id"]].energy for request in q_learning.list_request])
    e = np.asarray([request["energyCS"] for request in q_learning.list_request])
    second = nb_target_alive ** 2 / len(network.listTargets)
    third = np.sum(w * p_hat)
    first = np.sum(e * p / E)
    return first, second, third, charging_time

def init_function(nb_action=30):
    """
    Khởi tạo bảng Q với kích thước nhất định.

    Args:
        nb_action (int): Số lượng hành động.

    Returns:
        numpy.ndarray: Bảng Q được khởi tạo với kích thước đã chỉ định.
    """
    return np.zeros((nb_action, nb_action), dtype=float)

def get_weight(net, mc, q_learning, action_id, charging_time):
    """
    Tính toán trọng số và số lượng mục tiêu còn sống trong mạng cho một hành động cụ thể.

    Args:
        net (object): Đối tượng mạng.
        mc (object): Đối tượng MC.
        q_learning (object): Q-learning.
        action_id (int): ID của hành động cần tính toán.
        charging_time (float): Thời gian sạc dự kiến.

    Returns:
        tuple: Bao gồm trọng số cho mỗi yêu cầu và số lượng mục tiêu còn sống.
    """
    p = get_charge_per_sec(net, q_learning, action_id)
    all_path = get_all_path(net)
    time_move = distance.euclidean(q_learning.action_list[mc.state],
                                   q_learning.action_list[action_id]) / mc.velocity
    list_dead = []
    w = [0 for _ in q_learning.list_request]
    for request_id, request in enumerate(q_learning.list_request):
        temp = (net.listNodes[request["id"]].energy - time_move * request["energyCS"]) + (
                p[request_id] - request["energyCS"]) * charging_time
        if temp < 0:
            list_dead.append(request["id"])
    for request_id, request in enumerate(q_learning.list_request):
        nb_path = 0
        for path in all_path:
            if request["id"] in path:
                nb_path += 1
        w[request_id] = nb_path
    total_weight = sum(w) + len(w) * 10 ** -3
    w = np.asarray([(item + 10 ** -3) / total_weight for item in w])
    nb_target_alive = 0
    for path in all_path:
        if para.base in path and not (set(list_dead) & set(path)):
            nb_target_alive += 1
    return w, nb_target_alive

def get_path(net, sensor_id):
    """
    Tìm đường dẫn từ một nút cảm biến đến trạm cơ sở hoặc đến nút cảm biến khác.

    Args:
        net (object): Đối tượng mạng.
        sensor_id (int): ID của nút cảm biến.

    Returns:
        list: Danh sách đường dẫn từ nút cảm biến đến đích.
    """
    path = [sensor_id]
    if distance.euclidean(net.listNodes[sensor_id].location, para.base) <= net.listNodes[sensor_id].com_range:
        path.append(para.base)
    else:
        receive_id = find_receiver(node=net.listNodes[sensor_id])
        if receive_id != -1:
            path.extend(get_path(net, receive_id))
    return path

def get_all_path(net):
    """
    Tìm tất cả các đường dẫn trong mạng từ mỗi nút cảm biến đến trạm cơ sở hoặc nút cảm biến khác.

    Args:
        net (object): Đối tượng mạng.

    Returns:
        list: Danh sách tất cả các đường dẫn từ mỗi nút cảm biến.
    """
    list_path = []
    for sensor_id, node in enumerate(net.listNodes):
        list_path.append(get_path(net, sensor_id))
    return list_path

def get_charge_per_sec(net, q_learning, state):
    """
    Tính toán tỷ lệ sạc cho mỗi yêu cầu tại một trạng thái cụ thể.

    Args:
        net (object): Đối tượng mạng.
        q_learning (object): Q-learning.
        state (int): Trạng thái cần tính toán.

    Returns:
        list: Danh sách tỷ lệ sạc cho mỗi yêu cầu tại trạng thái cụ thể.
    """
    arr = []
    for request in q_learning.list_request:
        arr.append(para.alpha / (distance.euclidean(net.listNodes[request["id"]].location, q_learning.action_list[state]) + para.beta) ** 2)
    return arr

def get_charging_time(network=None, mc=None, q_learning=None, time_stem=0, state=None, alpha=0.5): 
    """
    Tính toán thời gian sạc dự kiến cho một trạng thái cụ thể của MC trong mạng.

    Args:
        network (object): Đối tượng mạng.
        mc (object): Đối tượng MC.
        q_learning (object): Q-learning.
        time_stem (int): Thời gian.
        state (int): Trạng thái của MC.
        alpha (float): Tham số điều chỉnh.

    Returns:
        float: Thời gian sạc dự kiến.
    """
    time_move = distance.euclidean(mc.location, q_learning.action_list[state]) / mc.velocity
    energy_min = network.listNodes[0].threshold + alpha * network.listNodes[0].capacity
    
    avg_energy_consumption = []
    for node in network.listNodes:
        d = distance.euclidean(q_learning.action_list[state], node.location)
        p = para.alpha / (d + para.beta) ** 2
        avg_energy_consumption.append(p - node.energyCS)
    
    charging_times = []
    for i, pos in enumerate(q_learning.action_list):
        charging_time = 0
        for j, node in enumerate(network.listNodes):
            if avg_energy_consumption[j] > 0:
                d = distance.euclidean(pos, node.location)
                time_to_charge = (energy_min - node.energy) / avg_energy_consumption[j]
                charging_time = max(charging_time, time_to_charge)
        charging_times.append(charging_time)
    
    total_charging_time = max(charging_times) + time_move
    
    return total_charging_time

def network_clustering_v2(optimizer, network=None, nb_cluster=81):
    """
    Phân cụm các nút mạng để tạo các vị trí sạc tiềm năng.

    Args:
        optimizer (object): Đối tượng tối ưu hóa.
        network (object): Đối tượng mạng.
        nb_cluster (int): Số lượng cụm mong muốn.

    Returns:
        list: Danh sách các vị trí sạc tiềm năng.
    """
    X = []
    Y = []
    min_node = 1000
    for node in network.listNodes:
        node.set_check_point(200)
        if node.avg_energy != 0:
            min_node = min(min_node, node.avg_energy)
    for node in network.listNodes:
        repeat = int(node.avg_energy / min_node)
        for _ in range(repeat):
            X.append(node.location)
            Y.append(node.avg_energy)
    X = np.array(X)
    Y = np.array(Y)
    d = np.linalg.norm(Y)
    Y = Y / d
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(X)
    charging_pos = []
    for pos in kmeans.cluster_centers_:
        charging_pos.append((int(pos[0]), int(pos[1])))
    charging_pos.append(para.depot)
    network_plot(network=network, charging_pos=charging_pos)
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