from ChargePositions import *
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from math import sqrt
from itertools import combinations
import yaml
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.spatial import distance

import pandas as pd
from physical_env.mc.MobileCharger import MobileCharger
from physical_env.network.NetworkIO import NetworkIO
from optimizer.q_learning_heuristic import Q_learning
import sys
import os
import time
import copy
import networkx as nx
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def draw_battery_mc(ax, x, y, width, height, percent_remaining):
    ax.text(x - width / 2 - 5, y + height / 2 - 5, f"{percent_remaining:.2f}%", fontsize=8)

def log(net, mcs, q_learning):
    plt.figure(figsize=(10, 10))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Circles and Intersections in Hanoi')
    plt.grid(True)
    plt.ion()

    ax = plt.gca()

    counter = 0
    while True:
        yield net.env.timeout(50)
        counter += 1
        print_state_net(net, mcs)
        nodes_info = {}
        for node in net.listNodes:
            node_id = node.id
            energy_consume = node.energy
            nodes_info[node_id] = energy_consume

        nodes = net.listNodes
        location_nodes = [node.location for node in nodes]
        radius_nodes = []
        for node in nodes:
            radius_nodes.append(node.radius)

        set_arr_non_intersecting_circles = find_non_intersecting_circles(location_nodes, radius_nodes)
        set_arr_non_intersecting_circles = remove_duplicate_sets(set_arr_non_intersecting_circles)
        set_arr_non_intersecting_circles = remove_common_elements(set_arr_non_intersecting_circles, location_nodes)

        ax.clear()
        draw_mc_icon(ax, mcs[0].current[0],mcs[0].current[1], "D:\Code\Python\BaoCaoNghienCuu\WRSN\images\mc.png",75)
        plot_circles(net, location_nodes, set_arr_non_intersecting_circles, radius_nodes, base_station_icon_path="images/bs.png")
        percent_energy_remaining_per_node = {}
        for node_id, energy_consumption in nodes_info.items():
            percent_remaining = (energy_consumption / 30000) * 100
            percent_energy_remaining_per_node[node_id] = percent_remaining

        for node_id, percent_remaining in percent_energy_remaining_per_node.items():
            x, y = location_nodes[node_id]
            draw_battery_mc(ax, x, y, 15, 40, percent_remaining)

        optimal_charge_points = net.network_cluster
        for point in optimal_charge_points:
            ax.plot(point[0], point[1], marker='*', color='blue', markersize=7.5)

        plt.legend()
        plt.pause(1)
        plt.draw()
        
def print_node_energy(net):
    for node in net.listNodes:
        print("Node {}: Energy = {}".format(node.id, node.energy))

def print_state_net(net, mcs):
    print("[Network] Simulating time: {}s".format(net.env.now))
    for mc in net.mc_list:
        if mc.chargingTime != 0 and mc.get_status() == "charging":
            print("\t\tMC energy:{} is {} at {} state:{}".format(mc.energy, mc.get_status(), mc.current, mc.state))
        elif mc.moving_time != 0 and mc.get_status() == "moving":
            print("\t\tMC energy:{} is {} to {} state:{}".format(mc.energy, mc.get_status(), mc.end, mc.state))
        else:
            print("\t\tMC energy:{} is {} at {} state:{}".format(mc.energy, mc.get_status(), mc.current, mc.state))

# def train_q_learning(q_learning, episodes=100):
#     for episode in range(episodes):
#         # Khởi tạo môi trường và agent (MC)
#         env, net = networkIO.makeNetwork()
#         mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(1)]
#         for id, mc in enumerate(mcs):
#             mc.env = env
#             mc.net = net
#             mc.id = id
#             mc.cur_phy_action = [net.baseStation.location[0], net.baseStation.location[1], 0]
#         net.mc_list = mcs

#         # Huấn luyện trong một episode
#         x = env.process(net.operate(optimizer=q_  learning))
#         env.run(until=x)

#         # Lưu bảng Q sau mỗi episode
#         np.save(f'q_table_episode_{episode}.npy', q_learning.q_table)

networkIO = NetworkIO("physical_env/network/network_scenarios/hanoi1000n50.yaml")
env, net = networkIO.makeNetwork()

with open("physical_env\mc\mc_types\default.yaml", 'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(1)]
print(mc for mc in mcs)
for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
    mc.cur_phy_action = [net.baseStation.location[0], net.baseStation.location[1], 0]
q_learning = Q_learning(net=net, nb_action=31, alpha=0.1, q_gamma=0.1, epsilon=0)
# train_q_learning(q_learning, episodes=100)
print("start network simulation")
net.mc_list = mcs
x = env.process(net.operate(optimizer=q_learning))
env.process(log(net, mcs, q_learning))
env.run(until=x)
