import copy
import numpy as np
from scipy.spatial import distance

import optimizer
from physical_env.network.utils import network_clustering, network_cluster_id_node


class Network:
    def __init__(self, env, listNodes, baseStation, listTargets, mc_list=None, max_time=None):
        self.env = env
        self.listNodes = listNodes
        self.baseStation = baseStation
        self.listTargets = listTargets
        self.targets_active = [1 for _ in range(len(self.listTargets))]
        self.alive = 1
        baseStation.env = self.env
        baseStation.net = self
        self.max_time = max_time
        self.mc_list = mc_list
        self.network_cluster = []
        self.network_cluster_id_node = []

        self.frame = np.array([self.baseStation.location[0], self.baseStation.location[0], self.baseStation.location[1],
                               self.baseStation.location[1]], np.float64)
        it = 0
        for node in self.listNodes:
            node.id = it
            node.env = self.env
            node.net = self
            it += 1
            self.frame[0] = min(self.frame[0], node.location[0])
            self.frame[1] = max(self.frame[1], node.location[0])
            self.frame[2] = min(self.frame[2], node.location[1])
            self.frame[3] = max(self.frame[3], node.location[1])
        self.nodes_density = len(self.listNodes) / ((self.frame[1] - self.frame[0]) * (self.frame[3] - self.frame[2]))
        it = 0

        for target in listTargets:
            target.id = it
            it += 1
            
    def setLevels(self):
        for node in self.listNodes:
            node.level = -1
        tmp1 = []
        tmp2 = []
        for node in self.baseStation.direct_nodes:
            if node.status == 1:
                node.level = 1
                tmp1.append(node)

        for i in range(len(self.targets_active)):
            self.targets_active[i] = 0

        while True:
            if len(tmp1) == 0:
                break
            for node in tmp1:
                for target in node.listTargets:
                    self.targets_active[target.id] = 1
                for neighbor in node.neighbors:
                    if neighbor.status == 1 and neighbor.level == -1:
                        tmp2.append(neighbor)
                        neighbor.level = node.level + 1

            tmp1 = tmp2[:]
            tmp2.clear()
        return

    def reset(self):
        for node in self.listNodes:
            node.energy = 30000
        for mc in self.mc_list:
            mc.reset()

    def operate(self, t=1, optimizer=None):
        request_id = []
        for node in self.listNodes:
            self.env.process(node.operate(t=t))
        self.env.process(self.baseStation.operate(t=t))
        first_step = 0
        energy_warning = self.listNodes[0].threshold * 30

        while True:
            yield self.env.timeout(t / 10.0)
            self.setLevels()
            self.alive = self.check_targets()
            if not self.network_cluster:
                yield self.env.timeout(10)
                self.network_cluster = network_clustering(network=self)
                self.network_cluster_id_node = network_cluster_id_node(network=self)
                optimizer.action_list = self.network_cluster
            yield self.env.timeout(9.0 * t / 10.0)
            for index, node in enumerate(self.listNodes):
                if node.energy <= energy_warning:
                    node.request(optimizer=optimizer, t=t)
                    request_id.append(index)
                else:
                    node.is_request = False
            arr_active_mc = []
            for mc in self.mc_list:
                if mc.cur_action_type == "deactive":
                    arr_active_mc.append(0)
                else:
                    arr_active_mc.append(1)
            a = 0
            b = 0
            len_list_request_before = len(optimizer.list_request)
            if optimizer and self.alive:
                for mc in self.mc_list:
                    mc.runv2(network=self, time_stem=self.env.now, net=self, optimizer=optimizer)
                    if mc.cur_action_type == "charging":
                        for node_id in mc.target_nodes:
                            node = self.listNodes[node_id]
                            print(f"Node {node_id} is being charged by MC.")
            if self.alive == 0:
                break
            dead_nodes = self.get_dead_nodes()
            if dead_nodes:
                print("Các nút có khả năng chết:")
                for node_info in dead_nodes:
                    print(f"Node {node_info['id']} - Năng lượng: {node_info['energy']}")
        return

    def delete_request(self, id_cluster, optimizer):
        for i, item in enumerate(optimizer.list_request):
            for id_node in self.network_cluster_id_node[id_cluster]:
                if item['id'] == id_node:
                    del optimizer.list_request[i]
                    break

    def check_cluster(self, id_node):
        for id_cluster, cluster in enumerate(self.network_cluster_id_node):
            for node_cluster in cluster:
                if node_cluster == id_node:
                    return id_cluster

    def check_targets(self):
        return min(self.targets_active)

    def check_nodes(self):
        tmp = 0
        for node in self.listNodes:
            if node.status == 0:
                tmp += 1
        return tmp

    def avg_network(self):
        sum = 0
        for node in self.listNodes:
            sum += node.energy
        return sum / len(self.listNodes)

    def min_node(self):
        id_node_min = -1
        min_energy = 1000000000
        for id, node in enumerate(self.listNodes):
            if node.energy < min_energy:
                min_energy = node.energy
                id_node_min = id
        return id_node_min
    
    def get_dead_nodes(self):
        dead_nodes = []
        for node in self.listNodes:
            if node.status == 0:
                dead_nodes.append({"id": node.id, "energy": node.energy})
        return dead_nodes