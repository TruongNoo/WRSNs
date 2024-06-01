from scipy.spatial.distance import euclidean
import numpy as np

class BaseStation:
    def __init__(self, location):
        """
        Việc khởi tạo cho trạm cơ sở (BS)
        :param location: tọa độ của trạm cơ sở
        """
        self.env = None
        self.net = None
        self.location = np.array(location)
        self.monitored_target = []
        self.direct_nodes = []

    def probe_neighbors(self):
        for node in self.net.listNodes:
            if euclidean(self.location, node.location) <= node.com_range:
                self.direct_nodes.append(node)

    def receive_package(self, package):
        return

    def operate(self, t=1):
        self.probe_neighbors()
        while True:
            yield self.env.timeout(t)