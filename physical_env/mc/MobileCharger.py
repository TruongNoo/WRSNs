import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import copy

class MobileCharger:
    """
    Lớp đại diện cho một bộ sạc di động để sạc các nút trong mạng.

    Attributes:
        location (np.array): Vị trí hiện tại của bộ sạc di động.
        energy (float): Năng lượng hiện tại của bộ sạc di động.
        capacity (float): Năng lượng tối đa mà bộ sạc di động có thể lưu trữ.
        alpha (float): Tham số alpha trong tính toán năng lượng.
        beta (float): Tham số beta trong tính toán năng lượng.
        threshold (float): Ngưỡng năng lượng tối thiểu trước khi cần sạc lại.
        velocity (float): Vận tốc di chuyển của bộ sạc di động.
        pm (float): Tham số pm trong tính toán năng lượng.
        chargingRate (float): Tốc độ sạc của bộ sạc di động.
        chargingRange (float): Phạm vi sạc của bộ sạc di động.
        epsilon (float): Tham số epsilon trong tính toán.
        status (int): Trạng thái hiện tại của bộ sạc di động.
        connected_nodes (list): Danh sách các nút cảm biến đang kết nối để sạc.
        incentive (float): Lợi ích hoặc phần thưởng từ việc sạc.
        end (np.array): Điểm đến của bộ sạc di động.
        start (np.array): Điểm bắt đầu của bộ sạc di động.
        state (int): Trạng thái hiện tại của bộ sạc di động.
        q_table (list): Bảng Q-learning cho việc điều khiển.
        next_phy_action (list): Hành động vật lý tiếp theo của bộ sạc di động.
        save_state (list): Danh sách các trạng thái đã lưu.
        e (float): Tham số e trong tính toán.
        is_active (bool): Trạng thái hoạt động của bộ sạc di động.
        is_self_charge (bool): Trạng thái tự sạc của bộ sạc di động.
        is_stand (bool): Trạng thái đứng yên của bộ sạc di động.
        current (np.array): Vị trí hiện tại của bộ sạc di động.
        end_time (float): Thời gian kết thúc hành động hiện tại.
        moving_time (float): Thời gian di chuyển.
        arrival_time (float): Thời gian đến điểm đến.
        e_move (float): Năng lượng tiêu hao khi di chuyển.
        next_location (list): Vị trí tiếp theo của bộ sạc di động.
    """

    def __init__(self, location, mc_phy_spe):
        """
        Khởi tạo đối tượng MobileCharger.

        Parameters:
            location (tuple): Vị trí ban đầu của bộ sạc di động.
            mc_phy_spe (dict): Thông số vật lý của bộ sạc di động.
        """
        self.chargingTime = 0
        self.env = None
        self.net = None
        self.id = None
        self.cur_phy_action = [500, 500, 0]
        self.location = np.array(location)
        self.energy = mc_phy_spe['capacity']
        self.capacity = mc_phy_spe['capacity']

        self.alpha = mc_phy_spe['alpha']
        self.beta = mc_phy_spe['beta']
        self.threshold = mc_phy_spe['threshold']
        self.velocity = mc_phy_spe['velocity']
        self.pm = mc_phy_spe['pm']
        self.chargingRate = 0
        self.chargingRange = mc_phy_spe['charging_range']
        self.epsilon = mc_phy_spe['epsilon']
        self.status = 1
        self.checkStatus()
        self.cur_action_type = "deactive"
        self.connected_nodes = []
        self.incentive = 0
        self.end = self.location
        self.start = self.location
        self.state = 30
        self.q_table = []
        self.next_phy_action = [500, 500, 0]
        self.save_state = []
        self.e = mc_phy_spe['e']
        self.is_active = False
        self.is_self_charge = False
        self.is_stand = False
        self.current = self.location
        self.end_time = 0
        self.moving_time = 0
        self.arrival_time = 0
        self.e_move = mc_phy_spe['velocity']
        self.next_location = [500, 500]

    def charge_step(self, t):
        """
        Thực hiện bước sạc trong thời gian t.

        Parameters:
            t (float): Thời gian sạc.

        Returns:
            None
        """
        for node in self.connected_nodes:
            node.charger_connection(self)
        yield self.env.timeout(t)
        self.energy = self.energy - self.chargingRate * t
        self.cur_phy_action[2] = max(0, self.cur_phy_action[2] - t)
        for node in self.connected_nodes:
            node.charger_disconnection(self)
        self.chargingRate = 0
        return

    def chargev2(self, net):
        """
        Thực hiện sạc các nút trong mạng.

        Parameters:
            net (object): Đối tượng mạng.

        Returns:
            None
        """
        for nd in net.listNodes:
            p = nd.charge(mc=self)
            self.energy -= p

    def update_location(self):
        """
        Cập nhật vị trí hiện tại của bộ sạc di động.

        Returns:
            None
        """
        self.current = self.get_location()
        self.energy -= self.e_move

    def get_location(mc):
        """
        Lấy vị trí hiện tại của bộ sạc di động.

        Parameters:
            mc (MobileCharger): Đối tượng MobileCharger.

        Returns:
            tuple: Vị trí hiện tại của bộ sạc di động.
        """
        d = distance.euclidean(mc.start, mc.end)
        time_move = d / mc.velocity
        if time_move == 0:
            return mc.current
        elif distance.euclidean(mc.current, mc.end) < 10 ** -3:
            return mc.end
        else:
            x_hat = (mc.end[0] - mc.start[0]) / time_move + mc.current[0]
            y_hat = (mc.end[1] - mc.start[1]) / time_move + mc.current[1]
            if (mc.end[0] - mc.current[0]) * (mc.end[0] - x_hat) < 0 or (
                    (mc.end[0] - mc.current[0]) * (mc.end[0] - x_hat) == 0 and (mc.end[1] - mc.current[1]) * (
                    mc.end[1] - y_hat) <= 0):
                return mc.end
            else:
                return x_hat, y_hat

    def move_step(self, vector, t):
        """
        Thực hiện bước di chuyển trong thời gian t.

        Parameters:
            vector (np.array): Vector di chuyển.
            t (float): Thời gian di chuyển.

        Returns:
            None
        """
        yield self.env.timeout(t)
        self.location = self.location + vector
        self.energy -= self.pm * t * self.velocity

    def move(self, destination):
        """
        Di chuyển đến vị trí đích.

        Parameters:
            destination (np.array): Vị trí đích.

        Returns:
            float: Thời gian đến vị trí đích.
        """
        moving_time = euclidean(destination, self.location) / self.velocity
        self.arrival_time = moving_time
        moving_vector = destination - self.location
        total_moving_time = moving_time
        while True:
            if moving_time <= 0:
                break
            if self.status == 0:
                yield self.env.timeout(moving_time)
                break
            moving_time = euclidean(destination, self.location) / self.velocity
            span = min(min(moving_time, 1.0), (self.energy - self.threshold) / (self.pm * self.velocity))
            yield self.env.process(self.move_step(moving_vector / total_moving_time * span, t=span))
            moving_time -= span
            self.checkStatus()
        return self.arrival_time

    def move_time(self, destination):
        """
        Tính toán thời gian di chuyển đến vị trí đích.

        Parameters:
            destination (np.array): Vị trí đích.

        Returns:
            float: Thời gian đến vị trí đích.
        """
        moving_time = euclidean(destination, self.location) / self.velocity
        self.arrival_time = moving_time
        return self.arrival_time

    def recharge(self):
        """
        Sạc lại năng lượng khi năng lượng thấp.

        Returns:
            None
        """
        print("Energy of MC is low, need to come back base station and re-charge")
        if euclidean(self.location, self.net.baseStation.location) <= self.epsilon:
            self.location = copy.deepcopy(self.net.baseStation.location)
            self.energy = self.capacity
        self.is_self_charge = True
        yield self.env.timeout(0)

    def update_q_table(self, optimizer, net, time_stem):
        """
        Cập nhật bảng Q-learning.

        Parameters:
            optimizer (object): Đối tượng tối ưu hóa.
            net (object): Đối tượng mạng.
            time_stem (float): Thời gian hiện tại.

        Returns:
            float: Giá trị cập nhật.
        """
        result = optimizer.update_v2(self, net, time_stem)
        self.q_table = result[1]
        self.next_phy_action = []
        self.next_phy_action = [result[2][0], result[2][1], result[3]]
        return result[0]

    def check_cur_action(self):
        """
        Kiểm tra hành động hiện tại của bộ sạc di động.

        Returns:
            None
        """
        if not self.cur_action_type == 'moving':
            self.cur_action_type = 'charging'
        elif not self.cur_action_type == 'charging':
            self.cur_action_type = 'recharging'
        elif not self.cur_action_type == 'recharging':
            self.cur_action_type = 'deactive'

    def get_status(self):
        """
        Lấy trạng thái hiện tại của bộ sạc di động.

        Returns:
            str: Trạng thái hiện tại của bộ sạc di động.
        """
        if not self.is_active:
            return "deactivated"
        if not self.is_stand:
            return "moving"
        if not self.is_self_charge:
            return "charging"
        return "self_charging"

    def checkStatus(self):
        """
        Kiểm tra trạng thái của bộ sạc di động.

        Returns:
            None
        """
        if self.energy <= self.threshold:
            self.status = 0
            self.energy = self.threshold

    def get_next_location(self, network, time_stem, optimizer=None):
        """
        Lấy vị trí tiếp theo của bộ sạc di động.

        Parameters:
            network (object): Đối tượng mạng.
            time_stem (float): Thời gian hiện tại.
            optimizer (object): Đối tượng tối ưu hóa (tuỳ chọn).

        Returns:
            None
        """
        next_location, charging_time = optimizer.update(self, network, time_stem)
        self.start = self.current
        self.end = next_location
        self.moving_time = distance.euclidean(self.location, self.end) / self.velocity
        self.end_time = time_stem + self.moving_time + charging_time
        print("[Mobile Charger] MC end time {}".format(self.end_time))
        self.chargingTime = charging_time
        self.arrival_time = time_stem + self.moving_time

    def runv2(self, network, time_stem, net=None, optimizer=None):
        """
        Chạy phiên bản 2 của bộ sạc di động.

        Parameters:
            network (object): Đối tượng mạng.
            time_stem (float): Thời gian hiện tại.
            net (object): Đối tượng mạng (tuỳ chọn).
            optimizer (object): Đối tượng tối ưu hóa (tuỳ chọn).

        Returns:
            None
        """
        if (((not self.is_active) and optimizer.list_request) or (np.abs(time_stem - self.end_time) < 1)):
            self.is_active = True
            new_list_request = []
            for request in optimizer.list_request:
                if net.listNodes[request["id"]].energy < net.listNodes[request["id"]].threshold * 30:
                    new_list_request.append(request)
                else:
                    net.listNodes[request["id"]].is_request = False
            optimizer.list_request = new_list_request
            if not optimizer.list_request:
                self.is_active = False
            self.get_next_location(network=network, time_stem=time_stem, optimizer=optimizer)
        else:
            if self.is_active:
                if not self.is_stand:
                    self.update_location()
                elif not self.is_self_charge:
                    self.chargev2(net)
                else:
                    self.recharge()
        if np.any(self.energy < self.threshold) and not self.is_self_charge and np.any(self.end != self.net.baseStation.location):
            self.start = self.current
            self.end = self.net.baseStation.location
            self.is_stand = False
            charging_time = 0
            moving_time = distance.euclidean(self.start, self.end) / self.velocity
            self.end_time = time_stem + moving_time + charging_time
        self.check_state()

    def __str__(self):
        """
        Trả về chuỗi mô tả đối tượng MobileCharger.

        Returns:
            str: Chuỗi mô tả đối tượng MobileCharger.
        """
        return f"MobileCharger(location={self.location}, cur_action_type={self.cur_action_type})"

    def check_state(self):
        """
        Kiểm tra và cập nhật trạng thái của bộ sạc di động.

        Returns:
            None
        """
        if distance.euclidean(self.current, self.end) < 1:
            self.is_stand = True
            self.current = self.end
        else:
            self.is_stand = False
        if distance.euclidean(self.net.baseStation.location, self.end) < 10 ** -3:
            self.is_self_charge = True
        else:
            self.is_self_charge = False
