import random
import numpy as np
from scipy.spatial import distance
from optimizer.utils import init_function, q_max_function, reward_function, network_clustering_v2
from physical_env.network import Node

class Q_learning:
    """
    Thuật toán học Q-learning cho tối ưu hóa năng lượng trong một môi trường mạng.

    Args:
        init_func (function): Hàm khởi tạo bảng Q.
        nb_action (int): Số lượng hành động.
        alpha (float): Tỉ lệ học.
        q_alpha (float): Trọng số cho việc cập nhật giá trị Q.
        q_gamma (float): Hệ số chiết khấu cho các phần thưởng tương lai.
        epsilon (float): Tỷ lệ khám phá.
        load_checkpoint (bool): Có nên tải dữ liệu kiểm tra không.
        net (object): Đối tượng mạng.

    Attributes:
        action_list (numpy.ndarray): Mảng các hành động có thể thực hiện.
        nb_action (int): Số lượng hành động.
        q_table (numpy.ndarray): Bảng Q để lưu trữ các giá trị Q.
        charging_time (list): Danh sách thời gian sạc cho mỗi hành động.
        reward (numpy.ndarray): Mảng các phần thưởng cho mỗi hành động.
        reward_max (list): Danh sách các phần thưởng tối đa cho mỗi hành động.
        list_request (list): Danh sách yêu cầu.
        alpha (float): Tỉ lệ học.
        q_alpha (float): Trọng số cho việc cập nhật giá trị Q.
        q_gamma (float): Hệ số chiết khấu cho các phần thưởng tương lai.
        epsilon (float): Tỷ lệ khám phá.

    Methods:
        reset_q_table(): Đặt lại bảng Q.
        update_v2(mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function): Cập nhật bảng Q và chọn hành động tiếp theo.
        update(mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function): Cập nhật bảng Q và chọn hành động tiếp theo.
        q_max(mc, q_max_func=q_max_function): Tính toán giá trị Q lớn nhất.
        set_reward(mc=None, time_stem=0, network=None): Đặt các phần thưởng cho mỗi hành động.
        choose_next_state(mc, network): Chọn trạng thái tiếp theo dựa trên giá trị Q.
        choose_next_state_v2(mc, network): Chọn trạng thái tiếp theo dựa trên chính sách epsilon-greedy.
    """

    def __init__(self, init_func=init_function, nb_action=31, alpha=0.5, q_alpha=0.1, q_gamma=0.01, 
                 epsilon=0, load_checkpoint=False, net=None):
        self.action_list = np.zeros(nb_action + 1)
        self.nb_action = nb_action + 1
        self.q_table = init_func(nb_action=nb_action + 1)
        self.charging_time = [0.0 for _ in range(nb_action + 1)]
        self.reward = np.asarray([0.0 for _ in range(nb_action + 1)])
        self.reward_max = [0.0 for _ in range(nb_action + 1)]
        self.list_request = []
        self.alpha = alpha
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.epsilon = epsilon
    
    def reset_q_table(self):
        """
        Đặt lại bảng Q.
        """
        self.q_table = init_function(nb_action=self.nb_action)

    def update_v2(self, mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function):
        """
        Cập nhật bảng Q và chọn hành động tiếp theo.

        Args:
            mc (object): Đối tượng MC.
            network (object): Đối tượng mạng.
            time_stem (int): Thời gian.
            alpha (float): Tỉ lệ học.
            gamma (float): Hệ số chiết khấu.
            q_max_func (function): Hàm tính giá trị Q lớn nhất.

        Returns:
            tuple: Bao gồm giá trị Q lớn nhất, bảng Q được cập nhật, vị trí của MC và thời gian sạc.
        """
        if mc.state == -1:
            return np.max(self.q_table), self.q_table, network.baseStation.location, 0

        self.set_reward(mc=mc, time_stem=time_stem, network=network)
        temp = self.q_max(mc, q_max_func)
        self.q_table[mc.state] = (1 - self.q_alpha) * self.q_table[mc.state] + self.q_alpha * (
            self.reward)
        self.choose_next_state_v2(mc, network)
        charging_time = self.charging_time[mc.state]
        return np.max(self.q_table), self.q_table, self.action_list[mc.state], charging_time

    def update(self, mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function):
        """
        Cập nhật bảng Q và chọn hành động tiếp theo.

        Args:
            mc (object): Đối tượng MC.
            network (object): Đối tượng mạng.
            time_stem (int): Thời gian.
            alpha (float): Tỉ lệ học.
            gamma (float): Hệ số chiết khấu.
            q_max_func (function): Hàm tính giá trị Q lớn nhất.

        Returns:
            tuple: Bao gồm hành động được chọn và thời gian sạc.
        """
        if not self.list_request:
            return self.action_list[mc.state], 0
        self.set_reward(mc=mc, time_stem=time_stem, network=network)
        self.q_table[mc.state] = (1 - self.q_alpha) * self.q_table[mc.state] + self.q_alpha * (
                self.reward + self.q_gamma * self.q_max(mc, q_max_func))
        self.choose_next_state(mc, network)
        if mc.state == len(self.action_list) - 1:
            charging_time = 0
        else:
            charging_time = self.charging_time[mc.state]
        print("[Optimizer] MC is sent to point {} (id={}) and charge for {:.2f}s".format(self.action_list[mc.state], mc.state, charging_time))
        return self.action_list[mc.state], charging_time

    def q_max(self, mc, q_max_func=q_max_function):
        """
        Tính toán giá trị Q lớn nhất.

        Args:
            mc (object): Đối tượng MC.
            q_max_func (function): Hàm tính giá trị Q lớn nhất.

        Returns:
            float: Giá trị Q lớn nhất.
        """
        return q_max_function(q_table=self.q_table, state=mc.state)

    def set_reward(self, mc=None, time_stem=0, network=None):
        """
        Đặt các phần thưởng cho mỗi hành động.

        Args:
            mc (object): Đối tượng MC.
            time_stem (int): Thời gian.
            network (object): Đối tượng mạng.
        """
        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)
        third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index, row in enumerate(self.q_table):
            temp = reward_function(network=network, mc=mc, q_learning=self, state=index, time_stem=time_stem)
            first[index] = temp[0]
            second[index] = temp[1]
            third[index] = temp[2]
            self.charging_time[index] = temp[3]
        first = first / np.sum(first)
        second = second / np.sum(second)
        third = third / np.sum(third)
        self.reward = first + second + third
        self.reward_max = list(zip(first, second, third))

    def choose_next_state(self, mc, network):
        """
        Chọn trạng thái tiếp theo dựa trên giá trị Q.

        Args:
            mc (object): Đối tượng MC.
            network (object): Đối tượng mạng.
        """
        if mc.energy < 540:
            mc.state = len(self.q_table) - 1
            print('[Optimizer] MC energy is running low ({:.2f}), and needs to rest!'.format(mc.energy))
        else:
            if random.uniform(0, 1) < self.epsilon:
                mc.state = random.randrange(len(self.q_table)-1)
                print('[Optimizer] Randomly choosing next state {} due to epsilon-greedy policy.'.format(mc.state))
            else:
                mc.state = np.argmax(self.q_table[mc.state])
                print('[Optimizer] Choosing next state {} based on Q-table.'.format(mc.state))
            if mc.state == len(self.q_table) - 1:
                mc.state = random.randrange(len(self.q_table)-1)

    def choose_next_state_v2(self, mc, network):
        """
        Chọn trạng thái tiếp theo dựa trên chính sách epsilon-greedy.

        Args:
            mc (object): Đối tượng MC.
            network (object): Đối tượng mạng.
        """
        if random.uniform(0, 1) < self.epsilon:
            mc.state = random.randrange(len(self.q_table)-1)
            print('[Optimizer] Randomly choosing next state {} due to epsilon-greedy policy.'.format(mc.state))
        else:
            mc.state = np.nanargmax(self.q_table[mc.state])
            print('[Optimizer] Choosing next state {} based on Q-table.'.format(mc.state))
