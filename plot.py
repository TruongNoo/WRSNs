import matplotlib.pyplot as plt
import numpy as np

# Data for the first plot
categories = ['50', '100', '150', '200']
targets_monitored = [1, 2, 3, 4]
q_learning_epsilon_greedy = [15603, 5150, 4000, 1536]
q_learning_tham_lam = [14300, 4850, 3818, 1521] 
sarsa = [14550, 4950, 3950, 1450]
# 4865, 
# Data for the second plot (epsilon values)
epsilon0 = [14300, 4850, 3800, 1500]
epsilon001 = [15603, 5150, 4000, 1536]
epsilon01 = [14900, 4700, 3600, 1500]
epsilon02 = [14800, 4700, 3600, 1500]
epsilon03 = [14500, 4700, 3550, 1500]
epsilon05 = [13200, 4600, 3550, 1450]

# First plot: Comparison of Q-Learning and Sarsa Methods
plt.figure(figsize=(10, 6))
plt.plot(targets_monitored, q_learning_epsilon_greedy, marker='o', label='Q-Learning-epsilon greedy')
plt.plot(targets_monitored, q_learning_tham_lam, marker='o', label='Thuật toán tham lam')
plt.plot(targets_monitored, sarsa, marker='o', label='Sarsa')

plt.xlabel('Số lượng mục tiêu giám sát')
plt.ylabel('Thời gian sống của mạng')
plt.title('So sánh thuật toán')
plt.xticks(targets_monitored, categories)
plt.legend()
plt.grid(True)
plt.savefig('comparison_q_learning_sarsa.png')
plt.show()

# Second plot: Effect of Epsilon on Network Lifetime
plt.figure(figsize=(10, 6))
plt.plot(targets_monitored, epsilon0, marker='o', label='epsilon = 0')
plt.plot(targets_monitored, epsilon001, marker='o', label='epsilon = 0.01')
plt.plot(targets_monitored, epsilon01, marker='o', label='epsilon = 0.1')
plt.plot(targets_monitored, epsilon02, marker='o', label='epsilon = 0.2')
plt.plot(targets_monitored, epsilon03, marker='o', label='epsilon = 0.3')
plt.plot(targets_monitored, epsilon05, marker='o', label='epsilon = 0.5')

plt.xlabel('Số lượng mục tiêu giám sát')
plt.ylabel('Thời gian sống của mạng')
plt.title('Ảnh hưởng của epsilon đến thời gian sống của mạng')
plt.xticks(targets_monitored, categories)
plt.legend()
plt.grid(True)
plt.savefig('effect_of_epsilon_on_network_lifetime.png')
plt.show()

q_learning_tham_lam = [14300, 4850, 3818, 1534]
no_charge = [11622, 4603, 3550, 1490]
plt.figure(figsize=(10, 6))
plt.plot(targets_monitored, q_learning_epsilon_greedy, marker='o', label='Q-Learning')
plt.plot(targets_monitored, q_learning_tham_lam, marker='o', label='Thuật toán tham lam')
plt.plot(targets_monitored, no_charge, marker='o', label='No charge')

plt.xlabel('Số lượng mục tiêu giám sát')
plt.ylabel('Thời gian sống của mạng')
plt.title('So sánh thuật toán')
plt.xticks(targets_monitored, categories)
plt.legend()
plt.grid(True)
plt.savefig('comparison_qlearning_thamlam_nocharge.png')
plt.show()

# alpha: 15481s, 15702s, 14941s, 14570s, 13871s, 14621s, 14621, 13782s, 12695s
# gamma: 15481s, 14994s, 14718s, 14559s, 13479s, 13481s, 13489s, 13489s, 13489s

# Second plot: Effect of Epsilon on Network Lifetime
# Data for alpha and gamma
# alpha = [15481, 15702, 14941, 14570, 13871, 14621, 14621, 13782, 12695]
gamma = [15428, 15481, 14787, 14994, 15102, 14718, 14567, 14559, 13479, 13479, 13479,13481, 13481,13489, 13489, 13489, 13489,13489]
# 4873,  4860, 4882, 4803, 4823, 4797, 4790,  4777, 4777, 4777, 

# Categories for alpha and gamma
alpha_gamma_categories = ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9']
targets_monitored = list(range(1, 19))

# Plotting the data
plt.figure(figsize=(10, 6))
# plt.plot(targets_monitored, alpha, marker='o', label='alpha')
plt.plot(targets_monitored, gamma, marker='o', label='gamma')

plt.xlabel('alpha và gamma')
plt.ylabel('Thời gian sống của mạng')
plt.title('Ảnh hưởng của 2 tham số alpha và gamma đến thời gian sống của mạng')
plt.xticks(targets_monitored, alpha_gamma_categories)
plt.legend()
plt.grid(True)
plt.savefig('effect_of_alpha_gamma_on_network_lifetime.png')
plt.show()