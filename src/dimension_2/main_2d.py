import numpy as np
import matplotlib.pyplot as plt
import random

import generate_field as gf

# 模拟参数
grid_size = 201  # 网格大小，奇数以便居中
center = grid_size // 2
max_particles = 1000  # 最大沉积粒子数
max_steps_per_particle = 10000  # 每个粒子的最大步数
attach_prob = 1.0  # 粘附概率，后续可以调控
radius_buffer = 5  # 控制生成新粒子的半径缓冲区

# 初始化网格（0为空，1为枝晶）
grid = np.zeros((grid_size, grid_size), dtype=int)
grid[center, center] = 1  # 初始晶核

# 用于控制粒子的生成半径
cluster_radius = 5

def spawn_particle(radius):
	angle = 2 * np.pi * random.random()
	x = int(center + (radius + radius_buffer) * np.cos(angle))
	y = int(center + (radius + radius_buffer) * np.sin(angle))
	return x, y

def is_valid(x, y):
	return 0 <= x < grid_size and 0 <= y < grid_size

def is_adjacent_to_cluster(x, y):
	for dx in [-1, 0, 1]:
		for dy in [-1, 0, 1]:
			if (dx != 0 or dy != 0) and is_valid(x + dx, y + dy):
				if grid[x + dx, y + dy] == 1:
					return True
	return False


def biased_move_with_field(x, y, field_x, field_y):
	directions = [(1,0), (-1,0), (0,1), (0,-1)]
	weights = []

	for dx, dy in directions:
		nx, ny = x + dx, y + dy
		if is_valid(nx, ny):
			ex = field_x[x, y]
			ey = field_y[x, y]
			dot = dx * ex + dy * ey  # 电场方向与移动方向的投影
			weight = max(0.001, dot)  # 投影越大越偏好
			weights.append(weight)
		else:
			weights.append(0.0)

	if sum(weights) == 0:
		return x, y

	probs = [w / sum(weights) for w in weights]
	dx, dy = random.choices(directions, weights=probs)[0]
	return x + dx, y + dy


def dla_with_custom_field(field_x, field_y):
	global cluster_radius
	particle_count = 1
	if particle_count == 2:
		print(2)

	while particle_count < max_particles:
		x, y = spawn_particle(cluster_radius)
		steps = 0

		while steps < max_steps_per_particle:
			x, y = biased_move_with_field(x, y, field_x, field_y)
			steps += 1

			if not is_valid(x, y):
				break

			if is_adjacent_to_cluster(x, y) and random.random() < attach_prob:
				grid[x, y] = 1
				cluster_radius = max(cluster_radius, int(np.sqrt((x - center)**2 + (y - center)**2)))
				particle_count += 1
				break


def plot_cluster():
	plt.figure(figsize=(6, 6))
	plt.imshow(grid, cmap='binary', origin='lower')
	plt.title("Simulated Lithium Dendrite Growth")
	plt.axis('off')
	plt.show()


# 点电荷在中心
field_x, field_y = gf.point_charge(grid_size, charge_pos=(center, center), k=1.0)
# 或平行板电场
# field_x, field_y = generate_parallel_plate_field(grid_size, direction="down", strength=1.0)

# 运行模拟
dla_with_custom_field(field_x, field_y)
plot_cluster()
