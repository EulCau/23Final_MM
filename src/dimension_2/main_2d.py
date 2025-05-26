import random

import matplotlib.pyplot as plt
import numpy as np

import generate_field as gf

# 模拟参数
grid_size = 201  # 网格大小, 奇数以便居中
center = grid_size // 2
max_particles = 1000  # 最大沉积粒子数
max_steps_per_particle = 10000  # 每个粒子的最大步数
attach_prob = 1.0  # 粘附概率, 后续可以调控
radius_buffer = 5  # 控制生成新粒子的半径缓冲区

# 初始化网格 (0为空, 1为枝晶)
grid = np.zeros((grid_size, grid_size), dtype=int)

# 用于控制粒子的生成半径
cluster_radius = 5

def spawn_particle(radius, field_type="point"):
	if field_type == "point":
		# 点电荷: 从团簇外围随机角度注入
		angle = 2 * np.pi * random.random()
		x = int(center + (radius + radius_buffer) * np.cos(angle))
		y = int(center + (radius + radius_buffer) * np.sin(angle))
	elif field_type == "uniform_down":
		# 匀强场竖直向下: 从顶部一条水平线随机注入
		x = random.randint(0, grid_size - 1)
		y = grid_size - 1  # 顶部注入
	elif field_type == "uniform_up":
		x = random.randint(0, grid_size - 1)
		y = 0  # 底部注入
	elif field_type == "uniform_left":
		x = grid_size - 1
		y = random.randint(0, grid_size - 1)
	elif field_type == "uniform_right":
		x = 0
		y = random.randint(0, grid_size - 1)
	else:
		raise ValueError(f"Unknown field type: {field_type}")
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
	ex = field_x[x, y]
	ey = field_y[x, y]
	alpha = abs(ex) + abs(ey)
	if alpha == 0:
		return x, y

	for dx, dy in directions:
		nx, ny = x + dx, y + dy
		if is_valid(nx, ny):
			dot = dx * ex + dy * ey  # 电场方向与移动方向的投影
			weight = np.exp(dot / alpha)  # 放大权重
			weights.append(weight)
		else:
			weights.append(0.0)

	total = sum(weights)
	if total == 0:
		return x, y

	probs = [w / total for w in weights]
	dx, dy = random.choices(directions, weights=probs)[0]
	return x + dx, y + dy


def dla_with_custom_field(field_x, field_y, field_type="point"):
	global cluster_radius
	particle_count = 1

	while particle_count < max_particles:
		x, y = spawn_particle(cluster_radius, field_type)
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
	plt.imshow(grid.T, cmap='binary', origin='lower')
	plt.title("Simulated Lithium Dendrite Growth")
	plt.axis('off')
	plt.show()


# 坐标转换总是有问题, 我每天再想
# def plot_field(field_x, field_y, stride=10):
# 	X, Y = np.mgrid[0:grid_size, 0:grid_size]
# 	plt.figure(figsize=(6, 6))
# 	plt.quiver(X[::stride, ::stride], -Y[::stride, ::stride],
# 			   field_x[::stride, ::stride], field_y[::stride, ::stride],
# 			   pivot='middle', color='red', alpha=0.6)
# 	plt.title("Electric Field Visualization")
# 	plt.gca().invert_yaxis()
# 	plt.show()
#
#
# def plot_field_2(field_x, field_y, step=5):
# 	plt.figure(figsize=(6, 6))
# 	X, Y = np.meshgrid(np.arange(0, grid_size), np.arange(0, grid_size))
#
# 	# 归一化箭头长度
# 	magnitude = np.sqrt(field_x ** 2 + field_y ** 2)
# 	field_x_vis = field_x / (magnitude + 1e-6)
# 	field_y_vis = field_y / (magnitude + 1e-6)
#
# 	# 使用转置绘制笛卡尔坐标系下的
# 	plt.quiver(X[::step, ::step], Y[::step, ::step],
# 			   field_x_vis.T[::step, ::step], field_y_vis.T[::step, ::step],
# 			   pivot='middle', color='blue', scale=20)
#
# 	plt.xlim(0, grid_size)
# 	plt.ylim(0, grid_size)
# 	plt.xlabel("x")
# 	plt.ylabel("y")
# 	plt.title("Electric Field Vectors")
# 	plt.gca().set_aspect('equal')
# 	plt.grid(False)
# 	plt.show()
#
#
# def plot_field_3(field_x, field_y, step=5, scale=1.0, scale_factor=5.0):
# 	grid_size_x, grid_size_y = field_x.shape
# 	X, Y = np.meshgrid(np.arange(grid_size_y), np.arange(grid_size_x))  # 注意 x 是 axis=0
#
# 	# 放大电场用于显示
# 	fx_plot = field_x * scale_factor
# 	fy_plot = field_y * scale_factor
#
# 	plt.figure(figsize=(6, 6))
# 	plt.quiver(X[::step, ::step], Y[::step, ::step],
# 			   fy_plot[::step, ::step], fx_plot[::step, ::step],
# 			   pivot='middle', color='blue', scale=scale)
#
# 	plt.xlim(0, grid_size_y)
# 	plt.ylim(0, grid_size_x)
# 	plt.gca().set_aspect('equal')
# 	plt.title("Electric Field Visualization (X first indexing)")
# 	plt.xlabel("x")
# 	plt.ylabel("y")
# 	plt.grid(False)
# 	plt.show()


def test_point():
	# 点电荷在中心
	field_x, field_y = gf.point_field(grid_size, charge_pos=(center, center), k=1.0)
	grid[center, center] = 1  # 初始晶核

	# plot_field_3(field_x, field_y)

	# 运行模拟
	dla_with_custom_field(field_x, field_y, "point")
	plot_cluster()


def test_parallel():
	# 平行板电场
	field_x, field_y = gf.uniform_field(grid_size, direction="down", strength=1.0)
	for i in range(grid_size):
		grid[i, 0] = 1  # 初始晶核

	# plot_field_3(field_x, field_y)

	# 运行模拟
	dla_with_custom_field(field_x, field_y, "uniform_down")
	plot_cluster()


if __name__ == "__main__":
	# 两个 test 暂时不能同时进行, 因为运行一个之后会改变全局变量 grid. 明天我会重构代码, 改变参数设置的方式
	test_point()
	# test_parallel()
