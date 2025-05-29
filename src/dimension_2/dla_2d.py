import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

import generate_field_2d as gf2


# 模拟参数
# 长宽比不重要, 为方便令网格为正方形, 见 class ElectricField
@dataclass
class DLASimulator:
	sei_thickness: np.ndarray = field(init=False)  # 记录每个格点的SEI膜厚度
	sei_growth_rate: float = 0.01                 # SEI膜生长速率
	sei_max_thickness: float = 1.0                # SEI膜最大厚度
	sei_resistance_factor: float = 0.01            # SEI膜对离子通过的阻碍系数
	max_particles: int = 1000					# 最大沉积粒子数
	max_steps_per_particle: int = 10000			# 每个粒子的最大步数
	attach_prob: float = 1.0					# 粘附概率, 后续可以调控
	radius_buffer: int = 5						# 控制生成新粒子的半径缓冲区
	electric_field: gf2.ElectricField2d = None	# 控制电场形状

	cluster_radius: int = field(init=False)

	def __post_init__(self):
		self.grid_size = self.electric_field.grid_size
		self.center = self.grid_size // 2
		self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
		self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
		self.weights = self.weight_culculator()
		self.cluster_radius = 0
		self.sei_thickness = np.zeros((self.grid_size, self.grid_size), dtype=float)

		if self.electric_field.field_type == gf2.FieldType2d.POINT:
			self.grid[self.center, self.center] = 1
		elif self.electric_field.field_type == gf2.FieldType2d.UNIFORM_UP:
			self.grid[:, -1] = 1
		elif self.electric_field.field_type == gf2.FieldType2d.UNIFORM_DOWN:
			self.grid[:, 0] = 1
		elif self.electric_field.field_type == gf2.FieldType2d.UNIFORM_LEFT:
			self.grid[0, :] = 1
		elif self.electric_field.field_type == gf2.FieldType2d.UNIFORM_RIGHT:
			self.grid[-1, :] = 1
		else:
			raise ValueError(f"Unsupported field type: {self.electric_field.field_type}")

	def weight_culculator(self):
		weights = np.zeros((self.grid_size, self.grid_size, len(self.directions)))

		for i in range(self.grid_size):
			for j in range(self.grid_size):
				field_vec = np.array([self.electric_field.field_x[i, j], self.electric_field.field_y[i, j]])
				dot_products = []

				for dx, dy in self.directions:
					ni, nj = i + dx, j + dy
					if self.is_valid(ni, nj):
						dir_vec = np.array([dx, dy], dtype=float)
						dir_vec /= np.linalg.norm(dir_vec)  # 单位向量
						dot = np.dot(field_vec, dir_vec)
						dot_products.append(dot)
					else:
						dot_products.append(float('-inf'))  # 出界方向, 稍后设为 0 权重

				# 归一化前的 softmax 权重（防止数值爆炸）
				max_dot = max(dot for dot in dot_products if dot != float('-inf'))
				exp_weights = [np.exp(dot - max_dot) if dot != float('-inf') else 0.0 for dot in dot_products]

				sum_weights = sum(exp_weights)
				if sum_weights > 0:
					norm_weights = [w / sum_weights for w in exp_weights]
				else:
					norm_weights = [0.0] * len(self.directions)

				weights[i, j, :] = norm_weights

		return weights

	def spawn_particle(self):
		ft = self.electric_field.field_type
		if ft == gf2.FieldType2d.POINT:
			angle = 2 * np.pi * random.random()
			x = int(self.center + (self.cluster_radius + self.radius_buffer) * np.cos(angle))
			y = int(self.center + (self.cluster_radius + self.radius_buffer) * np.sin(angle))
		elif ft == gf2.FieldType2d.UNIFORM_DOWN:
			x = random.randint(0, self.grid_size - 1)
			y = self.grid_size - 1
		elif ft == gf2.FieldType2d.UNIFORM_UP:
			x = random.randint(0, self.grid_size - 1)
			y = 0
		elif ft == gf2.FieldType2d.UNIFORM_LEFT:
			x = self.grid_size - 1
			y = random.randint(0, self.grid_size - 1)
		elif ft == gf2.FieldType2d.UNIFORM_RIGHT:
			x = 0
			y = random.randint(0, self.grid_size - 1)
		else:
			raise ValueError(f"Unknown field type: {ft}")
		return x, y

	def is_valid(self, x, y):
		return 0 <= x < self.grid_size and 0 <= y < self.grid_size

	def is_dendrite(self, x, y):
		return self.is_valid(x, y) and self.grid[x, y] == 1

	def is_adjacent_to_cluster(self, x, y):
		directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
		for dx, dy in directions:
			if self.is_dendrite(x + dx, y + dy):
				return True
		return False

	def calculate_curvature(self, x, y):
		neighbors = 0
		for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
			if self.is_dendrite(x + dx, y + dy):
				neighbors += 1
		return max(0, 4 - neighbors)  # 孤立点曲率=4，平坦区域曲率=0

	def biased_move_with_field(self, x, y):
		dx, dy = random.choices(self.directions, weights=self.weights[x, y])[0]
		return x + dx, y + dy

	def update_sei_thickness(self):
		for x in range(self.grid_size):
			for y in range(self.grid_size):
				if self.is_dendrite(x, y) and self.sei_thickness[x, y] < self.sei_max_thickness:
					growth = self.sei_growth_rate
					curvature = self.calculate_curvature(x, y)
					growth *= (1.0 + curvature)

					# 更新SEI膜厚度（限制最大值）
					self.sei_thickness[x, y] = min(
						self.sei_thickness[x, y] + growth,
						self.sei_max_thickness)

	def simulate(self):
		particle_count = 1

		while particle_count < self.max_particles:
			x, y = self.spawn_particle()
			steps = 0

			while steps < self.max_steps_per_particle:
				x, y = self.biased_move_with_field(x, y)
				self.update_sei_thickness()  # 更新SEI膜厚度
				steps += 1

				if not self.is_valid(x, y):
					break

				if self.is_adjacent_to_cluster(x, y) and not self.is_dendrite(x, y):
					base_prob = self.attach_prob
					thickness=0
					directions = [(1,0), (0,1), (-1,0), (0,-1)]
					for dx, dy in directions:
						nx, ny = x + dx, y + dy
						if self.is_valid(nx, ny):
							thickness +=self.sei_thickness[nx,ny]
					sei_effect = np.exp(-self.sei_resistance_factor * thickness)
					effective_prob = base_prob * sei_effect

					if random.random() < effective_prob:
						self.grid[x, y] = 1
						dist = int(np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2))
						self.cluster_radius = max(self.cluster_radius, dist)
						particle_count += 1
						break

	def plot_cluster(self):
		plt.figure(figsize=(6, 6))
		plt.imshow(self.grid.T, cmap='binary', origin='lower')
		plt.title("Simulated Lithium Dendrite Growth")
		plt.axis('off')
		plt.show()


def test_point():
	# 点电荷在中心
	field_point = gf2.ElectricField2d(
		field_type=gf2.FieldType2d.POINT,
		grid_size=201,
		strength=1.0
	)

	dla_point = DLASimulator(
		max_particles = 1000,
		max_steps_per_particle = 10000,
		attach_prob = 1.0,
		radius_buffer = 5,
		electric_field= field_point
	)

	# 运行模拟
	dla_point.simulate()
	dla_point.plot_cluster()


def test_parallel():
	# 平行板电场
	field_parallel = gf2.ElectricField2d(
		field_type=gf2.FieldType2d.UNIFORM_DOWN,
		grid_size=201,
		strength=1.0
	)

	dla_parallel = DLASimulator(
		max_particles = 1000,
		max_steps_per_particle = 10000,
		attach_prob = 1.0,
		electric_field= field_parallel
	)

	# 运行模拟
	dla_parallel.simulate()
	dla_parallel.plot_cluster()


if __name__ == "__main__":
	test_point()
	test_parallel()
