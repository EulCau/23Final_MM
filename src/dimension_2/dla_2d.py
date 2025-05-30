import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

import generate_field_2d as gf2


# 模拟参数
# 长宽比不重要, 为方便令网格为正方形, 见 class ElectricField
@dataclass
class DLASimulator:
	sei_growth_rate: float = 0.01					# SEI膜生长速率
	sei_max_thickness: float = 1.0					# SEI膜最大厚度
	sei_resistance_factor: float = 0.1				# SEI膜对离子通过的阻碍系数
	max_particles: int = 1000						# 最大沉积粒子数
	max_steps_per_particle: int = 10000				# 每个粒子的最大步数
	attach_prob: float = 1.0						# 粘附概率, 后续可以调控
	radius_buffer: int = 5							# 控制生成新粒子的半径缓冲区
	electric_field: gf2.ElectricField2d = None		# 控制电场形状

	sei_thickness: np.ndarray = field(init=False)	# 记录每个格点的SEI膜厚度
	curvature: np.ndarray = field(init=False)
	cluster_radius: int = field(init=False)

	def __post_init__(self):
		self.grid_size = self.electric_field.grid_size
		self.center = self.grid_size // 2
		self.directions_edge = [(1, 0), (0, 1), (-1, 0), (0, -1)]
		self.directions_vertex = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
		self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
		self.weights = self.weight_calculator()
		self.sei_thickness = np.zeros((self.grid_size, self.grid_size), dtype=float)
		self.curvature = np.zeros((self.grid_size, self.grid_size), dtype=float)
		self.cluster_radius = 0

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

		self.update_curvature_initial()

	def weight_calculator(self):
		h, w = self.grid_size, self.grid_size
		num_dirs = len(self.directions_edge)

		# shape (h, w, 2) 电场向量场
		field_vecs = np.stack([self.electric_field.field_x, self.electric_field.field_y], axis=-1)

		# 所有方向单位向量: shape (num_dirs, 2)
		dir_array = np.array(self.directions_edge, dtype=float)
		dir_norms = np.linalg.norm(dir_array, axis=1, keepdims=True)
		dir_unit = dir_array / np.maximum(dir_norms, 1e-8)

		dot_products = np.zeros((h, w, num_dirs), dtype=float)
		mask_valid = np.ones((h, w, num_dirs), dtype=bool)

		for k, (dx, dy) in enumerate(self.directions_edge):
			# 使用 np.roll 移动 field 向量
			shifted_x = np.roll(field_vecs[..., 0], shift=-dx, axis=0)
			shifted_y = np.roll(field_vecs[..., 1], shift=-dy, axis=1)

			# 检查越界（设置无效方向 mask）
			if dx > 0:
				mask_valid[-dx:, :, k] = False
			elif dx < 0:
				mask_valid[:-dx, :, k] = False
			if dy > 0:
				mask_valid[:, -dy:, k] = False
			elif dy < 0:
				mask_valid[:, :-dy, k] = False

			# 点积
			dot_products[..., k] = shifted_x * dir_unit[k, 0] + shifted_y * dir_unit[k, 1]

		# 越界方向设置为 -inf, 保证 softmax 中其权重为 0
		dot_products[~mask_valid] = -np.inf

		# softmax 归一化
		max_dot = np.max(dot_products, axis=2, keepdims=True)
		exp_weights = np.exp(dot_products - max_dot)
		exp_weights[~mask_valid] = 0.0  # 越界方向权重为 0
		sum_exp = np.sum(exp_weights, axis=2, keepdims=True)
		weights = np.divide(exp_weights, sum_exp, out=np.zeros_like(exp_weights), where=(sum_exp > 0))

		return weights

	def spawn_particle(self):
		ft = self.electric_field.field_type
		r = self.cluster_radius + self.radius_buffer

		if ft == gf2.FieldType2d.POINT:
			x = y = self.grid_size - 1
			try_max = 300
			try_time = 0
			while try_time < try_max:
				try_time += 1
				angle = 2 * np.pi * random.random()

				x_try = int(self.center + r * np.cos(angle))
				y_try = int(self.center + r * np.sin(angle))

				if self.is_valid(x_try, y_try):
					x, y = x_try, y_try
					break

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
		for dx, dy in self.directions_edge + self.directions_vertex:
			if self.is_dendrite(x + dx, y + dy):
				return True
		return False

	def calculate_curvature_at(self, x, y):
		neighbors = 0
		for dx, dy in self.directions_edge:
			if self.is_dendrite(x + dx, y + dy):
				neighbors += 1
		return max(0, 4 - neighbors)  # 孤立点曲率=4，平坦区域曲率=0

	def update_curvature_initial(self):
		for x in range(self.grid_size):
			for y in range(self.grid_size):
				if self.is_dendrite(x, y):
					self.curvature[x, y] = self.calculate_curvature_at(x, y)

	def update_curvature_around(self, x, y):
		for dx, dy in self.directions_edge + [(0, 0)]:
			nx, ny = x + dx, y + dy
			if self.is_dendrite(nx, ny):
				self.curvature[nx, ny] = self.calculate_curvature_at(nx, ny)

	def biased_move_with_field(self, x, y):
		dx, dy = random.choices(self.directions_edge, weights=self.weights[x, y])[0]
		return x + dx, y + dy

	def update_sei_thickness(self):
		mask = (self.grid == 1) & (self.sei_thickness < self.sei_max_thickness)
		growth = self.sei_growth_rate * (1.0 + self.curvature)
		self.sei_thickness = np.where(
			mask,
			np.minimum(self.sei_thickness + growth, self.sei_max_thickness),
			self.sei_thickness
		)

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
					thickness = 0
					for dx, dy in self.directions_edge:
						nx, ny = x + dx, y + dy
						if self.is_valid(nx, ny):
							thickness += self.sei_thickness[nx, ny]
					sei_effect = np.exp(-self.sei_resistance_factor * thickness)
					effective_prob = base_prob * sei_effect

					if random.random() < effective_prob:
						particle_count += 1
						self.grid[x, y] = 1
						self.update_curvature_around(x, y)
						# self.update_sei_thickness()
						dist2 = (x - self.center) ** 2 + (y - self.center) ** 2
						if dist2 > self.cluster_radius ** 2:
							self.cluster_radius = np.sqrt(dist2)
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
