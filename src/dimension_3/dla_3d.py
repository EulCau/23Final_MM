import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

import generate_field_3d as gf3


# 模拟参数
# 长宽比不重要, 为方便令网格为正方体, 见 class ElectricField
@dataclass
class DLASimulator:
	max_particles: int = 30000					# 最大沉积粒子数
	max_steps_per_particle: int = 100000		# 每个粒子的最大步数
	attach_prob: float = 1.0					# 粘附概率, 后续可以调控
	radius_buffer: int = 5						# 控制生成新粒子的半径缓冲区
	electric_field: gf3.ElectricField3d = None	# 控制电场形状

	cluster_radius: int = field(init=False)


	def __post_init__(self):
		self.grid_size = self.electric_field.grid_size
		self.center = self.grid_size // 2
		self.directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
		self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=int)
		self.weights = self.weight_culculator()
		self.cluster_radius = 0

		if self.electric_field.field_type == gf3.FieldType3d.POINT:
			self.grid[self.center, self.center] = 1
		elif self.electric_field.field_type == gf3.FieldType3d.UNIFORM_UP:
			self.grid[:, :, -1] = 1
		elif self.electric_field.field_type == gf3.FieldType3d.UNIFORM_DOWN:
			self.grid[:, :, 0] = 1
		elif self.electric_field.field_type == gf3.FieldType3d.UNIFORM_LEFT:
			self.grid[:, 0, :] = 1
		elif self.electric_field.field_type == gf3.FieldType3d.UNIFORM_RIGHT:
			self.grid[:, -1, :] = 1
		elif self.electric_field.field_type == gf3.FieldType3d.UNIFORM_FRONT:
			self.grid[-1, :, :] = 1
		elif self.electric_field.field_type == gf3.FieldType3d.UNIFORM_BACK:
			self.grid[0, :, :] = 1
		else:
			raise ValueError(f"Unsupported field type: {self.electric_field.field_type}")

	def weight_culculator(self):
		weights = np.zeros((self.grid_size, self.grid_size, self.grid_size, len(self.directions)))

		for i in range(self.grid_size):
			for j in range(self.grid_size):
				for k in range(self.grid_size):
					field_vec = np.array([
						self.electric_field.field_x[i, j, k],
						self.electric_field.field_y[i, j, k],
						self.electric_field.field_z[i, j, k]
					])
					dot_products = []

					for dx, dy, dz in self.directions:
						ni, nj, nk = i + dx, j + dy, k + dz
						if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
							dir_vec = np.array([dx, dy, dz], dtype=float)
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

					weights[i, j, k, :] = norm_weights

		return weights

	def spawn_particle(self):
		ft = self.electric_field.field_type
		r = self.cluster_radius + self.radius_buffer

		if ft == gf3.FieldType3d.POINT:
			theta = np.arccos(2 * random.random() - 1)
			phi = 2 * np.pi * random.random()

			x = int(self.center + r * np.sin(theta) * np.cos(phi))
			y = int(self.center + r * np.sin(theta) * np.sin(phi))
			z = int(self.center + r * np.cos(theta))

		elif ft == gf3.FieldType3d.UNIFORM_FRONT:
			x = 0
			y = random.randint(0, self.grid_size - 1)
			z = random.randint(0, self.grid_size - 1)

		elif ft == gf3.FieldType3d.UNIFORM_BACK:
			x = self.grid_size - 1
			y = random.randint(0, self.grid_size - 1)
			z = random.randint(0, self.grid_size - 1)

		elif ft == gf3.FieldType3d.UNIFORM_RIGHT:
			x = random.randint(0, self.grid_size - 1)
			y = 0
			z = random.randint(0, self.grid_size - 1)

		elif ft == gf3.FieldType3d.UNIFORM_LEFT:
			x = random.randint(0, self.grid_size - 1)
			y = self.grid_size - 1
			z = random.randint(0, self.grid_size - 1)

		elif ft == gf3.FieldType3d.UNIFORM_UP:
			x = random.randint(0, self.grid_size - 1)
			y = random.randint(0, self.grid_size - 1)
			z = 0

		elif ft == gf3.FieldType3d.UNIFORM_DOWN:
			x = random.randint(0, self.grid_size - 1)
			y = random.randint(0, self.grid_size - 1)
			z = self.grid_size - 1

		else:
			raise ValueError(f"Unknown field type: {ft}")

		return x, y, z

	def is_valid(self, x, y, z):
		return 0 <= x < self.grid_size and 0 <= y < self.grid_size and 0 <= z < self.grid_size

	def is_dendrite(self, x, y, z):
		return self.is_valid(x, y, z) and self.grid[x, y, z] == 1

	def is_adjacent_to_cluster(self, x, y, z):
		directions = [
			(dx, dy, dz)
			for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]
			if not (dx == dy == dz == 0)]

		for dx, dy, dz in directions:
			if self.is_dendrite(x + dx, y + dy, z + dz):
				return True
		return False

	def biased_move_with_field(self, x, y, z):
		dx, dy, dz = random.choices(self.directions, weights=self.weights[x, y, z])[0]
		return x + dx, y + dy, z + dz

	def simulate(self):
		particle_count = 1

		while particle_count < self.max_particles:
			x, y, z = self.spawn_particle()
			steps = 0

			while steps < self.max_steps_per_particle:
				x, y, z = self.biased_move_with_field(x, y, z)
				steps += 1

				if not self.is_valid(x, y, z):
					break

				if self.is_adjacent_to_cluster(x, y, z) and random.random() < self.attach_prob:
					self.grid[x, y, z] = 1
					dist = round(np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2 + (z - self.center) ** 2))
					self.cluster_radius = max(self.cluster_radius, dist)
					particle_count += 1
					break

	def plot_cluster(self):
		fig = plt.figure(figsize=(6, 6))
		ax = fig.add_subplot(111, projection='3d')
		x, y, z = np.where(self.grid == 1)
		ax.scatter(x, y, z, c='blue', marker='o', s=2)
		ax.set_title("3D Simulated Lithium Dendrite Growth")
		ax.set_xlim(0, self.grid_size)
		ax.set_ylim(0, self.grid_size)
		ax.set_zlim(0, self.grid_size)
		plt.axis('off')
		plt.show()


def test_point():
	# 点电荷在中心
	field_point = gf3.ElectricField3d(
		field_type=gf3.FieldType3d.POINT,
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
	field_parallel = gf3.ElectricField3d(
		field_type=gf3.FieldType3d.UNIFORM_DOWN,
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
