import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

import generate_field as gf


# 模拟参数
# 长宽比不重要, 为方便令网格为正方形, 见 class ElectricField
@dataclass
class DLASimulator:
	max_particles: int = 1000					# 最大沉积粒子数
	max_steps_per_particle: int = 10000			# 每个粒子的最大步数
	attach_prob: float = 1.0					# 粘附概率, 后续可以调控
	radius_buffer: int = 5						# 控制生成新粒子的半径缓冲区
	electric_field: gf.ElectricField = None		# 控制电场形状

	cluster_radius: int = field(init=False)

	def __post_init__(self):
		self.grid_size = self.electric_field.grid_size
		self.center = self.grid_size // 2
		self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
		self.cluster_radius = 0

		if self.electric_field.field_type == gf.FieldType.POINT:
			self.grid[self.center, self.center] = 1
		elif self.electric_field.field_type == gf.FieldType.UNIFORM_UP:
			self.grid[:, -1] = 1
		elif self.electric_field.field_type == gf.FieldType.UNIFORM_DOWN:
			self.grid[:, 0] = 1
		elif self.electric_field.field_type == gf.FieldType.UNIFORM_LEFT:
			self.grid[0, :] = 1
		elif self.electric_field.field_type == gf.FieldType.UNIFORM_RIGHT:
			self.grid[-1, :] = 1
		else:
			raise ValueError(f"Unsupported field type: {self.electric_field.field_type}")

	def spawn_particle(self):
		ft = self.electric_field.field_type
		if ft == gf.FieldType.POINT:
			angle = 2 * np.pi * random.random()
			x = int(self.center + (self.cluster_radius + self.radius_buffer) * np.cos(angle))
			y = int(self.center + (self.cluster_radius + self.radius_buffer) * np.sin(angle))
		elif ft == gf.FieldType.UNIFORM_DOWN:
			x = random.randint(0, self.grid_size - 1)
			y = self.grid_size - 1
		elif ft == gf.FieldType.UNIFORM_UP:
			x = random.randint(0, self.grid_size - 1)
			y = 0
		elif ft == gf.FieldType.UNIFORM_LEFT:
			x = self.grid_size - 1
			y = random.randint(0, self.grid_size - 1)
		elif ft == gf.FieldType.UNIFORM_RIGHT:
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

	def biased_move_with_field(self, x, y):
		directions = [(1,0), (0,1), (-1,0), (0,-1)]
		weights = []
		ex = self.electric_field.field_x[x, y]
		ey = self.electric_field.field_y[x, y]
		alpha = abs(ex) + abs(ey)
		if alpha == 0:
			weights = [1, 1, 1, 1]
		else:
			for dx, dy in directions:
				nx, ny = x + dx, y + dy
				if self.is_valid(nx, ny):
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

	def simulate(self):
		particle_count = 1

		while particle_count < self.max_particles:
			x, y = self.spawn_particle()
			steps = 0

			while steps < self.max_steps_per_particle:
				x, y = self.biased_move_with_field(x, y)
				steps += 1

				if not self.is_valid(x, y):
					break

				if self.is_adjacent_to_cluster(x, y) and random.random() < self.attach_prob:
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
	field_point = gf.ElectricField(
		field_type=gf.FieldType.POINT,
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
	field_point = gf.ElectricField(
		field_type=gf.FieldType.UNIFORM_DOWN,
		grid_size=201,
		strength=1.0
	)

	dla_point = DLASimulator(
		max_particles = 1000,
		max_steps_per_particle = 10000,
		attach_prob = 1.0,
		electric_field= field_point
	)

	# 运行模拟
	dla_point.simulate()
	dla_point.plot_cluster()


if __name__ == "__main__":
	test_point()
	test_parallel()
