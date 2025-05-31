import itertools
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

import generate_field_3d as gf3


# 模拟参数
# 长宽比不重要, 为方便令网格为正方体, 见 class ElectricField
def safe_roll(array, shift, axis):
	rolled = np.roll(array, shift=shift, axis=axis)
	if shift > 0:
		slicer = [slice(None)] * 3
		slicer[axis] = slice(0, shift)
		rolled[tuple(slicer)] = 0
	elif shift < 0:
		slicer = [slice(None)] * 3
		slicer[axis] = slice(shift, None)
		rolled[tuple(slicer)] = 0
	return rolled


@dataclass
class DLASimulator:
	sei_growth_rate: float = 0.01					# SEI膜生长速率
	sei_max_thickness: float = 1.0					# SEI膜最大厚度
	sei_resistance_factor: float = 0.1				# SEI膜对离子通过的阻碍系数
	max_particles: int = 1000						# 最大沉积粒子数
	max_steps_per_particle: int = 10000				# 每个粒子的最大步数
	attach_prob: float = 1.0						# 粘附概率, 后续可以调控
	radius_buffer: int = 5							# 控制生成新粒子的半径缓冲区
	electric_field: gf3.ElectricField3d = None		# 控制电场形状
	weight_zoom: int = 50
	use_sei: bool = True							# 控制是否启用 SEI 模型

	sei_thickness: np.ndarray = field(init=False)	# 记录每个格点的SEI膜厚度
	curvature: np.ndarray = field(init=False)
	cluster_radius: int = field(init=False)
	dendrite_indices: tuple = field(init=False)

	def __post_init__(self):
		self.grid_size = self.electric_field.grid_size
		self.center = self.grid_size // 2
		self.all_dirs = list(itertools.product([-1, 0, 1], repeat=3))
		self.directions_face = [d for d in self.all_dirs if sum(abs(i) for i in d) == 1]
		self.directions_edge = [d for d in self.all_dirs if sum(abs(i) for i in d) == 2]
		self.directions_vertex = [d for d in self.all_dirs if sum(abs(i) for i in d) == 3]
		self.move_dirs = self.directions_face
		self.attach_dirs = self.directions_face + self.directions_edge
		self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=int)
		self.weights = self.weight_calculator()
		self.sei_thickness = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=float)
		self.curvature = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=float)
		self.cluster_radius = 0
		self.init_grid()
		self.dendrite_indices = np.where(self.grid == 1)
		self.update_curvature_initial()

	def init_grid(self):
		if self.electric_field.field_type == gf3.FieldType3d.POINT:
			self.grid[self.center, self.center, self.center] = 1
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

	def weight_calculator(self):
		N = self.grid_size
		D = len(self.move_dirs)
		dir_vecs = np.array(self.move_dirs, dtype=float)  # (D, 3)
		dir_vecs /= np.linalg.norm(dir_vecs, axis=1, keepdims=True)  # 单位方向向量 (D, 3)

		# 电场向量场 (N, N, N, 3)
		field_vecs = np.stack((
			self.electric_field.field_x,
			self.electric_field.field_y,
			self.electric_field.field_z
		), axis=-1)  # -> shape: (N, N, N, 3)

		# 点乘 field_vec 和 dir_vec: 结果 (N, N, N, D)
		# 使用 einsum 实现 batch 点积
		dot_products = np.einsum('xyzc,dc->xyzd', field_vecs, dir_vecs)

		# 处理越界 (例如 i + dx 可能越界), 我们通过掩码实现
		mask = np.ones((N, N, N, D), dtype=bool)
		for d, (dx, dy, dz) in enumerate(self.move_dirs):
			# 当前方向的越界条件
			valid_x = np.logical_and(0 <= np.arange(N) + dx, np.arange(N) + dx < N)
			valid_y = np.logical_and(0 <= np.arange(N) + dy, np.arange(N) + dy < N)
			valid_z = np.logical_and(0 <= np.arange(N) + dz, np.arange(N) + dz < N)

			mask_x = np.broadcast_to(valid_x[:, None, None], (N, N, N))
			mask_y = np.broadcast_to(valid_y[None, :, None], (N, N, N))
			mask_z = np.broadcast_to(valid_z[None, None, :], (N, N, N))

			mask[..., d] &= mask_x & mask_y & mask_z

		# 无效方向设为 -inf
		dot_products[~mask] = -np.inf

		# softmax 操作: 稳定性处理
		max_dot = np.max(dot_products, axis=-1, keepdims=True)
		exp_weights = np.exp(self.weight_zoom * (dot_products - max_dot))
		exp_weights[~mask] = 0.0

		sum_exp = np.sum(exp_weights, axis=-1, keepdims=True)
		# 避免除以0
		with np.errstate(divide='ignore', invalid='ignore'):
			weights = np.divide(
				exp_weights,
				sum_exp,
				out=np.zeros_like(exp_weights),
				where=(sum_exp > 0)
			)

		return weights

	def spawn_particle(self):
		ft = self.electric_field.field_type
		r = self.cluster_radius + self.radius_buffer

		if ft == gf3.FieldType3d.POINT:
			x = y = z = self.grid_size - 1
			try_max = 300
			try_time = 0
			while try_time < try_max:
				try_time += 1
				theta = np.arccos(2 * random.random() - 1)
				phi = 2 * np.pi * random.random()

				x_try = int(self.center + r * np.sin(theta) * np.cos(phi))
				y_try = int(self.center + r * np.sin(theta) * np.sin(phi))
				z_try = int(self.center + r * np.cos(theta))

				if self.is_valid(x_try, y_try, z_try):
					x, y, z = x_try, y_try, z_try
					break

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
		for dx, dy, dz in self.attach_dirs:
			if self.is_dendrite(x + dx, y + dy, z + dz):
				return True
		return False

	def calculate_curvature_at(self, x, y, z):
		neighbors = 0
		for dx, dy, dz in self.directions_face:
			nx, ny, nz = x + dx, y + dy, z + dz
			if self.is_dendrite(nx, ny, nz):
				neighbors += 1
		return max(0, 6 - neighbors)

	def update_curvature_initial(self):
		self.curvature.fill(0)
		dendrite_mask = self.grid

		# 向每个方向滚动后叠加, 得到的就是此点周围六个面中 True 的个数
		for dx, dy, dz in self.directions_face:
			shifted = safe_roll(dendrite_mask, dx, axis=0)
			shifted = safe_roll(shifted, dy, axis=1)
			shifted = safe_roll(shifted, dz, axis=2)
			self.curvature += shifted

		self.curvature = np.where(dendrite_mask, 6 - self.curvature, 0)

	def add_dendrite(self, x, y, z):
		self.grid[x, y, z] = 1
		self.dendrite_indices = (
			np.concatenate([self.dendrite_indices[0], [x]]),
			np.concatenate([self.dendrite_indices[1], [y]]),
			np.concatenate([self.dendrite_indices[2], [z]])
		)


	def update_curvature_around(self, x, y, z):
		for dx, dy, dz in self.directions_face + [(0,0,0)]:
			nx, ny, nz = x + dx, y + dy, z + dz
			if self.is_dendrite(nx, ny, nz):
				self.curvature[nx, ny, nz] = self.calculate_curvature_at(nx, ny, nz)

	def biased_move_with_field(self, x, y, z):
		dx, dy, dz = random.choices(self.move_dirs, weights=self.weights[x, y, z])[0]
		return x + dx, y + dy, z + dz

	def update_sei_if_enabled(self):
		if self.use_sei:
			self.sei_thickness[self.dendrite_indices] = np.minimum(
				self.sei_thickness[self.dendrite_indices] + self.sei_growth_rate * (1.0 + self.curvature[self.dendrite_indices]),
				self.sei_max_thickness
			)

	def should_attach(self, x, y, z):
		base_prob = self.attach_prob
		thickness = 0
		for dx, dy, dz in self.attach_dirs:
			nx, ny, nz = x + dx, y + dy, z + dz
			if self.is_valid(nx, ny, nz):
				thickness += self.sei_thickness[nx, ny, nz]
		sei_effect = np.exp(-self.sei_resistance_factor * thickness)
		effective_prob = base_prob * sei_effect
		return np.random.rand() < effective_prob

	def simulate_single_particle(self):
		attached = False
		x, y, z = self.spawn_particle()
		for _ in range(self.max_steps_per_particle):
			x, y, z = self.biased_move_with_field(x, y, z)
			self.update_sei_if_enabled()  # 更新 SEI 膜厚度

			if not self.is_valid(x, y, z):
				break

			if self.is_adjacent_to_cluster(x, y, z):
				if self.should_attach(x, y, z):
					attached = True
					self.add_dendrite(x, y, z)

					self.update_curvature_around(x, y, z)
					dist2 = (x - self.center) ** 2 + (y - self.center) ** 2 + (z - self.center) ** 2
					if dist2 > self.cluster_radius ** 2:
						self.cluster_radius = np.sqrt(dist2)
					break

		return attached

	def simulate(self):
		particle_count = 1

		while particle_count < self.max_particles:
			if (particle_count + 1) % 10 == 0:
				print(f"{particle_count + 1} / {self.max_particles}")
			particle_count += (1 if self.simulate_single_particle() else 0)

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
		max_particles = 5000,
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
		grid_size=41,
		strength=0.01
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
