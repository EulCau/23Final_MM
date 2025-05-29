from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt


class FieldType3d(Enum):
	POINT = "point"
	UNIFORM_UP = "uniform_up"
	UNIFORM_DOWN = "uniform_down"
	UNIFORM_LEFT = "uniform_left"
	UNIFORM_RIGHT = "uniform_right"
	UNIFORM_FRONT = "uniform_front"
	UNIFORM_BACK = "uniform_back"


@dataclass
class ElectricField3d:
	field_type: FieldType3d
	grid_size: int
	strength: float
	charge_pos: tuple = None

	field_x: np.ndarray = field(init=False)
	field_y: np.ndarray = field(init=False)
	field_z: np.ndarray = field(init=False)

	def __post_init__(self):
		if self.field_type == FieldType3d.POINT:
			if self.charge_pos is None:
				self.charge_pos = (self.grid_size // 2, self.grid_size // 2, self.grid_size // 2)
			self.field_x, self.field_y, self.field_z = self._point_field()
		elif self.field_type in {
			FieldType3d.UNIFORM_UP,
			FieldType3d.UNIFORM_DOWN,
			FieldType3d.UNIFORM_LEFT,
			FieldType3d.UNIFORM_RIGHT
		}:
			self.field_x, self.field_y, self.field_z = self._uniform_field()
		else:
			raise ValueError(f"Unsupported field type: {self.field_type}")

	def _point_field(self):
		x0, y0, z0 = self.charge_pos
		idx = np.arange(0, self.grid_size, 1)
		X, Y, Z = np.meshgrid(idx, idx, idx, indexing='ij')

		dx = X - x0
		dy = Y - y0
		dz = Z - z0
		r2 = dx ** 2 + dy ** 2 + dz ** 2
		r3 = r2 ** 1.5 + 1e-9

		field_x = -self.strength * dx / r3
		field_y = -self.strength * dy / r3
		field_z = -self.strength * dz / r3

		return field_x, field_y, field_z

	def _uniform_field(self):
		field_x = np.zeros((self.grid_size, self.grid_size, self.grid_size))
		field_y = np.zeros((self.grid_size, self.grid_size, self.grid_size))
		field_z = np.zeros((self.grid_size, self.grid_size, self.grid_size))

		if self.field_type == FieldType3d.UNIFORM_DOWN:
			field_z[:, :, :] = -self.strength
		elif self.field_type == FieldType3d.UNIFORM_UP:
			field_z[:, :, :] = self.strength
		elif self.field_type == FieldType3d.UNIFORM_LEFT:
			field_y[:, :, :] = -self.strength
		elif self.field_type == FieldType3d.UNIFORM_RIGHT:
			field_y[:, :, :] = self.strength
		elif self.field_type == FieldType3d.UNIFORM_BACK:
			field_x[:, :, :] = -self.strength
		elif self.field_type == FieldType3d.UNIFORM_FRONT:
			field_x[:, :, :] = self.strength

		return field_x, field_y, field_z

	def plot_field(self, density: int = 20, length: float = 20.0):
		idx = np.arange(0, self.grid_size, density)
		X, Y, Z = np.meshgrid(idx, idx, idx, indexing='ij')

		# 提取稀疏电场分量
		U = self.field_x[::density, ::density, ::density]
		V = self.field_y[::density, ::density, ::density]
		W = self.field_z[::density, ::density, ::density]

		# 计算模长归一化, 避免除零
		mag = np.sqrt(U ** 2 + V ** 2 + W ** 2) + 1e-6
		U = U / mag
		V = V / mag
		W = W / mag

		fig = plt.figure(figsize=(6, 6))
		ax = fig.add_subplot(111, projection='3d')
		ax.quiver(X, Y, Z, U, V, W, length=length, normalize=True, color='blue')

		ax.set_title(f"3D Electric Field: {self.field_type.value}")
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	electric_field = ElectricField3d(FieldType3d.POINT, 201, 1.0)
	electric_field.plot_field()
