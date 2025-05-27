import numpy as np
from enum import Enum
from dataclasses import dataclass, field

from matplotlib import pyplot as plt


class FieldType(Enum):
	POINT = "point"
	UNIFORM_UP = "uniform_up"
	UNIFORM_DOWN = "uniform_down"
	UNIFORM_LEFT = "uniform_left"
	UNIFORM_RIGHT = "uniform_right"


@dataclass
class ElectricField:
	field_type: FieldType
	grid_size: int
	strength: float
	charge_pos: tuple = None

	field_x: np.ndarray = field(init=False)
	field_y: np.ndarray = field(init=False)

	def __post_init__(self):
		if self.field_type == FieldType.POINT:
			if self.charge_pos is None:
				self.charge_pos = (self.grid_size // 2, self.grid_size // 2)
			self.field_x, self.field_y = self._point_field()
		elif self.field_type in {
			FieldType.UNIFORM_UP,
			FieldType.UNIFORM_DOWN,
			FieldType.UNIFORM_LEFT,
			FieldType.UNIFORM_RIGHT
		}:
			self.field_x, self.field_y = self._uniform_field()
		else:
			raise ValueError(f"Unsupported field type: {self.field_type}")

	def _point_field(self):
		field_x = np.zeros((self.grid_size, self.grid_size))
		field_y = np.zeros((self.grid_size, self.grid_size))
		x0, y0 = self.charge_pos

		for i in range(self.grid_size):
			for j in range(self.grid_size):
				dx = i - x0
				dy = j - y0
				r2 = dx ** 2 + dy ** 2 + 1e-6  # 避免除零
				field_x[i, j] = -self.strength * dx / (r2 ** 1.5)
				field_y[i, j] = -self.strength * dy / (r2 ** 1.5)

		return field_x, field_y

	def _uniform_field(self):
		field_x = np.zeros((self.grid_size, self.grid_size))
		field_y = np.zeros((self.grid_size, self.grid_size))

		if self.field_type == FieldType.UNIFORM_DOWN:
			field_y[:, :] = -self.strength
		elif self.field_type == FieldType.UNIFORM_UP:
			field_y[:, :] = self.strength
		elif self.field_type == FieldType.UNIFORM_LEFT:
			field_x[:, :] = -self.strength
		elif self.field_type == FieldType.UNIFORM_RIGHT:
			field_x[:, :] = self.strength

		return field_x, field_y

	def plot_field(self, density: int = 10, scale: float = 50.0):
		# 生成网格坐标
		x = np.arange(0, self.grid_size, 1)
		y = np.arange(0, self.grid_size, 1)
		X, Y = np.meshgrid(x, y, indexing='ij')  # 保持 X(i,j), Y(i,j) 对应 field_x[i,j]

		magnitude = np.sqrt(self.field_x ** 2 + self.field_y ** 2)
		field_x_vis = 2 * self.field_x / (magnitude + 1e-6)
		field_y_vis = 2 * self.field_y / (magnitude + 1e-6)

		# 稀疏采样
		slice_step = density
		X_slice = X[::slice_step, ::slice_step]
		Y_slice = Y[::slice_step, ::slice_step]
		U_slice = field_x_vis[::slice_step, ::slice_step]
		V_slice = field_y_vis[::slice_step, ::slice_step]

		plt.figure(figsize=(6, 6))
		plt.quiver(
			X_slice, Y_slice, U_slice, V_slice,
			scale=scale, angles='xy', pivot='middle', color='blue'
		)
		plt.title(f"Electric Field: {self.field_type.value}")
		plt.axis('equal')
		plt.xlim(0, self.grid_size)
		plt.ylim(0, self.grid_size)
		plt.gca().invert_yaxis()  # 保持与笛卡尔坐标系一致（y 向上）
		plt.show()


if __name__ == "__main__":
	electric_field = ElectricField(FieldType.POINT, 201, 1.0)
	electric_field.plot_field()
