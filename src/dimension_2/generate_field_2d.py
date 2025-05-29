import numpy as np
from enum import Enum
from dataclasses import dataclass, field

from matplotlib import pyplot as plt


class FieldType2d(Enum):
	POINT = "point"
	UNIFORM_UP = "uniform_up"
	UNIFORM_DOWN = "uniform_down"
	UNIFORM_LEFT = "uniform_left"
	UNIFORM_RIGHT = "uniform_right"


@dataclass
class ElectricField2d:
	field_type: FieldType2d
	grid_size: int
	strength: float
	charge_pos: tuple = None

	field_x: np.ndarray = field(init=False)
	field_y: np.ndarray = field(init=False)

	def __post_init__(self):
		if self.field_type == FieldType2d.POINT:
			if self.charge_pos is None:
				self.charge_pos = (self.grid_size // 2, self.grid_size // 2)
			self.field_x, self.field_y = self._point_field()
		elif self.field_type in {
			FieldType2d.UNIFORM_UP,
			FieldType2d.UNIFORM_DOWN,
			FieldType2d.UNIFORM_LEFT,
			FieldType2d.UNIFORM_RIGHT
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
				field_x[i, j] = -self.strength * dx / r2
				field_y[i, j] = -self.strength * dy / r2

		return field_x, field_y

	def _uniform_field(self):
		field_x = np.zeros((self.grid_size, self.grid_size))
		field_y = np.zeros((self.grid_size, self.grid_size))

		if self.field_type == FieldType2d.UNIFORM_DOWN:
			field_y[:, :] = -self.strength
		elif self.field_type == FieldType2d.UNIFORM_UP:
			field_y[:, :] = self.strength
		elif self.field_type == FieldType2d.UNIFORM_LEFT:
			field_x[:, :] = -self.strength
		elif self.field_type == FieldType2d.UNIFORM_RIGHT:
			field_x[:, :] = self.strength

		return field_x, field_y

	def plot_field(self, density: int = 10, length: float = 0.04):
		# 生成网格坐标
		idx = np.arange(0, self.grid_size, density)
		X, Y = np.meshgrid(idx, idx, indexing='ij')  # 保持 X(i,j), Y(i,j) 对应 field_x[i,j]

		# 提取稀疏电场分量
		U = self.field_x[::density, ::density]
		V = self.field_y[::density, ::density]

		# 计算模长归一化, 避免除零
		mag = np.sqrt(U ** 2 + V ** 2) + 1e-6
		U = U / mag
		V = V / mag

		plt.figure(figsize=(6, 6))
		plt.quiver(X, Y, U, V, scale=1.0 / length, angles='xy', pivot='middle', color='blue')

		plt.title(f"2D Electric Field: {self.field_type.value}")
		plt.xlim(0, self.grid_size)
		plt.ylim(0, self.grid_size)
		plt.gca().invert_yaxis()  # 保持与笛卡尔坐标系一致（y 向上）
		plt.show()


if __name__ == "__main__":
	electric_field = ElectricField2d(FieldType2d.POINT, 201, 1.0)
	electric_field.plot_field()
