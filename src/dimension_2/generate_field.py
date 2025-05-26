import numpy as np


def point_field(grid_size, charge_pos, k=1.0):
	field_x = np.zeros((grid_size, grid_size))
	field_y = np.zeros((grid_size, grid_size))
	x0, y0 = charge_pos

	for i in range(grid_size):
		for j in range(grid_size):
			dx = i - x0
			dy = j - y0
			r2 = dx**2 + dy**2 + 1e-6  # 避免除零
			field_x[i, j] = -k * dx / (r2 ** 1.5)
			field_y[i, j] = -k * dy / (r2 ** 1.5)

	return field_x, field_y


def uniform_field(grid_size, direction="down", strength=1.0):
	field_x = np.zeros((grid_size, grid_size))
	field_y = np.zeros((grid_size, grid_size))

	if direction == "down":
		field_y[:, :] = -strength
	elif direction == "up":
		field_y[:, :] = strength
	elif direction == "left":
		field_x[:, :] = -strength
	elif direction == "right":
		field_x[:, :] = strength

	return field_x, field_y
