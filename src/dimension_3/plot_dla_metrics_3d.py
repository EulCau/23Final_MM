import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from test_runner_3d import test


def plot_volume_ratios(labels, results, result_dir, ele_type="point"):
	plt.figure()
	for (times, volume_ratios, _, _, _), label in zip(results, labels):
		plt.plot(times, volume_ratios, label=label)
	plt.xlabel("Time")
	plt.ylabel("Volume Ratio")
	plt.title("Volume Ratio vs Time")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(result_dir / f"{ele_type}_volume_ratio_comparison.png")
	plt.show()


def interpolate_zero_speeds(times, speeds):
	speeds = np.array(speeds, dtype=np.float64)
	filled = speeds.copy()
	i = 0
	while i < len(speeds):
		if speeds[i] == 0:
			j = i
			while j < len(speeds) and speeds[j] == 0:
				j += 1
			if j < len(speeds):
				dt = times[j] - times[i]
				avg = (speeds[j] * (times[j] - times[j - 1])) / dt if dt > 0 else 0
				filled[i:j] = avg
			i = j
		else:
			i += 1
	return filled


def plot_speeds(labels, results, result_dir, ele_type="point"):
	plt.figure()
	for (times, _, _, speeds, _), label in zip(results, labels):
		times = np.array(times)
		speeds = np.array(speeds)
		plt.plot(np.array(times[1:]), np.array(speeds), label=label)

	plt.xlabel("Time")
	plt.ylabel("Growth Speed")
	plt.title("Smoothed Growth Speed vs Time")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(result_dir / f"{ele_type}_growth_speed_comparison.png")
	plt.show()


def plot_cumulative_growth(labels, results, result_dir, ele_type="point"):
	plt.figure()
	for (times, _, _, speeds, _), label in zip(results, labels):
		cumulative = np.cumsum(speeds)
		plt.plot(times[1:], cumulative, label=label)
	plt.xlabel("Time")
	plt.ylabel("Cumulative Growth")
	plt.title("Cumulative Growth vs Time")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(result_dir / f"{ele_type}_cumulative_growth_comparison.png")
	plt.show()


def plot_fractal_dimension(labels, results, result_dir, ele_type="point"):
	fds = [fd for (_, _, _, _, fd) in results]
	x = np.arange(len(labels))
	plt.figure()
	plt.bar(x, fds, tick_label=labels)
	plt.ylabel("Fractal Dimension")
	plt.title("Fractal Dimension Comparison")
	plt.xticks(rotation=20)
	plt.tight_layout()
	plt.savefig(result_dir / f"{ele_type}_fractal_dimension_comparison.png")
	plt.show()


def plot_densities(labels, results, result_dir, ele_type="point"):
	for (times, _, densities, _, _), label in zip(results, labels):
		data = np.stack(densities, axis=0)  # shape: (time, layers)
		plt.figure()
		plt.imshow(data.T, aspect="auto", origin="lower",
				   extent=(float(times[0]), float(times[-1]), 0.0, float(data.shape[1])))
		plt.xlabel("Time")
		plt.ylabel("Layer")
		plt.title(f"Density Profile: {label}")
		plt.colorbar(label="Density")
		plt.tight_layout()
		fname = f"{ele_type}_density_profile_{label.lower().replace(' ', '_')}.png"
		plt.savefig(result_dir / fname)
		plt.show()


def main(ele_type="point"):
	# 配置文件名
	config_names = [
		f"{ele_type}_3d_default.json",
		f"{ele_type}_3d_sei_growth_rate.json",
		f"{ele_type}_3d_sei_max_thickness.json",
		f"{ele_type}_3d_sei_resistance_factor.json",
		f"{ele_type}_3d_strength.json"
	]

	labels = [
		"Default",
		"SEI Growth Rate",
		"SEI Max Thickness",
		"SEI Resistance Factor",
		"Strength"
	]

	# 运行 test() 获取结果
	results = [test(name, False) for name in config_names]

	# 设置结果路径
	result_dir = Path("../../results")
	result_dir.mkdir(parents=True, exist_ok=True)
	plot_volume_ratios(labels, results, result_dir, ele_type)
	# plot_speeds(labels, results, result_dir, ele_type)
	plot_cumulative_growth(labels, results, result_dir, ele_type)
	plot_fractal_dimension(labels, results, result_dir, ele_type)
	plot_densities(labels, results, result_dir, ele_type)


if __name__ == "__main__":
	main("point")
	main("parallel")
