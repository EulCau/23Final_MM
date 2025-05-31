import os
from pathlib import Path

import numpy as np

from analysis_metrics_3d import DLAMetricsEvaluator
from dla_3d import test_point, test_parallel
import json


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"[警告] 配置文件不存在: {config_path}, 将使用默认参数.")
        return {
            "strength": 0.01,
            "sei_growth_rate": 0.01,
            "sei_max_thickness": 1.0,
            "sei_resistance_factor": 0.1,
            "max_particles": 2000,
            "attach_prob": 1.0,
            "plot_result": True
        }

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_save_path(result_name: str) -> Path:
    results_dir = Path("../../results/data")  # 保存目录
    results_dir.mkdir(exist_ok=True)
    return results_dir / f"{result_name}_result.npz"


def save_experiment_result(file_path: Path, dendrites_indices, sei_thicknesses, times, field_type):
    np.savez_compressed(
        file_path,
        dendrites_indices=np.array(dendrites_indices, dtype=object),
        sei_thicknesses=np.array(sei_thicknesses, dtype=object),
        times=np.array(times),
        field_type=np.array(field_type, dtype=object)
    )


def load_experiment_result(file_path: Path):
    data = np.load(file_path, allow_pickle=True)
    return (
        data["dendrites_indices"],
        data["sei_thicknesses"],
        data["times"],
        data["field_type"].item()
    )


def run_or_load_experiment(renew: bool = False, config_path = "config_point.json", target: str = "point") -> DLAMetricsEvaluator:
    base = os.path.splitext(os.path.basename(config_path))[0]
    file_path = get_save_path(base)
    params = load_config(config_path)

    # 如果文件已存在则读取
    if file_path.exists() and not renew:
        print(f"Loading existing experiment result from {file_path}")
        dendrites_indices, sei_thicknesses, times, field_type = load_experiment_result(file_path)
    else:
        print(f"Running new simulation for target: {target}")
        if target == "point":
            dendrites_indices, sei_thicknesses, times, field_type = test_point(**params)
        elif target == "parallel":
            dendrites_indices, sei_thicknesses, times, field_type = test_parallel(**params)
        else:
            raise ValueError(f"Unknown target '{target}', 'point' or 'parallel' only")

        save_experiment_result(file_path, dendrites_indices, sei_thicknesses, times, field_type)

    # 自动推断 grid_size
    grid_size = sei_thicknesses[0].shape[0]

    evaluator = DLAMetricsEvaluator(
        dendrites_indices=dendrites_indices,
        sei_thicknesses=sei_thicknesses,
        times=times,
        field_type=field_type,
        grid_size=grid_size
    )

    return evaluator


def test(config_name, renew=False):
    config_dir = Path("../../config")
    target = config_name.split('_')[0]
    evaluator = run_or_load_experiment(renew=renew, config_path=config_dir / config_name, target=config_name.split('_')[0])

    volume_ratios, densities = evaluator.compute_volume_ratio_and_density()
    speeds = evaluator.compute_growth_speed()
    fractal_dimension = evaluator.compute_fractal_dimension()

    return evaluator.times, volume_ratios, densities, speeds, fractal_dimension


if __name__ == "__main__":
    test("point_3d_default.json")
