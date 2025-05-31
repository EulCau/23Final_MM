from pathlib import Path

import numpy as np

from analysis_metrics_3d import DLAMetricsEvaluator
from dla_3d import test_point, test_parallel


def get_save_path(target: str) -> Path:
    results_dir = Path("../../results/data")  # 保存目录
    results_dir.mkdir(exist_ok=True)
    return results_dir / f"{target}_result.npz"


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


def run_or_load_experiment(renew: bool = False, target: str = "point") -> DLAMetricsEvaluator:
    file_path = get_save_path(target)

    # 如果文件已存在则读取
    if file_path.exists() and not renew:
        print(f"Loading existing experiment result from {file_path}")
        dendrites_indices, sei_thicknesses, times, field_type = load_experiment_result(file_path)
    else:
        print(f"Running new simulation for target: {target}")
        if target == "point":
            dendrites_indices, sei_thicknesses, times, field_type = test_point(
                1.0, 0.01, 1.0, 0.1,
                1000, 1.0, True)
        elif target == "parallel":
            dendrites_indices, sei_thicknesses, times, field_type = test_parallel(
                0.01, 0.01, 1.0, 0.1,
                1000, 1.0, True)
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


def main():
    evaluator = run_or_load_experiment(renew=True, target="point")

    volume_ratios, densities = evaluator.compute_volume_ratio_and_density()
    speeds = evaluator.compute_growth_speed()
    fractal_dimension = evaluator.compute_fractal_dimension()

    print("Volume Ratio: ", volume_ratios)
    print("Growth Speed", speeds)
    print("Dimension: ", fractal_dimension)


if __name__ == "__main__":
    main()
