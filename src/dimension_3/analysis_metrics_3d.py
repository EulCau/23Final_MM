from typing import List, Tuple

import numpy as np

import generate_field_3d as gf3

class DLAMetricsEvaluator:
    def __init__(
        self,
        dendrites_indices: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        sei_thicknesses: List[np.ndarray],
        times: List[int],
        field_type: gf3.FieldType3d,
        grid_size: int,
        num_layers: int = 10
    ):
        self.dendrites_indices = dendrites_indices
        self.sei_thicknesses = sei_thicknesses
        self.times = times
        self.field_type = field_type
        self.grid_size = grid_size
        self.center = grid_size // 2
        self.num_layers = num_layers

        # 判断是“点电荷中心对称”还是“底面起始垂直生长”
        self.axis_mode = self._determine_mode()

    def _determine_mode(self):
        if self.field_type == gf3.FieldType3d.POINT:
            return 'radial'
        else:
            return 'vertical'  # 默认使用 Z 轴垂直切片

    def _compute_extent(self, x, y, z):
        if self.axis_mode == 'radial':
            return np.sqrt((x - self.center)**2 + (y - self.center)**2 + (z - self.center)**2)
        else:
            return z  # 或 z - base_z

    def _get_slices(self, extent: np.ndarray):
        max_extent = extent.max()
        bins = np.linspace(0, max_extent, self.num_layers + 1)
        indices = np.digitize(extent, bins) - 1
        indices = np.minimum(indices, self.num_layers - 1)
        return indices, bins

    def compute_volume_ratio_and_density(self):
        volume_ratios = []
        densities = []

        for x, y, z in self.dendrites_indices:
            n = len(x)
            extent = self._compute_extent(x, y, z)
            indices, bins = self._get_slices(extent)

            # 包裹盒体积
            max_extent = extent.max()
            if self.axis_mode == 'radial':
                box_volume = 4 / 3 * np.pi * max_extent ** 3
            else:
                box_volume = (self.grid_size ** 2) * max_extent
            ratio = n / box_volume
            volume_ratios.append(ratio)

            # 密度分布
            density = np.zeros(self.num_layers)
            for i in range(self.num_layers):
                count = np.equal(indices, i).sum()
                if self.axis_mode == 'radial':
                    r1, r2 = bins[i], bins[i+1]
                    v_shell = 4 / 3 * np.pi * (r2**3 - r1**3)
                    density[i] = count / v_shell if v_shell > 0 else 0
                else:
                    h = bins[i+1] - bins[i]
                    v_layer = self.grid_size**2 * h
                    density[i] = count / v_layer if v_layer > 0 else 0
            densities.append(density)

        return volume_ratios, densities

    def compute_growth_speed(self):
        speeds = []
        for i in range(1, len(self.dendrites_indices)):
            x0, y0, z0 = self.dendrites_indices[i - 1]
            x1, y1, z1 = self.dendrites_indices[i]
            r0 = self._compute_extent(x0, y0, z0).max()
            r1 = self._compute_extent(x1, y1, z1).max()
            dt = self.times[i] - self.times[i - 1]
            speeds.append((r1 - r0) / dt if dt > 0 else 0)
        return speeds

    def compute_fractal_dimension(self):
        log_R = []
        log_N = []
        for (x, y, z) in self.dendrites_indices:
            extent = self._compute_extent(x, y, z).max()
            log_R.append(np.log(extent + 1e-8))
            log_N.append(np.log(len(x) + 1e-8))

        slope, _ = np.polyfit(log_R, log_N, 1)
        return slope
