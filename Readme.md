
# 3D Dendritic Growth Simulation with Electric Field and SEI Effects

本项目模拟三维空间中在电场和 SEI 膜影响下的枝晶生长过程, 并提供可视化界面, 数据评估与对比分析功能. 模拟基于 DLA 模型扩展, 支持电场引导和多参数调控, 适用于研究电化学系统中枝晶与 SEI 的生长行为. 

---

## 项目结构

````

project_root/
│
├── result/                             # 实验图像输出目录
│   ├── data/                           # 实验数据保存 (.npz 格式)
│   │   └── ...                         # 每次实验生成的压缩数据文件
│   │
│   └── ...                             # 实验结果评估图像
│
├── config/                             # 实验配置文件 (.json 格式)
│
├── src/
│   ├── dimension\_2/                   # 二维模拟, 仅用于验证模型直观效果
│   │   ├── dla\_2d.py
│   │   └── generate\_field\_2d.py
│   │
│   └── dimension\_3/                   # 三维模拟核心模块
│       ├── analysis\_metrics\_3d.py    # 枝晶形貌与 SEI 厚度分析指标
│       ├── dla\_3d.py                  # 三维 DLA 主体模拟逻辑
│       ├── generate\_field\_3d.py      # 电场生成模块 (点电荷或匀强场)
│       ├── plot\_dla\_metrics\_3d.py   # 可视化分析结果, 支持多组参数对比绘图
│       ├── test\_runner\_3d.py         # 实验运行控制器 (参数配置、数据保存、调用评估)
│       └── UI.py                       # 图形界面交互模拟器
│
├── report.pdf                          # 项目论文
│
├── requirements.txt                    # python 依赖文件
│
└── Readme.md                           # 当前说明文件

````

---

## 功能说明

### 1. 三维模拟核心 (`dla_3d.py`)
- 支持 **电场引导 DLA 生长**, 包含:
  - 点电荷场 (radial)
  - 匀强电场 (uniform)
- 引入 **SEI 膜建模**, 受生长速率和阻力因子调控

### 2. 电场生成 (`generate_field_3d.py`)
- 按参数自动生成电场对象并提供对外接口
- 可扩展支持不同电场类型

### 3. 实验控制与评估 (`test_runner_3d.py`)
- 自动读取或保存 `.npz` 实验结果
- 实现以下指标:
  - **体积占比与空间密度**
  - **生长速率 / 累积速率**
  - **分形维数**
  - **SEI 厚度分布**

### 4. 可视化 (`plot_dla_metrics_3d.py`)
- 加载一组参数配置文件, 自动绘制对比图
- 支持使用已有数据或强制重新实验

### 5. 图形界面 (`UI.py`)
- 基于 `PyQt` 提供直观参数配置与实验展示
- 支持旋转/缩放/交互式查看模拟结果

---

## 使用方法

### 1. 运行单组实验:

修改 `src/dimension_3/test_runner_3d.py` 主程序中的 `test` 的实参 `json` 文件名指向 `config` 中的某文件, 运行即可, 若想重复进行相同配置的数值实验, 可以为 `test` 增加实参 `renew=True`.

### 2. 对比不同实验参数的评估结果绘图:

修改 `plot_dla_metrics_3d.py` 中 `config_names` 和 `labels` 并运行即可.

### 3. 启动图形界面:

```bash
python src/dimension_3/UI.py
```

---

## 数据说明

所有实验数据保存在:

```
result/data/<config_name>_result.npz
```

包含:

* `dendrites_indices`: 每个时刻的枝晶体素坐标
* `sei_thicknesses`: 每个时刻的三维 SEI 膜厚度场
* `time`: 粒子总数 (时间标尺)
* `field_type`: 电场类型 (用于评估逻辑)

---

## 注意事项

* 实验结果默认基于 SEI 膜厚度推断网格大小
* 若需重重复进行已有结果的实验, 请设置 `renew=False`.
  在 `plot_dla_metrics_3d.py` 中修改 `results = [test(name, renew=False) for name in config_names]`  或在 `test_runner_3d.py` 中为 `test` 增加实参 `renew=True` 均可.
* 仅 `3D` 实验具有完整评估流程, `2D` 仅用于模型验证

---

## 依赖

* Python >= 3.8
* numpy
* matplotlib
* PyQt5 (用于 UI)
