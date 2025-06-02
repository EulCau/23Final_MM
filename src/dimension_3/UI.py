import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QComboBox, QGroupBox, QDoubleSpinBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import dla_3d as dla
import generate_field_3d as gf3


class DLA3DApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D DLA Simulator with SEI Model")
        self.setGeometry(100, 100, 1000, 800)
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Control panel
        self.control_panel = QGroupBox("Simulation Parameters")
        self.control_layout = QVBoxLayout()
        
        # Parameter inputs
        self.params = {
            'sei_growth_rate': ('SEI Growth Rate', 0.01, 0.0, 1.0, 0.01),
            'sei_max_thickness': ('SEI Max Thickness', 1.0, 0.1, 10.0, 0.1),
            'sei_resistance_factor': ('SEI Resistance Factor', 0.1, 0.0, 1.0, 0.01),
            'max_particles': ('Max Particles', 1000, 100, 10000, 100),
            'attach_prob': ('Attachment Probability', 1.0, 0.0, 1.0, 0.1),
            'field_strength': ('Electric Field Strength', 1.0, 0.01, 10.0, 0.1)
        }
        
        self.input_widgets = {}
        for param, (label, default, min_val, max_val, step) in self.params.items():
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(label))
            if param in ['max_particles', 'max_steps_per_particle']:
                spinbox = QDoubleSpinBox()
                spinbox.setDecimals(0)
            else:
                spinbox = QDoubleSpinBox()
            spinbox.setValue(default)
            spinbox.setMinimum(min_val)
            spinbox.setMaximum(max_val)
            spinbox.setSingleStep(step)
            hbox.addWidget(spinbox)
            self.input_widgets[param] = spinbox
            self.control_layout.addLayout(hbox)
        
        # Field type selection
        self.field_type_label = QLabel("Electric Field Type:")
        self.field_type_combo = QComboBox()
        self.field_type_combo.addItems(["Point Charge", "Uniform Up", "Uniform Down", 
                                      "Uniform Left", "Uniform Right", "Uniform Front", "Uniform Back"])
        self.control_layout.addWidget(self.field_type_label)
        self.control_layout.addWidget(self.field_type_combo)
        
        # Buttons
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.control_layout.addWidget(self.run_button)
        self.control_layout.addWidget(self.save_button)
        
        self.control_panel.setLayout(self.control_layout)
        
        # Visualization area
        self.figure = plt.figure(figsize=(8, 8))
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.figure)
        
        # Add to main layout
        self.main_layout.addWidget(self.control_panel, stretch=1)
        self.main_layout.addWidget(self.canvas, stretch=3)
        
        # Simulation results
        self.simulator = None
        self.results = None
    
    def get_parameters(self):
        params = {
            'sei_growth_rate': self.input_widgets['sei_growth_rate'].value(),
            'sei_max_thickness': self.input_widgets['sei_max_thickness'].value(),
            'sei_resistance_factor': self.input_widgets['sei_resistance_factor'].value(),
            'max_particles': int(self.input_widgets['max_particles'].value()),
            'attach_prob': self.input_widgets['attach_prob'].value(),
            'field_strength': self.input_widgets['field_strength'].value()
        }
        return params
    
    def get_field_type(self):
        mapping = {
            0: gf3.FieldType3d.POINT,
            1: gf3.FieldType3d.UNIFORM_UP,
            2: gf3.FieldType3d.UNIFORM_DOWN,
            3: gf3.FieldType3d.UNIFORM_LEFT,
            4: gf3.FieldType3d.UNIFORM_RIGHT,
            5: gf3.FieldType3d.UNIFORM_FRONT,
            6: gf3.FieldType3d.UNIFORM_BACK
        }
        return mapping[self.field_type_combo.currentIndex()]
    
    def run_simulation(self):
        params = self.get_parameters()
        field_type = self.get_field_type()
        
        # Create electric field
        grid_size = 101 if field_type == gf3.FieldType3d.POINT else 41
        electric_field = gf3.ElectricField3d(
            field_type=field_type,
            grid_size=grid_size,
            strength=params['field_strength']
        )
        
        # Create simulator
        self.simulator = dla.DLASimulator(
            sei_growth_rate=params['sei_growth_rate'],
            sei_max_thickness=params['sei_max_thickness'],
            sei_resistance_factor=params['sei_resistance_factor'],
            max_particles=params['max_particles'],
            attach_prob=params['attach_prob'],
            electric_field=electric_field
        )
        
        # Run simulation
        self.simulator.simulate()
        
        # Visualize results
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        x, y, z = np.where(self.simulator.grid == 1)
        ax.scatter(x, y, z, c='blue', marker='o', s=2)
        ax.set_title(f"3D DLA Simulation (Field Strength: {params['field_strength']})")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_zlim(0, grid_size)
        self.canvas.draw()
    
    def save_results(self):
        if self.simulator is None:
            return
        
        # Save the cluster data
        np.save('dla_cluster.npy', self.simulator.grid)
        
        # Save visualization
        self.figure.savefig('dla_simulation.png')
        print("Results saved to dla_cluster.npy and dla_simulation.png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DLA3DApp()
    window.show()
    sys.exit(app.exec_())
