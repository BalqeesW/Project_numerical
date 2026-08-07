import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QCheckBox, QLabel, 
                             QDoubleSpinBox, QSpinBox, QPushButton, QRadioButton,
                             QScrollArea, QFrame, QGridLayout)
from PyQt5.QtCore import Qt

class NumericalMethodsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane-Emden Numerical Methods Lab")
        self.setGeometry(100, 100, 1200, 900)
        
        # Apply Dark Theme styling to match Replit mockup
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #E0E0E0; font-family: 'Segoe UI', Arial, sans-serif; }
            QGroupBox { font-weight: bold; border: 1px solid #333; margin-top: 10px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; color: #FF5722; }
            QPushButton { background-color: #333; border: 1px solid #555; padding: 10px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #444; }
            #RunButton { background-color: #D84315; color: white; font-size: 16px; }
            #RunButton:hover { background-color: #E64A19; }
            QDoubleSpinBox, QSpinBox { background-color: #222; border: 1px solid #444; padding: 5px; }
        """)

        # Main Scroll Area (since the interface is tall)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Header
        header_label = QLabel("Lane-Emden Numerical Methods Lab")
        header_label.setStyleSheet("font-size: 24px; font-family: Georgia, serif; color: #E0E0E0; margin: 10px;")
        self.main_layout.addWidget(header_label)

        # Split Screen for Problem 1 and Problem 2
        self.split_layout = QHBoxLayout()
        
        self.prob1_widget = self.create_problem_panel("Problem 1: Isothermal Gas Sphere", "Lambda (λ)")
        self.prob2_widget = self.create_problem_panel("Problem 2: Polytropic Model", "Polytropic Index (n)")
        
        self.split_layout.addWidget(self.prob1_widget)
        self.split_layout.addWidget(self.prob2_widget)
        self.main_layout.addLayout(self.split_layout)

        # Execution Mode & Run Button
        self.main_layout.addWidget(self.create_execution_panel())

        # 3D Figure Gallery Placeholder
        self.gallery_group = QGroupBox("3D Figure Gallery")
        gallery_layout = QVBoxLayout()
        gallery_placeholder = QLabel("Configure parameters above and select an execution mode to begin.")
        gallery_placeholder.setAlignment(Qt.AlignCenter)
        gallery_placeholder.setMinimumHeight(200)
        gallery_placeholder.setStyleSheet("border: 1px dashed #555; color: #888;")
        gallery_layout.addWidget(gallery_placeholder)
        self.gallery_group.setLayout(gallery_layout)
        self.main_layout.addWidget(self.gallery_group)

        # Export Center
        self.main_layout.addWidget(self.create_export_panel())

        self.scroll_area.setWidget(self.main_widget)
        self.setCentralWidget(self.scroll_area)

    def create_problem_panel(self, title, param_name):
        panel = QGroupBox(title)
        layout = QVBoxLayout()
        
        # Enable Switch (Mockup)
        enable_cb = QCheckBox("Enabled")
        enable_cb.setChecked(True)
        enable_cb.setStyleSheet("color: #FF5722;")
        layout.addWidget(enable_cb, alignment=Qt.AlignRight)

        # Numerical Methods
        methods_group = QGroupBox("NUMERICAL METHODS")
        methods_layout = QGridLayout()
        self.select_all_cb = QCheckBox("Select All")
        methods_layout.addWidget(self.select_all_cb, 0, 0, 1, 2)
        
        methods = ["FDM", "FEM", "Chebyshev", "Shooting", "ADM", "B-Spline"]
        for i, method in enumerate(methods):
            row = (i // 2) + 1
            col = i % 2
            methods_layout.addWidget(QCheckBox(method), row, col)
        methods_group.setLayout(methods_layout)
        layout.addWidget(methods_group)

        # Parameter Variation
        params_group = QGroupBox("PARAMETER VARIATION")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel(f"{param_name} / xMax Sequence:"), 0, 0, 1, 3)
        params_layout.addWidget(QLabel("MIN"), 1, 0)
        params_layout.addWidget(QLabel("MAX"), 1, 1)
        params_layout.addWidget(QLabel("STEP"), 1, 2)
        
        params_layout.addWidget(QDoubleSpinBox(), 2, 0)
        params_layout.addWidget(QDoubleSpinBox(), 2, 1)
        params_layout.addWidget(QDoubleSpinBox(), 2, 2)
        
        params_layout.addWidget(QLabel("DOMAIN (XMAX)"), 3, 0)
        params_layout.addWidget(QLabel("TARGET (YTARGET)"), 3, 1)
        params_layout.addWidget(QDoubleSpinBox(), 4, 0)
        params_layout.addWidget(QDoubleSpinBox(), 4, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Termination Conditions
        term_group = QGroupBox("TERMINATION CONDITIONS")
        term_layout = QGridLayout()
        
        term_layout.addWidget(QLabel("E_A (APPROX. ERROR)"), 0, 0)
        term_layout.addWidget(QLabel("E_T (TRUE ERROR)"), 0, 1)
        ea_input = QDoubleSpinBox()
        ea_input.setDecimals(9)
        ea_input.setValue(0.000000001) # 1e-9
        et_input = QDoubleSpinBox()
        et_input.setDecimals(9)
        et_input.setValue(0.000000001)
        term_layout.addWidget(ea_input, 1, 0)
        term_layout.addWidget(et_input, 1, 1)
        
        term_layout.addWidget(QLabel("MAX ITERATIONS"), 2, 0)
        term_layout.addWidget(QLabel("GRID POINTS (N)"), 2, 1)
        max_iter = QSpinBox()
        max_iter.setMaximum(10000)
        max_iter.setValue(60)
        grid_pts = QSpinBox()
        grid_pts.setMaximum(10000)
        grid_pts.setValue(100)
        term_layout.addWidget(max_iter, 3, 0)
        term_layout.addWidget(grid_pts, 3, 1)

        term_group.setLayout(term_layout)
        layout.addWidget(term_group)
        
        panel.setLayout(layout)
        return panel

    def create_execution_panel(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 20, 0, 20)

        # Modes
        modes_group = QGroupBox("EXECUTION MODE")
        modes_layout = QHBoxLayout()
        
        self.radio_oneshot = QRadioButton("One-Shot")
        self.radio_oneshot.setChecked(True)
        
        self.radio_delay = QRadioButton("Automated Delay")
        self.delay_spinbox = QDoubleSpinBox()
        self.delay_spinbox.setSuffix(" sec")
        self.delay_spinbox.setValue(1.5)
        self.delay_spinbox.setToolTip("User-defined delay as per project requirements.")
        
        self.radio_manual = QRadioButton("Manual / Step-by-Step")
        
        modes_layout.addWidget(self.radio_oneshot)
        modes_layout.addWidget(self.radio_delay)
        modes_layout.addWidget(self.delay_spinbox)
        modes_layout.addWidget(self.radio_manual)
        modes_group.setLayout(modes_layout)
        
        # Run Button
        self.btn_run = QPushButton("RUN")
        self.btn_run.setObjectName("RunButton")
        self.btn_run.setMinimumHeight(60)
        
        layout.addWidget(modes_group, stretch=2)
        layout.addWidget(self.btn_run, stretch=1)
        return panel

    def create_export_panel(self):
        panel = QGroupBox("Export Center")
        layout = QHBoxLayout()
        
        btn_folder = QPushButton("Save Figures to Folder\n\nDownload all gallery figures")
        btn_word = QPushButton("Export to Word\n\nGenerate a .docx report with captions")
        btn_excel = QPushButton("Export to Excel\n\nExport solution data across multiple sheets")
        
        btn_folder.setMinimumHeight(100)
        btn_word.setMinimumHeight(100)
        btn_excel.setMinimumHeight(100)

        layout.addWidget(btn_folder)
        layout.addWidget(btn_word)
        layout.addWidget(btn_excel)
        
        panel.setLayout(layout)
        return panel

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NumericalMethodsGUI()
    window.show()
    sys.exit(app.exec_())