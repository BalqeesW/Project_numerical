# Lane-Emden Numerical Methods Lab - GUI Execution Guide

This project contains a comprehensive PyQt5 graphical user interface (GUI) designed to interact with multiple numerical solver engines (FDM, FEM, Chebyshev, Shooting RK4, ADM, and B-Spline) for the Lane-Emden Boundary Value Problems.

## Prerequisites

Ensure you have **Python 3.11** (or a compatible 3.x version) installed on your system. 

It is highly recommended to run this application within a Python Virtual Environment (`.venv`) to prevent conflicts with global system packages.

## 1. Environment Setup & Installation

Open your terminal (or VS Code terminal) and navigate to the root folder of this project.

If you are using a virtual environment, ensure it is activated. Your terminal prompt should show `(.venv)` at the beginning of the line.

Install all required dependencies by running the following command:

```bash
pip install PyQt5 matplotlib scipy numpy python-docx pandas openpyxl sympy mpmath
Note: If you encounter an error stating a module is not found (e.g., ModuleNotFoundError: No module named 'pandas'), verify that your terminal is actively using the correct virtual environment where the packages were installed.

2. Folder Structure
For the GUI to successfully import the mathematical engines, ensure your project directory maintains the following structure:

Plaintext
numericalProject/
│
├── solvers/
│   ├── main_app.py                        # The main GUI controller script
│   ├── finite_difference_Polytropic.py    # Solver module
│   ├── finite_element_Polytropic.py       # Solver module
│   ├── spectral_Polytropic.py             # Solver module
│   ├── shooting_RK4_Polytropic.py         # Solver module
│   ├── series_ADM_Polytropic.py           # Solver module
│   └── b_spline_Polytropic.py             # Solver module
│
└── requirements.txt                       # (Optional) List of dependencies
3. Running the Application
To launch the GUI, you must execute the main_app.py script.

From the root numericalProject directory, run:

Bash
python solvers/main_app.py
Alternatively, you can navigate directly into the solvers directory first:

Bash
cd solvers
python main_app.py
4. Usage Instructions
Problem Selection: Use the toggle switches on the left and right panels to enable Problem 1 (Isothermal) or Problem 2 (Polytropic).

Method Selection: Check the boxes for the numerical methods you wish to execute and compare.

Parameter Configuration: Adjust the Min, Max, and Samples to sweep through different parameter values (e.g., polytropic index n).

Execution Mode: Select how you want to view the generated 3D figures:

One-Shot: Generates and skips to the final figure immediately.

Automated Delay: Flips through the generation history automatically at the specified time interval.

Manual / Step-by-Step: Allows you to click through the iterative history and final results manually using the navigation bar.

Exporting: Use the Export Center at the bottom to save your generated 3D figures as PNGs, compile them into a formatted Word document (.docx), or export the raw numerical matrices to an Excel workbook (.xlsx).