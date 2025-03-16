Physics-Informed Neural Networks for Solving Forward and Inverse Problems in Muti-machine Power Systems

The increasing integration of renewable energy into traditional power grids presents new challenges, particularly in:
1)Dynamic transient simulations
2)Parameter estimation under uncertainty
This repository demonstrates how Physics-Informed Neural Networks (PINNs) can efficiently address these challenges, even with limited state measurements. By embedding power system swing dynamics into the neural network training process, we achieve:
✅ Accurate predictions with small datasets
✅ Fast computation compared to conventional solvers
✅ Generalizability to multi-machine power system models

🛠️ Key Features
✔️ Solving forward and inverse problems in power system dynamics
✔️ Training PINNs for accurate predictions with limited data
✔️ Comparison of PINNs vs RK45 ODE solver for different power grids
✔️ Implementation on multi-machine power systems (4-bus, 6-bus, 39-bus)

📊 Results & Findings
We evaluate PINNs based on:
📌 Accuracy in predicting system states
📌 Computational cost compared to numerical solvers
📌 Scalability for large-scale power grids

Repository Structure
📦 Physics-Informed-Neural-Networks-for-Power-Systems
┣ 📜 run_system_identification.py         # contains entire workflow
┣ 📜 create_example_parameters.py         # creates dictionary of parameters
┣ 📜 create_data.py                      # creates training data
 ┣ 📜 ode_solver.py                       # RK45ode solver
 ┣ 📜 PinnModel.py                        # PINN network model
 ┣ 📜 PinnLayer.py                        # Layer combining neural network with automatic differentiation
 ┣ 📜 forward problem results.py          # produces results of forward problem
 ┣ 📜 inverse problem results.py          # produces results of forward problem
 ┣ 📜 susceptance .py                    # creates susceptance matrix
 ┣ 📜 time_comparison.py               # compares ode time with pinns time
 ┣ 📜 README.md                          # Documentation

