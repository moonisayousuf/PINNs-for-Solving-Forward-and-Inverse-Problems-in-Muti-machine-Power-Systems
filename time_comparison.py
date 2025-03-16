from create_example_parameters import create_example_parameters
from create_data import create_data
from PinnModel import PinnModel
import matplotlib.pyplot as plt
import numpy as np
import time

# plots the results of time comparison between ode solver and pinns
simulation_parameters = create_example_parameters(n_buses=39)
x_training, y_training = create_data(simulation_parameters=simulation_parameters)
model = PinnModel(simulation_parameters=simulation_parameters)
model.load_weights('pinn_saved_weights.h5')



horizons = np.linspace(1, 20, 20)
computation_times_pinns = []
for horizon in horizons:
    start_time = time.time()
    predictions = model.predict(x_training)
    end_time = time.time()
    computation_times_pinns.append(end_time - start_time)

print("horizons of pinns are", horizons)
print("computation_times of pinns are", computation_times_pinns)

plt.figure(figsize=(8, 6))
plt.plot( horizons,computation_times_pinns, marker='o', color='b')
plt.yscale('log')
#plt.xscale('log')
plt.title('Computation Time vs. Prediction Horizon of PINNs')
plt.xlabel('Prediction Horizon (s)')
plt.ylabel('Computation Time (s)')
plt.grid(True)
plt.savefig('time_pinn.png', dpi=300)
plt.show()


#ode time to solve 39 bus problem
odetimes=[0.08462953567504883, 0.08651161193847656, 0.09261751174926758, 0.08460021018981934, 0.0780630111694336, 0.07822251319885254, 0.07898497581481934, 0.07777094841003418, 0.08284258842468262, 0.08040547370910645, 0.07866239547729492, 0.08051156997680664, 0.08481383323669434, 0.08629655838012695, 0.08071613311767578, 0.08258318901062012, 0.0828711986541748, 0.08196902275085449, 0.08398222923278809, 0.08585453033447266]

# Define colors
ode_color = "#1f77b4"  # Blue
pinn_color = "#d62728"  # Red

# Create figure
plt.figure(figsize=(10, 6))
plt.plot(horizons, odetimes, label="ODE Solver", marker="o", linestyle="-", markersize=8, color=ode_color, linewidth=2)
plt.plot(horizons, computation_times_pinns, label="PINNs", marker="s", linestyle="--", markersize=8, color=pinn_color, linewidth=2)

# Log scale for better visibility
plt.yscale("log")

# Labels and title with improved font styling
plt.xlabel("Time (seconds)", fontsize=14, fontweight="bold")
plt.ylabel("Average Evaluation Time (Seconds)", fontsize=14, fontweight="bold")
plt.title("Comparison of Prediction time: PINNs vs ODE Solver", fontsize=16, fontweight="bold")

# Customize ticks
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")

# Grid customization
plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

# Add legend with improved styling
plt.legend(fontsize=12, loc="best", frameon=True, fancybox=True, shadow=True, borderpad=1)

# Save and show
plt.savefig("time_compare_pinn_ode.png", dpi=300, bbox_inches="tight")
plt.show()