from matplotlib.lines import Line2D

from create_example_parameters import create_example_parameters
from create_data import create_data
from PinnModel import PinnModel
import matplotlib.pyplot as plt


simulation_parameters = create_example_parameters(n_buses=39)
x_training, y_training = create_data(simulation_parameters=simulation_parameters)
model = PinnModel(simulation_parameters=simulation_parameters)
model.load_weights('pinn_saved_weights.h5')

predictions=model.predict(x_training)
predicted_delta=predictions[0][:21]
actual_delta=y_training[0][:21]

n_buses = 39

# Create plot for each bus
for bus in range(n_buses):

    plt.plot(predicted_delta, label=f'Predicted Delta - Bus {bus+1}', linestyle=':', marker='o')
    plt.plot(actual_delta, label=f'Actual Delta - Bus {bus+1}', linestyle='-', marker='x')
legend_elements = [
    Line2D([0], [0], marker='o', color='b', markerfacecolor='b', markersize=10, label='Predicted Delta'),
    Line2D([0], [0], marker='x', color='b', markerfacecolor='r', markersize=10, label='Actual Delta')
]
# Adding title, labels, and legend
plt.title("Predicted vs Actual Delta", fontsize=14, fontweight='bold')
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Delta", fontsize=12)
plt.legend(handles=legend_elements)

plt.savefig('Delta.png', dpi=300)


# Show plot
plt.show()

predicted_omega=predictions[1][:21]
actual_omega=y_training[1][:21]

for bus in range(n_buses):
    # Assuming predictions and actual values are organized by buses
    plt.plot(predicted_omega, label=f'Predicted omega - Bus {bus+1}', linestyle=':', marker='o')
    plt.plot(actual_omega, label=f'Actual omega - Bus {bus+1}', linestyle='-', marker='x')
legend_elements = [
    Line2D([0], [0], marker='o', color='b', markerfacecolor='b', markersize=10, label='Predicted omega '),
    Line2D([0], [0], marker='x', color='b', markerfacecolor='r', markersize=10, label='Actual omega')
]
# Adding title, labels, and legend
plt.title("Predicted vs Actual omega", fontsize=14, fontweight='bold')
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Delta", fontsize=12)
plt.legend(handles=legend_elements)

plt.savefig('omega.png', dpi=300)


# Show plot
plt.show()



