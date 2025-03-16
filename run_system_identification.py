import numpy as np
import time
import matplotlib.pyplot as plt

from create_example_parameters import create_example_parameters
from create_data import create_data
from PinnModel import PinnModel


def run_system_identification():

    # load or create a file with all simulation parameters such that a simulation is repeatable
    # to illustrate the working principle, examples for 1 and 4 buses are implemented
    simulation_parameters = create_example_parameters(n_buses=39)

    # at this point the training data are provided
    # here we simulate a dataset based on the previously defined simulation parameters
    x_training, y_training = create_data(simulation_parameters=simulation_parameters)

    # creating the model including building it and setting the options for the optimiser, the loss function and the
    # loss weights --> see PinnModel.py
    model = PinnModel(simulation_parameters=simulation_parameters)

    np.set_printoptions(precision=3)
    print('Starting training')
    total_start_time = time.time()

    for n_epochs, batch_size in zip(simulation_parameters['training']['epoch_schedule'],
                                    simulation_parameters['training']['batching_schedule']):

        epoch_start_time = time.time()
        model.fit(x_training,
                  y_training,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=0,
                  shuffle=True)
        epoch_end_time = time.time()

        print(f'Trained for {n_epochs} epochs with batch size {batch_size} '
              f'in {epoch_end_time - epoch_start_time:.2f} seconds.')

        model.PinnLayer.print_relative_error()

    total_end_time = time.time()
    print(f'Total training time: {total_end_time - total_start_time:.1f} seconds')
    model.save_weights('pinn_saved_weights.h5')


    predictions = model.predict(x_training)

    y_delta_true = y_training[0][:21]

    predicted_omegas = predictions[1][:21]
    y_omega_true = y_training[1][:21]
    predicted_deltas = predictions[0][:21]

    print(f"Shape of predicted deltas: {predicted_deltas.shape}")
    print(f"Shape of true deltas: {y_delta_true.shape}")

    # Plot true vs predicted delta for each bus

    for bus in range(39):
        plt.plot(y_delta_true[:, bus], '--', label=f'Actual delta Bus {bus + 1}')
        plt.plot(predicted_deltas[:, bus], label=f'Predicted delta Bus {bus + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('delta')
    plt.title('Predicted vs Actual angle for 39 Buses')
    plt.legend(f'{bus + 1}')
    plt.grid(True)
    plt.show()

    for bus in range(39):
        plt.plot(y_omega_true[:, bus], '--', label=f'Actual omega Bus {bus + 1}')
        plt.plot(predicted_omegas[:, bus], label=f'Predicted omega Bus {bus + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('delta')
    plt.title('Predicted vs Actual omega for 39 Buses')
    plt.legend(f'{bus + 1}')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_system_identification()
