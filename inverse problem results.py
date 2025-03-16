import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 39 bus box plots
data = pd.DataFrame({
    "Parameter": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"] * 5,
    "Relative Error (%)": [
        5.405, 0.07, 1.926, 1.999, 11.294, 1.461, 3.003, 5.36, 2.632, 6.532
        ,
        # First trial
0.961, 0.262, 1.417, 9.78, 10.712, 3.408, 2.404, 2.925, 0.191, 1.034

        ,  # Second trial
2.953, 5.98, 0.312, 1.716, 2.268, 0.693, 0.911, 1.857, 0.191, 0.38


        ,  # Fourth trial
        1.199, 0.939, 2.368, 0.712, 0.277, 1.781, 1.213, 1.069, 3.494, 0.442
        # Fifth trial
        ,

4.077, 0.88, 1.628, 4.929, 3.125, 3.885, 2.247, 7.85, 5.414, 0.992

    ]
})

# Plot the box plot
plt.figure(figsize=(18, 8))
sns.boxplot(x="Parameter", y="Relative Error (%)", data=data, palette="Set3")
plt.xticks(fontsize=14, fontweight='bold')  # Make x-axis values bold
plt.yticks(fontsize=14, fontweight='bold')
# Customize the plot
plt.title("Relative Error (%) for Parameters M1 to M10", fontsize=18,fontweight="bold")
plt.xlabel("Parameters", fontsize=18, fontweight="bold")
plt.ylabel("Relative Error (%)", fontsize=18, fontweight="bold")
plt.grid(True, linestyle='-', alpha=0.5)
plt.savefig('M__different epochs.png', dpi=300)

# Show the plot
plt.show()


# Example relative error data for M1 to M10
data = pd.DataFrame({
    "Parameter": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28", "D29", "D30", "D31", "D32", "D33", "D34", "D35", "D36", "D37", "D38", "D39"
] * 5,
    "Relative Error (%)": [
0.376, 23.353, 27.55, 53.724, 23.047, 13.183, 6.525, 12.993, 4.933, 2.457, 6.314,
        1.049, 3.222, 12.918, 6.094, 1.569, 3.368, 5.105, 4.481, 8.826, 4.878, 9.631,
        2.051, 1.573, 4.261, 0.84, 0.637, 0.399, 1.622, 1.676, 12.448, 2.529, 3.885, 28.946, 2.227, 5.6,
        9.689, 3.658, 5.671

        ,
        # First trial
1.588e+00, 5.634e+00, 1.154e+00, 2.545e+00, 4.942e+00, 3.501e+00, 9.689e-01,
        1.910e+00, 2.174e+00, 1.564e+00, 2.303e+00, 6.344e-01, 4.886e+00, 1.073e+01, 4.271e+01,
        1.645e+02, 3.139e+01, 6.478e+00, 2.625e+01, 1.320e+01, 2.227e+01, 3.213e+00, 4.272e+00, 4.848e+01,
        2.012e+00, 8.309e+00, 5.382e+00, 2.027e+00, 8.334e-01, 1.389e-01, 1.734e+00, 1.599e+00, 2.160e+01,
        2.030e+01, 7.381e+00, 3.556e+00, 2.068e+00, 7.445e-02, 1.326e+00

,
2.119, 2.098, 5.485, 30.943, 77.363, 250.177, 62.046, 21.132, 3.267,
        40.923, 90.235, 19.355, 19.004, 1.33, 0.674, 5.097, 1.788, 1.981, 5.597, 1.733,
        1.283, 5.041, 2.084, 1.006, 2.592, 3.096, 2.887, 1.863, 1.321, 4.281, 15.475, 0.314, 0.329,
        7.497, 2.268, 2.505, 3.838, 0.369, 1.013


        ,
1.285, 2.679, 1.829, 4.127, 4.124, 3.386, 0.095, 3.548, 3.055, 1.475,
3.076, 2.784, 2.486, 5.738, 14.93, 72.312, 27.24, 3.614, 30.016, 8.333,
7.469, 0.556, 2.109, 15.873, 2.207, 1.749, 5.557, 2.474, 3.253, 0.722,
2.757, 3.571, 10.444, 8.739, 2.277, 3.099, 4.192, 7.966, 0.993

,
10.921, 25.791, 15.18, 9.107, 16.43, 15.809, 4.247, 3.51, 0.123, 0.672,
        0.333, 0.809, 0.084, 2.394, 9.397, 70.575, 26.958, 17.476, 25.56, 14.803, 30.296, 16.577, 3.829,
        8.384, 10.098, 1.925, 6.096, 2.77, 4.222, 10.291, 3.046, 1.88, 3.27, 9.997, 4.684, 3.882, 4.252, 8.023, 0.219


        # Fifth trial
    ]
})

# Plot the box plot
plt.figure(figsize=(15, 8))
sns.boxplot(x="Parameter", y="Relative Error (%)", data=data, palette="Set3")
plt.xticks(fontsize=8, fontweight='bold')  # Make x-axis values bold
plt.yticks(fontsize=8, fontweight='bold')
# Customize the plot
plt.title("Relative Error (%) for Parameters D1 to D39", fontsize=18,fontweight="bold")
plt.xlabel("Parameters", fontsize=18, fontweight="bold")
plt.ylabel("Relative Error (%)", fontsize=18, fontweight="bold")
plt.grid(True, linestyle='-', alpha=0.5)
plt.savefig('D__different epochs.png', dpi=300)

# Show the plot
plt.show()