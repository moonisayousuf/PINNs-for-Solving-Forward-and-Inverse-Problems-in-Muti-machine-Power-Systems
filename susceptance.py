import numpy as np


def create_susceptance_matrix(line_data, n_buses):
    """
    Creates a symmetric susceptance matrix for the IEEE 39-bus system.

    Parameters:
        line_data (list of tuples): Each tuple represents a line (from_bus, to_bus, X).
                                    'from_bus' and 'to_bus' are 1-indexed.
                                    'X' is the reactance in p.u.
        n_buses (int): Number of buses (order of the matrix).

    Returns:
        np.ndarray: Symmetric susceptance matrix of shape (n_buses, n_buses).
    """
    # Initialize the susceptance matrix with zeros
    susceptance_matrix = np.zeros((n_buses, n_buses))

    # Fill the off-diagonal elements with 1/X (susceptance)
    for from_bus, to_bus, reactance in line_data:
        susceptance =  1/reactance  # Calculate susceptance
        susceptance_matrix[from_bus - 1, to_bus - 1] = susceptance  # 1-indexed to 0-indexed
        susceptance_matrix[to_bus - 1, from_bus - 1] = susceptance  # Symmetric matrix

    # Set diagonal elements to 1
    np.fill_diagonal(susceptance_matrix, 1)

    return susceptance_matrix

# Each tuple is (from_bus, to_bus, reactance)
line_data = [
    (1, 2, 0.0411),
    (1, 39, 0.0250),
    (2, 3, 0.0151),
    (2, 25, 0.0086),
    (2, 30, 0.0181),
    (3, 4, 0.0213),
    (3, 18, 0.0133),
    (4, 5, 0.0128),
    (4, 14, 0.0129),
    (5, 6, 0.0026),
    (5, 8, 0.0112),
    (6, 7, 0.0092),
    (6, 11, 0.0082),
    (6, 31, 0.0250),
    (7, 8, 0.0046),
    (8, 9, 0.0363),
    (9, 39, 0.0250),
    (10, 11, 0.0043),
    (10, 13, 0.0043),
    (10, 32, 0.0200),
    (12, 11, 0.0435),
    (12, 13, 0.0435),
    (13, 14, 0.0101),
    (14, 15, 0.0217),
    (15, 16, 0.0094),
    (16, 17, 0.0089),
    (16, 19, 0.0195),
    (16, 21, 0.0135),
    (16, 24, 0.0059),
    (17, 18, 0.0082),
    (17, 27, 0.0173),
    (19, 20, 0.0138),
    (19, 33, 0.0142),
    (20, 34, 0.0180),
    (21, 22, 0.0140),
    (22, 23, 0.0096),
    (22, 35, 0.0143),
    (23, 24, 0.0350),
    (23, 36, 0.0272),
    (25, 26, 0.0323),
    (25, 37, 0.0232),
    (26, 27, 0.0147),
    (26, 28, 0.0474),
    (26, 29, 0.0625),
    (28, 29, 0.0151),
    (29, 38, 0.0156)
]

# Create the susceptance matrix for the IEEE 39-bus system
n_buses = 39
susceptance_matrix = create_susceptance_matrix(line_data, n_buses)

# Print the resulting matrix
print("Susceptance Matrix (lambda_B):")
print(susceptance_matrix)
