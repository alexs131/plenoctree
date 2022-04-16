from lshash import LSHash
import numpy as np

points = [
    np.array([0.65, 0.32, 0.71]),
    np.array([0.95, 0.37, 0.34]),
]

points = np.stack(points)
print(points)

lsh = LSHash(2, 3, matrices_file)
