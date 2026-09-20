import numpy as np

def calculate_distance(points, target):
    return np.sqrt(np.sum((points - target)**2, axis=1))


points = np.array([[0, 0], [3, 4], [1, 1]])
target = np.array([0, 0])

print(calculate_distance(points, target))