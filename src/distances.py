import numpy as np
import math

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    all_distances = []
    for exampleX in X:
        one_row = []
        for exampleY in Y:
            one_row.append(euclidean_distance(exampleX, exampleY))
        all_distances.append(one_row)
    return np.array(all_distances)
    # raise NotImplementedError()

def euclidean_distance(point1, point2):
    sum = 0
    for dimension in range(0, len(point1)):
        sum += math.pow(point1[dimension] - point2[dimension], 2)
    return math.sqrt(sum)

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    all_distances = []
    for exampleX in X:
        one_row = []
        for exampleY in Y:
            one_row.append(manhattan_distance(exampleX, exampleY))
        all_distances.append(one_row)
    return np.array(all_distances)
    # raise NotImplementedError()

def manhattan_distance(point1, point2):
    sum = 0
    for dimension in range(0, len(point1)):
        sum += abs(point1[dimension] - point2[dimension])
    return sum

def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    all_distances = []
    for exampleX in X:
        one_row = []
        for exampleY in Y:
            one_row.append(cosine_distance(exampleX, exampleY))
        all_distances.append(one_row)
    return np.array(all_distances)
    # raise NotImplementedError()

def cosine_distance(point1, point2):
    u_dot_v = np.dot(point1, point2)
    u_mag = math.sqrt(np.sum(np.square(point1)))
    v_mag = math.sqrt(np.sum(np.square(point2)))
    cosine_similarity = u_dot_v / (u_mag * v_mag)
    return 1 - cosine_similarity