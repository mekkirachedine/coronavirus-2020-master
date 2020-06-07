import numpy as np
import statistics
from .distances import euclidean_distances, manhattan_distances, cosine_distances

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances,
        if  'cosine', use cosine_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean', 'manhattan', or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.training_features = None
        self.training_targets = None

        # raise NotImplementedError()


    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        HINT: One use case of KNN is for imputation, where the features and the targets 
        are the same. See tests/test_collaborative_filtering for an example of this.
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        R = len(features)
        C = len(features[0])
        fixed_features = []
        for i in range(0, R):
            one_feature = []
            for j in range(1, C):
                one_feature.append(float(features[i][j]))
            fixed_features.append(one_feature)
        np_features = np.array(fixed_features)
        #print(np_features)
        self.training_features = np_features

        R = len(targets)
        fixed_targets = []
        for i in range(0, R):
            if targets[i] == 'N/A':
                fixed_targets.append(-1)
            else:
                fixed_targets.append(float(targets[i]))
        np_targets = np.array(fixed_targets)
        #print(np_targets)
        self.training_targets = np_targets
        # raise NotImplementedError()
        

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        the_labels = []
        the_locations = []
        # print("features: ", self.training_features, "\n")
        # print("targets: ", self.training_targets, "\n")
        # print("new features: ", features, "\n")
        for example in features:
            label, locations = class_predictor(self, example, ignore_first)
            the_labels.append(label)
            the_locations.append(locations)
        # print("labels: ", the_labels, "\n")
        return np.array(the_labels), locations
        # raise NotImplementedError()

def class_predictor(nearest_neighbor, example, ignore_first):
    if nearest_neighbor.distance_measure == 'euclidean':
        all_distances = euclidean_distances(nearest_neighbor.training_features, [example])
    elif nearest_neighbor.distance_measure == 'manhattan':
        all_distances = manhattan_distances(nearest_neighbor.training_features, [example])
    elif nearest_neighbor.distance_measure == 'cosine':
        all_distances = cosine_distances(nearest_neighbor.training_features, [example])
    else:
        raise ValueError("Invalid distance type")

    sorted_indices = sort_and_return_indices(all_distances)
    ri = []
    rt = []
    if ignore_first:
        index = 1
    else:
        index = 0
    while len(rt) < nearest_neighbor.n_neighbors:
        if nearest_neighbor.training_targets[sorted_indices[index]] != -1:
            ri.append(sorted_indices[index])
            rt.append(nearest_neighbor.training_targets[sorted_indices[index]])
        index += 1

    relevant_indices = np.array(ri)
    relevant_targets = np.array(rt)
    #print(relevant_indices)
    #print(relevant_targets)

    if nearest_neighbor.aggregator == 'mode':
        label = statistics.mode(relevant_targets)
    elif nearest_neighbor.aggregator == 'mean':
        label = statistics.mean(relevant_targets)
    elif nearest_neighbor.aggregator == 'median':
        label = statistics.median(relevant_targets)
    else:
        raise ValueError("Invalid aggregator type")
    return label, relevant_indices

def sort_and_return_indices(sl):
    indices = []
    some_list = sl.tolist()
    for index in range(0, len(some_list)):
        indices.append(index)

    for i in range(1, len(some_list)):
        current = some_list[i]
        currenti = indices[i]
        j = i - 1
        while (j >= 0 and some_list[j] > current):
            some_list[j + 1] = some_list[j]
            indices[j + 1] = indices[j]
            j = j - 1
        some_list[j + 1] = current
        indices[j + 1] = currenti

    return np.array(indices)







