import numpy as np

class Point:
    def __init__(self, coordinate):
        self.coordinate = coordinate

    def __lt__(self, other):
        return self.cluster_number < other.cluster_number

    def __le__(self, other):
        return self.cluster_number <= other.cluster_number

    def sub(self, other):
        return self.__sub__(other)

    def __sub__(self, other):
        return np.linalg.norm(self.coordinate - other.coordinate)

    def __repr__(self):
        return f"(coordinate: {self.coordinate}, cluster_number: {self.cluster_number}, index_of_base_sample: {self.base_sample_index})"
