from abc import ABC, abstractmethod


class BaseCluster(ABC):

    @abstractmethod
    def fit(self, X):
        pass

