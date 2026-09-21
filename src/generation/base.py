from abc import ABC, abstractmethod


class DefectGenerator(ABC):
    @abstractmethod
    def fit(self, dataset, config):
        ...

    @abstractmethod
    def generate(self, normal_images, num_samples, defect_type=None, seed=42):
        """Return generated images, masks and metadata."""
        ...
