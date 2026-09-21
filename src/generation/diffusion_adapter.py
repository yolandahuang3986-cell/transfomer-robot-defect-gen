from .base import DefectGenerator


class DiffusionDefectGenerator(DefectGenerator):
    """Adapter placeholder for AnomalyDiffusion."""

    def fit(self, dataset, config):
        raise NotImplementedError("Integrate pinned AnomalyDiffusion implementation.")

    def generate(self, normal_images, num_samples, defect_type=None, seed=42):
        raise NotImplementedError("Integrate pinned AnomalyDiffusion implementation.")
