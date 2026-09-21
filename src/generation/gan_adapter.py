from .base import DefectGenerator


class GANDefectGenerator(DefectGenerator):
    """Adapter placeholder for the upstream GAN defect-synthesis implementation."""

    def fit(self, dataset, config):
        raise NotImplementedError("Integrate pinned upstream GAN implementation.")

    def generate(self, normal_images, num_samples, defect_type=None, seed=42):
        raise NotImplementedError("Integrate pinned upstream GAN implementation.")
