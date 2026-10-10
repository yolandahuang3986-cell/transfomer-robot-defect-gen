"""Shared post-generation QC for synthetic image/mask batches."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
import time

import numpy as np
from scipy.ndimage import label

from .base import GeneratedBatch, validate_batch
from ..experiments.scope import require_in_scope


@dataclass(frozen=True)
class QCPolicy:
    """Versioned project-wide thresholds; change only for every method at once."""

    version: str = "synthetic-qc-v1"
    min_mask_object_area: float = 0.001
    max_mask_object_area: float = 0.20
    min_mask_foreground_overlap: float = 0.90
    max_attempts_per_target: int = 10
    foreground_border_fraction: float = 0.05
    foreground_distance_floor: float = 0.08
    foreground_mad_multiplier: float = 6.0


DEFAULT_QC_POLICY = QCPolicy()


@dataclass
class QCReport:
    policy_version: str
    target_count: int | None
    attempted_count: int
    accepted_count: int
    rejection_counts: dict[str, int]
    rejected_samples: list[dict] = field(default_factory=list)
    generation_seconds: float = 0.0
    qc_seconds: float = 0.0

    @property
    def target_met(self):
        return self.target_count is None or self.accepted_count == self.target_count

    @property
    def acceptance_rate(self):
        return self.accepted_count / self.attempted_count if self.attempted_count else 0.0

    def to_dict(self):
        result = asdict(self)
        result.update(target_met=self.target_met, acceptance_rate=self.acceptance_rate)
        return result


@dataclass
class QCResult:
    batch: GeneratedBatch
    report: QCReport

    def require_target(self):
        if not self.report.target_met:
            raise QCUnderfilledError(self)
        return self.batch


class QCUnderfilledError(RuntimeError):
    """Raised when a caller requires the requested accepted sample count."""

    def __init__(self, result: QCResult):
        self.result = result
        super().__init__(
            f"Synthetic QC underfilled: accepted {result.report.accepted_count}/"
            f"{result.report.target_count} after {result.report.attempted_count} attempts"
        )


def _check_rgb_image(image):
    image = np.asarray(image)
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Normal source images must be float32 CHW RGB")
    if min(image.shape[1:]) < 2 or not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
        raise ValueError("Normal source images must be finite, nonempty, and in [0,1]")
    return image


def estimate_object_foreground(image, policy=DEFAULT_QC_POLICY):
    """Estimate the object silhouette from one training-normal image.

    The median RGB value of a 5% border band estimates the background. Pixels
    whose RGB distance exceeds the greater of 0.08 and six robust MAD scales
    are foreground; the largest 8-connected component removes isolated noise.
    """
    image = _check_rgb_image(image)
    height, width = image.shape[1:]
    band = max(1, int(round(min(height, width) * policy.foreground_border_fraction)))
    hwc = np.moveaxis(image, 0, -1)
    border = np.zeros((height, width), dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    background_pixels = hwc[border]
    background = np.median(background_pixels, axis=0)
    border_distance = np.linalg.norm(background_pixels - background, axis=1)
    distance_median = float(np.median(border_distance))
    mad = float(np.median(np.abs(border_distance - distance_median)))
    threshold = max(
        policy.foreground_distance_floor,
        distance_median + policy.foreground_mad_multiplier * 1.4826 * mad,
    )
    foreground = np.linalg.norm(hwc - background, axis=2) > threshold
    components, count = label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        raise ValueError("Could not estimate a nonempty object foreground from normal image")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    foreground = components == int(np.argmax(sizes))
    if not foreground.any():
        raise ValueError("Could not estimate a nonempty object foreground from normal image")
    return foreground


def estimate_foregrounds(normal_images, source_image_ids, policy=DEFAULT_QC_POLICY):
    """Create the shared foreground map keyed by training-normal source ID."""
    images = np.asarray(normal_images)
    if images.ndim != 4 or images.shape[1] != 3 or len(images) == 0:
        raise ValueError("Expected nonempty NCHW RGB training-normal images")
    if len(source_image_ids) != len(images) or any(not isinstance(x, str) or not x for x in source_image_ids):
        raise ValueError("One nonempty source image ID is required per normal image")
    for source_id in source_image_ids:
        path = PurePosixPath(source_id)
        if path.is_absolute() or ".." in path.parts or path.parts[-3:-1] != ("train", "good"):
            raise ValueError("QC source IDs must be relative MVTec train/good paths")
    if len(set(source_image_ids)) != len(source_image_ids):
        raise ValueError("Training-normal source image IDs must be unique")
    return {
        source_id: estimate_object_foreground(image, policy)
        for source_id, image in zip(source_image_ids, images)
    }


def _resize_binary_mask(mask, shape):
    """Nearest-neighbor resize a source foreground mask to the output grid."""
    source_h, source_w = mask.shape
    target_h, target_w = shape
    if not np.isclose(source_h / source_w, target_h / target_w, rtol=0, atol=1e-6):
        raise ValueError("QC source/output aspect ratios must match")
    rows = np.minimum(((np.arange(target_h) + 0.5) * source_h / target_h).astype(int), source_h - 1)
    cols = np.minimum(((np.arange(target_w) + 0.5) * source_w / target_w).astype(int), source_w - 1)
    return mask[np.ix_(rows, cols)]


def filter_generated_batch(batch, foreground_by_source_id, target_count=None, policy=DEFAULT_QC_POLICY):
    """Apply the same mask/object area and object-overlap rules to a batch."""
    attempted = len(batch.images) if isinstance(batch, GeneratedBatch) else -1
    validate_batch(batch, attempted)
    if target_count is not None and (type(target_count) is not int or target_count < 1):
        raise ValueError("target_count must be a positive integer or None")

    started = time.perf_counter()
    accepted, rejected = [], []
    rejection_counts = {"empty_mask": 0, "mask_too_small": 0, "mask_too_large": 0,
                        "low_foreground_overlap": 0}
    foreground_cache = {}
    height, width = batch.images.shape[2:]
    for index, (mask, metadata) in enumerate(zip(batch.masks[:, 0], batch.metadata)):
        source_id = metadata["source_image_id"]
        if source_id not in foreground_by_source_id:
            raise ValueError(f"No training-normal foreground for source_image_id={source_id!r}")
        if source_id not in foreground_cache:
            object_mask = np.asarray(foreground_by_source_id[source_id])
            valid_dtype = object_mask.dtype == np.bool_ or np.issubdtype(object_mask.dtype, np.number)
            if object_mask.ndim != 2 or not valid_dtype or not np.isfinite(object_mask).all():
                raise ValueError(f"Invalid foreground shape/values for source_image_id={source_id!r}")
            if not np.isin(object_mask, [0, 1, False, True]).all() or not np.any(object_mask):
                raise ValueError(f"Foreground must be a nonempty binary mask for source_image_id={source_id!r}")
            if object_mask.shape != (height, width):
                object_mask = _resize_binary_mask(object_mask.astype(bool, copy=False), (height, width))
            if not object_mask.any():
                raise ValueError(f"Foreground became empty on output grid for source_image_id={source_id!r}")
            foreground_cache[source_id] = object_mask.astype(bool, copy=False)
        object_mask = foreground_cache[source_id]
        defect_mask = mask.astype(bool, copy=False)
        area = int(defect_mask.sum())
        reasons = []
        if area == 0:
            reasons.append("empty_mask")
        else:
            object_area = int(object_mask.sum())
            area_ratio = area / object_area
            overlap = int(np.logical_and(defect_mask, object_mask).sum()) / area
            if area_ratio < policy.min_mask_object_area:
                reasons.append("mask_too_small")
            if area_ratio > policy.max_mask_object_area:
                reasons.append("mask_too_large")
            if overlap < policy.min_mask_foreground_overlap:
                reasons.append("low_foreground_overlap")
        if reasons:
            for reason in reasons:
                rejection_counts[reason] += 1
            rejected.append({"index": index, "source_image_id": source_id, "reasons": reasons})
        else:
            accepted.append(index)

    accepted_images = batch.images[accepted] if accepted else np.empty((0, *batch.images.shape[1:]), dtype=np.float32)
    accepted_masks = batch.masks[accepted] if accepted else np.empty((0, *batch.masks.shape[1:]), dtype=np.float32)
    accepted_metadata = [batch.metadata[i] for i in accepted]
    output = GeneratedBatch(accepted_images, accepted_masks, accepted_metadata)
    validate_batch(output, len(accepted))
    report = QCReport(
        policy_version=policy.version,
        target_count=target_count,
        attempted_count=attempted,
        accepted_count=len(accepted),
        rejection_counts=rejection_counts,
        rejected_samples=rejected,
        qc_seconds=time.perf_counter() - started,
    )
    return QCResult(output, report)


def generate_with_qc(generator, normal_images, source_image_ids, target_count,
                     defect_type=None, seed=42, **kwargs):
    """Generate up to target_count accepted samples with the shared 10x cap."""
    category = kwargs.get("category")
    if not isinstance(defect_type, str) or not defect_type:
        raise ValueError("Shared-QC generation requires one explicit defect_type")
    if category is None and source_image_ids:
        categories = {PurePosixPath(source_id).parts[0] for source_id in source_image_ids
                      if isinstance(source_id, str) and source_id}
        if len(categories) == 1:
            category = categories.pop()
    if not isinstance(category, str) or not category:
        raise ValueError("Shared-QC generation requires a category or category-scoped source IDs")
    require_in_scope(category, defect_type)
    if type(target_count) is not int or target_count < 1:
        raise ValueError("target_count must be a positive integer")
    images = np.asarray(normal_images)
    if images.ndim != 4 or images.shape[1] != 3 or len(images) == 0:
        raise ValueError("Expected nonempty NCHW RGB training-normal images")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    policy = DEFAULT_QC_POLICY
    qc_started = time.perf_counter()
    foregrounds = estimate_foregrounds(images, source_image_ids, policy)
    foreground_seconds = time.perf_counter() - qc_started
    attempt_budget = target_count * policy.max_attempts_per_target
    attempted = accepted_count = batch_number = 0
    accepted_images, accepted_masks, accepted_metadata = [], [], []
    rejection_counts = {"empty_mask": 0, "mask_too_small": 0, "mask_too_large": 0,
                        "low_foreground_overlap": 0}
    rejected_samples = []
    generation_seconds = 0.0
    qc_seconds = foreground_seconds
    generation_kwargs = dict(kwargs)
    generation_kwargs["category"] = category
    while accepted_count < target_count and attempted < attempt_budget:
        request = min(target_count - accepted_count, attempt_budget - attempted)
        generated = generator.generate(
            images, request, defect_type=defect_type, seed=seed + batch_number,
            source_image_ids=source_image_ids, **generation_kwargs,
        )
        validate_batch(generated, request)
        result = filter_generated_batch(generated, foregrounds, policy=policy)
        remaining = target_count - accepted_count
        take = min(remaining, result.report.accepted_count)
        if take:
            accepted_images.append(result.batch.images[:take])
            accepted_masks.append(result.batch.masks[:take])
            accepted_metadata.extend(result.batch.metadata[:take])
        attempted += request
        accepted_count += take
        batch_number += 1
        generation_seconds += sum(float(row["generation_seconds"]) for row in generated.metadata)
        qc_seconds += result.report.qc_seconds
        for reason, count in result.report.rejection_counts.items():
            rejection_counts[reason] += count
        rejected_samples.extend(
            {**sample, "attempt_index": attempted - request + sample["index"]}
            for sample in result.report.rejected_samples
        )

    if accepted_images:
        output_images = np.concatenate(accepted_images, axis=0)
        output_masks = np.concatenate(accepted_masks, axis=0)
    else:
        output_images = np.empty((0, *images.shape[1:]), dtype=np.float32)
        output_masks = np.empty((0, 1, *images.shape[2:]), dtype=np.float32)
    output = GeneratedBatch(output_images, output_masks, accepted_metadata)
    validate_batch(output, accepted_count)
    report = QCReport(
        policy_version=policy.version,
        target_count=target_count,
        attempted_count=attempted,
        accepted_count=accepted_count,
        rejection_counts=rejection_counts,
        rejected_samples=rejected_samples,
        generation_seconds=generation_seconds,
        qc_seconds=qc_seconds,
    )
    return QCResult(output, report)
