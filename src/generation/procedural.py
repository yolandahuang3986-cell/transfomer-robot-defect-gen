"""CutPaste-style colored patch baseline; not full CutPaste or DRAEM."""
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .base import DefectGenerator, GeneratedBatch, validate_batch


def estimate_foreground(image, category):
    """Conservative Otsu foreground, using only this normal image.

    This is an object-region heuristic, NOT a defect ground-truth mask.
    Metal nuts have a dark background; screws have a bright background.
    """
    if category not in {"metal_nut", "screw"}:
        raise ValueError("Foreground placement supports metal_nut and screw only")
    gray = np.rint(image.mean(axis=0) * 255).astype(np.uint8)
    if int(gray.max()) - int(gray.min()) < 8:
        raise ValueError("Insufficient contrast to estimate foreground")
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    weight = np.cumsum(hist)
    moment = np.cumsum(hist * np.arange(256))
    denominator = weight * (weight[-1] - weight)
    score = np.zeros(256)
    valid = denominator > 0
    score[valid] = (moment[-1] * weight[valid] - moment[valid] * weight[-1])**2 / denominator[valid]
    threshold = int(np.argmax(score))
    foreground = gray > threshold if category == "metal_nut" else gray <= threshold
    if category == "screw":
        # Development correction after v1 shadow-placement review. A global
        # threshold includes cast shadows; split that dark population again
        # and place only on its darkest material core. No held-out inputs.
        dark = gray[foreground]
        if int(dark.max()) > int(dark.min()):
            core_hist = np.bincount(dark, minlength=256).astype(float)
            core_weight = np.cumsum(core_hist)
            core_moment = np.cumsum(core_hist * np.arange(256))
            core_denom = core_weight * (core_weight[-1] - core_weight)
            core_score = np.full(256, -np.inf)
            valid_core = core_denom > 0
            core_score[valid_core] = (core_moment[-1] * core_weight[valid_core] -
                core_moment[valid_core] * core_weight[-1])**2 / core_denom[valid_core]
            threshold = int(np.argmax(core_score))
            foreground = gray <= threshold
    # Stay away from estimated object edges, and explicitly exclude image borders.
    foreground = np.asarray(Image.fromarray(foreground.astype(np.uint8)*255).filter(ImageFilter.MinFilter(5))) > 0
    foreground = foreground.copy()
    foreground[:2] = foreground[-2:] = False
    foreground[:, :2] = foreground[:, -2:] = False
    if foreground.sum() < 16 or foreground.mean() > .9:
        raise ValueError("Cannot estimate a reliable foreground for this normal image")
    return foreground, threshold


class ProceduralGenerator(DefectGenerator):
    def __init__(self, area_range=(0.005, 0.03), opacity=0.8, placement="foreground"):
        if not 0 < area_range[0] <= area_range[1] <= 0.25:
            raise ValueError("Invalid area range")
        if not 0 < opacity <= 1:
            raise ValueError("Invalid opacity")
        if placement not in {"foreground", "uniform"}:
            raise ValueError("Unknown placement")
        self.area_range, self.opacity, self.placement = tuple(area_range), opacity, placement

    def fit(self, dataset=None, config=None):
        # No learned parameters and no real defect images required.
        return self

    def generate(self, normal_images, num_samples, defect_type=None, seed=42, **kwargs):
        start = time.perf_counter()
        a = np.asarray(normal_images)
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be positive")
        if a.ndim != 4 or a.shape[0] < 1 or a.shape[1] != 3 or min(a.shape[2:]) < 16:
            raise ValueError("Expected nonempty NCHW images with dimensions >=16")
        if a.dtype != np.float32 or not np.isfinite(a).all() or not ((a>=0)&(a<=1)).all():
            raise ValueError("Expected finite float32 images in [0,1]")
        if defect_type not in (None, "synthetic_patch"):
            raise ValueError("This baseline only supports synthetic_patch, not semantic defect types")
        ids = kwargs.get("source_image_ids")
        category = kwargs.get("category")
        if not isinstance(category,str) or not category or ids is None or len(ids)!=len(a):
            raise ValueError("category and aligned source_image_ids are required")
        if any(not isinstance(s,str) or not s for s in ids):
            raise ValueError("Invalid source IDs")
        rng = np.random.default_rng(seed)
        _,_,h,w=a.shape
        if self.placement == "foreground":
            estimates = [estimate_foreground(x, category) for x in a]
        else:
            estimates = [(np.ones((h,w), dtype=bool), None) for x in a]
        # Patch scale and placement support are separate: a conservative material
        # core must not shrink the requested patch area to nearly zero.
        size_reference = None
        if category == "screw" and self.placement == "foreground":
            from .qc import estimate_foregrounds
            size_reference = estimate_foregrounds(a, ids)
        images, masks, metadata=[],[],[]
        for _ in range(num_samples):
            si,di = rng.integers(len(a),size=2)
            source=a[si]
            foreground, threshold = estimates[si]
            donor_foreground, _ = estimates[di]
            area=float(rng.uniform(*self.area_range))
            reference_area = int(size_reference[ids[si]].sum()) if size_reference is not None else int(foreground.sum())
            aspect=float(np.exp(rng.uniform(np.log(.3),np.log(3))))
            ph=max(3,min(h,int(np.sqrt(area*reference_area/aspect))))
            pw=max(3,min(w,int(np.sqrt(area*reference_area*aspect))))
            cy,cx=np.argwhere(foreground)[int(rng.integers(int(foreground.sum())))]
            top,left=int(np.clip(cy-ph//2,0,h-ph)),int(np.clip(cx-pw//2,0,w-pw))
            donor_points=np.argwhere(donor_foreground)
            dy,dx=donor_points[int(rng.integers(len(donor_points)))]
            dt,dl=int(np.clip(dy-ph//2,0,h-ph)),int(np.clip(dx-pw//2,0,w-pw))
            patch=a[di,:,dt:dt+ph,dl:dl+pw].copy()
            tint=rng.uniform(0,1,(3,1,1)).astype(np.float32)
            patch=np.clip(.5*patch+.5*tint,0,1)
            shape=Image.new("L",(pw,ph),0)
            points=[]
            for angle in np.linspace(0,2*np.pi,8,endpoint=False):
                radius=float(rng.uniform(.65,1))
                points.append(((pw-1)/2*(1+radius*np.cos(angle)),(ph-1)/2*(1+radius*np.sin(angle))))
            ImageDraw.Draw(shape).polygon(points,fill=1)
            local=np.asarray(shape,dtype=bool) & foreground[top:top+ph,left:left+pw]
            output=source.copy()
            region=output[:,top:top+ph,left:left+pw]
            mixed=(1-self.opacity)*region+self.opacity*patch
            region[:,local]=mixed[:,local]
            # The output mask labels pixels actually altered, not an enclosing rectangle.
            mask=np.any(output!=source,axis=0).astype(np.float32)[None]
            if not mask.any():
                raise ValueError("Empty generated anomaly")
            images.append(output); masks.append(mask)
            metadata.append(dict(method="procedural",category=category,defect_type="synthetic_patch",
                seed=seed,source_image_id=ids[si],generation_seconds=0.,
                generation_config=dict(algorithm=("screw_dark_core_colored_polygon_v4" if category=="screw" and self.placement=="foreground" else "foreground_colored_polygon_cutpaste_v2"),area_range=list(self.area_range),
                    opacity=self.opacity,donor_image_id=ids[di],target_box=[top,left,ph,pw],
                    donor_box=[dt,dl,ph,pw],placement=self.placement, foreground_threshold=threshold,
                    area_reference=("shared_normal_foreground_pixels" if size_reference is not None else "estimated_foreground_pixels"), area_reference_pixels=reference_area, foreground_pixels=int(foreground.sum()))))
        for m in metadata:m["generation_seconds"]=(time.perf_counter()-start)/num_samples
        return validate_batch(GeneratedBatch(np.stack(images),np.stack(masks),metadata),num_samples)
