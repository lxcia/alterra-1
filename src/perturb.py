"""Perturbations. Volume-level ones act on RAS+ float arrays (x, y, z) BEFORE the reference
preprocessing; stack-level ones act on a prepared VolumeInput (list of shown slices).

Every function returns a new object; nothing is modified in place.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage

from pipeline import VolumeInput


# ----------------------------------------------------------------- volume level (RAS+)
def lr_flip(vol: np.ndarray) -> np.ndarray:
    """Mirror patient left<->right (RAS+ axis 0). Displayed image is mirrored horizontally."""
    return vol[::-1, :, :].copy()


def ap_flip(vol: np.ndarray) -> np.ndarray:
    """Mirror anterior<->posterior (a non-lateral flip; should not change side answers)."""
    return vol[:, ::-1, :].copy()


def invert(vol: np.ndarray) -> np.ndarray:
    """Intensity inversion (bright <-> dark)."""
    return vol.max() - vol


def gamma(vol: np.ndarray, g: float) -> np.ndarray:
    v = vol - vol.min()
    v = v / max(v.max(), 1e-6)
    return v ** g


def blank_like(vol: np.ndarray) -> np.ndarray:
    return np.zeros_like(vol)


def noise_like(vol: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(vol.shape).astype(np.float32)


def crop_fov(vol: np.ndarray, frac: float) -> np.ndarray:
    """Keep the central `frac` of the in-plane field of view (zoom in)."""
    nx, ny = vol.shape[:2]
    cx, cy = int(nx * (1 - frac) / 2), int(ny * (1 - frac) / 2)
    return vol[cx:nx - cx, cy:ny - cy, :].copy()


def rotate_inplane(vol: np.ndarray, deg: float) -> np.ndarray:
    return ndimage.rotate(vol, deg, axes=(0, 1), reshape=False, order=1, mode="constant",
                          cval=float(vol.min()))


def resample_to(src: nib.Nifti1Image, ref: nib.Nifti1Image, order: int = 1) -> np.ndarray:
    """Resample src into ref's world grid (both already RAS+). Used to co-display sequences."""
    return np.asarray(resample_from_to(src, ref, order=order).dataobj, dtype=np.float32)


# ----------------------------------------------------------------- stack level
def _with(vin: VolumeInput, slices, z, **tags) -> VolumeInput:
    return VolumeInput(list(slices), list(z), vin.seq, vin.subject, {**vin.tags, **tags})


def reverse_order(vin: VolumeInput) -> VolumeInput:
    return _with(vin, vin.slices[::-1], vin.z_indices[::-1], order="reversed")


def shuffle_order(vin: VolumeInput, seed: int = 0) -> VolumeInput:
    p = np.random.default_rng(seed).permutation(len(vin.slices))
    return _with(vin, [vin.slices[i] for i in p], [vin.z_indices[i] for i in p], order=f"shuffled{seed}")


def keep_z(vin: VolumeInput, keep: set[int]) -> VolumeInput:
    """Keep only shown slices whose RAS z index is in `keep` (order preserved)."""
    idx = [i for i, z in enumerate(vin.z_indices) if z in keep]
    return _with(vin, [vin.slices[i] for i in idx], [vin.z_indices[i] for i in idx])


def stride(vin: VolumeInput, step: int, offset: int) -> VolumeInput:
    idx = list(range(offset, len(vin.slices), step))
    return _with(vin, [vin.slices[i] for i in idx], [vin.z_indices[i] for i in idx],
                 stride=step, offset=offset)


def single_slice(vin: VolumeInput, z: int) -> VolumeInput:
    i = vin.z_indices.index(z)
    return _with(vin, [vin.slices[i]], [z], single=z)


def set_lesion(vol: np.ndarray, mask: np.ndarray, ratio: float | None):
    """SYNTHETIC contrast edit inside a lesion mask (used by expA2).

    Scales lesion intensities so the lesion mean equals `ratio` x the mean of a 4-voxel ring
    around it; ratio=None fills the lesion with the ring mean (crude erasure). Applied through
    a softly blurred mask (sigma 1 voxel) to avoid hard edges.
    Returns (edited volume, original lesion/ring ratio)."""
    ring = ndimage.binary_dilation(mask, iterations=4) & ~mask
    ring_mean, les_mean = vol[ring].mean(), vol[mask].mean()
    soft = np.clip(ndimage.gaussian_filter(mask.astype(np.float32), 1.0), 0, 1)
    if ratio is None:
        target = np.full_like(vol, ring_mean)
    else:
        target = vol * (ratio * ring_mean / max(les_mean, 1e-6))
    return vol * (1 - soft) + target * soft, float(les_mean / ring_mean)


def sphere_mask(img: nib.Nifti1Image, shape, center_world: np.ndarray, radius_mm: float) -> np.ndarray:
    """Boolean mask of voxels whose centre lies within radius_mm (world) of center_world."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    xyz = nib.affines.apply_affine(img.affine, ijk).reshape(*shape, 3)
    return np.linalg.norm(xyz - center_world, axis=-1) <= radius_mm


def insert_sphere(img: nib.Nifti1Image, vol: np.ndarray, center_world: np.ndarray, radius_mm: float,
                  value: float, sigma_mm: float | None = None) -> np.ndarray:
    """SYNTHETIC lesion: blend a sphere (world-mm radius) centred at center_world (RAS+ mm) to
    intensity `value`, through a softly blurred mask.

    sigma_mm=None: blur sigma 1 VOXEL on every axis (the pre-patch behaviour, used by expC2, expC3
    and expP; on 6 mm slices this is a 6 mm blur in z, so spheres of <=10 mm radius never reach
    `value`). sigma_mm=s: blur sigma s mm on every axis (used by expC4 after the patch, P08)."""
    sphere = sphere_mask(img, vol.shape, center_world, radius_mm).astype(np.float32)
    sigma = 1.0 if sigma_mm is None else [sigma_mm / z for z in img.header.get_zooms()[:3]]
    soft = np.clip(ndimage.gaussian_filter(sphere, sigma), 0, 1)
    return vol * (1 - soft) + value * soft


def pelvis_blanked(vol: np.ndarray) -> np.ndarray:
    """SYNTHETIC: keep the body outline but erase the pelvic contents. An ellipse covering the
    central 70% of the in-plane FOV is filled, on every slice, with the median body intensity
    (median of voxels above the volume's 50th percentile). Used by expB4 / figure 2."""
    nx, ny = vol.shape[:2]
    xx, yy = np.mgrid[:nx, :ny]
    ell = ((xx - nx / 2) / (0.35 * nx)) ** 2 + ((yy - ny / 2) / (0.35 * ny)) ** 2 <= 1
    body = vol[vol > np.percentile(vol, 50)]
    out = vol.copy()
    out[ell, :] = float(np.median(body))
    return out
