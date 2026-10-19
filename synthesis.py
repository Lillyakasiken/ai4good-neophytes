"""Core of the copy-paste synthesis: one plant + one background window -> image + label mask.

The training loader will call synthesize() with whatever the sampling draws (species,
background type, flight, augmentation); this module only does the compositing.

Methods, each a step towards a realistic composite (more follow):
    hard   binary plant mask
    soft   Gaussian-blurred mask edge (sigma in px)

Label: the plant pixels get the class, an ignore ring of `ring` px around the plant
gets IGNORE (labels at plant edges are inexact, and the soft edge mixes plant and
background there), everything else keeps the background mask.
"""

import numpy as np
from scipy import ndimage

IGNORE = -1
METHODS = ("hard", "soft")


def synthesize(background, bg_mask, patch, alpha, cls, y, x, method="soft", sigma=2.0, ring=3):
    """Paste `patch` (h, w, 3 uint8, already at the background GSD) with binary mask `alpha`
    (h, w) at top-left (y, x) of `background` (H, W, 3 uint8) / `bg_mask` (H, W int).

    Returns the composite image and its label mask (copies; inputs are not changed).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}', available: {METHODS}")
    h, w = alpha.shape
    win = (slice(y, y + h), slice(x, x + w))
    image, mask = background.copy(), bg_mask.copy()

    # %% image
    if method == "hard":
        image[win] = np.where(alpha[..., None], patch, image[win])
    else:
        a = np.clip(ndimage.gaussian_filter(alpha.astype(np.float32), sigma), 0, 1)[..., None]
        image[win] = (a * patch + (1 - a) * image[win]).round().astype(np.uint8)

    # %% label: class on the plant, ignore ring around it
    if ring > 0:
        # # the plant bbox is tight, so grow the ring in a window padded by `ring` (clipped to the image)
        y0, x0 = max(y - ring, 0), max(x - ring, 0)
        y1, x1 = min(y + h + ring, mask.shape[0]), min(x + w + ring, mask.shape[1])
        plant = np.zeros((y1 - y0, x1 - x0), bool)
        plant[y - y0:y - y0 + h, x - x0:x - x0 + w] = alpha
        grown = ndimage.binary_dilation(plant, iterations=ring)
        mask[y0:y1, x0:x1][grown & ~plant] = IGNORE
    mask[win][alpha] = cls
    return image, mask
