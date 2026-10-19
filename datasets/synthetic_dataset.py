"""Synthetic training samples: real plants pasted onto background windows of other sites.

Each sample is built on the fly, so every draw is a new composite. The draw is
balanced, unlike the real data where most species sit at one or two sites:

    1. species        uniform over the six
    2. source site    uniform over the training sites that have this species, then a
                      random instance there (plant pool of build_plant_pool.py)
    3. background     site uniform over the training sites, then type uniform over the
                      types present at that site, then a random usable window of that site
                      and type (pool_<fold>_typed.csv)
    4. 1 to n plants  each repeats 1-2; the source site is drawn among the species' sites
                      except the background site, so no plant lands on its own site
    5. compositing    synthesis.synthesize (soft paste by default, no ignore ring); the
                      plant is scaled to the background GSD (x jitter), rotated by a
                      multiple of 90 degrees and flipped, as preview_paste.pick_instance

The label follows the real masks: background mask as the dataset reads it (254/255 ->
ignore), the pasted plant pixels get their class. The train augmentations then run
without the crop step, since a background window already has the crop size.
"""

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from scipy import ndimage
from torch.utils.data import Dataset

from datasets.neophyte_dataset import _tile_month
from preview_paste import place
from synthesis import synthesize
from utils.utils import extract_dataset_name


class SyntheticDataset(Dataset):
    """`length` virtual samples (the index is ignored; each call draws a new composite)."""

    def __init__(self, plants_csv, backgrounds_csv, transform, length, value_mapping=None, size=512,
                 n_plants=(1, 3), method="soft", sigma=2.0, ring=0, max_size_frac=0.6, min_bg_frac=0.8,
                 scale_jitter=(0.9, 1.1), max_tries=20):
        self.transform, self.length, self.size = transform, length, size
        self.value_mapping = value_mapping
        self.n_plants, self.method, self.sigma, self.ring = tuple(n_plants), method, sigma, ring
        self.max_side = max_size_frac * size
        self.min_bg_frac, self.scale_jitter, self.max_tries = min_bg_frac, tuple(scale_jitter), max_tries
        self.rng = None  # per worker, created on first use

        plants = pd.read_csv(plants_csv)
        # # species -> source site -> instance rows
        self.plants = {int(c): {s: g.reset_index(drop=True) for s, g in gc.groupby("site")}
                       for c, gc in plants.groupby("cls")}
        bg = pd.read_csv(backgrounds_csv)
        bg = bg[bg["usable"] == 1]
        # # background site -> type -> window rows
        self.backgrounds = {s: {t: g.reset_index(drop=True) for t, g in gs.groupby("type")}
                            for s, gs in bg.groupby("site")}
        print(f"SyntheticDataset: {len(plants)} plant instances of {len(self.plants)} species, "
              f"{len(bg)} background windows of {len(self.backgrounds)} sites")

    def __len__(self):
        return self.length

    # %% reading
    def _read(self, path, window, mask=False):
        with rasterio.open(path) as src:
            if mask:
                return src.read(1, window=window).astype(np.int64)
            img = src.read([1, 2, 3], window=window)
            return np.ascontiguousarray(img.transpose(1, 2, 0)), abs(src.transform.a) * 1000.0

    def _map_mask(self, mask, path):
        """Same value mapping as NeophyteDataset (254/255 -> ignore)."""
        if self.value_mapping:
            mapping = self.value_mapping.get(extract_dataset_name(path, self.value_mapping.keys()))
            for old, new in (mapping or {}).items():
                mask[mask == int(old)] = new
        return mask

    # %% drawing
    def _draw_plant(self, exclude_site, bg_gsd):
        """A random instance of a uniform species from a uniform source site != exclude_site,
        scaled to bg_gsd and rotated/flipped. Returns (cls, patch, alpha, source site) or None."""
        cls = int(self.rng.choice(list(self.plants)))
        sites = [s for s in self.plants[cls] if s != exclude_site]
        if not sites:
            return None
        site = sites[self.rng.integers(len(sites))]
        r = self.plants[cls][site].iloc[self.rng.integers(len(self.plants[cls][site]))]
        scale = r["gsd_mm"] / bg_gsd * self.rng.uniform(*self.scale_jitter)
        h, w = r["y1"] - r["y0"], r["x1"] - r["x0"]
        if max(h, w) * scale > self.max_side:
            return None
        win = Window(int(r["x0"]), int(r["y0"]), int(w), int(h))
        image, _ = self._read(r["img_path"], win)
        mask = self._map_mask(self._read(r["mask_path"], win, mask=True), r["mask_path"])
        comp, _ = ndimage.label(mask == cls)
        region = comp == comp[int(r["seed_y"] - r["y0"]), int(r["seed_x"] - r["x0"])]
        new_wh = (max(1, round(w * scale)), max(1, round(h * scale)))
        patch = cv2.resize(image, new_wh, interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(region.astype(np.float32), new_wh, interpolation=cv2.INTER_LINEAR) > 0.5
        k = int(self.rng.integers(4))
        patch, alpha = np.rot90(patch, k), np.rot90(alpha, k)
        if self.rng.random() < 0.5:
            patch, alpha = patch[:, ::-1], alpha[:, ::-1]
        return cls, np.ascontiguousarray(patch), np.ascontiguousarray(alpha), site

    def _draw_background(self):
        """A usable window of a uniform site and a uniform type present there."""
        sites = list(self.backgrounds)
        site = sites[self.rng.integers(len(sites))]
        types = list(self.backgrounds[site])
        rows = self.backgrounds[site][types[self.rng.integers(len(types))]]
        return rows.iloc[self.rng.integers(len(rows))]

    def __getitem__(self, idx):
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % (1 << 32))
        for _ in range(self.max_tries):
            bg = self._draw_background()
            win = Window(int(bg["x0"]), int(bg["y0"]), self.size, self.size)
            image, bg_gsd = self._read(bg["img_path"], win)
            mask = self._map_mask(self._read(bg["mask_path"], win, mask=True), bg["mask_path"])
            placed = 0
            for _ in range(int(self.rng.integers(self.n_plants[0], self.n_plants[1] + 1))):
                plant = None
                for _ in range(self.max_tries):
                    plant = self._draw_plant(bg["site"], bg_gsd)
                    if plant is not None:
                        break
                if plant is None:
                    continue
                cls, patch, alpha, _ = plant
                yx = place(mask, alpha, self.min_bg_frac, self.rng)
                if yx is None:
                    continue
                image, mask = synthesize(image, mask, patch, alpha, cls, *yx, method=self.method,
                                         sigma=self.sigma, ring=self.ring)
                placed += 1
            if placed:
                break

        # %% same augmentations as the real samples (without the crop)
        if self.transform:
            from utils.transform_utils import set_tile_context
            set_tile_context(self.transform, gsd=bg_gsd, month=_tile_month(bg["img_path"]))
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return {"image": image, "mask": mask.long(), "meta": {}, "name": "synthetic:" + bg["img_path"]}
