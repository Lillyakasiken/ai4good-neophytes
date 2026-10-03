# Tile explorer

Local viewer for the drone tiles the baseline trains on (U-Net + MiT-b2, fold cv3 by default). It does not use the training conda env. Everything below runs from this folder, in `explorer/.venv`.

The catalog is one Parquet row per tile, queried with DuckDB. Thumbnails are 256px WebP files, so the grid never opens a full GeoTIFF.

## Setup

```bash
cd explorer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build the catalog and thumbnails

```bash
python build_catalog.py
python build_thumbnails.py
python build_grid.py
```

`build_catalog.py` reads `../data/Neophytes/stats_imagewise.csv` and the `split_cv*.yaml` files. A tile's CV role is the role that file assigns to its folder, which is not the same thing as the on-disk `train` / `val` / `test` folder. For cv3, training uses the train and test folders of sites that are not held out; the test set is the train folders of Dornach_1, Rivera_1, Basel_2, and Waltenschwil_1.

`build_thumbnails.py` is resumable. It skips a tile that already has a thumbnail, a mask, and a row in `cache/raster_meta.parquet`. `--limit 20` processes a short trial run. The full set is about 42,000 tiles and takes a while.

`build_grid.py` reads each GeoTIFF header and writes tile origins. The map view uses that file to stitch one flight.

## Run

From `explorer/`, after the venv and `web/node_modules` exist (`npm install` in `web/`):

```bash
./run.sh
```

That starts the API on port 8000 and the Vite dev server on port 5173, and stops both on Ctrl+C. Open http://127.0.0.1:5173 . The dev server proxies `/api`, `/thumbs`, and `/masks` to the API.

The page opens on cv3 train, sorted by labelled area, with the class mask drawn over each thumbnail. Map stitches one flight; it needs `cache/grid.parquet` from `build_grid.py`.
