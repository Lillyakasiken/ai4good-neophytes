# Tile explorer

Local viewer for the drone tiles the baseline trains on (U-Net + MiT-b2, fold cv3 by default). It does not use the training conda env. Everything below runs from this folder, in `explorer/.venv`.

The catalog is one Parquet row per tile, queried with DuckDB. Thumbnails are 256px WebP files, so the grid never opens a full GeoTIFF.

## Setup

```bash
cd explorer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd web && npm install
```

## Run

```bash
cd explorer
./run.sh
```

`run.sh` prepares the cache, then starts the API on port 8000 and the Vite dev server on port 5173. Ctrl+C stops both. Open http://127.0.0.1:5173 . The dev server proxies `/api`, `/thumbs`, and `/masks` to the API.

Prepare runs only what is missing:

- `build_catalog.py` when `cache/tiles.parquet` is absent.
- `build_thumbnails.py` every time. It skips a tile that already has a thumbnail, a mask, and a row in `cache/raster_meta.parquet`.
- `build_grid.py` when `cache/grid.parquet` is absent, or older than the catalog.

The first thumbnail pass is about 42,000 tiles and takes a while. Later starts only check that the cache is complete.

The page opens on cv3 train, sorted by labelled area, with the class mask drawn over each thumbnail. Map stitches one flight from `cache/grid.parquet`.

Open a tile and choose Vision to preview OpenCV filters and a gray-value histogram on that tile. The preview is computed in memory and is not written to disk. It does not change the thumbnails, the catalog, or the training data.

## The prepare scripts

You can run them by hand from `explorer/` with `.venv/bin/python`. `run.sh` calls the same commands.

```bash
python build_catalog.py
python build_thumbnails.py
python build_grid.py
```

`build_catalog.py` reads `../data/Neophytes/stats_imagewise.csv` and the `split_cv*.yaml` files. A tile's CV role is the role that file assigns to its folder, which is not the same thing as the on-disk `train` / `val` / `test` folder. For cv3, training uses the train and test folders of sites that are not held out; the test set is the train folders of Dornach_1, Rivera_1, Basel_2, and Waltenschwil_1.

`build_thumbnails.py` is resumable. `--limit 20` processes a short trial run. `--workers` sets the process count.

`build_grid.py` reads each GeoTIFF header and writes tile origins. The map view uses that file to stitch one flight.
