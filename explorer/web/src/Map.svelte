<script>
  import { untrack } from 'svelte';

  let { facets, filters, showLabels = $bindable(), onopen } = $props();

  const OVERVIEW_MAX = 72;

  let site = $state('');
  let flight = $state('');
  let mosaic = $state(null);
  let loading = $state(false);
  let imagesStarted = $state(false);
  let retained = $state([]);
  let readySheets = $state({});
  let error = $state('');
  let cell = $state(32);
  let panX = $state(0);
  let panY = $state(0);
  let viewportW = $state(0);
  let viewportH = $state(0);
  let viewport = $state(null);
  let world = $state(null);
  let drag = $state(null);
  let hover = $state(null);
  let imageOpacity = $state(1);

  const IGNORE = [
    { name: 'unlabelled', color: '#737373', binomial: false },
    { name: 'uncertain', color: '#facc15', binomial: false },
  ];
  const samples = new WeakMap();

  function rgbOf(hex) {
    const n = Number.parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  let sites = $derived([...new Set((facets?.flights ?? []).map((row) => row.site_name))]);
  let flightsForSite = $derived(
    (facets?.flights ?? []).filter((row) => row.site_name === site),
  );
  let byCell = $derived(
    new Map((mosaic?.tiles ?? []).map((tile) => [`${tile.col},${tile.row}`, tile])),
  );
  let palette = $derived([
    ...(facets?.species ?? []).map((spec) => ({
      name: spec.name,
      color: spec.color,
      rgb: rgbOf(spec.color),
      binomial: true,
    })),
    ...IGNORE.map((item) => ({ ...item, rgb: rgbOf(item.color) })),
  ]);
  let overview = $derived(cell < OVERVIEW_MAX);
  let chunks = $derived.by(() => {
    if (!mosaic || overview || !viewportW) return [];
    const sheet = mosaic.sheet || 8;
    const thumb = mosaic.thumb_px || 256;
    const pad = mosaic.sheet_pad || 0;
    const c0 = Math.max(0, Math.floor(-panX / cell) - 1);
    const r0 = Math.max(0, Math.floor(-panY / cell) - 1);
    const c1 = Math.min(mosaic.ncols - 1, Math.ceil((viewportW - panX) / cell) + 1);
    const r1 = Math.min(mosaic.nrows - 1, Math.ceil((viewportH - panY) / cell) + 1);
    const occupied = new Set();
    for (const tile of mosaic.tiles) {
      occupied.add(`${Math.floor(tile.col / sheet)},${Math.floor(tile.row / sheet)}`);
    }
    const out = [];
    for (let cr = Math.floor(r0 / sheet); cr <= Math.floor(r1 / sheet); cr++) {
      for (let cc = Math.floor(c0 / sheet); cc <= Math.floor(c1 / sheet); cc++) {
        if (!occupied.has(`${cc},${cr}`)) continue;
        const col = cc * sheet;
        const row = cr * sheet;
        const cols = Math.min(sheet, mosaic.ncols - col);
        const rows = Math.min(sheet, mosaic.nrows - row);
        const photo = rows * thumb;
        out.push({
          key: `${cc},${cr}`,
          cc,
          cr,
          col,
          row,
          cols,
          rows,
          heightPct: ((photo * 2 + pad) / photo) * 100,
          maskTopPct: -((photo + pad) / photo) * 100,
        });
      }
    }
    return out;
  });
  let visibleKeys = $derived(new Set(chunks.map((chunk) => chunk.key)));
  let detailPending = $derived(
    !overview && chunks.length > 0 && chunks.every((chunk) => !readySheets[chunk.key]),
  );

  let fittedFor = '';

  $effect(() => {
    const flights = facets?.flights ?? [];
    const wantedSite = filters.site;
    const wantedFlight = filters.flight;
    if (!flights.length) return;
    untrack(() => {
      const match =
        wantedSite &&
        wantedFlight &&
        flights.some(
          (row) => row.site_name === wantedSite && String(row.flight) === String(wantedFlight),
        );
      if (match) {
        site = wantedSite;
        flight = String(wantedFlight);
        return;
      }
      if (site && flight) return;
      const first =
        (wantedSite && flights.find((row) => row.site_name === wantedSite)) || flights[0];
      site = first.site_name;
      flight = String(first.flight);
    });
  });

  $effect(() => {
    if (!site || flight === '') return;
    const siteName = site;
    const flightName = flight;
    const fold = filters.fold;
    let dead = false;
    loading = true;
    imagesStarted = false;
    retained = [];
    readySheets = {};
    error = '';
    mosaic = null;
    fetch(
      `/api/mosaic?site=${encodeURIComponent(siteName)}&flight=${encodeURIComponent(flightName)}&fold=${encodeURIComponent(fold)}`,
    )
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.text()) || res.statusText);
        return res.json();
      })
      .then((data) => {
        if (dead) return;
        mosaic = data;
        if (!data?.tiles?.length) imagesStarted = true;
      })
      .catch((err) => {
        if (!dead) {
          error = String(err.message || err);
          imagesStarted = true;
        }
      })
      .finally(() => {
        if (!dead) loading = false;
      });
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    const node = viewport;
    if (!node) return;
    const observer = new ResizeObserver(() => {
      viewportW = node.clientWidth;
      viewportH = node.clientHeight;
    });
    observer.observe(node);
    viewportW = node.clientWidth;
    viewportH = node.clientHeight;

    function onWheel(event) {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      const rect = node.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const next = Math.max(8, Math.min(256, cell * factor));
      const scale = next / cell;
      panX = px - (px - panX) * scale;
      panY = py - (py - panY) * scale;
      applyCell(next);
    }
    node.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      observer.disconnect();
      node.removeEventListener('wheel', onWheel);
    };
  });

  $effect(() => {
    const data = mosaic;
    const width = viewportW;
    const height = viewportH;
    if (!data || !width || !height) return;
    const key = `${data.site}/${data.flight}`;
    if (key === fittedFor) return;
    fittedFor = key;
    const next = Math.max(8, Math.min(256, Math.min(width / data.ncols, height / data.nrows)));
    applyCell(next);
    panX = (width - data.ncols * next) / 2;
    panY = (height - data.nrows * next) / 2;
  });

  $effect(() => {
    const incoming = chunks;
    if (!incoming.length) return;
    untrack(() => {
      const have = new Set(retained.map((chunk) => chunk.key));
      const add = incoming.filter((chunk) => !have.has(chunk.key));
      if (add.length) retained = [...retained, ...add];
    });
  });

  function onImageStart() {
    imagesStarted = true;
  }

  function noteImage(node) {
    if (node.complete) onImageStart();
  }

  function applyCell(next) {
    cell = next;
  }

  function watchTile(node, key) {
    let current = key;
    const done = () => {
      imagesStarted = true;
      if (readySheets[current]) return;
      readySheets = { ...readySheets, [current]: true };
    };
    if (node.complete) done();
    node.addEventListener('load', done);
    node.addEventListener('error', done);
    return {
      update(nextKey) {
        current = nextKey;
        if (node.complete) done();
      },
      destroy() {
        node.removeEventListener('load', done);
        node.removeEventListener('error', done);
      },
    };
  }

  function chooseSite(value) {
    site = value;
    const rows = (facets?.flights ?? []).filter((row) => row.site_name === value);
    if (!rows.some((row) => String(row.flight) === flight)) {
      flight = rows[0] ? String(rows[0].flight) : '';
    }
  }

  function fit() {
    if (!mosaic || !viewportW || !viewportH) return;
    fittedFor = '';
    const next = Math.max(
      8,
      Math.min(256, Math.min(viewportW / mosaic.ncols, viewportH / mosaic.nrows)),
    );
    applyCell(next);
    panX = (viewportW - mosaic.ncols * next) / 2;
    panY = (viewportH - mosaic.nrows * next) / 2;
    fittedFor = `${mosaic.site}/${mosaic.flight}`;
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    drag = { x: event.clientX, y: event.clientY, panX, panY, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function nearestLabel(r, g, b, a) {
    if (a < 140) return null;
    let best = null;
    let bestD = 55 * 55;
    for (const item of palette) {
      const d =
        (r - item.rgb[0]) ** 2 + (g - item.rgb[1]) ** 2 + (b - item.rgb[2]) ** 2;
      if (d < bestD) {
        bestD = d;
        best = item;
      }
    }
    return best;
  }

  function sampleImage(img, u, v) {
    if (!img || !img.complete || !img.naturalWidth) return null;
    let entry = samples.get(img);
    if (!entry) {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      try {
        ctx.drawImage(img, 0, 0);
        ctx.getImageData(0, 0, 1, 1);
      } catch {
        samples.set(img, { failed: true });
        return null;
      }
      entry = { ctx, width: canvas.width, height: canvas.height };
      samples.set(img, entry);
    }
    if (entry.failed) return null;
    const x = Math.min(entry.width - 1, Math.max(0, Math.floor(u * entry.width)));
    const y = Math.min(entry.height - 1, Math.max(0, Math.floor(v * entry.height)));
    const px = entry.ctx.getImageData(x, y, 1, 1).data;
    return nearestLabel(px[0], px[1], px[2], px[3]);
  }

  function labelAt(event) {
    if (!showLabels || !mosaic || !world || !viewport) return null;
    const rect = world.getBoundingClientRect();
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = (event.clientY - rect.top) / rect.height;
    if (fx < 0 || fy < 0 || fx >= 1 || fy >= 1) return null;
    let img;
    let u;
    let v;
    if (overview) {
      img = world.querySelector('.overview.mask');
      u = fx;
      v = fy;
    } else {
      const sheet = mosaic.sheet || 8;
      const thumb = mosaic.thumb_px || 256;
      const pad = mosaic.sheet_pad || 0;
      const col = Math.min(mosaic.ncols - 1, Math.floor(fx * mosaic.ncols));
      const row = Math.min(mosaic.nrows - 1, Math.floor(fy * mosaic.nrows));
      const cc = Math.floor(col / sheet);
      const cr = Math.floor(row / sheet);
      img = world.querySelector(`.chunk[data-cc="${cc}"][data-cr="${cr}"] img.mask`);
      const cols = Math.min(sheet, mosaic.ncols - cc * sheet);
      const rows = Math.min(sheet, mosaic.nrows - cr * sheet);
      const photo = rows * thumb;
      u = (fx * mosaic.ncols - cc * sheet) / cols;
      v = (photo + pad + (fy * mosaic.nrows - cr * sheet) * thumb) / (photo * 2 + pad);
    }
    const hit = sampleImage(img, u, v);
    if (!hit) return null;
    const vp = viewport.getBoundingClientRect();
    let x = event.clientX - vp.left + 14;
    let y = event.clientY - vp.top + 16;
    if (x > vp.width - 180) x = event.clientX - vp.left - 170;
    if (y > vp.height - 32) y = event.clientY - vp.top - 28;
    return { name: hit.name, color: hit.color, binomial: hit.binomial, x, y };
  }

  function onPointerMove(event) {
    if (drag) {
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
      panX = drag.panX + dx;
      panY = drag.panY + dy;
      hover = null;
      return;
    }
    hover = labelAt(event);
  }

  function onPointerUp(event) {
    if (!drag || !mosaic || !world) {
      drag = null;
      return;
    }
    const moved = drag.moved;
    drag = null;
    if (moved) return;
    const rect = world.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left) / (rect.width / mosaic.ncols));
    const row = Math.floor((event.clientY - rect.top) / (rect.height / mosaic.nrows));
    const tile = byCell.get(`${col},${row}`);
    if (tile) onopen(tile.tile_id);
  }
</script>

<div class="map">
  <div class="map-bar">
    <label class="field">
      Site
      <select value={site} onchange={(event) => chooseSite(event.currentTarget.value)}>
        {#each sites as name (name)}
          <option value={name}>{name}</option>
        {/each}
      </select>
    </label>
    <label class="field">
      Flight
      <select value={flight} onchange={(event) => (flight = event.currentTarget.value)}>
        {#each flightsForSite as row (`${row.site_name}-${row.flight}`)}
          <option value={row.flight}>{row.flight}</option>
        {/each}
      </select>
    </label>
    <div class="map-actions">
      <button type="button" class="fit" onclick={fit}>Fit</button>
      {#if error}
        <p class="map-meta">{error}</p>
      {:else if mosaic}
        <p class="map-meta">
          {mosaic.acq_date || 'undated'}
          · {mosaic.tiles.length.toLocaleString()} tiles
          · {mosaic.ncols}×{mosaic.nrows}
        </p>
      {:else if loading}
        <p class="map-meta">Placing tiles…</p>
      {/if}
      <label class="map-check">
        <input type="checkbox" bind:checked={showLabels} />
        Show labels
      </label>
      <label class="map-slider">
        Image
        <input type="range" min="0" max="1" step="0.05" bind:value={imageOpacity} />
      </label>
      <p class="map-hint">Whole flight, every folder. Drag to move, scroll to zoom, click a tile.</p>
    </div>
  </div>

  <div
    class="viewport"
    role="application"
    aria-label="Flight map. Drag to move, scroll to zoom, click a tile."
    class:dragging={drag}
    class:reading={hover && !drag}
    bind:this={viewport}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointerleave={() => (hover = null)}
    onpointercancel={() => {
      drag = null;
      hover = null;
    }}
  >
    {#if mosaic}
      <div
        class="world"
        bind:this={world}
        style:width="{mosaic.ncols * cell}px"
        style:height="{mosaic.nrows * cell}px"
        style:transform="translate({panX}px, {panY}px)"
      >
        <img
          class="overview"
          class:parked={!overview}
          src="/api/mosaic/{encodeURIComponent(mosaic.site)}/{encodeURIComponent(mosaic.flight)}/overview.webp"
          alt=""
          draggable="false"
          style:opacity={imageOpacity}
          use:noteImage
          onload={onImageStart}
          onerror={onImageStart}
        />
        {#if showLabels}
          <img
            class="overview mask"
            class:parked={!overview}
            src="/api/mosaic/{encodeURIComponent(mosaic.site)}/{encodeURIComponent(mosaic.flight)}/overview-mask.webp"
            alt=""
            draggable="false"
          />
        {/if}
        {#each retained as chunk (chunk.key)}
          <div
            class="chunk"
            class:parked={overview || !visibleKeys.has(chunk.key)}
            data-cc={chunk.cc}
            data-cr={chunk.cr}
            style:left="{chunk.col * cell}px"
            style:top="{chunk.row * cell}px"
            style:width="{chunk.cols * cell}px"
            style:height="{chunk.rows * cell}px"
          >
            <img
              class="photo"
              src="/api/mosaic/{encodeURIComponent(mosaic.site)}/{encodeURIComponent(mosaic.flight)}/sheet/{chunk.cc}/{chunk.cr}/image.webp"
              alt=""
              draggable="false"
              style:opacity={imageOpacity}
              style:height="{chunk.heightPct}%"
              use:watchTile={chunk.key}
            />
            {#if showLabels}
              <img
                class="mask"
                src="/api/mosaic/{encodeURIComponent(mosaic.site)}/{encodeURIComponent(mosaic.flight)}/sheet/{chunk.cc}/{chunk.cr}/image.webp"
                alt=""
                draggable="false"
                style:height="{chunk.heightPct}%"
                style:top="{chunk.maskTopPct}%"
              />
            {/if}
          </div>
        {/each}
      </div>
    {/if}
    {#if (!imagesStarted || detailPending) && !error}
      <div class="map-loading" role="status" aria-live="polite">
        <span class="map-spin" aria-hidden="true"></span>
        {imagesStarted ? 'Loading tiles' : 'Loading flight'}
      </div>
    {/if}
    {#if hover}
      <p class="map-tip" style:left="{hover.x}px" style:top="{hover.y}px">
        <i class="swatch" style:--swatch={hover.color}></i>
        {#if hover.binomial}<span class="binomial">{hover.name}</span>{:else}{hover.name}{/if}
      </p>
    {/if}
  </div>
</div>
