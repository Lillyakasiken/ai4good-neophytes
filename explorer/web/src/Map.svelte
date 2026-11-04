<script>
  import { tick, untrack } from 'svelte';
  import MapRuns from './MapRuns.svelte';

  let { facets, filters = $bindable(), showLabels = $bindable(), onopen } = $props();

  const OVERVIEW_MAX = 72;
  // Sheet builds are slow and share the browser's connection pool with the
  // tile preview. Keep only a few in flight so a click is not queued behind them.
  const SHEET_SLOTS = 3;

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
  let showGrid = $state(false);
  let showRoleColors = $state(true);
  let foldPinned = $state(false);
  let showRun = $state(false);
  let runName = $state('');
  let runList = $state([]);
  let runsError = $state('');
  let runTiles = $state(null);
  let runTilesLoading = $state(false);
  let runTilesError = $state('');

  const MONTHS = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const ROLES = [
    ['all', 'All roles'],
    ['train', 'Train'],
    ['val', 'Validation'],
    ['test', 'Test'],
    ['unused', 'Unused'],
  ];
  const ROLE_COLOR = {
    train: '#0072B2',
    val: '#E69F00',
    test: '#CC79A7',
    unused: '#1e293b',
  };
  const FOLD_KEY = [
    ['train', 'Train', false],
    ['val', 'Validation', false],
    ['test', 'Test', false],
    ['unused', 'Unused', true],
  ];
  const RUN_KEY = {
    train_labelled: 'Train - labelled',
    train_bg: 'Train - background',
    val: 'Validation',
    unused: 'Not picked',
    test: 'Test',
  };
  const RUN_MARKS = [
    ['train_labelled', 'trained, labelled', '#146c36', false],
    ['train_bg', 'trained, background', '#7dcea0', false],
    ['val', 'validation', '#2b6cb0', false],
    ['unused', 'not used', '#64748b', true],
    ['test', 'test', '#d97706', false],
  ];
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
    (facets?.flights ?? [])
      .filter((row) => row.site_name === site)
      .slice()
      .sort((a, b) => String(a.acq_date ?? '').localeCompare(String(b.acq_date ?? ''))
        || String(a.flight).localeCompare(String(b.flight))),
  );
  let years = $derived(facets?.years ?? []);
  let months = $derived(facets?.months ?? []);
  let flightAt = $derived.by(() => {
    const at = new Map();
    for (const row of flightsForSite) {
      if (row.year == null || row.month == null) continue;
      const key = `${row.year}-${row.month}`;
      if (!at.has(key)) at.set(key, row);
    }
    return at;
  });
  let timeline = $derived(
    years.map((year) => ({
      year,
      quiet: !months.some((month) => flightAt.has(`${year}-${month}`)),
      months: months.map((month) => ({
        month,
        row: flightAt.get(`${year}-${month}`) ?? null,
      })),
    })),
  );
  let heldOut = $derived((facets?.folds?.[filters.fold]?.test_sites ?? []).includes(site));
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
  let gridPath = $derived.by(() => {
    if (!showGrid || !mosaic) return '';
    let path = '';
    for (const tile of mosaic.tiles) {
      const x = tile.col * cell;
      const y = tile.row * cell;
      path += `M${x} ${y}h${cell}v${cell}h${-cell}z`;
    }
    return path;
  });
  function coversFlight(keys, siteName, flightName) {
    if (!keys || !keys.length) return true;
    const flightKey = flightName ? `${siteName}_${flightName}` : siteName;
    return keys.includes(siteName) || keys.includes(flightKey);
  }

  async function errorText(res) {
    const text = await res.text();
    try {
      const body = JSON.parse(text);
      if (body.detail) return String(body.detail);
    } catch {
      /* the body is already plain text */
    }
    return text || res.statusText;
  }

  let placedRun = '';

  function siteNameOf(key) {
    if (sites.includes(key)) return key;
    return sites.find((name) => String(key).startsWith(`${name}_`)) || '';
  }

  function selectRun(run) {
    if (!run) return;
    if (run.exp_name !== runName) {
      runTiles = null;
      runTilesError = '';
      placedRun = '';
    }
    runName = run.exp_name;
    showGrid = true;
    showRun = true;
    foldPinned = false;
    showRoleColors = false;
    const keys = run.train_site || [];
    if (keys.length !== 1) return;
    const target = siteNameOf(keys[0]);
    if (target && target !== site) chooseSite(target);
  }

  function loadRuns() {
    fetch('/api/runs/tile-sets')
      .then(async (res) => {
        if (!res.ok) throw new Error(await errorText(res));
        return res.json();
      })
      .then((data) => {
        runList = data.runs || [];
        runsError = '';
      })
      .catch((err) => {
        runsError = String(err.message || err);
      });
  }

  function setRoleColors(on) {
    showRoleColors = on;
    foldPinned = on;
  }

  function toggleRun(on) {
    showRun = on;
    if (!on) {
      setRoleColors(true);
      return;
    }
    showGrid = true;
    if (!runList.length) loadRuns();
  }

  function runForSite(run, siteName) {
    const keys = run.train_site || [];
    if (!keys.length) return false;
    return keys.includes(siteName) || keys.some((key) => String(key).startsWith(`${siteName}_`));
  }

  let runsInFold = $derived(runList.filter((run) => run.fold === filters.fold));

  function runCount(siteName) {
    return runsInFold.filter((run) => runForSite(run, siteName)).length;
  }

  let mapHelp = $derived(
    'The map opens with the fold colours from Tiles. Click a run to fill that run. Color roles, or Fold in the legend, brings the fold colours back and keeps them while you change dates. Drag to move, scroll to zoom, click a tile.',
  );

  function runRoleMarks() {
    const tileset = runTiles;
    if (!tileset || tileset.exp_name !== runName || !mosaic) return [];
    const labelled = new Set(tileset.train_labelled || []);
    const background = new Set(tileset.train_bg || []);
    const validation = new Set(tileset.val || []);
    const keys = tileset.train_site || [];
    const inRun = coversFlight(keys, mosaic.site, String(mosaic.flight ?? ''));
    const paths = {};
    const counts = {};
    for (const [id] of RUN_MARKS) {
      paths[id] = '';
      counts[id] = 0;
    }
    for (const tile of mosaic.tiles) {
      let id = null;
      if (labelled.has(tile.tile_id)) id = 'train_labelled';
      else if (background.has(tile.tile_id)) id = 'train_bg';
      else if (validation.has(tile.tile_id)) id = 'val';
      else if (tile.cv_role === 'train' && inRun) id = 'unused';
      else if (tile.cv_role === 'test') id = 'test';
      if (!id) continue;
      counts[id] += 1;
      const x = tile.col * cell;
      const y = tile.row * cell;
      paths[id] += `M${x} ${y}h${cell}v${cell}h${-cell}z`;
    }
    const totals = tileset.totals || {};
    const totalOf = {
      train_labelled: totals.train_labelled,
      train_bg: totals.train_bg,
      val: totals.val,
      unused: totals.pool_unused,
    };
    return RUN_MARKS
      .filter(([id]) => counts[id] || (id !== 'test' && totalOf[id]))
      .map(([id, label, color, outline]) => ({
        role: id,
        label,
        path: paths[id],
        color,
        outline,
        count: counts[id],
        countText:
          totalOf[id] == null
            ? counts[id].toLocaleString()
            : `${counts[id].toLocaleString()} (run ${Number(totalOf[id]).toLocaleString()})`,
      }));
  }

  let runMarks = $derived.by(() => {
    if (!showRun || !mosaic || !runName) return [];
    return runRoleMarks();
  });

  let runIdSets = $derived.by(() => {
    const tileset = runTiles;
    if (!showRun || !tileset || tileset.exp_name !== runName) return null;
    return {
      labelled: new Set(tileset.train_labelled || []),
      background: new Set(tileset.train_bg || []),
      val: new Set(tileset.val || []),
    };
  });

  let flightRoles = $derived.by(() => {
    const counts = { train: 0, val: 0, test: 0, unused: 0 };
    if (!mosaic) return counts;
    for (const tile of mosaic.tiles) {
      const role = tile.cv_role;
      if (role in counts) counts[role] += 1;
    }
    return counts;
  });

  function flightUse(flightName) {
    const tileset = runTiles;
    if (!showRun || !runName || !tileset || tileset.exp_name !== runName) return null;
    const row = (tileset.by_flight || []).find(
      (item) => item.site === site && String(item.flight) === String(flightName),
    );
    const trained = (row?.train_labelled || 0) + (row?.train_bg || 0);
    const val = row?.val || 0;
    return { trained, val, touched: trained + val > 0 };
  }

  let roleMarks = $derived.by(() => {
    if (!mosaic) return [];
    const wanted = filters.role && filters.role !== 'all' ? filters.role : null;
    const paths = { train: '', val: '', test: '', unused: '' };
    const counts = { train: 0, val: 0, test: 0, unused: 0 };
    for (const tile of mosaic.tiles) {
      const role = tile.cv_role;
      if (!(role in paths)) continue;
      if (wanted && role !== wanted) continue;
      counts[role] += 1;
      if (!showRoleColors) continue;
      const x = tile.col * cell;
      const y = tile.row * cell;
      paths[role] += `M${x} ${y}h${cell}v${cell}h${-cell}z`;
    }
    return ['train', 'val', 'test', 'unused']
      .filter((role) => counts[role])
      .map((role) => ({
        role,
        label: role,
        path: paths[role],
        color: ROLE_COLOR[role],
        outline: role === 'unused',
        count: counts[role],
        countText: counts[role].toLocaleString(),
      }));
  });
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
  let sheetSlots = $derived.by(() => {
    const allow = new Set();
    if (overview) return allow;
    for (const chunk of chunks) {
      if (readySheets[chunk.key]) continue;
      allow.add(chunk.key);
      if (allow.size >= SHEET_SLOTS) break;
    }
    return allow;
  });
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
    loadRuns();
  });

  $effect(() => {
    const name = runName;
    const options = runsInFold;
    if (!name || !runList.length) return;
    if (!options.some((run) => run.exp_name === name)) {
      runName = '';
      runTiles = null;
      placedRun = '';
    }
  });

  $effect(() => {
    const tiles = runTiles;
    const name = runName;
    const here = site;
    if (!name || !tiles || tiles.exp_name !== name || placedRun === name) return;
    const trained = (tiles.by_flight || []).filter(
      (row) => row.site === here && (row.train_labelled || 0) + (row.train_bg || 0) > 0,
    );
    placedRun = name;
    if (!trained.length) return;
    if (trained.some((row) => String(row.flight) === String(flight))) return;
    const dated = trained
      .map((row) => {
        const info = (facets?.flights ?? []).find(
          (item) => item.site_name === row.site && String(item.flight) === String(row.flight),
        );
        return { flight: String(row.flight), date: info?.acq_date || '' };
      })
      .sort((a, b) => a.date.localeCompare(b.date) || a.flight.localeCompare(b.flight));
    flight = dated[0].flight;
  });

  $effect(() => {
    const name = runName;
    const visible = showRun;
    if (!name || !visible) {
      runTilesLoading = false;
      return;
    }
    if (untrack(() => runTiles?.exp_name) === name) return;
    let dead = false;
    runTilesLoading = true;
    runTilesError = '';
    fetch(`/api/runs/${encodeURIComponent(name)}/tiles`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await errorText(res));
        return res.json();
      })
      .then((data) => {
        if (!dead) runTiles = data;
      })
      .catch((err) => {
        if (!dead) {
          runTilesError = String(err.message || err);
          if (!foldPinned) setRoleColors(true);
        }
      })
      .finally(() => {
        if (!dead) runTilesLoading = false;
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

  function showSheet(key) {
    return !!readySheets[key] || sheetSlots.has(key);
  }

  function sheetUrl(cc, cr) {
    return `/api/mosaic/${encodeURIComponent(mosaic.site)}/${encodeURIComponent(mosaic.flight)}/sheet/${cc}/${cr}/image.webp`;
  }

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

  function dayOf(acq) {
    if (!acq || String(acq).length < 10) return '';
    const day = Number(String(acq).slice(8, 10));
    return Number.isFinite(day) ? String(day) : '';
  }

  function chooseFlight(value) {
    flight = String(value);
    if (runName && !foldPinned) showRoleColors = false;
  }

  function flightDateLabel(row, year, month) {
    const day = dayOf(row.acq_date);
    const when = day ? `${day} ${MONTHS[month]} ${year}` : `${MONTHS[month] ?? month} ${year}`;
    return `${site}, ${when}`;
  }

  function chooseSite(value) {
    site = value;
    const rows = (facets?.flights ?? [])
      .filter((row) => row.site_name === value)
      .slice()
      .sort((a, b) => String(a.acq_date ?? '').localeCompare(String(b.acq_date ?? ''))
        || String(a.flight).localeCompare(String(b.flight)));
    flight = rows[0] ? String(rows[0].flight) : '';
  }

  async function onTimelineKey(event) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const group = event.currentTarget.closest('.map-time');
    const rows = flightsForSite.filter((row) => row.year != null && row.month != null);
    const index = rows.findIndex((row) => String(row.flight) === String(flight));
    const next = rows[index + (event.key === 'ArrowRight' ? 1 : -1)];
    if (!next) return;
    event.preventDefault();
    chooseFlight(next.flight);
    await tick();
    group?.querySelector('.map-month.on')?.focus();
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
    const label = labelAt(event);
    const role = phraseFor(tileUnder(event));
    if (label) {
      hover = { ...label, role };
      return;
    }
    if (!role || !viewport) {
      hover = null;
      return;
    }
    const vp = viewport.getBoundingClientRect();
    let x = event.clientX - vp.left + 14;
    let y = event.clientY - vp.top + 16;
    if (x > vp.width - 180) x = event.clientX - vp.left - 170;
    if (y > vp.height - 32) y = event.clientY - vp.top - 28;
    hover = { name: '', color: '', binomial: false, role, x, y };
  }

  function tileUnder(event) {
    if (!mosaic || !world) return null;
    const rect = world.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const col = Math.floor((event.clientX - rect.left) / (rect.width / mosaic.ncols));
    const row = Math.floor((event.clientY - rect.top) / (rect.height / mosaic.nrows));
    if (col < 0 || row < 0 || col >= mosaic.ncols || row >= mosaic.nrows) return null;
    return byCell.get(`${col},${row}`) ?? null;
  }

  function phraseFor(tile) {
    const sets = runIdSets;
    if (!sets || !tile || !mosaic) return '';
    const id = tile.tile_id;
    if (sets.labelled.has(id)) return 'train - labelled';
    if (sets.background.has(id)) return 'train - background';
    if (sets.val.has(id)) return 'validation';
    const keys = runTiles?.train_site || [];
    if (tile.cv_role === 'train' && coversFlight(keys, mosaic.site, String(mosaic.flight ?? ''))) {
      return 'in pool, not picked';
    }
    if (tile.cv_role === 'test') return 'test';
    return '';
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
          <option value={name}>
            {name}{runCount(name) ? ` (${runCount(name)} ${runCount(name) === 1 ? 'run' : 'runs'})` : ''}
          </option>
        {/each}
      </select>
    </label>
    <label class="field">
      Fold
      <select bind:value={filters.fold} onchange={() => setRoleColors(true)}>
        {#each ['cv1', 'cv2', 'cv3', 'cv4', 'cv5'] as fold}
          <option value={fold}>{fold}</option>
        {/each}
      </select>
    </label>
    <label class="field">
      CV role
      <select bind:value={filters.role}>
        {#each ROLES as [value, label]}
          <option {value}>{label}</option>
        {/each}
      </select>
    </label>
    <div class="map-actions">
      <button type="button" class="fit" onclick={fit}>Fit map to screen</button>
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
      <label class="map-check">
        <input type="checkbox" bind:checked={showGrid} />
        Show grid
      </label>
      <label class="map-check">
        <input
          type="checkbox"
          checked={showRoleColors}
          onchange={(event) => setRoleColors(event.currentTarget.checked)}
        />
        Color roles
      </label>
      <label class="map-check">
        <input
          type="checkbox"
          checked={showRun}
          onchange={(event) => toggleRun(event.currentTarget.checked)}
        />
        Show runs
      </label>
      <label class="map-slider">
        Image
        <input type="range" min="0" max="1" step="0.05" bind:value={imageOpacity} />
      </label>
      <button
        type="button"
        class="info"
        title={mapHelp}
        aria-label="How to read the map"
      >i</button>
    </div>
    {#if showRun && runTilesLoading}
      <p class="map-meta run-status">Rebuilding this run's tiles from its config and seed…</p>
    {:else if showRun && runTilesError}
      <p class="map-meta run-status">{runTilesError}</p>
    {:else if showRun && runsError && !runList.length}
      <p class="map-meta run-status">{runsError}</p>
    {:else if showRun && runTiles}
      <p class="map-meta run-status">Rebuilt from config and seed.</p>
    {/if}
  </div>

  <div class="map-time" class:with-run={showRun && !!runName} role="group" aria-label="Flights of {site} by date">
    {#each timeline as group (group.year)}
      <div class="map-year" class:quiet={group.quiet}>
        <span class="map-year-label">{group.year}</span>
        <div class="map-months">
          {#each group.months as cell (`${group.year}-${cell.month}`)}
            {#if cell.row}
              {@const use = flightUse(cell.row.flight)}
              <button
                type="button"
                class="map-month"
                class:on={String(cell.row.flight) === String(flight)}
                class:idle={use && !use.touched}
                aria-pressed={String(cell.row.flight) === String(flight)}
                aria-label={flightDateLabel(cell.row, group.year, cell.month)}
                onclick={() => chooseFlight(cell.row.flight)}
                onkeydown={onTimelineKey}
              >
                <span class="map-dot" aria-hidden="true"></span>
                <span class="map-month-name">{MONTHS[cell.month]}</span>
                <span class="map-month-day">{dayOf(cell.row.acq_date)}</span>
                {#if use?.touched}
                  <span class="map-month-run">{use.trained} trained · {use.val} val</span>
                {/if}
              </button>
            {:else}
              <span class="map-month off">
                <span class="map-dot" aria-hidden="true"></span>
                <span class="map-month-name">{MONTHS[cell.month]}</span>
                <span class="map-month-day"></span>
              </span>
            {/if}
          {/each}
        </div>
      </div>
    {/each}
  </div>

  <div class="map-body" class:with-runs={showRun}>
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
            {#if showSheet(chunk.key)}
              <img
                class="photo"
                src={sheetUrl(chunk.cc, chunk.cr)}
                alt=""
                draggable="false"
                fetchpriority="low"
                style:opacity={imageOpacity}
                style:height="{chunk.heightPct}%"
                use:watchTile={chunk.key}
              />
              {#if showLabels && readySheets[chunk.key]}
                <img
                  class="mask"
                  src={sheetUrl(chunk.cc, chunk.cr)}
                  alt=""
                  draggable="false"
                  fetchpriority="low"
                  style:height="{chunk.heightPct}%"
                  style:top="{chunk.maskTopPct}%"
                />
              {/if}
            {/if}
          </div>
        {/each}
        {#if showGrid || showRoleColors || (showRun && runName)}
          <svg class="tile-grid" aria-hidden="true">
            {#if showRoleColors}
              {#each roleMarks as mark (mark.role)}
                <path
                  class="mark"
                  class:outline={mark.outline}
                  d={mark.path}
                  style:fill={mark.outline ? 'none' : mark.color}
                  style:stroke={mark.color}
                />
              {/each}
            {/if}
            {#if showRun}
              {#each runMarks as mark (mark.role)}
                <path
                  class="mark run-mark"
                  class:outline={showRoleColors || mark.outline}
                  d={mark.path}
                  style:fill={showRoleColors || mark.outline ? 'none' : mark.color}
                  style:stroke={mark.color}
                />
              {/each}
            {/if}
            {#if showGrid}
              <path class="edge" d={gridPath} />
            {/if}
          </svg>
        {/if}
      </div>
    {/if}
    {#if (!imagesStarted || detailPending) && !error}
      <div class="map-loading" role="status" aria-live="polite">
        <span class="map-spin" aria-hidden="true"></span>
        {imagesStarted ? 'Loading tiles' : 'Loading flight'}
      </div>
    {/if}
    {#if mosaic && (showRoleColors || heldOut || runMarks.some((mark) => mark.count))}
      {@const runOn = runMarks.some((mark) => mark.count)}
      {@const canSwitch = runOn}
      {@const wanted = filters.role && filters.role !== 'all' ? filters.role : null}
      {@const foldRows = FOLD_KEY.filter(([role]) => flightRoles[role] > 0 && (heldOut || !wanted || role === wanted))}
      <aside class="map-key" aria-label="Map legend" onpointerdown={(event) => event.stopPropagation()}>
        {#if heldOut}
          <p class="held">Held-out site</p>
        {/if}
        {#if foldRows.length || heldOut}
          <section class:off={canSwitch && !showRoleColors}>
            <h2>
              {#if canSwitch}
                <button type="button" class:on={showRoleColors} onclick={() => setRoleColors(true)}>
                  Fold {filters.fold}
                </button>
              {:else}
                Fold {filters.fold}
              {/if}
            </h2>
            <ul>
              {#each foldRows as [role, label, outline] (role)}
                <li>
                  <i
                    class:outline
                    style:background={outline ? 'transparent' : ROLE_COLOR[role]}
                    style:--swatch={ROLE_COLOR[role]}
                  ></i>
                  {label}
                  <span>{flightRoles[role].toLocaleString()}</span>
                </li>
              {/each}
            </ul>
          </section>
        {/if}
        {#if runOn}
          <section class:off={showRoleColors}>
            <h2>
              <button type="button" class:on={!showRoleColors} onclick={() => setRoleColors(false)}>
                This run
              </button>
            </h2>
            <ul>
              {#each runMarks.filter((mark) => mark.count) as mark (mark.role)}
                <li>
                  <i
                    class:outline={mark.outline}
                    style:background={mark.outline ? 'transparent' : mark.color}
                    style:--swatch={mark.color}
                  ></i>
                  {RUN_KEY[mark.role]}
                  <span>{mark.count.toLocaleString()}</span>
                </li>
              {/each}
            </ul>
            <p>
              Whole run:
              {(runTiles?.totals?.train_labelled ?? 0).toLocaleString()} train, labelled,
              {(runTiles?.totals?.train_bg ?? 0).toLocaleString()} train, background,
              {(runTiles?.totals?.val ?? 0).toLocaleString()} validation
            </p>
          </section>
        {/if}
      </aside>
    {/if}
    {#if hover}
      <p class="map-tip" style:left="{hover.x}px" style:top="{hover.y}px">
        {#if hover.name}
          <i class="swatch" style:--swatch={hover.color}></i>
          {#if hover.binomial}<span class="binomial">{hover.name}</span>{:else}{hover.name}{/if}
        {/if}
        {#if hover.role}<span class="tip-role">{hover.role}</span>{/if}
      </p>
    {/if}
  </div>
  {#if showRun}
    <MapRuns runs={runsInFold} selected={runName} onselect={selectRun} />
  {/if}
  </div>
</div>
