<script>
  let { facets = null, fold: initialFold = 'cv3', onopen } = $props();

  const PAGE = 48;
  const ROLES = [
    ['all', 'All roles'],
    ['train', 'Train'],
    ['val', 'Validation'],
    ['test', 'Test'],
    ['unused', 'Unused'],
  ];
  const PRESETS = [
    ['all', 'All'],
    ['predicted', 'Predicted'],
    ['best', 'Best F1'],
    ['worst', 'Worst F1'],
    ['fp', 'False positives'],
    ['fn', 'Missed'],
    ['wrong', 'Confused'],
    ['unpredicted', 'Not run'],
  ];
  const SORTS = [
    ['f1_species', 'Species F1'],
    ['iou_species', 'Species IoU'],
    ['binary_f1', 'Plant vs background F1'],
    ['recall_species', 'Species recall'],
    ['precision_species', 'Species precision'],
    ['agreement', 'Pixel agreement'],
    ['fp_px', 'False positive px'],
    ['fn_px', 'Missed px'],
    ['wrong_species_px', 'Confused px'],
    ['pred_px', 'Predicted px'],
  ];
  const OVERLAYS = [
    ['prediction', 'Prediction'],
    ['agreement', 'Agreement'],
    ['labels', 'Labels'],
    ['photo', 'Photo'],
  ];
  const SPECIES_OPS = [
    ['or', 'OR', 'Any selected species'],
    ['and', 'AND', 'Every selected species'],
    ['xor', 'XOR', 'An odd number of the selected species'],
    ['nor', 'NOR', 'None of the selected species'],
    ['nand', 'NAND', 'Not every selected species'],
  ];

  // Starts from the tile browser's fold, then stays local to this tab.
  // svelte-ignore state_referenced_locally
  let fold = $state(initialFold || 'cv3');
  let role = $state('test');
  let site = $state('');
  let labelled = $state(true);
  let backgroundOnly = $state(false);
  let species = $state([]);
  let speciesOp = $state('or');
  let opMenu = $state(false);
  let opEl = $state(null);
  let minF1 = $state('');
  let maxF1 = $state('');
  let preset = $state('all');
  let only = $state('all');
  let sortKey = $state('f1_species');
  let sortDir = $state('desc');
  let speciesScore = $state('');
  let speciesField = $state('f1');
  let overlay = $state('prediction');
  let strength = $state(0.85);
  let scoresOpen = $state(false);
  let optionsOpen = $state(true);
  let filtersOpen = $state(true);
  let summaryOpen = $state(true);
  let page = $state(0);

  let checkpoints = $state([]);
  let checkpointsLoading = $state(true);
  let runs = $state([]);
  let scoresReady = $state(false);
  let scoresNote = $state('');
  let ckptQuery = $state('');
  let ckpt = $state('');
  let listError = $state('');

  let records = $state([]);
  let loading = $state(false);
  let loadError = $state('');
  let runError = $state('');
  let running = $state(false);
  let runDone = $state(0);
  let runTotal = $state(0);
  let runRate = $state(0);
  let runLeft = $state(0);

  let fetchGen = 0;
  let runToken = 0;
  let pauseAsked = false;
  let alive = true;
  let pickedCkpt = false;
  let pageWatch = '';

  let colors = $derived(Object.fromEntries((facets?.species ?? []).map((spec) => [spec.stats_name, spec.color])));
  let ranked = $derived(rankCheckpoints(checkpoints, runs));
  let selectedCkpt = $derived(ranked.find((item) => item.path === ckpt));
  let visibleCkpts = $derived.by(() => {
    const needle = ckptQuery.trim().toLowerCase();
    let rows = ranked;
    if (needle) {
      rows = ranked.filter((item) => `${item.exp_name} ${item.file} ${item.kind}`.toLowerCase().includes(needle));
    }
    if (ckpt && !rows.some((item) => item.path === ckpt)) {
      const selected = ranked.find((item) => item.path === ckpt);
      if (selected) rows = [selected, ...rows];
    }
    return rows;
  });
  let query = $derived.by(() => {
    const params = new URLSearchParams();
    params.set('fold', fold || 'cv3');
    params.set('role', role || 'all');
    if (site) params.set('site', site);
    params.set('labelled', labelled ? 'true' : 'false');
    params.set('background_only', backgroundOnly ? 'true' : 'false');
    for (const name of species) params.append('species', name);
    if (species.length) params.set('species_op', speciesOp);
    return params.toString();
  });
  let filtered = $derived.by(() => {
    const rows = records.filter((tile) => passesOnly(tile) && inF1Range(tile));
    rows.sort((a, b) => compareTiles(a, b));
    return rows;
  });
  let viewSummary = $derived(aggregate(filtered));
  let scopeSummary = $derived(aggregate(records));
  let bins = $derived(histogram(records));
  let binPeak = $derived(Math.max(1, ...bins.map((bin) => bin.n)));
  let pageCount = $derived(filtered.length ? Math.ceil(filtered.length / PAGE) : 0);
  let pageRows = $derived(filtered.slice(page * PAGE, page * PAGE + PAGE));
  let rangeStart = $derived(filtered.length ? page * PAGE + 1 : 0);
  let rangeEnd = $derived(Math.min(filtered.length, (page + 1) * PAGE));
  let remaining = $derived(records.filter((tile) => !tile.metrics).length);
  let links = $derived.by(() => pageLinks(pageCount, page));
  let runLabel = $derived.by(() => {
    if (!running) return '';
    const rate = runRate > 0 ? ` · ${runRate.toFixed(1)} tiles/s` : '';
    const left = runLeft >= 1 ? ` · about ${eta(runLeft)} left` : '';
    return `${runDone.toLocaleString()} / ${runTotal.toLocaleString()}${rate}${left}`;
  });

  $effect(() => {
    return () => {
      alive = false;
      runToken += 1;
    };
  });

  $effect(() => {
    const key = [
      query,
      only,
      preset,
      sortKey,
      sortDir,
      speciesScore,
      speciesField,
      minF1,
      maxF1,
      speciesOp,
      species.join(','),
    ].join('|');
    if (key === pageWatch) return;
    pageWatch = key;
    page = 0;
  });

  $effect(() => {
    const count = filtered.length;
    const pages = Math.max(1, Math.ceil(count / PAGE) || 1);
    if (page > pages - 1) page = pages - 1;
  });

  $effect(() => {
    const ctrl = new AbortController();
    fetch('/api/checkpoints', { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        checkpoints = data.checkpoints ?? [];
        listError = '';
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        listError = String(err.message || err);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) checkpointsLoading = false;
      });
    return () => ctrl.abort();
  });

  $effect(() => {
    const ctrl = new AbortController();
    fetch('/api/wandb/runs', { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        runs = data.runs ?? [];
        scoresNote = data.available ? '' : (data.detail || '');
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        scoresNote = String(err.message || err);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) scoresReady = true;
      });
    return () => ctrl.abort();
  });

  $effect(() => {
    if (pickedCkpt || !scoresReady || !checkpoints.length) return;
    pickedCkpt = true;
    const rows = rankCheckpoints(checkpoints, runs);
    const best = rows.find((item) => item.kind === 'best' && item.metrics?.test_f1_species != null) ?? rows[0];
    if (!best) return;
    ckpt = best.path;
    const found = foldIn(best.exp_name);
    if (found) fold = found;
  });

  $effect(() => {
    const path = ckpt;
    const search = query;
    const gen = ++fetchGen;
    runToken += 1;
    if (!path) {
      records = [];
      loading = false;
      return;
    }
    const ctrl = new AbortController();
    loading = true;
    loadError = '';
    records = [];
    fetch(`/api/batch/tiles?ckpt=${encodeURIComponent(path)}&${search}`, { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        if (gen !== fetchGen) return;
        records = data.tiles ?? [];
        loadError = '';
      })
      .catch((err) => {
        if (err.name === 'AbortError' || gen !== fetchGen) return;
        loadError = String(err.message || err);
        records = [];
      })
      .finally(() => {
        if (gen === fetchGen) loading = false;
      });
    return () => ctrl.abort();
  });

  function rankCheckpoints(rows, wandbRuns) {
    const byName = new Map((wandbRuns ?? []).map((run) => [run.name, run.metrics || {}]));
    return rows.map((item) => ({
      ...item,
      metrics: byName.get(item.exp_name) || {},
    })).sort((a, b) => compareRows(a, b));
  }

  function compareRows(a, b) {
    const byTest = compareScore(a, b, 'test_f1_species');
    if (byTest) return byTest;
    const byVal = compareScore(a, b, 'val_f1_species');
    if (byVal) return byVal;
    if (a.kind !== b.kind) {
      if (a.kind === 'best') return -1;
      if (b.kind === 'best') return 1;
    }
    return (b.mtime || 0) - (a.mtime || 0);
  }

  function compareScore(a, b, key) {
    const left = a.metrics?.[key];
    const right = b.metrics?.[key];
    if (left == null && right == null) return 0;
    if (left == null) return 1;
    if (right == null) return -1;
    return right - left;
  }

  function titleOf(item) {
    return item.exp_name.replace(/^\d{8}_[\d.]+_/, '') || item.exp_name;
  }

  function f1(item, key) {
    const value = item.metrics?.[key];
    if (value == null || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(3);
  }

  function optionLabel(item) {
    return `test ${f1(item, 'test_f1_species')} · val ${f1(item, 'val_f1_species')} · ${item.kind} · ${titleOf(item)}${item.ds2 ? ' · half res' : ''}`;
  }

  function foldIn(name) {
    const match = String(name || '').match(/cv[1-5]/);
    return match ? match[0] : '';
  }

  function onCkpt(event) {
    const item = ranked.find((row) => row.path === event.currentTarget.value);
    const found = foldIn(item?.exp_name);
    if (found) fold = found;
  }

  function readError(res) {
    return res.text().then((text) => {
      try {
        const data = JSON.parse(text);
        if (typeof data?.detail === 'string') return data.detail;
      } catch {
        /* plain text */
      }
      return text || res.statusText;
    });
  }

  let speciesOpRow = $derived(SPECIES_OPS.find((item) => item[0] === speciesOp) ?? SPECIES_OPS[0]);
  let filterPeek = $derived.by(() => {
    const bits = [];
    if (species.length) bits.push(`${speciesOpRow[1]} · ${species.length} species`);
    const presetRow = PRESETS.find((item) => item[0] === preset);
    if (preset !== 'all' && presetRow) bits.push(presetRow[1]);
    if (String(minF1) !== '' || String(maxF1) !== '') {
      bits.push(`F1 ${minF1 === '' ? '0' : minF1}–${maxF1 === '' ? '1' : maxF1}`);
    }
    const sortRow = SORTS.find((item) => item[0] === sortKey);
    if (sortRow) bits.push(`${sortRow[1]} · ${sortDir === 'desc' ? 'high first' : 'low first'}`);
    return bits.join(' · ');
  });

  function toggleSpecies(name) {
    species = species.includes(name) ? species.filter((item) => item !== name) : [...species, name];
  }

  function setSpeciesOp(value) {
    speciesOp = value;
    opMenu = false;
  }

  $effect(() => {
    if (!opMenu) return;
    function onPointer(event) {
      if (opEl?.contains(event.target)) return;
      opMenu = false;
    }
    function onKey(event) {
      if (event.key === 'Escape') opMenu = false;
    }
    window.addEventListener('pointerdown', onPointer);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('pointerdown', onPointer);
      window.removeEventListener('keydown', onKey);
    };
  });

  function setLabelled(checked) {
    labelled = checked;
    if (checked) backgroundOnly = false;
  }

  function setBackground(checked) {
    backgroundOnly = checked;
    if (checked) labelled = false;
  }

  function applyPreset(name) {
    preset = name;
    if (name === 'all') only = 'all';
    else if (name === 'predicted') only = 'predicted';
    else if (name === 'unpredicted') only = 'unpredicted';
    else if (name === 'fp') {
      only = 'fp';
      sortKey = 'fp_px';
      sortDir = 'desc';
      speciesScore = '';
    } else if (name === 'fn') {
      only = 'fn';
      sortKey = 'fn_px';
      sortDir = 'desc';
      speciesScore = '';
    } else if (name === 'wrong') {
      only = 'wrong';
      sortKey = 'wrong_species_px';
      sortDir = 'desc';
      speciesScore = '';
    } else if (name === 'best') {
      only = 'scored';
      sortKey = 'f1_species';
      sortDir = 'desc';
      speciesScore = '';
    } else if (name === 'worst') {
      only = 'scored';
      sortKey = 'f1_species';
      sortDir = 'asc';
      speciesScore = '';
    }
  }

  function onSortKey() {
    if (preset === 'best' || preset === 'worst') preset = '';
  }

  function onSortDir() {
    if (preset === 'best' || preset === 'worst') preset = '';
    sortDir = sortDir === 'desc' ? 'asc' : 'desc';
  }

  function passesOnly(tile) {
    const metrics = tile.metrics;
    if (only === 'all') return true;
    if (only === 'unpredicted') return !metrics;
    if (!metrics) return false;
    if (only === 'predicted') return true;
    if (only === 'scored') return metrics.f1_species != null;
    if (only === 'fp') return (metrics.fp_px || 0) > 0;
    if (only === 'fn') return (metrics.fn_px || 0) > 0;
    if (only === 'wrong') return (metrics.wrong_species_px || 0) > 0;
    return true;
  }

  function countFor(tile, name) {
    return (tile.metrics?.counts || []).find((row) => row.stats_name === name);
  }

  function bound(raw) {
    if (raw == null || raw === '') return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }

  function inF1Range(tile) {
    const lo = bound(minF1);
    const hi = bound(maxF1);
    if (lo == null && hi == null) return true;
    const value = tile.metrics?.f1_species;
    if (value == null) return false;
    if (lo != null && value < lo) return false;
    if (hi != null && value > hi) return false;
    return true;
  }

  function sortValue(tile) {
    const metrics = tile.metrics;
    if (!metrics) return null;
    if (speciesScore) {
      const row = countFor(tile, speciesScore);
      if (!row) return null;
      if (speciesField === 'fp') return row.fp ?? null;
      if (speciesField === 'fn') return row.fn ?? null;
      return row.f1 ?? null;
    }
    const value = metrics[sortKey];
    return value == null ? null : value;
  }

  function compareTiles(a, b) {
    const left = sortValue(a);
    const right = sortValue(b);
    if (left == null && right == null) return a.tile_id < b.tile_id ? -1 : 1;
    if (left == null) return 1;
    if (right == null) return -1;
    const diff = sortDir === 'asc' ? left - right : right - left;
    if (diff) return diff;
    return a.tile_id < b.tile_id ? -1 : 1;
  }

  function pixelSort() {
    if (speciesScore) return speciesField === 'fp' || speciesField === 'fn';
    return String(sortKey).endsWith('_px');
  }

  function badgeLabel() {
    if (speciesScore) {
      if (speciesField === 'fp') return 'FP';
      if (speciesField === 'fn') return 'Miss';
      return 'F1';
    }
    const labels = {
      f1_species: 'F1',
      iou_species: 'IoU',
      binary_f1: 'F1',
      recall_species: 'Rec',
      precision_species: 'Prec',
      agreement: 'Agr',
      fp_px: 'FP',
      fn_px: 'Miss',
      wrong_species_px: 'Mix',
      pred_px: 'Px',
    };
    return labels[sortKey] || 'Score';
  }

  function badge(tile) {
    if (!tile.metrics) return 'Not run';
    const value = sortValue(tile);
    if (value == null) return '—';
    const text = pixelSort() ? `${Math.round(value).toLocaleString()} px` : Number(value).toFixed(3);
    return `${badgeLabel()} ${text}`;
  }

  function share(part, item) {
    const sum = (item.tp || 0) + (item.fp || 0) + (item.fn || 0);
    if (!sum) return 0;
    return (100 * part) / sum;
  }

  function portion(part, item) {
    const sum = item.tp + item.fp + item.fn + item.wrong;
    if (!sum) return 0;
    return (100 * part) / sum;
  }

  function fmt(value) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(3);
  }

  function ratio(num, den) {
    if (den == null || den <= 0 || num == null) return null;
    return num / den;
  }

  function mean(values) {
    if (!values.length) return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  function f1of(tp, fp, fn) {
    const den = 2 * tp + fp + fn;
    return den > 0 ? (2 * tp) / den : null;
  }

  function iouof(tp, fp, fn) {
    const den = tp + fp + fn;
    return den > 0 ? tp / den : null;
  }

  function aggregate(tiles) {
    const totals = {
      valid_px: 0,
      correct_px: 0,
      pred_px: 0,
      label_px: 0,
      tp_px: 0,
      fp_px: 0,
      fn_px: 0,
      wrong_species_px: 0,
      bin_tp: 0,
      bin_fp: 0,
      bin_fn: 0,
    };
    let predicted = 0;
    let scored = 0;
    const byId = new Map();
    for (const tile of tiles) {
      const metrics = tile.metrics;
      if (!metrics) continue;
      predicted += 1;
      if (metrics.correct_px != null && metrics.valid_px != null) {
        scored += 1;
        for (const key of Object.keys(totals)) {
          if (metrics[key] != null) totals[key] += Number(metrics[key]) || 0;
        }
      } else if (metrics.pred_px != null) {
        totals.pred_px += Number(metrics.pred_px) || 0;
      }
      for (const row of metrics.counts || []) {
        if (row.tp == null) continue;
        const slot = byId.get(row.id) || {
          id: row.id,
          stats_name: row.stats_name || '',
          name: row.name || '',
          color: row.color || '#5d6e74',
          tp: 0,
          fp: 0,
          fn: 0,
        };
        slot.tp += Number(row.tp) || 0;
        slot.fp += Number(row.fp) || 0;
        slot.fn += Number(row.fn) || 0;
        byId.set(row.id, slot);
      }
    }
    const order = (facets?.species ?? []).map((spec) => spec.stats_name);
    const speciesRows = [...byId.values()].map((row) => ({
      ...row,
      f1: f1of(row.tp, row.fp, row.fn),
      iou: iouof(row.tp, row.fp, row.fn),
    })).sort((a, b) => {
      const ia = order.indexOf(a.stats_name);
      const ib = order.indexOf(b.stats_name);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    const f1s = speciesRows.map((row) => row.f1).filter((value) => value != null);
    const ious = speciesRows.map((row) => row.iou).filter((value) => value != null);
    const precs = [];
    const recs = [];
    for (const row of speciesRows) {
      if (row.tp + row.fp > 0) precs.push(row.tp / (row.tp + row.fp));
      if (row.tp + row.fn > 0) recs.push(row.tp / (row.tp + row.fn));
    }
    return {
      predicted,
      scored,
      ...totals,
      f1_species: mean(f1s),
      iou_species: mean(ious),
      precision_species: mean(precs),
      recall_species: mean(recs),
      binary_f1: f1of(totals.bin_tp, totals.bin_fp, totals.bin_fn),
      agreement: ratio(totals.correct_px, totals.valid_px),
      species: speciesRows,
    };
  }

  function histogram(tiles) {
    const bins = Array.from({ length: 8 }, (_, index) => ({
      index,
      lo: index / 8,
      hi: (index + 1) / 8,
      n: 0,
    }));
    for (const tile of tiles) {
      const value = tile.metrics?.f1_species;
      if (value == null) continue;
      const index = Math.min(7, Math.max(0, Math.floor(Number(value) * 8)));
      bins[index].n += 1;
    }
    return bins;
  }

  function pickBin(bin) {
    if (minF1 === bin.lo.toFixed(2) && maxF1 === (bin.hi >= 1 ? '1' : bin.hi.toFixed(2))) {
      minF1 = '';
      maxF1 = '';
      return;
    }
    minF1 = bin.lo.toFixed(2);
    maxF1 = bin.hi >= 1 ? '1' : bin.hi.toFixed(2);
    only = 'scored';
    preset = '';
  }

  function pageLinks(count, current) {
    if (count < 1) return [];
    const here = current + 1;
    const wanted = [1, count, here - 2, here - 1, here, here + 1, here + 2];
    const nums = [...new Set(wanted.filter((n) => n >= 1 && n <= count))].sort((a, b) => a - b);
    const out = [];
    for (let i = 0; i < nums.length; i++) {
      if (i > 0 && nums[i] - nums[i - 1] > 1) out.push({ kind: 'gap', id: `gap-${nums[i]}` });
      out.push({ kind: 'page', n: nums[i], id: `page-${nums[i]}` });
    }
    return out;
  }

  function showPage(next) {
    const pages = Math.max(1, pageCount || 1);
    page = Math.min(Math.max(0, next), pages - 1);
  }

  function bust(url, saved) {
    if (!url) return '';
    return `${url}?v=${saved ? Math.floor(saved) : 0}`;
  }

  function eta(seconds) {
    const whole = Math.round(seconds);
    if (whole < 60) return `${whole}s`;
    const minutes = Math.floor(whole / 60);
    const rest = whole % 60;
    return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
  }

  function applyResult(row) {
    records = records.map((item) => (
      item.tile_id === row.tile_id
        ? { ...item, metrics: row.metrics, pred: row.pred, comp: row.comp, saved_at: row.saved_at }
        : item
    ));
  }

  async function run(force) {
    if (running || !ckpt) return;
    const ids = records.filter((tile) => force || !tile.metrics).map((tile) => tile.tile_id);
    if (!ids.length) return;
    const token = runToken;
    pauseAsked = false;
    running = true;
    runError = '';
    runDone = 0;
    runTotal = ids.length;
    runRate = 0;
    runLeft = 0;
    const started = performance.now();
    const CHUNK = 32;
    try {
      for (let start = 0; start < ids.length; start += CHUNK) {
        if (!alive || token !== runToken || pauseAsked) break;
        const chunk = ids.slice(start, start + CHUNK);
        const res = await fetch('/api/batch/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ckpt, tile_ids: chunk, force }),
        });
        if (!alive || token !== runToken) return;
        if (!res.ok) {
          runError = await readError(res);
          break;
        }
        const data = await res.json();
        const rows = Array.isArray(data.tiles) ? data.tiles : [data];
        let failed = false;
        for (const row of rows) {
          if (!row || row.error) {
            runError = String(row?.error || 'prediction failed');
            failed = true;
            break;
          }
          applyResult(row);
          runDone += 1;
        }
        const elapsed = (performance.now() - started) / 1000;
        runRate = elapsed > 0 ? runDone / elapsed : 0;
        runLeft = runRate > 0 ? (runTotal - runDone) / runRate : 0;
        if (failed || pauseAsked) break;
      }
    } catch (err) {
      if (alive && err.name !== 'AbortError') runError = String(err.message || err);
    } finally {
      if (alive) running = false;
    }
  }

  function pause() {
    pauseAsked = true;
  }

  function rerun() {
    if (!records.length || running) return;
    const count = records.length;
    if (count > 15 && !confirm(`Re-run all ${count.toLocaleString()} tiles in this filter? Each saved score is replaced when that tile finishes.`)) {
      return;
    }
    run(true);
  }

  async function clearCache() {
    if (!ckpt || running) return;
    if (!confirm('Delete saved batch predictions for this checkpoint?')) return;
    runToken += 1;
    runError = '';
    try {
      const res = await fetch(`/api/batch/cache?ckpt=${encodeURIComponent(ckpt)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await readError(res));
      records = records.map((item) => ({ ...item, metrics: null, pred: null, comp: null, saved_at: null }));
    } catch (err) {
      runError = String(err.message || err);
    }
  }
</script>

<div class="batch">
  <section class="panel" class:shut={!optionsOpen} aria-label="Prediction options">
    <h2>
      <button
        type="button"
        class="head"
        aria-expanded={optionsOpen}
        aria-controls="prediction-options"
        onclick={() => (optionsOpen = !optionsOpen)}
      >
        <i class="caret" aria-hidden="true"></i>
        Prediction options
        {#if !optionsOpen && running}
          <span class="peek">{runLabel}</span>
        {:else if !optionsOpen && selectedCkpt}
          <span class="peek">test {f1(selectedCkpt, 'test_f1_species')} · val {f1(selectedCkpt, 'val_f1_species')}</span>
        {/if}
      </button>
    </h2>
    {#if optionsOpen}
    <div id="prediction-options">
    <div class="row">
      <label class="field ckpt">
        Checkpoint
        <input
          type="search"
          placeholder="Filter checkpoints"
          bind:value={ckptQuery}
          disabled={running || checkpointsLoading}
        />
        <select bind:value={ckpt} onchange={onCkpt} disabled={running || !visibleCkpts.length}>
          {#if !ckpt}
            <option value="">Choose a checkpoint</option>
          {/if}
          {#each visibleCkpts as item (item.path)}
            <option value={item.path} title={item.exp_name}>{optionLabel(item)}</option>
          {/each}
        </select>
      </label>
      <div class="ckpt-score" aria-label="Species F1 for this checkpoint">
        <span>Test<strong>{selectedCkpt ? f1(selectedCkpt, 'test_f1_species') : '—'}</strong></span>
        <span>Val<strong>{selectedCkpt ? f1(selectedCkpt, 'val_f1_species') : '—'}</strong></span>
      </div>
    </div>
    <div class="row">
      <label class="field">
        Fold
        <select bind:value={fold} disabled={running}>
          {#each ['cv1', 'cv2', 'cv3', 'cv4', 'cv5'] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <label class="field">
        CV role
        <select bind:value={role} disabled={running}>
          {#each ROLES as [value, label]}
            <option {value}>{label}</option>
          {/each}
        </select>
      </label>
      <label class="field">
        Site
        <select bind:value={site} disabled={running}>
          <option value="">All sites</option>
          {#each facets?.sites ?? [] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <label class="check">
        <input type="checkbox" checked={labelled} disabled={running} onchange={(e) => setLabelled(e.currentTarget.checked)} />
        Labelled only
      </label>
      <label class="check">
        <input type="checkbox" checked={backgroundOnly} disabled={running} onchange={(e) => setBackground(e.currentTarget.checked)} />
        Background only
      </label>
      <div class="runbox">
        {#if running}
          <button type="button" onclick={pause}>Pause</button>
          <p class="status" role="status"><span class="spin" aria-hidden="true"></span>{runLabel}</p>
        {:else}
          <button type="button" class="go" onclick={() => run(false)} disabled={!ckpt || loading || !remaining}>
            Run {remaining.toLocaleString()} not scored
          </button>
          <button type="button" onclick={rerun} disabled={!ckpt || loading || !records.length}>Re-run all</button>
          <button type="button" onclick={clearCache} disabled={!ckpt || loading}>Clear saved</button>
        {/if}
      </div>
    </div>
    </div>
    {/if}
  </section>

  <section class="panel" class:shut={!filtersOpen} aria-label="Filter options">
    <h2>
      <button
        type="button"
        class="head"
        aria-expanded={filtersOpen}
        aria-controls="filter-options"
        onclick={() => (filtersOpen = !filtersOpen)}
      >
        <i class="caret" aria-hidden="true"></i>
        Filter options
        {#if !filtersOpen && filterPeek}
          <span class="peek">{filterPeek}</span>
        {/if}
      </button>
    </h2>
    {#if filtersOpen}
    <div id="filter-options">
      <div class="species">
        <div class="species-op" bind:this={opEl}>
          <button
            type="button"
            class="op"
            class:set={speciesOp !== 'or'}
            aria-haspopup="listbox"
            aria-expanded={opMenu}
            aria-label="Match selected species with {speciesOpRow[1]}: {speciesOpRow[2]}"
            title={speciesOpRow[2]}
            disabled={running}
            onclick={() => (opMenu = !opMenu)}
          >{speciesOpRow[1]}</button>
          {#if opMenu}
            <div class="op-menu" role="listbox" aria-label="How selected species combine">
              {#each SPECIES_OPS as [value, label, hint]}
                <button
                  type="button"
                  role="option"
                  aria-selected={speciesOp === value}
                  class:on={speciesOp === value}
                  onclick={() => setSpeciesOp(value)}
                >
                  <strong>{label}</strong>
                  <span>{hint}</span>
                </button>
              {/each}
            </div>
          {/if}
        </div>
        {#each facets?.species ?? [] as spec}
          <button
            type="button"
            class:on={species.includes(spec.stats_name)}
            style:--swatch={spec.color}
            disabled={running}
            onclick={() => toggleSpecies(spec.stats_name)}
          >
            <i class="swatch" style:--swatch={spec.color}></i>
            <span class="binomial">{spec.name}</span>
          </button>
        {/each}
      </div>
      <div class="row">
      <div class="presets" role="group" aria-label="Result filter">
        {#each PRESETS as [value, label]}
          <button type="button" class:on={preset === value} aria-pressed={preset === value} onclick={() => applyPreset(value)}>{label}</button>
        {/each}
      </div>
      <label class="field">
        Sort
        <select bind:value={sortKey} onchange={onSortKey}>
          {#each SORTS as [value, label]}
            <option {value}>{label}</option>
          {/each}
        </select>
      </label>
      <button type="button" class="dir" aria-label={sortDir === 'desc' ? 'High values first' : 'Low values first'} onclick={onSortDir}>
        {sortDir === 'desc' ? 'High first' : 'Low first'}
      </button>
      <label class="field">
        Species score
        <select bind:value={speciesScore} onchange={() => { if (speciesScore) preset = ''; }}>
          <option value="">Whole tile</option>
          {#each facets?.species ?? [] as spec}
            <option value={spec.stats_name}>{spec.name}</option>
          {/each}
        </select>
      </label>
      <label class="field">
        For that species
        <select bind:value={speciesField} disabled={!speciesScore}>
          <option value="f1">F1</option>
          <option value="fp">False positives</option>
          <option value="fn">Missed</option>
        </select>
      </label>
      <label class="field tiny">
        Min F1
        <input type="number" min="0" max="1" step="0.05" placeholder="0" bind:value={minF1} />
      </label>
      <label class="field tiny">
        Max F1
        <input type="number" min="0" max="1" step="0.05" placeholder="1" bind:value={maxF1} />
      </label>
      </div>
    </div>
    {/if}
  </section>

  {#if scoresNote && scoresReady}
    <p class="hint">{scoresNote}</p>
  {/if}
  {#if listError || loadError || runError}
    <p class="problem" role="alert">{listError || loadError || runError}</p>
  {/if}
  <section class="summary" class:shut={!summaryOpen} aria-label="Scores">
    <h2>
      <button
        type="button"
        class="head"
        aria-expanded={summaryOpen}
        aria-controls="score-summary"
        onclick={() => (summaryOpen = !summaryOpen)}
      >
        <i class="caret" aria-hidden="true"></i>
        Scores
        {#if !summaryOpen}
          <span class="peek">F1 {fmt(viewSummary.f1_species)} · {viewSummary.predicted.toLocaleString()} predicted</span>
        {/if}
      </button>
    </h2>
    {#if summaryOpen}
    <div id="score-summary">
    <div class="summary-bar">
      <div class="stats">
        <div><span>Species F1</span><strong>{fmt(viewSummary.f1_species)}</strong></div>
        <div><span>IoU</span><strong>{fmt(viewSummary.iou_species)}</strong></div>
        <div><span>Precision</span><strong>{fmt(viewSummary.precision_species)}</strong></div>
        <div><span>Recall</span><strong>{fmt(viewSummary.recall_species)}</strong></div>
        <div><span>Plant F1</span><strong>{fmt(viewSummary.binary_f1)}</strong></div>
        <div><span>Agreement</span><strong>{fmt(viewSummary.agreement)}</strong></div>
        <div><span>Predicted</span><strong>{viewSummary.predicted.toLocaleString()}</strong></div>
      </div>
      {#if bins.some((bin) => bin.n > 0)}
        <div class="hist" role="group" aria-label="Species F1 across predicted tiles">
          {#each bins as bin (bin.index)}
            <button
              type="button"
              class:on={minF1 === bin.lo.toFixed(2) && maxF1 === (bin.hi >= 1 ? '1' : bin.hi.toFixed(2))}
              title="{bin.n.toLocaleString()} tiles with species F1 from {bin.lo.toFixed(2)} to {bin.hi >= 1 ? '1' : bin.hi.toFixed(2)}"
              onclick={() => pickBin(bin)}
            >
              <i style:height="{bin.n ? Math.max(8, (100 * bin.n) / binPeak) : 0}%"></i>
              <span>{bin.lo.toFixed(1)}</span>
            </button>
          {/each}
        </div>
      {/if}
      <button
        type="button"
        class="info"
        aria-label="How these scores are calculated"
        aria-expanded={scoresOpen}
        aria-controls="score-note"
        onclick={() => (scoresOpen = !scoresOpen)}
      >i</button>
      <div class="view">
        <div class="overlays" role="group" aria-label="Overlay">
          {#each OVERLAYS as [value, label]}
            <button type="button" class:on={overlay === value} aria-pressed={overlay === value} onclick={() => (overlay = value)}>{label}</button>
          {/each}
        </div>
        {#if overlay !== 'photo'}
          <label class="slider">
            Overlay
            <input type="range" min="0" max="1" step="0.05" bind:value={strength} />
          </label>
        {/if}
      </div>
    </div>
    {#if scoresOpen}
      <p class="hint" id="score-note">
        Pixels from the {viewSummary.scored.toLocaleString()} scored tiles in this view are added together, then turned into F1 and IoU.
        Species F1 for this view is {fmt(viewSummary.f1_species)}.
        False positives are plants on background. Missed pixels are labelled plants the model called background. Confused pixels are the wrong species.
        {#if scopeSummary.predicted !== viewSummary.predicted}
          This view is {viewSummary.predicted.toLocaleString()} of {scopeSummary.predicted.toLocaleString()} predicted tiles loaded here.
          Added up the same way, those {scopeSummary.predicted.toLocaleString()} tiles have species F1 {fmt(scopeSummary.f1_species)}.
        {/if}
      </p>
    {/if}
    {#if viewSummary.species.length}
      <ul class="species-scores">
        {#each viewSummary.species as row (row.id)}
          <li>
            <i class="swatch" style:--swatch={row.color}></i>
            <span class="binomial">{row.name}</span>
            <span class="split" title="{row.tp.toLocaleString()} correct, {row.fp.toLocaleString()} extra, {row.fn.toLocaleString()} missed">
              <i style:width="{share(row.tp, row)}%" style:background={row.color}></i>
              <i class="fp" style:width="{share(row.fp, row)}%"></i>
              <i class="fn" style:width="{share(row.fn, row)}%"></i>
            </span>
            <span class="num">{fmt(row.f1)}</span>
          </li>
        {/each}
      </ul>
    {/if}
    </div>
    {/if}
  </section>

  <div class="scroller">
    {#if loading && !records.length}
      <div class="mosaic" aria-hidden="true">
        {#each { length: 12 } as _, i (i)}
          <div class="skel"></div>
        {/each}
      </div>
    {:else if loadError}
      <p class="empty">The tile list could not be loaded.</p>
    {:else if !ckpt && !checkpointsLoading}
      <p class="empty">No checkpoints in lightning_logs yet.</p>
    {:else if !filtered.length}
      <p class="empty">{loading ? 'Loading tiles…' : 'No tiles match these filters.'}</p>
    {:else}
      <div class="mosaic" class:loading>
        {#each pageRows as tile (tile.tile_id)}
          {@const predUrl = bust(tile.pred, tile.saved_at)}
          {@const compUrl = bust(tile.comp, tile.saved_at)}
          <button type="button" class="cell" onclick={() => onopen?.(tile.tile_id, ckpt)}>
            <span class="frame">
              <img src={tile.thumb} alt="" />
              {#if overlay === 'labels'}
                <img class="mask" src={tile.mask} alt="" style:opacity={strength} />
              {/if}
              {#if overlay === 'prediction' && predUrl}
                <img class="mask" src={predUrl} alt="" style:opacity={strength} />
              {/if}
              {#if overlay === 'agreement' && compUrl}
                <img class="mask" src={compUrl} alt="" style:opacity={strength} />
              {/if}
              <span class="score">{badge(tile)}</span>
            </span>
            <span class="caption">
              <strong>{tile.site_name}_{tile.flight}</strong>
              <span class="dots">
                {#each tile.species as name}
                  <i class="swatch" style:--swatch={colors[name] || '#999'}></i>
                {/each}
              </span>
            </span>
            {#if tile.metrics?.tp_px != null}
              {@const bits = {
                tp: tile.metrics.tp_px || 0,
                fp: tile.metrics.fp_px || 0,
                fn: tile.metrics.fn_px || 0,
                wrong: tile.metrics.wrong_species_px || 0,
              }}
              <span class="bar" title="Correct, false positive, wrong species, missed">
                <i style:width="{portion(bits.tp, bits)}%"></i>
                <i class="fp" style:width="{portion(bits.fp, bits)}%"></i>
                <i class="wrong" style:width="{portion(bits.wrong, bits)}%"></i>
                <i class="fn" style:width="{portion(bits.fn, bits)}%"></i>
              </span>
            {/if}
          </button>
        {/each}
      </div>
    {/if}
  </div>

  {#if filtered.length > 0}
    <nav class="pager" aria-label="Batch pages">
      <button type="button" disabled={page === 0} onclick={() => showPage(page - 1)}>Previous</button>
      <div class="pages">
        {#each links as item (item.id)}
          {#if item.kind === 'gap'}
            <span class="gap">…</span>
          {:else}
            <button type="button" aria-current={item.n === page + 1 ? 'page' : undefined} onclick={() => showPage(item.n - 1)}>{item.n}</button>
          {/if}
        {/each}
      </div>
      <button type="button" disabled={page >= pageCount - 1} onclick={() => showPage(page + 1)}>Next</button>
      <p class="count">{rangeStart.toLocaleString()}–{rangeEnd.toLocaleString()} of {filtered.length.toLocaleString()}</p>
    </nav>
  {/if}
</div>

<style>
  .batch {
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  .panel, .summary {
    background: var(--sheet);
    border-bottom: 1px solid var(--line);
  }

  .panel { padding: 8px 18px 10px; }

  .panel h2, .summary h2 {
    margin: 0 0 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--mist);
  }

  .shut h2 { margin: 0; }

  .head {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    margin: 0;
    padding: 0;
    border: 0;
    background: none;
    color: inherit;
    font: inherit;
    letter-spacing: inherit;
    text-transform: inherit;
    text-align: left;
    cursor: pointer;
  }

  .caret {
    width: 0;
    height: 0;
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
    border-left: 5px solid currentColor;
    transform: rotate(90deg);
  }

  .shut .caret { transform: none; }

  .peek {
    letter-spacing: 0;
    text-transform: none;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
  }

  .row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 12px;
    align-items: end;
  }

  .row + .row { margin-top: 8px; }

  #filter-options .species { margin-bottom: 8px; }

  .field {
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin: 0;
    font-size: 12px;
    color: var(--mist);
  }

  .ckpt { min-width: min(280px, 100%); flex: 1 1 360px; }
  .ckpt input, .ckpt select { width: 100%; }

  .ckpt-score {
    display: flex;
    gap: 14px;
    padding-bottom: 4px;
    font-size: 12px;
    color: var(--mist);
  }

  .ckpt-score strong {
    display: block;
    margin-top: 1px;
    color: var(--ink);
    font-size: 15px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .field input,
  .field select,
  .dir, .runbox button, .presets button, .overlays button {
    border: 1px solid var(--line);
    border-radius: 2px;
    background: var(--sheet);
    color: var(--ink);
  }

  .field input, .field select { padding: 5px 8px; }

  .ckpt input { margin-bottom: 4px; }

  .tiny { width: 76px; }
  .tiny input { width: 100%; }

  .check {
    display: flex;
    align-items: center;
    gap: 6px;
    padding-bottom: 6px;
    font-size: 13px;
  }

  .runbox {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }

  .runbox button, .dir, .presets button, .overlays button {
    padding: 5px 10px;
    font-size: 12px;
  }

  .go {
    border-color: var(--canopy);
    background: var(--canopy);
    color: var(--sheet);
  }

  button:disabled { opacity: 0.55; }

  .status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin: 0;
    font-variant-numeric: tabular-nums;
    font-size: 13px;
  }

  .spin {
    width: 14px;
    height: 14px;
    border: 2px solid color-mix(in srgb, var(--mist) 45%, white);
    border-top-color: var(--canopy);
    border-radius: 50%;
    box-sizing: border-box;
    animation: spin 0.75s linear infinite;
  }

  .presets, .overlays { display: flex; flex-wrap: wrap; gap: 6px; }

  .presets button.on, .overlays button.on, .hist button.on {
    background: color-mix(in srgb, var(--canopy) 12%, white);
    border-color: var(--canopy);
  }

  .slider {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 4px;
    font-size: 12px;
    color: var(--mist);
  }

  .summary { padding: 8px 18px 10px; }

  .summary-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px 18px;
  }

  .summary-bar .view {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 12px;
  }

  .summary-bar .slider input { width: 88px; }

  .summary-bar .info { flex: none; }

  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 18px;
  }

  .stats div { display: flex; flex-direction: column; gap: 1px; }
  .stats span, .hint { color: var(--mist); font-size: 12px; }
  .stats strong { font-variant-numeric: tabular-nums; font-size: 16px; font-weight: 600; }
  .hint { margin: 6px 0 0; max-width: 90ch; }

  .hist {
    display: flex;
    align-items: end;
    gap: 4px;
    height: 52px;
    flex: none;
  }

  .hist button {
    width: 28px;
    height: 100%;
    padding: 0 0 14px;
    border: 0;
    background: transparent;
    position: relative;
  }

  .hist button i {
    display: block;
    width: 100%;
    background: var(--canopy);
    position: absolute;
    left: 0;
    bottom: 14px;
  }

  .hist button span {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    font-size: 9px;
    color: var(--mist);
  }

  .species-scores {
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: 6px 16px;
    margin: 8px 0 0;
    padding: 0;
  }

  .species-scores li {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
  }

  .split, .bar {
    display: flex;
    width: 72px;
    height: 7px;
    overflow: hidden;
    background: color-mix(in srgb, var(--line) 70%, white);
  }

  .bar { width: 100%; margin-top: 2px; }
  .split i, .bar i { display: block; height: 100%; }
  .bar i:first-child { background: var(--canopy); }
  .split .fp, .bar .fp { background: #f4f8f9; box-shadow: inset 0 0 0 1px #b42318; }
  .bar .wrong { background: #8a5a00; }
  .split .fn, .bar .fn { background: repeating-linear-gradient(-45deg, #8a5a00, #8a5a00 1px, transparent 1px, transparent 3px); }

  .num {
    font-variant-numeric: tabular-nums;
    color: var(--mist);
  }

  .score {
    position: absolute;
    z-index: 2;
    left: 6px;
    top: 6px;
    padding: 2px 6px;
    background: rgba(18, 32, 36, 0.72);
    color: var(--sheet);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }

  .problem { color: #8c2f2f; font-size: 13px; margin: 8px 18px; }
  .empty { margin: 24px 18px; color: var(--mist); }

  @keyframes spin { to { transform: rotate(360deg); } }

  @media (prefers-reduced-motion: reduce) {
    .spin { animation: none; }
  }
</style>
