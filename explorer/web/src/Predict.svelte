<script>
  import CopyButton from './CopyButton.svelte';
  import Zoom from './Zoom.svelte';
  import { paintImportance, paintIsolate } from './isolate.js';

  let { tileId, thumb, mask, focusColor = $bindable(''), onopenmodel = null } = $props();

  let checkpoints = $state([]);
  let checkpointsLoading = $state(true);
  let listError = $state('');
  let runs = $state([]);
  let scoresLoading = $state(true);
  let scoresNote = $state('');
  let query = $state('');
  let sortBy = $state('test');
  let touched = $state(false);

  const SORTS = [
    ['test', 'Test F1'],
    ['val', 'Val F1'],
    ['newest', 'Newest'],
    ['name', 'Name'],
  ];
  const VIEWS = [
    ['photo', 'Photo'],
    ['labels', 'Labels'],
    ['prediction', 'Prediction'],
    ['compare', 'Compare'],
    ['importance', 'Importance'],
  ];
  const EXPLAINS = [
    ['gradcam', 'Grad-CAM (Features)'],
    ['saliency', 'Input Saliency (Pixels)'],
    ['confidence', 'Class Confidence (Probability)'],
    ['entropy', 'Uncertainty (Entropy)'],
    ['none', 'Off'],
  ];
  const EXPLAIN_HINTS = {
    gradcam: 'Warm color marks encoder features that raised the score for the chosen class.',
    saliency: 'Warm color marks input pixels that raised that score. A large tile is reduced to 1024 px for this pass.',
    confidence: 'Warm color marks where the chosen class is probable. With all species, where any species is probable.',
    entropy: 'Warm color marks where the model is split between classes.',
  };
  let ckpt = $state('');
  let explain = $state('gradcam');
  let targetClass = $state('');
  let maskStyle = $state('heat');
  let view = $state('prediction');
  let compareKind = $state('wipe');
  let strength = $state(0.65);
  let wipe = $state(0.5);
  let styledUrl = $state('');
  let styledFor = $state('');
  let labelPaint = $state('');
  let labelPaintFor = $state('');
  let predPaint = $state('');
  let predPaintFor = $state('');
  let comparePaint = $state('');
  let comparePaintFor = $state('');
  let busy = $state(false);
  let loadingSaved = $state(false);
  let error = $state('');
  let result = $state(null);
  let resultFor = $state('');
  let zoomOpen = $state(false);
  let seamDrag = null;
  let seenKey = '';
  let fetchGen = 0;
  let training = $state(false);
  let cacheCount = $state(0);
  let cacheBytes = $state(0);
  let savedFor = $state('');
  let savedCkpts = $state([]);
  let gpuNote = $state('');

  let ranked = $derived.by(() => rankCheckpoints(checkpoints, runs, sortBy));
  let savedSet = $derived(savedFor === tileId ? new Set(savedCkpts) : new Set());
  let visible = $derived.by(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return ranked;
    return ranked.filter((item) => {
      const hay = `${item.exp_name} ${item.file} ${item.kind}`.toLowerCase();
      return hay.includes(needle);
    });
  });
  let selected = $derived(ranked.find((item) => item.path === ckpt) ?? null);
  let activeKey = $derived(`${tileId}:${ckpt}`);
  let shown = $derived(result && resultFor === activeKey ? result : null);
  let imageUrl = $derived(shown?.image ? `data:image/jpeg;base64,${shown.image}` : thumb);
  let labelUrl = $derived(shown?.label ? `data:image/png;base64,${shown.label}` : mask);
  let overlayUrl = $derived(shown?.overlay ? `data:image/png;base64,${shown.overlay}` : '');
  let comparisonUrl = $derived(shown?.comparison ? `data:image/png;base64,${shown.comparison}` : '');
  let wantsTarget = $derived(explain !== 'none' && explain !== 'entropy');
  let chosenTarget = $derived.by(() => {
    if (!wantsTarget || targetClass === '') return null;
    const value = Number(targetClass);
    return Number.isFinite(value) ? value : null;
  });
  let importanceFresh = $derived.by(() => {
    if (!shown?.importance || shown.explain !== explain || explain === 'none') return false;
    if (explain === 'entropy') return true;
    return (shown.target_class ?? null) === chosenTarget;
  });
  let rawImportance = $derived(shown?.importance ? `data:image/png;base64,${shown.importance}` : '');
  let importanceUrl = $derived(styledFor === `${rawImportance}|${maskStyle}` ? styledUrl : '');
  let shownView = $derived.by(() => {
    if ((view === 'prediction' || view === 'compare') && !overlayUrl) return 'labels';
    if (view === 'importance' && (!rawImportance || explain === 'none')) return overlayUrl ? 'prediction' : 'labels';
    if (view === 'compare' && compareKind === 'agreement' && !comparisonUrl) return 'compare';
    return view;
  });
  let wipeOn = $derived(shownView === 'compare' && (compareKind === 'wipe' || !comparisonUrl));
  let agreementOn = $derived(shownView === 'compare' && compareKind === 'agreement' && !!comparisonUrl);
  let layerOpacity = $derived(wipeOn ? 1 : strength);
  let dimPhoto = $derived(shownView === 'labels' || shownView === 'prediction' || shownView === 'compare');
  let labelShown = $derived.by(() => {
    if (!focusColor) return labelUrl;
    return labelPaintFor === `${labelUrl}|${focusColor}` ? labelPaint : '';
  });
  let predShown = $derived.by(() => {
    if (!focusColor) return overlayUrl;
    return predPaintFor === `${overlayUrl}|${focusColor}` ? predPaint : '';
  });
  let compareShown = $derived.by(() => {
    if (!focusColor) return comparisonUrl;
    return comparePaintFor === `${comparisonUrl}|${focusColor}` ? comparePaint : '';
  });
  let wipeClip = $derived(wipeOn ? `inset(0 ${(1 - wipe) * 100}% 0 0)` : undefined);
  let labelClip = $derived(wipeOn ? `inset(0 0 0 ${wipe * 100}%)` : undefined);
  let importanceHint = $derived.by(() => {
    const base = EXPLAIN_HINTS[explain] || '';
    if (!base || shownView !== 'importance') return '';
    const style = maskStyle === 'spotlight'
      ? ' Unimportant parts of the photo are dimmed.'
      : maskStyle === 'grayscale'
        ? ' White is important and black is not.'
        : ' Clear areas did not matter for the score.';
    const reduced = shown?.importance_reduced && importanceFresh
      ? ' This map was computed on a smaller copy because the GPU was full.'
      : '';
    const stale = shown && !importanceFresh
      ? ' Run again to refresh this map.'
      : '';
    return base + style + reduced + stale;
  });
  let modeNote = $derived.by(() => {
    if (shownView === 'compare' && agreementOn) return '';
    if (wipeOn) return 'Left is the prediction. Right is the label. Drag the line to move the split.';
    if (shownView === 'importance') return '';
    if (focusColor) return 'Other classes are faded. Click the class again to show all.';
    return '';
  });
  let sliderLabel = $derived.by(() => {
    if (wipeOn) return 'Split';
    if (shownView === 'labels') return 'Labels';
    if (shownView === 'importance') return 'Importance';
    if (agreementOn) return 'Agreement';
    if (shownView === 'prediction') return 'Prediction';
    return 'Overlay';
  });

  $effect(() => {
    const id = tileId;
    const path = ckpt;
    const key = `${id}:${path}`;
    if (seenKey && seenKey !== key) focusColor = '';
    seenKey = key;
    result = null;
    resultFor = '';
    error = '';
    styledUrl = '';
    styledFor = '';
    zoomOpen = false;
    gpuNote = '';
    if (!id || !path) return;
    const gen = ++fetchGen;
    const ctrl = new AbortController();
    loadingSaved = true;
    fetch(`/api/tiles/${id}/predict/saved?ckpt=${encodeURIComponent(path)}`, { signal: ctrl.signal })
      .then(async (res) => {
        if (res.status === 404) return null;
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        if (gen !== fetchGen || !data) return;
        result = data;
        resultFor = key;
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
      })
      .finally(() => {
        if (gen === fetchGen) loadingSaved = false;
      });
    return () => {
      ctrl.abort();
    };
  });

  $effect(() => {
    const src = rawImportance;
    const style = maskStyle;
    if (!src) return;
    let dead = false;
    paintImportance(src, style).then((url) => {
      if (dead) return;
      styledUrl = url;
      styledFor = `${src}|${style}`;
    }).catch(() => {
      if (dead) return;
      styledUrl = '';
      styledFor = '';
    });
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    const src = labelUrl;
    const color = focusColor;
    if (!src || !color) return;
    let dead = false;
    paintIsolate(src, color).then((url) => {
      if (dead) return;
      labelPaint = url;
      labelPaintFor = `${src}|${color}`;
    }).catch(() => {});
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    const src = overlayUrl;
    const color = focusColor;
    if (!src || !color) return;
    let dead = false;
    paintIsolate(src, color).then((url) => {
      if (dead) return;
      predPaint = url;
      predPaintFor = `${src}|${color}`;
    }).catch(() => {});
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    const src = comparisonUrl;
    const color = focusColor;
    if (!src || !color) return;
    let dead = false;
    paintIsolate(src, color).then((url) => {
      if (dead) return;
      comparePaint = url;
      comparePaintFor = `${src}|${color}`;
    }).catch(() => {});
    return () => {
      dead = true;
    };
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
        if (!touched) ckpt = bestPath(rankCheckpoints(checkpoints, runs));
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
        scoresNote = data.available ? '' : (data.detail || 'Scores are unavailable.');
        if (!touched) ckpt = bestPath(rankCheckpoints(checkpoints, runs));
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        scoresNote = String(err.message || err);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) scoresLoading = false;
      });
    return () => ctrl.abort();
  });

  $effect(() => {
    const id = tileId;
    const ctrl = new AbortController();
    refreshMarks(id, ctrl.signal);
    return () => ctrl.abort();
  });

  $effect(() => {
    const ctrl = new AbortController();
    refreshCache(ctrl.signal);
    fetch('/api/gpu', { signal: ctrl.signal })
      .then(async (res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) training = !!data.training;
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
      });
    return () => ctrl.abort();
  });

  function rankCheckpoints(rows, wandbRuns, sortBy = 'test') {
    const byName = new Map((wandbRuns ?? []).map((run) => [run.name, run.metrics || {}]));
    return rows.map((item) => ({
      ...item,
      metrics: byName.get(item.exp_name) || {},
    })).sort((a, b) => compareRows(a, b, sortBy));
  }

  function compareRows(a, b, sortBy) {
    if (sortBy === 'name') {
      const byName = titleOf(a).localeCompare(titleOf(b));
      if (byName) return byName;
    } else if (sortBy === 'newest') {
      const byTime = (b.mtime || 0) - (a.mtime || 0);
      if (byTime) return byTime;
    } else if (sortBy === 'val') {
      const byVal = compareScore(a, b, 'val_f1_species');
      if (byVal) return byVal;
      const byTest = compareScore(a, b, 'test_f1_species');
      if (byTest) return byTest;
    } else {
      const byTest = compareScore(a, b, 'test_f1_species');
      if (byTest) return byTest;
      const byVal = compareScore(a, b, 'val_f1_species');
      if (byVal) return byVal;
    }
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

  function bestPath(rows) {
    const best = rows.find((item) => item.kind === 'best' && item.metrics?.test_f1_species != null);
    return (best ?? rows[0])?.path ?? '';
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

  function titleOf(item) {
    return item.exp_name.replace(/^\d{8}_[\d.]+_/, '') || item.exp_name;
  }

  function f1(item, key) {
    const value = item.metrics?.[key];
    if (value == null || Number.isNaN(value)) return '—';
    return Number(value).toFixed(3);
  }

  function choose(path) {
    touched = true;
    ckpt = path;
  }

  function pickClass(id, color) {
    if (busy) return;
    const on = focusColor === color;
    focusColor = on ? '' : color;
    if (!wantsTarget) return;
    targetClass = on ? '' : String(id);
  }

  function viewDisabled(mode) {
    if (mode === 'prediction' || mode === 'compare') return !overlayUrl;
    if (mode === 'importance') return explain === 'none' || !rawImportance;
    return false;
  }

  function openZoom(event) {
    if (event.target.closest('.seam')) return;
    zoomOpen = true;
  }

  function moveSeam(event, stage) {
    const rect = stage.getBoundingClientRect();
    if (!rect.width) return;
    const x = (event.clientX - rect.left) / rect.width;
    wipe = Math.min(1, Math.max(0, x));
  }

  function onSeamDown(event) {
    event.preventDefault();
    event.stopPropagation();
    const stage = event.currentTarget.closest('.stage');
    if (!stage) return;
    seamDrag = { id: event.pointerId, stage };
    event.currentTarget.setPointerCapture(event.pointerId);
    moveSeam(event, stage);
  }

  function onSeamMove(event) {
    if (!seamDrag || event.pointerId !== seamDrag.id) return;
    moveSeam(event, seamDrag.stage);
  }

  function onSeamUp(event) {
    if (!seamDrag || event.pointerId !== seamDrag.id) return;
    seamDrag = null;
  }

  async function refreshMarks(id, signal) {
    if (!id) {
      savedCkpts = [];
      savedFor = '';
      return;
    }
    try {
      const res = await fetch(`/api/tiles/${id}/predict/marks`, { signal });
      if (!res.ok) return;
      const data = await res.json();
      if (signal?.aborted) return;
      savedFor = id;
      savedCkpts = data.checkpoints ?? [];
    } catch (err) {
      if (err.name === 'AbortError') return;
    }
  }

  async function refreshCache(signal) {
    try {
      const res = await fetch('/api/predictions/cache', { signal });
      if (!res.ok) return;
      const data = await res.json();
      cacheCount = data.count ?? 0;
      cacheBytes = data.bytes ?? 0;
    } catch (err) {
      if (err.name === 'AbortError') return;
    }
    refreshMarks(tileId, signal);
  }

  function applyStats(data) {
    cacheCount = data.count ?? 0;
    cacheBytes = data.bytes ?? 0;
  }

  async function run() {
    if (!ckpt || busy) return;
    const key = `${tileId}:${ckpt}`;
    const rerun = !!shown;
    const gen = ++fetchGen;
    loadingSaved = false;
    busy = true;
    error = '';
    gpuNote = '';
    try {
      const res = await fetch('/api/tiles/' + tileId + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ckpt,
          explain,
          target_class: chosenTarget,
          force: rerun,
        }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const data = await res.json();
      if (gen !== fetchGen) return;
      result = data;
      resultFor = key;
      refreshCache();
    } catch (err) {
      error = String(err.message || err);
    } finally {
      busy = false;
    }
  }

  async function clearSaved() {
    if (!tileId || !ckpt || busy) return;
    fetchGen += 1;
    loadingSaved = false;
    error = '';
    try {
      const res = await fetch(`/api/tiles/${tileId}/predict/saved?ckpt=${encodeURIComponent(ckpt)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await readError(res));
      applyStats(await res.json());
      result = null;
      resultFor = '';
    } catch (err) {
      error = String(err.message || err);
    }
  }

  async function clearAll() {
    if (busy || !cacheCount) return;
    if (!confirm(`Delete all ${cacheCount} saved predictions?`)) return;
    fetchGen += 1;
    loadingSaved = false;
    error = '';
    try {
      const res = await fetch('/api/predictions/cache', { method: 'DELETE' });
      if (!res.ok) throw new Error(await readError(res));
      applyStats(await res.json());
      result = null;
      resultFor = '';
    } catch (err) {
      error = String(err.message || err);
    }
  }

  async function freeGpu() {
    if (busy || training) return;
    error = '';
    gpuNote = '';
    try {
      const res = await fetch('/api/predict/release', { method: 'POST' });
      if (!res.ok) throw new Error(await readError(res));
      const data = await res.json();
      gpuNote = data.released ? 'Prediction model unloaded. The next run loads it again.' : 'No prediction model was loaded.';
    } catch (err) {
      error = String(err.message || err);
    }
  }

  function percent(value) {
    if (value == null || Number.isNaN(value)) return '';
    return `${Math.round(value * 1000) / 10}%`;
  }

  function sizeLabel(bytes) {
    if (!bytes) return '0 MB';
    if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  }

  function savedWhen(ts) {
    if (!ts) return 'Saved';
    const date = new Date(ts * 1000);
    if (Number.isNaN(date.getTime())) return 'Saved';
    return `Saved · ${date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}`;
  }

  function coverageNote() {
    const rows = shown?.counts ?? [];
    if (!rows.some((row) => row.fp != null)) return '';
    let predPx = 0;
    let labelPx = 0;
    for (const row of rows) {
      predPx += row.px || 0;
      labelPx += (row.tp || 0) + (row.fn || 0);
    }
    const gap = predPx - labelPx;
    if (gap > 0) return `The prediction covers ${gap.toLocaleString()} more species pixels than the label.`;
    if (gap < 0) return `The label covers ${(-gap).toLocaleString()} more species pixels than the prediction.`;
    return 'The prediction and the label cover the same number of species pixels.';
  }

  function share(part, item) {
    const total = (item.tp || 0) + (item.fp || 0) + (item.fn || 0);
    if (!total) return 0;
    return (100 * part) / total;
  }
</script>

{#snippet spinner()}
  <span class="spin" aria-hidden="true"></span>
{/snippet}

{#snippet agreementKey(onDark)}
  <ul class="key" class:ondark={onDark} aria-label="Agreement marks">
    <li><i class="mark both"></i> Both</li>
    <li><i class="mark extra"></i> White edge: prediction only</li>
    <li><i class="mark miss"></i> Outline: label only</li>
    <li><i class="mark wrong"></i> Colored edge: prediction, edge is the label</li>
  </ul>
{/snippet}

{#snippet viewBar(onDark)}
  <div class="views" class:ondark={onDark} role="group" aria-label="Prediction view">
    {#each VIEWS as [mode, label]}
      {@const waiting = loadingSaved && (mode === 'prediction' || mode === 'compare' || mode === 'importance')}
      <button
        type="button"
        class:on={shownView === mode}
        class:wait={waiting}
        aria-pressed={shownView === mode}
        aria-busy={waiting}
        disabled={viewDisabled(mode)}
        onclick={() => (view = mode)}
      >{#if waiting}{@render spinner()}{/if}{label}</button>
    {/each}
  </div>
{/snippet}

{#snippet layers(alt)}
  <img class="photo" class:dim={dimPhoto} src={imageUrl} alt={alt} draggable="false" />
  {#if labelShown && (shownView === 'labels' || wipeOn)}
    <img class="mask" src={labelShown} alt="" style:opacity={layerOpacity} style:clip-path={labelClip} draggable="false" />
  {/if}
  {#if predShown && (shownView === 'prediction' || wipeOn)}
    <img class="mask pred" src={predShown} alt="" style:opacity={layerOpacity} style:clip-path={wipeClip} draggable="false" />
  {/if}
  {#if compareShown && agreementOn}
    <img class="mask pred" src={compareShown} alt="" style:opacity={layerOpacity} draggable="false" />
  {/if}
  {#if importanceUrl && shownView === 'importance'}
    <img class="mask importance" src={importanceUrl} alt="" style:opacity={strength} draggable="false" />
  {/if}
  {#if wipeOn}
    {#if wipe > 0.02}<span class="badge left">Prediction</span>{/if}
    {#if wipe < 0.98}<span class="badge right">Labels</span>{/if}
    <div
      class="seam"
      style:left="{wipe * 100}%"
      aria-hidden="true"
      onpointerdown={onSeamDown}
      onpointermove={onSeamMove}
      onpointerup={onSeamUp}
      onpointercancel={onSeamUp}
    ></div>
  {/if}
{/snippet}

<div class="lab">
  <div class="preview">
    <div class="shot">
      <button type="button" class="stage" aria-busy={busy || loadingSaved} aria-label="Enlarge prediction" onclick={openZoom}>
        {@render layers('')}
        {#if busy}
          <div class="predicting" role="status">{@render spinner()} Running prediction</div>
        {:else if loadingSaved}
          <div class="predicting quiet" role="status">{@render spinner()} Loading saved prediction</div>
        {/if}
        {#if shown?.saved_at && !busy}
          <span class="saved">{savedWhen(shown.saved_at)}</span>
        {/if}
      </button>
      <CopyButton overlay />
    </div>
    <Zoom open={zoomOpen} onclose={() => (zoomOpen = false)} label="Prediction">
      {#snippet toolbar()}
        {@render viewBar(true)}
        {#if wipeOn}
          <label class="zoom-slider">
            Split
            <input type="range" min="0" max="1" step="0.01" bind:value={wipe} aria-label="Comparison split" />
          </label>
        {:else if shownView !== 'photo'}
          <label class="zoom-slider">
            {sliderLabel}
            <input type="range" min="0" max="1" step="0.05" bind:value={strength} aria-label="{sliderLabel} opacity" />
          </label>
          {#if agreementOn}
            {@render agreementKey(true)}
          {/if}
        {/if}
      {/snippet}
      {@render layers('Prediction')}
    </Zoom>
    {@render viewBar(false)}
    {#if shownView === 'compare' && overlayUrl}
      <div class="styles" role="group" aria-label="Comparison display">
        <button type="button" class:on={compareKind === 'wipe'} aria-pressed={compareKind === 'wipe'} onclick={() => (compareKind = 'wipe')}>Wipe</button>
        <button type="button" class:on={compareKind === 'agreement'} aria-pressed={compareKind === 'agreement'} disabled={!comparisonUrl} onclick={() => (compareKind = 'agreement')}>Agreement</button>
      </div>
    {/if}
    {#if shownView !== 'photo'}
      <label class="slider">
        {sliderLabel}
        {#if wipeOn}
          <input type="range" min="0" max="1" step="0.01" bind:value={wipe} aria-label="Comparison split" />
        {:else}
          <input type="range" min="0" max="1" step="0.05" bind:value={strength} aria-label="{sliderLabel} opacity" />
        {/if}
        <span class="readout">{wipeOn ? `${Math.round(wipe * 100)}%` : `${Math.round(strength * 100)}%`}</span>
      </label>
    {/if}
    {#if shownView === 'importance'}
      <div class="styles" role="group" aria-label="Importance display">
        <button type="button" class:on={maskStyle === 'heat'} aria-pressed={maskStyle === 'heat'} disabled={!rawImportance} onclick={() => (maskStyle = 'heat')}>Heat</button>
        <button type="button" class:on={maskStyle === 'spotlight'} aria-pressed={maskStyle === 'spotlight'} disabled={!rawImportance} onclick={() => (maskStyle = 'spotlight')}>Spotlight</button>
        <button type="button" class:on={maskStyle === 'grayscale'} aria-pressed={maskStyle === 'grayscale'} disabled={!rawImportance} onclick={() => (maskStyle = 'grayscale')}>Grayscale</button>
      </div>
    {/if}
    {#if importanceHint}
      <p class="hint">{importanceHint}</p>
    {/if}
    {#if agreementOn}
      {@render agreementKey(false)}
      {#if coverageNote()}
        <p class="hint">{coverageNote()}</p>
      {/if}
    {/if}
    {#if modeNote}
      <p class="hint">{modeNote}</p>
    {/if}
    {#if shown?.agreement != null || shown?.agreement_species != null}
      <p class="agree">
        {#if shown?.agreement != null}
          <span>Agrees with the label on {percent(shown.agreement)} of labelled pixels, including background.</span>
        {/if}
        {#if shown?.agreement_species != null}
          <span>Agrees on {percent(shown.agreement_species)} of labelled species pixels.</span>
        {/if}
      </p>
    {/if}
    {#if selected?.ds2 || shown?.ds2}
      <p class="hint">Same pass as test.py for a half-resolution run: the tile is scaled down before the forward pass.</p>
    {:else}
      <p class="hint">Same pass as test.py: the whole tile, at the file's own resolution.</p>
    {/if}
  </div>

  <div class="controls">
    {#if onopenmodel}
      <button type="button" class="model-link" onclick={() => onopenmodel(tileId)}>
        See how this model reads the photo
      </button>
    {/if}
    <div class="finder">
      <label class="field grow">
        Checkpoint
        <input
          type="search"
          placeholder="Search runs"
          bind:value={query}
          disabled={busy || !checkpoints.length}
          aria-controls="checkpoint-list"
        />
      </label>
      <label class="field sort">
        Sort
        <select bind:value={sortBy} disabled={busy || !checkpoints.length}>
          {#each SORTS as [value, label]}
            <option {value}>{label}</option>
          {/each}
        </select>
      </label>
    </div>
    {#if visible.length}
      <ul class="models" id="checkpoint-list" role="listbox" aria-label="Checkpoints">
        {#each visible as item (item.path)}
          <li>
            <button
              type="button"
              role="option"
              class:on={item.path === ckpt}
              aria-selected={item.path === ckpt}
              disabled={busy}
              title={savedSet.has(item.path) ? `${item.exp_name}. Saved prediction for this tile.` : item.exp_name}
              onclick={() => choose(item.path)}
            >
              <span class="id">
                <span class="name-row">
                  <span class="name">{titleOf(item)}</span>
                  {#if savedSet.has(item.path)}<span class="kept">Saved</span>{/if}
                </span>
                <span class="sub">{item.kind}{item.ds2 ? ' · half res' : ''} · {item.file}</span>
              </span>
              <span class="scores">
                {#if scoresLoading}
                  {@render spinner()}
                {:else}
                  <span>test {f1(item, 'test_f1_species')}</span>
                  <span>val {f1(item, 'val_f1_species')}</span>
                {/if}
              </span>
            </button>
          </li>
        {/each}
      </ul>
    {:else if checkpointsLoading}
      <p class="hint loading" role="status">{@render spinner()} Loading checkpoints</p>
    {:else if checkpoints.length}
      <p class="hint">No checkpoint matches.</p>
    {/if}
    {#if scoresLoading}
      <p class="hint loading" role="status">{@render spinner()} Loading test and val F1</p>
    {:else if scoresNote}
      <p class="hint">{scoresNote}</p>
    {:else if ranked.some((item) => item.metrics?.test_f1_species != null)}
      <p class="hint">test and val are that run's species F1. The test score is from the best checkpoint.</p>
    {/if}
    <label class="field">
      Importance
      <select bind:value={explain} disabled={busy || !ckpt}>
        {#each EXPLAINS as [value, label]}
          <option {value}>{label}</option>
        {/each}
      </select>
    </label>
    <div class="actions">
      <button type="button" class="run" onclick={run} disabled={busy || loadingSaved || !ckpt} aria-busy={busy || loadingSaved}>
        {#if busy || loadingSaved}{@render spinner()}{/if}
        {shown || loadingSaved ? 'Re-run' : 'Run'}
      </button>
      <button type="button" onclick={clearSaved} disabled={busy || !shown}>Clear saved</button>
      <button type="button" onclick={clearAll} disabled={busy || !cacheCount}>Clear all saved ({cacheCount} · {sizeLabel(cacheBytes)})</button>
    </div>
    <div class="actions">
      <button type="button" onclick={freeGpu} disabled={busy || training} title={training ? 'A training job is using the GPU' : 'Unload the prediction model from GPU memory'}>
        Free GPU
      </button>
    </div>
    {#if gpuNote}
      <p class="hint">{gpuNote}</p>
    {/if}
    <div class="hist-head">
      <h3>Predicted classes</h3>
    </div>
    {#if shown?.counts?.length}
      <ul class="hits">
        {#each shown.counts as item (item.id)}
          <li>
            <button
              type="button"
              class:on={focusColor === item.color}
              aria-pressed={focusColor === item.color}
              disabled={busy}
              title={wantsTarget ? 'Show this class and use it for the importance map' : 'Show only this class'}
              onclick={() => pickClass(item.id, item.color)}
            >
              <i class="swatch" style:--swatch={item.color}></i>
              <span class="binomial">{item.name}</span>
              {#if item.tp != null}
                <span class="split" title="{item.tp.toLocaleString()} correct, {item.fp.toLocaleString()} extra, {item.fn.toLocaleString()} missed">
                  <i style:width="{share(item.tp, item)}%" style:background={item.color}></i>
                  <i class="fp" style:width="{share(item.fp, item)}%"></i>
                  <i class="fn" style:width="{share(item.fn, item)}%"></i>
                </span>
              {/if}
              <span class="num">{item.px.toLocaleString()} px</span>
            </button>
          </li>
        {/each}
      </ul>
      {#if shown.counts.some((item) => item.tp != null)}
        <p class="hint">Bar: correct, extra, missed. Click a class to fade the others.</p>
      {/if}
    {:else if shown}
      <p class="hint">No species pixels in this prediction.</p>
    {:else if !checkpoints.length && !listError}
      <p class="hint">No checkpoints in lightning_logs yet.</p>
    {/if}
    {#if listError || error}
      <p class="problem" role="alert">{listError || error}</p>
    {/if}
  </div>
</div>

<style>
  .lab {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(220px, 0.85fr);
    gap: 14px 16px;
    align-items: start;
    margin-bottom: 14px;
  }

  .preview, .controls { min-width: 0; }

  .model-link {
    margin: 0 0 10px;
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--canopy);
    font-size: 13px;
    text-decoration: underline;
    text-underline-offset: 2px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 10px;
    font-size: 12px;
    color: var(--mist);
  }

  .finder {
    display: flex;
    gap: 8px;
    align-items: end;
  }

  .field.grow { flex: 1; min-width: 0; }
  .field.sort { width: 108px; flex: none; }

  input[type='search'], select {
    width: 100%;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: var(--sheet);
    padding: 6px 8px;
    color: var(--ink);
  }

  .models {
    list-style: none;
    margin: 0 0 10px;
    padding: 0;
    max-height: 240px;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: var(--sheet);
  }

  .models button {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 7px 8px;
    border: 0;
    border-bottom: 1px solid var(--line);
    background: transparent;
    color: var(--ink);
    text-align: left;
  }

  .models li:last-child button { border-bottom: 0; }

  .models button.on {
    background: var(--canopy);
    color: var(--sheet);
  }

  .id { min-width: 0; display: flex; flex-direction: column; gap: 2px; }

  .name-row {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }

  .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
  }

  .kept {
    flex: none;
    padding: 1px 5px;
    border-radius: 2px;
    background: color-mix(in srgb, var(--canopy) 16%, white);
    color: var(--canopy);
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .models button.on .kept {
    background: color-mix(in srgb, var(--sheet) 24%, transparent);
    color: var(--sheet);
  }

  .sub {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--mist);
    font-size: 11px;
  }

  .models button.on .sub { color: color-mix(in srgb, var(--sheet) 78%, transparent); }

  .scores {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 2px;
    font-variant-numeric: tabular-nums;
    font-size: 12px;
    white-space: nowrap;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }

  .actions button, .run {
    border: 1px solid var(--line);
    border-radius: 2px;
    background: transparent;
    color: var(--ink);
    padding: 6px 10px;
    font-size: 12px;
  }

  .run {
    border-color: var(--canopy);
    background: var(--canopy);
    color: var(--sheet);
  }

  .actions button:disabled, .run:disabled { opacity: 0.55; }

  .hist-head h3 {
    margin: 14px 0 8px;
    font-size: 13px;
    font-weight: 600;
  }

  .hits {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .hits li { border-bottom: 1px solid var(--line); }

  .hits button {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 4px 2px;
    border: 0;
    background: transparent;
    color: var(--ink);
    text-align: left;
  }

  .hits button.on {
    background: color-mix(in srgb, var(--canopy) 12%, white);
  }

  .views, .styles {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
  }

  .views button, .styles button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--line);
    background: transparent;
    border-radius: 2px;
    padding: 3px 8px;
    font-size: 12px;
    color: var(--ink);
  }

  .views button.on, .styles button.on {
    background: color-mix(in srgb, var(--canopy) 12%, white);
    border-color: var(--canopy);
  }

  .views.ondark button {
    color: var(--sheet);
    border-color: rgba(244, 248, 249, 0.35);
  }

  .views.ondark button.on {
    background: rgba(244, 248, 249, 0.16);
    border-color: var(--sheet);
  }

  .views button:disabled, .styles button:disabled {
    opacity: 0.45;
    cursor: default;
  }

  .views button.wait {
    opacity: 1;
  }

  .split {
    display: flex;
    width: 72px;
    height: 7px;
    overflow: hidden;
    background: color-mix(in srgb, var(--line) 70%, white);
    flex: none;
  }

  .split i { display: block; height: 100%; }
  .split .fp { background: #f4f8f9; box-shadow: inset 0 0 0 1px #b42318; }
  .split .fn { background: repeating-linear-gradient(-45deg, #8a5a00, #8a5a00 1px, transparent 1px, transparent 3px); }

  .num {
    margin-left: auto;
    color: var(--mist);
    font-variant-numeric: tabular-nums;
  }

  .readout {
    min-width: 2.6em;
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
  }

  .key {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 12px;
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: 12px;
    color: var(--ink);
  }

  .key li {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .key.ondark { color: var(--sheet); }

  .mark {
    width: 16px;
    height: 12px;
    flex: none;
    box-sizing: border-box;
  }

  .mark.both { background: color-mix(in srgb, var(--canopy) 55%, white); }

  .mark.extra {
    background: var(--canopy);
    box-shadow: inset 0 0 0 2px #fff, inset 0 0 0 3px rgba(18, 32, 36, 0.45);
  }

  .mark.miss {
    border: 2px solid var(--canopy);
    background: transparent;
  }

  .mark.wrong {
    background: var(--canopy);
    box-shadow: inset 0 0 0 2px #e2256a;
  }

  .agree, .hint { margin: 8px 0 0; color: var(--mist); font-size: 12px; }

  .agree {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .problem { color: #8c2f2f; font-size: 13px; }

  .importance { z-index: 2; }

  .predicting {
    z-index: 3;
    position: absolute;
    left: 8px;
    bottom: 8px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: rgba(18, 32, 36, 0.72);
    color: var(--sheet);
    font-size: 12px;
  }

  .spin {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid color-mix(in srgb, var(--mist) 45%, white);
    border-top-color: var(--canopy);
    border-radius: 50%;
    flex: none;
    box-sizing: border-box;
    animation: spin 0.75s linear infinite;
  }

  .predicting .spin,
  .models button.on .spin,
  .views.ondark .spin {
    border-color: rgba(244, 248, 249, 0.35);
    border-top-color: var(--sheet);
  }

  .hint.loading {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .run {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  @media (prefers-reduced-motion: reduce) {
    .spin { animation: none; opacity: 0.85; }
  }

  .predicting.quiet { bottom: 32px; }

  .saved {
    z-index: 3;
    position: absolute;
    left: 8px;
    top: 8px;
    padding: 3px 7px;
    background: rgba(18, 32, 36, 0.72);
    color: var(--sheet);
    font-size: 11px;
    pointer-events: none;
  }

  .badge {
    position: absolute;
    z-index: 3;
    bottom: 8px;
    padding: 2px 6px;
    background: rgba(18, 32, 36, 0.72);
    color: var(--sheet);
    font-size: 11px;
    pointer-events: none;
  }

  .badge.left { left: 8px; }
  .badge.right { right: 8px; }

  .seam {
    position: absolute;
    z-index: 4;
    top: 0;
    bottom: 0;
    width: 14px;
    margin-left: -7px;
    padding: 0;
    border: 0;
    background: transparent;
    cursor: ew-resize;
    pointer-events: auto;
  }

  .seam::after {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: 6px;
    width: 2px;
    background: #f4f8f9;
    box-shadow: 0 0 0 1px rgba(18, 32, 36, 0.65);
  }

  .pred { z-index: 1; }

  @media (max-width: 720px) {
    .lab { grid-template-columns: 1fr; }
  }
</style>
