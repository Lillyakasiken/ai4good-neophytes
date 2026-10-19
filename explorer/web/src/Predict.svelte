<script>
  import CopyButton from './CopyButton.svelte';
  import Zoom from './Zoom.svelte';

  let { tileId, thumb, mask } = $props();

  let checkpoints = $state([]);
  let listError = $state('');
  let runs = $state([]);
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
  const EXPLAINS = [
    ['gradcam', 'Grad-CAM (Features)'],
    ['saliency', 'Input Saliency (Pixels)'],
    ['confidence', 'Class Confidence (Probability)'],
    ['entropy', 'Uncertainty (Entropy)'],
    ['none', 'Off'],
  ];
  const EXPLAIN_HINTS = {
    gradcam: 'Bright areas are encoder features that raised the score for the chosen class.',
    saliency: 'Bright areas are input pixels that raised that score. A large tile is reduced to 1024 px for this pass.',
    confidence: 'Bright areas are where the chosen class is probable. With all species, where any species is probable.',
    entropy: 'Bright areas are where the model is split between classes.',
  };
  let ckpt = $state('');
  let explain = $state('gradcam');
  let targetClass = $state('');
  let maskStyle = $state('spotlight');
  let importanceOpacity = $state(0.85);
  let spotUrl = $state('');
  let spotFor = $state('');
  let labelOpacity = $state(0.55);
  let predOpacity = $state(0.75);
  let busy = $state(false);
  let error = $state('');
  let result = $state(null);
  let resultFor = $state('');
  let zoomOpen = $state(false);

  let ranked = $derived.by(() => rankCheckpoints(checkpoints, runs, sortBy));
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
  let wantsTarget = $derived(explain !== 'none' && explain !== 'entropy');
  let chosenTarget = $derived.by(() => {
    if (!wantsTarget || targetClass === '') return null;
    const value = Number(targetClass);
    return Number.isFinite(value) ? value : null;
  });
  let classChoices = $derived(shown?.classes ?? []);
  let importanceFresh = $derived.by(() => {
    if (!shown?.importance || shown.explain !== explain || explain === 'none') return false;
    if (explain === 'entropy') return true;
    return (shown.target_class ?? null) === chosenTarget;
  });
  let rawImportance = $derived(importanceFresh ? `data:image/png;base64,${shown.importance}` : '');
  let importanceUrl = $derived.by(() => {
    if (!rawImportance) return '';
    if (maskStyle === 'grayscale') return rawImportance;
    return spotFor === rawImportance ? spotUrl : '';
  });
  let importanceHint = $derived.by(() => {
    const base = EXPLAIN_HINTS[explain] || '';
    if (!base) return '';
    const style = maskStyle === 'spotlight'
      ? ' Unimportant parts of the photo are dimmed.'
      : ' White is important and black is not.';
    const reduced = shown?.importance_reduced && importanceFresh
      ? ' This map was computed on a smaller copy because the GPU was full.'
      : '';
    const stale = shown && !importanceFresh
      ? ' Run again to refresh this map.'
      : '';
    return base + style + reduced + stale;
  });

  $effect(() => {
    tileId;
    result = null;
    resultFor = '';
    error = '';
    spotUrl = '';
    spotFor = '';
    zoomOpen = false;
  });

  $effect(() => {
    const src = rawImportance;
    if (maskStyle !== 'spotlight' || !src) return;
    let dead = false;
    paintSpotlight(src).then((url) => {
      if (dead) return;
      spotUrl = url;
      spotFor = src;
    }).catch(() => {
      if (dead) return;
      spotUrl = '';
      spotFor = '';
    });
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

  function sameClass(id) {
    return targetClass !== '' && Number(targetClass) === Number(id);
  }

  function pickClass(id) {
    if (busy || !wantsTarget) return;
    targetClass = sameClass(id) ? '' : String(id);
  }

  function paintSpotlight(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = image.data;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i + 3] === 0) continue;
          const gray = data[i];
          data[i] = 0;
          data[i + 1] = 0;
          data[i + 2] = 0;
          data[i + 3] = 255 - gray;
        }
        ctx.putImageData(image, 0, 0);
        resolve(canvas.toDataURL('image/png'));
      };
      img.onerror = () => reject(new Error('importance failed to load'));
      img.src = src;
    });
  }

  async function run() {
    if (!ckpt || busy) return;
    const key = `${tileId}:${ckpt}`;
    busy = true;
    error = '';
    try {
      const res = await fetch('/api/tiles/' + tileId + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ckpt,
          explain,
          target_class: chosenTarget,
        }),
      });
      if (!res.ok) throw new Error(await readError(res));
      result = await res.json();
      resultFor = key;
    } catch (err) {
      error = String(err.message || err);
    } finally {
      busy = false;
    }
  }

  function percent(value) {
    if (value == null || Number.isNaN(value)) return '';
    return `${Math.round(value * 1000) / 10}%`;
  }
</script>

{#snippet layers(alt)}
  <img src={imageUrl} alt={alt} draggable="false" />
  {#if labelUrl}
    <img class="mask" src={labelUrl} alt="" style:opacity={labelOpacity} draggable="false" />
  {/if}
  {#if overlayUrl}
    <img class="mask pred" src={overlayUrl} alt="" style:opacity={predOpacity} draggable="false" />
  {/if}
  {#if importanceUrl}
    <img class="mask importance" src={importanceUrl} alt="" style:opacity={importanceOpacity} draggable="false" />
  {/if}
{/snippet}

<div class="lab">
  <div class="preview">
    <div class="shot">
      <button type="button" class="stage" aria-busy={busy} aria-label="Enlarge prediction" onclick={() => (zoomOpen = true)}>
        {@render layers('')}
        {#if busy}
          <div class="predicting" role="status">Running prediction</div>
        {/if}
      </button>
      <CopyButton overlay />
    </div>
    <Zoom open={zoomOpen} onclose={() => (zoomOpen = false)} label="Prediction" bind:opacity={labelOpacity}>
      {#snippet toolbar()}
        <label class="zoom-slider">
          Prediction
          <input type="range" min="0" max="1" step="0.05" bind:value={predOpacity} disabled={!overlayUrl} aria-label="Prediction opacity" />
        </label>
        <label class="zoom-slider">
          Importance
          <input type="range" min="0" max="1" step="0.05" bind:value={importanceOpacity} disabled={!importanceUrl} aria-label="Importance opacity" />
        </label>
      {/snippet}
      {@render layers('Prediction')}
    </Zoom>
    <label class="slider">
      Labels
      <input type="range" min="0" max="1" step="0.05" bind:value={labelOpacity} />
    </label>
    <label class="slider">
      Prediction
      <input type="range" min="0" max="1" step="0.05" bind:value={predOpacity} disabled={!overlayUrl} />
    </label>
    <label class="slider">
      Importance
      <input type="range" min="0" max="1" step="0.05" bind:value={importanceOpacity} disabled={!importanceUrl} />
    </label>
    <div class="styles" role="group" aria-label="Importance display">
      <button type="button" class:on={maskStyle === 'spotlight'} aria-pressed={maskStyle === 'spotlight'} disabled={!rawImportance} onclick={() => (maskStyle = 'spotlight')}>Spotlight</button>
      <button type="button" class:on={maskStyle === 'grayscale'} aria-pressed={maskStyle === 'grayscale'} disabled={!rawImportance} onclick={() => (maskStyle = 'grayscale')}>Grayscale</button>
    </div>
    {#if importanceHint}
      <p class="hint">{importanceHint}</p>
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
              title={item.exp_name}
              onclick={() => choose(item.path)}
            >
              <span class="id">
                <span class="name">{titleOf(item)}</span>
                <span class="sub">{item.kind}{item.ds2 ? ' · half res' : ''} · {item.file}</span>
              </span>
              <span class="scores">
                <span>test {f1(item, 'test_f1_species')}</span>
                <span>val {f1(item, 'val_f1_species')}</span>
              </span>
            </button>
          </li>
        {/each}
      </ul>
    {:else if checkpoints.length}
      <p class="hint">No checkpoint matches.</p>
    {/if}
    {#if scoresNote}
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
    <label class="field">
      Class
      <select bind:value={targetClass} disabled={busy || !ckpt || !wantsTarget}>
        <option value="">All predicted species</option>
        {#each classChoices as spec (spec.id)}
          <option value={String(spec.id)}>{spec.name}</option>
        {/each}
      </select>
    </label>
    <button type="button" class="run" onclick={run} disabled={busy || !ckpt}>
      {busy ? 'Running…' : 'Run'}
    </button>
    <div class="hist-head">
      <h3>Predicted classes</h3>
    </div>
    {#if shown?.counts?.length}
      <ul class="hits">
        {#each shown.counts as item (item.id)}
          <li>
            <button
              type="button"
              class:on={wantsTarget && sameClass(item.id)}
              aria-pressed={wantsTarget && sameClass(item.id)}
              disabled={busy || !wantsTarget}
              title={wantsTarget ? 'Use this class for the importance map' : ''}
              onclick={() => pickClass(item.id)}
            >
              <i class="swatch" style:--swatch={item.color}></i>
              <span class="binomial">{item.name}</span>
              <span class="num">{item.px.toLocaleString()} px</span>
            </button>
          </li>
        {/each}
      </ul>
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

  .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
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

  .run {
    border: 0;
    border-radius: 2px;
    background: var(--canopy);
    color: var(--sheet);
    padding: 7px 14px;
  }

  .run:disabled { opacity: 0.55; }

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

  .hits button:disabled {
    opacity: 1;
    cursor: default;
  }

  .styles {
    display: flex;
    gap: 6px;
    margin: 0 0 8px;
  }

  .styles button {
    border: 1px solid var(--line);
    background: transparent;
    border-radius: 2px;
    padding: 3px 8px;
    font-size: 12px;
    color: var(--ink);
  }

  .styles button.on {
    background: color-mix(in srgb, var(--canopy) 12%, white);
    border-color: var(--canopy);
  }

  .styles button:disabled {
    opacity: 0.45;
    cursor: default;
  }

  .num {
    margin-left: auto;
    color: var(--mist);
    font-variant-numeric: tabular-nums;
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
    padding: 4px 8px;
    background: rgba(18, 32, 36, 0.72);
    color: var(--sheet);
    font-size: 12px;
  }

  .pred { z-index: 1; }

  @media (max-width: 720px) {
    .lab { grid-template-columns: 1fr; }
  }
</style>
