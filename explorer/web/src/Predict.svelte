<script>
  import CopyButton from './CopyButton.svelte';

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
  let ckpt = $state('');
  let labelOpacity = $state(0.55);
  let predOpacity = $state(0.75);
  let busy = $state(false);
  let error = $state('');
  let result = $state(null);
  let resultFor = $state('');

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

  $effect(() => {
    tileId;
    result = null;
    resultFor = '';
    error = '';
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

  async function run() {
    if (!ckpt || busy) return;
    const key = `${tileId}:${ckpt}`;
    busy = true;
    error = '';
    try {
      const res = await fetch('/api/tiles/' + tileId + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ckpt }),
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

<div class="lab">
  <div class="preview">
    <div class="shot">
      <div class="stage" aria-busy={busy}>
        <img src={imageUrl} alt="" draggable="false" />
        {#if labelUrl}
          <img class="mask" src={labelUrl} alt="" style:opacity={labelOpacity} draggable="false" />
        {/if}
        {#if overlayUrl}
          <img class="mask pred" src={overlayUrl} alt="" style:opacity={predOpacity} draggable="false" />
        {/if}
        {#if busy}
          <div class="predicting" role="status">Running prediction</div>
        {/if}
      </div>
      <CopyButton overlay />
    </div>
    <label class="slider">
      Labels
      <input type="range" min="0" max="1" step="0.05" bind:value={labelOpacity} />
    </label>
    <label class="slider">
      Prediction
      <input type="range" min="0" max="1" step="0.05" bind:value={predOpacity} disabled={!overlayUrl} />
    </label>
    {#if shown?.agreement != null}
      <p class="agree">Agrees with the label on {percent(shown.agreement)} of labelled pixels.</p>
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
            <i class="swatch" style:--swatch={item.color}></i>
            <span class="binomial">{item.name}</span>
            <span class="num">{item.px.toLocaleString()} px</span>
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

  .hits li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid var(--line);
  }

  .num {
    margin-left: auto;
    color: var(--mist);
    font-variant-numeric: tabular-nums;
  }

  .agree, .hint { margin: 8px 0 0; color: var(--mist); font-size: 12px; }

  .problem { color: #8c2f2f; font-size: 13px; }

  .predicting {
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
