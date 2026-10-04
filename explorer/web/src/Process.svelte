<script>
  import { tick } from 'svelte';

  let { tileId, mask } = $props();

  const FALLBACK_EDGES = [256, 512, 1024];

  let catalog = $state(null);
  let maxEdge = $state(512);
  let chain = $state([]);
  let menuOpen = $state(false);
  let query = $state('');
  let pickerEl = $state(null);
  let searchEl = $state(null);
  let wipe = $state(100);
  let opacity = $state(0.8);
  let original = $state(null);
  let shown = $state(null);
  let shownFor = $state('');
  let sourceBusy = $state(false);
  let filterBusy = $state(false);
  let loadError = $state('');
  let filterError = $state('');
  let catalogError = $state('');
  let canvas = $state(null);
  let segmentCanvas = $state(null);
  let range = $state(null);
  let sliderDraft = $state(null);
  let dragAnchor = 0;
  let dragging = false;
  let grayPixels = { src: '', data: null };

  let specByName = $derived(new Map((catalog?.ops ?? []).map((op) => [op.name, op])));
  let edges = $derived(catalog?.edges ?? FALLBACK_EDGES);
  let maxOps = $derived(catalog?.max_ops ?? 12);

  let chainError = $derived.by(() => {
    if (!catalog || !chain.length) return '';
    let mode = 'color';
    for (const step of chain) {
      const spec = specByName.get(step.name);
      if (!spec) return 'Unknown filter.';
      if (spec.needs === 'color' && mode !== 'color') {
        return `${spec.label} needs a color image. An earlier step already turned the preview gray.`;
      }
      if (spec.uses_elev && original && original.has_ndsm === false) {
        return `${spec.label} needs nDSM, which is missing for this tile.`;
      }
      if (spec.makes === 'gray') mode = 'gray';
    }
    if (chain.length > maxOps) return `At most ${maxOps} filters.`;
    return '';
  });

  let problem = $derived(chainError || filterError || loadError || catalogError);
  let busy = $derived(sourceBusy || filterBusy);
  let pickerDisabled = $derived(busy || !catalog || chain.length >= maxOps);

  let menuGroups = $derived.by(() => {
    const needle = query.trim().toLowerCase();
    const groups = catalog?.groups ?? [];
    const ops = catalog?.ops ?? [];
    return groups
      .map((group) => {
        const groupHit = needle && group.label.toLowerCase().includes(needle);
        return {
          ...group,
          ops: ops.filter((op) => {
            if (op.group !== group.id) return false;
            if (!needle || groupHit) return true;
            const label = op.label.toLowerCase();
            const name = op.name.replaceAll('_', ' ');
            return label.includes(needle) || name.includes(needle);
          }),
        };
      })
      .filter((group) => group.ops.length);
  });

  let view = $derived.by(() => {
    if (!chain.length) return original;
    if (shown && shownFor === `${tileId}:${Number(maxEdge)}`) return shown;
    return original;
  });

  let canWipe = $derived(
    Boolean(chain.length && !chainError && original?.image && shown?.image && shownFor === `${tileId}:${Number(maxEdge)}`),
  );

  let filterKey = $derived(
    `${tileId}:${Number(maxEdge)}:${JSON.stringify(chain.map((step) => [step.name, step.params]))}`,
  );

  let segmentShare = $derived.by(() => {
    const bins = view?.histogram?.gray;
    if (!range || !bins) return null;
    let total = 0;
    let hit = 0;
    for (let i = 0; i < 256; i++) {
      const count = bins[i] || 0;
      total += count;
      if (i >= range.lo && i <= range.hi) hit += count;
    }
    if (!total) return null;
    return (100 * hit) / total;
  });

  function urlOf(data) {
    return data?.image ? `data:image/jpeg;base64,${data.image}` : '';
  }

  async function readError(res) {
    const text = await res.text();
    try {
      const data = JSON.parse(text);
      if (typeof data?.detail === 'string') return data.detail;
    } catch {
      /* The body is plain text. */
    }
    return text || res.statusText;
  }

  function fetchPreview(id, edge, ops, signal) {
    return fetch(`/api/tiles/${id}/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_edge: edge, ops }),
      signal,
    }).then(async (res) => {
      if (!res.ok) throw new Error(await readError(res));
      return res.json();
    });
  }

  function drawHist(node, histogram) {
    const ctx = node.getContext('2d');
    const width = node.width;
    const height = node.height;
    ctx.clearRect(0, 0, width, height);
    const gray = histogram.gray ?? [];
    const series = [];
    if (histogram.r) series.push([histogram.r, '#b42318']);
    if (histogram.g) series.push([histogram.g, '#1b7a45']);
    if (histogram.b) series.push([histogram.b, '#1d4e89']);
    let max = 1;
    for (const value of gray) if (value > max) max = value;
    for (const [data] of series) {
      for (const value of data) if (value > max) max = value;
    }
    const bar = width / 256;
    ctx.fillStyle = series.length ? 'rgba(18, 32, 36, 0.28)' : 'rgba(18, 32, 36, 0.72)';
    for (let i = 0; i < 256; i++) {
      const barHeight = ((gray[i] || 0) / max) * (height - 4);
      ctx.fillRect(i * bar, height - barHeight, Math.max(bar, 1), barHeight);
    }
    ctx.lineWidth = 1.5;
    for (const [data, color] of series) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      for (let i = 0; i < 256; i++) {
        const x = (i + 0.5) * bar;
        const y = height - ((data[i] || 0) / max) * (height - 4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  function drawBand(node, lo, hi) {
    const ctx = node.getContext('2d');
    const bar = node.width / 256;
    ctx.fillStyle = 'rgba(225, 29, 138, 0.28)';
    ctx.fillRect(lo * bar, 0, (hi - lo + 1) * bar, node.height);
    ctx.strokeStyle = 'rgba(225, 29, 138, 0.9)';
    ctx.lineWidth = 1;
    ctx.strokeRect(lo * bar + 0.5, 0.5, (hi - lo + 1) * bar - 1, node.height - 1);
  }

  function binFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const t = rect.width ? (event.clientX - rect.left) / rect.width : 0;
    return Math.max(0, Math.min(255, Math.floor(t * 256)));
  }

  function startSegment(event) {
    if (busy || !view?.histogram || !canvas) return;
    dragging = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragAnchor = binFromEvent(event);
    range = { lo: dragAnchor, hi: dragAnchor };
  }

  function moveSegment(event) {
    if (!dragging) return;
    const bin = binFromEvent(event);
    range = { lo: Math.min(dragAnchor, bin), hi: Math.max(dragAnchor, bin) };
  }

  function endSegment() {
    dragging = false;
  }

  function resetSegment() {
    range = null;
  }

  function loadGray(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const scratch = document.createElement('canvas');
        scratch.width = img.naturalWidth;
        scratch.height = img.naturalHeight;
        const ctx = scratch.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        resolve(ctx.getImageData(0, 0, scratch.width, scratch.height));
      };
      img.onerror = () => reject(new Error('gray plane failed to load'));
      img.src = `data:image/png;base64,${src}`;
    });
  }

  function paintSegment(node, pixels, lo, hi) {
    const width = pixels.width;
    const height = pixels.height;
    if (node.width !== width || node.height !== height) {
      node.width = width;
      node.height = height;
    }
    const ctx = node.getContext('2d');
    const src = pixels.data;
    const out = ctx.createImageData(width, height);
    const dst = out.data;
    for (let i = 0; i < src.length; i += 4) {
      if (src[i + 3] === 0) continue;
      const gray = src[i];
      if (gray < lo || gray > hi) continue;
      dst[i] = 225;
      dst[i + 1] = 29;
      dst[i + 2] = 138;
      dst[i + 3] = 160;
    }
    ctx.putImageData(out, 0, 0);
  }

  function formatParam(param, value) {
    if (param.integer) return String(value);
    const text = String(param.step);
    const places = text.includes('.') ? text.split('.')[1].length : 0;
    return Number(value).toFixed(places);
  }

  function stat(value) {
    return value == null ? '—' : String(value);
  }

  function addFilter(name) {
    const spec = specByName.get(name);
    if (!spec || chain.length >= maxOps) return;
    const params = {};
    for (const param of spec.params) params[param.key] = param.default;
    chain = [...chain, { id: crypto.randomUUID(), name: spec.name, params }];
    menuOpen = false;
    query = '';
  }

  async function toggleMenu() {
    if (pickerDisabled) return;
    menuOpen = !menuOpen;
    if (!menuOpen) return;
    query = '';
    await tick();
    searchEl?.focus();
  }

  function closeMenu(event) {
    if (!menuOpen || pickerEl?.contains(event.target)) return;
    menuOpen = false;
  }

  function onMenuKey(event) {
    if (event.key === 'Escape') {
      menuOpen = false;
      return;
    }
    if (event.key !== 'Enter') return;
    const only = menuGroups.length === 1 && menuGroups[0].ops.length === 1 ? menuGroups[0].ops[0] : null;
    if (!only) return;
    event.preventDefault();
    addFilter(only.name);
  }

  function resetChain() {
    chain = [];
    query = '';
    menuOpen = false;
    filterError = '';
  }

  function move(index, dir) {
    const next = index + dir;
    if (next < 0 || next >= chain.length) return;
    const copy = chain.slice();
    const [item] = copy.splice(index, 1);
    copy.splice(next, 0, item);
    chain = copy;
  }

  function remove(index) {
    chain = chain.filter((_, i) => i !== index);
  }

  function setParam(index, key, value) {
    chain = chain.map((step, i) => (
      i === index ? { ...step, params: { ...step.params, [key]: value } } : step
    ));
  }

  function paramNumber(param, raw) {
    const value = Number(raw);
    return param.integer ? Math.round(value) : value;
  }

  function sliderValue(step, param) {
    if (sliderDraft && sliderDraft.id === step.id && sliderDraft.key === param.key) return sliderDraft.value;
    return step.params[param.key];
  }

  function previewParam(step, param, raw) {
    sliderDraft = { id: step.id, key: param.key, value: paramNumber(param, raw) };
  }

  function commitParam(index, key, value) {
    sliderDraft = null;
    if (chain[index]?.params?.[key] === value) return;
    setParam(index, key, value);
  }

  $effect(() => {
    let dead = false;
    fetch('/api/vision/ops')
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        if (!dead) catalog = data;
      })
      .catch((err) => {
        if (!dead) catalogError = String(err.message || err);
      });
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    const id = tileId;
    const edge = Number(maxEdge);
    const ctrl = new AbortController();
    let dead = false;
    sourceBusy = true;
    original = null;
    shown = null;
    shownFor = '';
    loadError = '';
    fetchPreview(id, edge, [], ctrl.signal)
      .then((data) => {
        if (dead) return;
        original = data;
        sourceBusy = false;
      })
      .catch((err) => {
        if (dead || err.name === 'AbortError') return;
        loadError = String(err.message || err);
        sourceBusy = false;
      });
    return () => {
      dead = true;
      ctrl.abort();
    };
  });

  $effect(() => {
    const id = tileId;
    const edge = Number(maxEdge);
    const ops = chain.map((step) => ({ name: step.name, ...step.params }));
    if (chainError || ops.length === 0) {
      filterBusy = false;
      return;
    }
    filterBusy = true;
    const ctrl = new AbortController();
    let dead = false;
    const timer = setTimeout(() => {
      fetchPreview(id, edge, ops, ctrl.signal)
        .then((data) => {
          if (dead) return;
          shown = data;
          shownFor = `${id}:${edge}`;
          filterError = '';
          filterBusy = false;
        })
        .catch((err) => {
          if (dead || err.name === 'AbortError') return;
          filterError = String(err.message || err);
          filterBusy = false;
        });
    }, 200);
    return () => {
      dead = true;
      clearTimeout(timer);
      ctrl.abort();
    };
  });

  $effect(() => {
    filterKey;
    range = null;
    grayPixels = { src: '', data: null };
  });

  $effect(() => {
    if (pickerDisabled) menuOpen = false;
  });

  $effect(() => {
    const histogram = view?.histogram;
    const selected = range;
    if (!canvas) return;
    if (!histogram) {
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    drawHist(canvas, histogram);
    if (selected) drawBand(canvas, selected.lo, selected.hi);
  });

  $effect(() => {
    const src = view?.gray;
    const selected = range;
    const node = segmentCanvas;
    if (!node) return;
    let dead = false;
    const ctx = node.getContext('2d');

    async function run() {
      if (!src || !selected) {
        ctx.clearRect(0, 0, node.width, node.height);
        return;
      }
      let data = grayPixels.src === src ? grayPixels.data : null;
      if (!data) {
        data = await loadGray(src);
        if (dead) return;
        grayPixels = { src, data };
      }
      if (dead) return;
      paintSegment(node, data, selected.lo, selected.hi);
    }

    run();
    return () => {
      dead = true;
    };
  });
</script>

<svelte:window onclick={closeMenu} onkeydown={(event) => menuOpen && event.key === 'Escape' && (menuOpen = false)} />

<div class="lab">
  <div class="preview">
    <label class="field">
      Preview
      <select value={maxEdge} disabled={busy} onchange={(e) => (maxEdge = Number(e.currentTarget.value))}>
        {#each edges as edge}
          <option value={edge}>{edge}px</option>
        {/each}
      </select>
    </label>
    <div class="stage" aria-busy={busy}>
      {#if view?.image}
        <img class="base" src={urlOf(view)} alt="Filtered preview" draggable="false" />
      {:else if !problem}
        <p class="wait">Rendering preview…</p>
      {/if}
      {#if canWipe}
        <img
          class="wipe"
          src={urlOf(original)}
          alt=""
          draggable="false"
          style:clip-path={`inset(0 0 0 ${wipe}%)`}
        />
      {/if}
      {#if mask}
        <img class="mask" src={mask} alt="" draggable="false" style:opacity={opacity} />
      {/if}
      <canvas
        class="segment"
        bind:this={segmentCanvas}
        aria-hidden="true"
        style:clip-path={canWipe ? `inset(0 ${100 - wipe}% 0 0)` : undefined}
      ></canvas>
    </div>
    <label class="slider">
      Original
      <input type="range" min="0" max="100" step="1" bind:value={wipe} disabled={!canWipe} aria-label="Compare original and filtered preview" />
      Filtered
    </label>
    <label class="slider">
      Labels
      <input type="range" min="0" max="1" step="0.05" bind:value={opacity} />
    </label>
  </div>

  <div class="controls">
    <div class="hist-head">
      <h3>Gray values</h3>
      <span class="hist-actions">
        {#if busy}<span class="busy">Updating…</span>{/if}
        <button type="button" onclick={resetSegment} disabled={busy || !range}>Reset segment</button>
      </span>
    </div>
    <canvas
      bind:this={canvas}
      class="hist"
      class:locked={busy}
      width="512"
      height="148"
      aria-label="Gray value histogram. Drag to segment a range."
      onpointerdown={startSegment}
      onpointermove={moveSegment}
      onpointerup={endSegment}
      onpointercancel={endSegment}
    ></canvas>
    <div class="axis"><span>0</span><span>255</span></div>
    <p class="segment-line">
      {#if range}
        {range.lo}–{range.hi}{#if segmentShare != null} · {segmentShare.toFixed(1)}% of pixels{/if}
      {:else}
        Drag the histogram to segment those gray values.
      {/if}
    </p>
    <p class="stats">
      {stat(view?.stats?.min)}–{stat(view?.stats?.max)}
      · mean {stat(view?.stats?.mean)}
      · sd {stat(view?.stats?.std)}
    </p>
    <div class="keys">
      <span><i class="swatch" style:--swatch="#122024"></i> Gray</span>
      {#if view?.histogram?.r}
        <span><i class="swatch" style:--swatch="#b42318"></i> Red</span>
        <span><i class="swatch" style:--swatch="#1b7a45"></i> Green</span>
        <span><i class="swatch" style:--swatch="#1d4e89"></i> Blue</span>
      {/if}
    </div>

    {#if problem}
      <p class="problem" role="alert">{problem}</p>
    {/if}

    <div class="add-row">
      <div class="picker grow" bind:this={pickerEl}>
        <span class="picker-label">Add filter</span>
        <button
          type="button"
          class="picker-trigger"
          aria-haspopup="listbox"
          aria-expanded={menuOpen}
          disabled={pickerDisabled}
          onclick={toggleMenu}
        >
          {catalog ? 'Choose a filter…' : 'Loading filters…'}
        </button>
        {#if menuOpen}
          <div class="picker-menu">
            <input
              bind:this={searchEl}
              class="picker-search"
              type="search"
              placeholder="Search filters…"
              aria-label="Search filters"
              autocomplete="off"
              bind:value={query}
              onkeydown={onMenuKey}
            />
            <div class="picker-list" role="listbox" aria-label="Filters">
              {#each menuGroups as group (group.id)}
                <p class="picker-group">{group.label}</p>
                {#each group.ops as op (op.name)}
                  <button type="button" role="option" aria-selected={false} class="picker-option" onclick={() => addFilter(op.name)}>
                    {op.label}
                  </button>
                {/each}
              {:else}
                <p class="picker-empty">No matching filters.</p>
              {/each}
            </div>
          </div>
        {/if}
      </div>
      <button type="button" onclick={resetChain} disabled={busy || !chain.length}>Reset</button>
    </div>
    {#if original && original.has_ndsm === false}
      <p class="unsaved">No DSM/DTM for this tile, so nDSM height is unavailable.</p>
    {/if}
    <p class="unsaved">This preview is not saved.</p>

    {#if chain.length}
      <ol class="chain">
        {#each chain as step, index (step.id)}
          {@const spec = specByName.get(step.name)}
          <li>
            <div class="step-head">
              <strong>{index + 1}. {spec?.label ?? step.name}</strong>
              <span class="step-actions">
                <button type="button" aria-label="Move up" disabled={busy || index === 0} onclick={() => move(index, -1)}>Up</button>
                <button type="button" aria-label="Move down" disabled={busy || index === chain.length - 1} onclick={() => move(index, 1)}>Down</button>
                <button type="button" aria-label="Remove filter" disabled={busy} onclick={() => remove(index)}>Remove</button>
              </span>
            </div>
            {#if spec?.hint}<p class="step-hint">{spec.hint}</p>{/if}
            {#each spec?.params ?? [] as param}
              {#if param.kind === 'select'}
                <label class="field">
                  {param.label}
                  <select
                    value={step.params[param.key]}
                    disabled={busy}
                    onchange={(e) => setParam(index, param.key, e.currentTarget.value)}
                  >
                    {#each param.options as option}
                      <option value={option.value}>{option.label}</option>
                    {/each}
                  </select>
                </label>
              {:else}
                <label class="param">
                  <span>{param.label}</span>
                  <input
                    type="range"
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    value={sliderValue(step, param)}
                    disabled={busy}
                    oninput={(e) => previewParam(step, param, e.currentTarget.value)}
                    onchange={(e) => commitParam(index, param.key, paramNumber(param, e.currentTarget.value))}
                    aria-label={param.label}
                  />
                  <span class="readout">{formatParam(param, sliderValue(step, param))}</span>
                </label>
              {/if}
            {/each}
          </li>
        {/each}
      </ol>
    {/if}
  </div>
</div>

<style>
  .lab {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(260px, 0.85fr);
    gap: 14px 16px;
    align-items: start;
    margin-bottom: 14px;
  }

  .preview, .controls { min-width: 0; }

  .field { margin-bottom: 8px; }

  .stage { display: flex; align-items: center; justify-content: center; }

  .base, .wipe {
    position: absolute;
    inset: 0;
  }

  .wipe, .segment { pointer-events: none; }

  .segment {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    z-index: 2;
  }

  .wait {
    position: relative;
    z-index: 1;
    margin: 0;
    padding: 16px;
    color: var(--sheet);
    font-size: 13px;
  }

  .hist-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
  }

  .hist-actions {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }

  .hist-actions button {
    border: 1px solid var(--line);
    background: transparent;
    border-radius: 2px;
    padding: 2px 6px;
    font-size: 11px;
    color: var(--mist);
  }

  .hist-actions button:disabled { opacity: 0.4; cursor: default; }

  h3 {
    margin: 0 0 6px;
    font-size: 13px;
  }

  .busy {
    color: var(--quiet);
    font-size: 11px;
  }

  button:disabled,
  select:disabled,
  input:disabled {
    opacity: 0.45;
    cursor: default;
  }

  canvas.hist {
    width: 100%;
    height: auto;
    display: block;
    cursor: crosshair;
    touch-action: none;
    background: color-mix(in srgb, var(--line) 40%, white);
  }

  canvas.hist.locked {
    cursor: default;
    opacity: 0.45;
    pointer-events: none;
  }

  .segment-line {
    margin: 4px 0 0;
    color: var(--quiet);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }

  .axis {
    display: flex;
    justify-content: space-between;
    color: var(--quiet);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }

  .stats {
    margin: 4px 0 8px;
    color: var(--mist);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }

  .keys {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 12px;
    margin-bottom: 10px;
    color: var(--mist);
    font-size: 12px;
  }

  .keys span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .problem {
    margin: 0 0 10px;
    color: #8a2a22;
    font-size: 12px;
    line-height: 1.4;
  }

  .add-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
  }

  .grow { flex: 1; margin: 0; min-width: 0; }

  .picker { position: relative; }

  .picker-label {
    display: block;
    margin-bottom: 3px;
    font-size: 12px;
    color: var(--mist);
  }

  .picker-trigger {
    width: 100%;
    text-align: left;
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 5px 28px 5px 8px;
    color: var(--ink);
    background-image: linear-gradient(45deg, transparent 50%, var(--mist) 50%), linear-gradient(135deg, var(--mist) 50%, transparent 50%);
    background-position: calc(100% - 14px) calc(50% - 1px), calc(100% - 9px) calc(50% - 1px);
    background-size: 5px 5px, 5px 5px;
    background-repeat: no-repeat;
  }

  .picker-trigger[aria-expanded="true"] {
    background-image: linear-gradient(135deg, transparent 50%, var(--mist) 50%), linear-gradient(45deg, var(--mist) 50%, transparent 50%);
    background-position: calc(100% - 14px) calc(50% + 1px), calc(100% - 9px) calc(50% + 1px);
  }

  .picker-menu {
    position: absolute;
    z-index: 3;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    display: flex;
    flex-direction: column;
    max-height: min(42dvh, 320px);
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 2px;
    box-shadow: var(--shadow);
  }

  .picker-search {
    margin: 6px;
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 5px 8px;
    background: white;
  }

  .picker-list {
    overflow: auto;
    padding-bottom: 4px;
  }

  .picker-group {
    margin: 6px 8px 2px;
    color: var(--quiet);
    font-size: 11px;
  }

  .picker-option {
    display: block;
    width: 100%;
    text-align: left;
    border: 0;
    background: transparent;
    border-radius: 0;
    padding: 5px 10px;
    font-size: 13px;
  }

  .picker-option:hover,
  .picker-option:focus-visible {
    background: color-mix(in srgb, var(--canopy) 12%, white);
  }

  .picker-empty {
    margin: 2px 8px 8px;
    color: var(--quiet);
    font-size: 12px;
  }

  .add-row > button {
    border: 1px solid var(--line);
    background: transparent;
    border-radius: 2px;
    padding: 6px 10px;
  }

  .add-row > button:disabled { opacity: 0.45; cursor: default; }

  .unsaved {
    margin: 8px 0;
    color: var(--quiet);
    font-size: 11px;
  }

  .chain {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: min(52dvh, 560px);
    overflow: auto;
  }

  .chain li {
    margin: 0;
    padding: 8px 8px 6px;
    border: 1px solid var(--line);
    border-radius: 2px;
  }

  .step-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }

  .step-head strong { font-size: 13px; font-weight: 600; }

  .step-actions { display: inline-flex; gap: 4px; }

  .step-actions button {
    border: 1px solid var(--line);
    background: transparent;
    border-radius: 2px;
    padding: 2px 6px;
    font-size: 11px;
    color: var(--mist);
  }

  .step-actions button:disabled { opacity: 0.4; cursor: default; }

  .step-hint {
    margin: 0 0 6px;
    color: var(--quiet);
    font-size: 11px;
    line-height: 1.35;
  }

  .param {
    display: grid;
    grid-template-columns: 72px minmax(0, 1fr) 36px;
    align-items: center;
    gap: 6px;
    margin: 2px 0;
    color: var(--mist);
    font-size: 12px;
  }

  .param input { width: 100%; accent-color: var(--canopy); }

  .readout {
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
  }

  @media (max-width: 720px) {
    .lab { grid-template-columns: 1fr; }
  }
</style>
