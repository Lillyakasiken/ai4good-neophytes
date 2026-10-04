<script>
  let { tileId, mask } = $props();

  const FALLBACK_EDGES = [256, 512, 1024];

  let catalog = $state(null);
  let catalogError = $state('');
  let modelId = $state('yoloe-26s');
  let maxEdge = $state(512);
  let conf = $state(0.25);
  let selected = $state([]);
  let extra = $state('');
  let phrase = $state('');
  let tool = $state('include-point');
  let points = $state([]);
  let boxes = $state([]);
  let drag = $state(null);
  let skipClick = false;
  let runId = $state(0);
  let submitted = $state(null);

  let base = $state(null);
  let baseBusy = $state(false);
  let baseError = $state('');
  let pred = $state(null);
  let predFor = $state('');
  let predBusy = $state(false);
  let predError = $state('');
  let labelOpacity = $state(0.55);
  let predOpacity = $state(0.75);

  let models = $derived(catalog?.models ?? []);
  let current = $derived(models.find((item) => item.id === modelId) ?? models[0] ?? null);
  let promptKind = $derived(current?.prompt ?? (modelId === 'sam2.1-t' ? 'points' : 'text'));
  let edges = $derived(catalog?.edges ?? FALLBACK_EDGES);
  let presets = $derived(catalog?.presets ?? []);
  let confMin = $derived(catalog?.conf_min ?? 0.05);
  let confMax = $derived(catalog?.conf_max ?? 0.95);
  let confStep = $derived(catalog?.conf_step ?? 0.05);
  let maxPoints = $derived(catalog?.max_points ?? 16);
  let maxBoxes = $derived(catalog?.max_boxes ?? 8);

  let texts = $derived.by(() => {
    const chosen = presets
      .filter((row) => selected.includes(row.stats_name))
      .map((row) => row.prompt);
    if (phrase && !chosen.some((item) => item.toLowerCase() === phrase.toLowerCase())) chosen.push(phrase);
    return chosen;
  });

  let textKey = $derived(`${tileId}:${Number(maxEdge)}:text:${Number(conf)}:${JSON.stringify(texts)}`);
  let samKey = $derived(
    submitted ? `${tileId}:${submitted.edge}:sam:${submitted.model}:${runId}` : '',
  );
  let activeKey = $derived(promptKind === 'points' ? samKey : textKey);

  let baseUrl = $derived(base?.image ? `data:image/jpeg;base64,${base.image}` : '');
  let overlayUrl = $derived(
    pred && predFor === activeKey && pred.overlay ? `data:image/png;base64,${pred.overlay}` : '',
  );
  let instances = $derived(pred && predFor === activeKey ? pred.instances ?? [] : []);
  let problem = $derived(catalogError || baseError || predError);
  let busy = $derived(baseBusy || predBusy);
  let hasInclude = $derived(
    points.some((point) => point.label === 1) || boxes.some((box) => box.label === 1),
  );
  let boxTool = $derived(tool === 'include-box' || tool === 'exclude-box');

  function commitPhrase() {
    const next = extra.trim().replace(/\s+/g, ' ');
    if (next !== phrase) phrase = next;
  }

  function onPhraseKey(event) {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    commitPhrase();
  }

  function readError(res) {
    return res.text().then((text) => {
      try {
        const data = JSON.parse(text);
        if (typeof data?.detail === 'string') return data.detail;
      } catch {
        /* The body is plain text. */
      }
      return text || res.statusText;
    });
  }

  function togglePreset(name) {
    selected = selected.includes(name)
      ? selected.filter((item) => item !== name)
      : [...selected, name];
  }

  function nextMarkId() {
    const ids = [...points, ...boxes].map((mark) => mark.id);
    return ids.length ? Math.max(...ids) + 1 : 1;
  }

  function eventPoint(event) {
    const img = event.currentTarget.querySelector('img');
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    const naturalW = img.naturalWidth || rect.width;
    const naturalH = img.naturalHeight || rect.height;
    const scale = Math.min(rect.width / naturalW, rect.height / naturalH);
    const drawnW = naturalW * scale;
    const drawnH = naturalH * scale;
    const left = rect.left + (rect.width - drawnW) / 2;
    const top = rect.top + (rect.height - drawnH) / 2;
    const x = (event.clientX - left) / drawnW;
    const y = (event.clientY - top) / drawnH;
    if (x < 0 || y < 0 || x > 1 || y > 1) return null;
    return { x, y };
  }

  function onPreviewClick(event) {
    if (skipClick) {
      skipClick = false;
      return;
    }
    if (promptKind !== 'points' || predBusy || boxTool || points.length >= maxPoints || event.detail === 0) return;
    const point = eventPoint(event);
    if (!point) return;
    points = [...points, { id: nextMarkId(), ...point, label: tool === 'exclude-point' ? 0 : 1 }];
  }

  function onPointerDown(event) {
    if (promptKind !== 'points' || predBusy || !boxTool || boxes.length >= maxBoxes || event.button !== 0) return;
    const point = eventPoint(event);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag = { x1: point.x, y1: point.y, x2: point.x, y2: point.y, label: tool === 'exclude-box' ? 0 : 1 };
  }

  function onPointerMove(event) {
    if (!drag) return;
    const point = eventPoint(event);
    if (!point) return;
    drag = { ...drag, x2: point.x, y2: point.y };
  }

  function onPointerUp(event) {
    if (!drag) return;
    const box = drag;
    drag = null;
    if (event?.type === 'pointerup') skipClick = true;
    const x1 = Math.min(box.x1, box.x2);
    const y1 = Math.min(box.y1, box.y2);
    const x2 = Math.max(box.x1, box.x2);
    const y2 = Math.max(box.y1, box.y2);
    if (x2 - x1 < 0.01 || y2 - y1 < 0.01) return;
    boxes = [...boxes, { id: nextMarkId(), x1, y1, x2, y2, label: box.label }];
  }

  function removePoint(id) {
    points = points.filter((point) => point.id !== id);
  }

  function removeBox(id) {
    boxes = boxes.filter((box) => box.id !== id);
  }

  function clearMarks() {
    points = [];
    boxes = [];
    drag = null;
  }

  function runSam() {
    if (!hasInclude || predBusy) return;
    submitted = {
      model: modelId,
      edge: Number(maxEdge),
      conf: Number(conf),
      points: points.map(({ x, y, label }) => ({ x, y, label })),
      boxes: boxes.map(({ x1, y1, x2, y2, label }) => ({ x1, y1, x2, y2, label })),
    };
    runId += 1;
  }

  function percent(share) {
    if (share == null) return '—';
    return `${(100 * share).toFixed(1)}%`;
  }

  function score(value) {
    if (value == null) return '—';
    return Number(value).toFixed(2);
  }

  $effect(() => {
    let dead = false;
    fetch('/api/segment/models')
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
    points = [];
    boxes = [];
    drag = null;
    submitted = null;
    runId = 0;
    pred = null;
    predFor = '';
    predError = '';
    base = null;
    void id;
  });

  $effect(() => {
    const id = tileId;
    const edge = Number(maxEdge);
    let dead = false;
    const controller = new AbortController();
    baseBusy = true;
    baseError = '';
    fetch(`/api/tiles/${id}/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_edge: edge, ops: [] }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        if (!dead) base = data;
      })
      .catch((err) => {
        if (dead || err.name === 'AbortError') return;
        baseError = String(err.message || err);
      })
      .finally(() => {
        if (!dead) baseBusy = false;
      });
    return () => {
      dead = true;
      controller.abort();
    };
  });

  $effect(() => {
    if (promptKind !== 'text') return;
    const id = tileId;
    const edge = Number(maxEdge);
    const phrases = texts;
    const confidence = Number(conf);
    const model = modelId;
    const key = `${id}:${edge}:text:${confidence}:${JSON.stringify(phrases)}`;
    if (!phrases.length) {
      pred = null;
      predFor = '';
      predError = '';
      predBusy = false;
      return;
    }
    let dead = false;
    const controller = new AbortController();
    predError = '';
    const timer = setTimeout(() => {
      if (dead) return;
      predBusy = true;
      predError = '';
      fetch(`/api/tiles/${id}/segment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          max_edge: edge,
          conf: confidence,
          texts: phrases,
          points: [],
        }),
        signal: controller.signal,
      })
        .then(async (res) => {
          if (!res.ok) throw new Error(await readError(res));
          return res.json();
        })
        .then((data) => {
          if (dead) return;
          pred = data;
          predFor = key;
        })
        .catch((err) => {
          if (dead || err.name === 'AbortError') return;
          predError = String(err.message || err);
        })
        .finally(() => {
          if (!dead) predBusy = false;
        });
    }, 400);
    return () => {
      dead = true;
      predBusy = false;
      clearTimeout(timer);
      controller.abort();
    };
  });

  $effect(() => {
    if (promptKind !== 'points') return;
    const id = tileId;
    const edge = Number(maxEdge);
    const seq = runId;
    const job = submitted;
    if (!seq || !job || job.edge !== edge || job.model !== modelId) {
      pred = null;
      predFor = '';
      predBusy = false;
      predError = '';
      return;
    }
    const key = `${id}:${job.edge}:sam:${job.model}:${seq}`;
    let dead = false;
    const controller = new AbortController();
    predBusy = true;
    predError = '';
    fetch(`/api/tiles/${id}/segment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: job.model,
        max_edge: job.edge,
        conf: job.conf,
        texts: [],
        points: job.points,
        boxes: job.boxes,
      }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        if (dead) return;
        pred = data;
        predFor = key;
      })
      .catch((err) => {
        if (dead || err.name === 'AbortError') return;
        predError = String(err.message || err);
      })
      .finally(() => {
        if (!dead) predBusy = false;
      });
    return () => {
      dead = true;
      predBusy = false;
      controller.abort();
    };
  });
</script>

{#snippet frameBody()}
  <img class="base" src={baseUrl} alt="Tile preview" draggable="false" />
  {#if mask}
    <img class="mask" src={mask} alt="" draggable="false" style:opacity={labelOpacity} />
  {/if}
  {#if overlayUrl}
    <img class="mask pred" src={overlayUrl} alt="" draggable="false" style:opacity={predOpacity} />
  {/if}
  {#if promptKind === 'points'}
    {#each points as point (point.id)}
      <span
        class="dot"
        class:neg={point.label === 0}
        style:left={`${point.x * 100}%`}
        style:top={`${point.y * 100}%`}
      ></span>
    {/each}
    {#each boxes as box (box.id)}
      <span
        class="region"
        class:neg={box.label === 0}
        style:left={`${box.x1 * 100}%`}
        style:top={`${box.y1 * 100}%`}
        style:width={`${(box.x2 - box.x1) * 100}%`}
        style:height={`${(box.y2 - box.y1) * 100}%`}
      ></span>
    {/each}
    {#if drag}
      <span
        class="region live"
        class:neg={drag.label === 0}
        style:left={`${Math.min(drag.x1, drag.x2) * 100}%`}
        style:top={`${Math.min(drag.y1, drag.y2) * 100}%`}
        style:width={`${Math.abs(drag.x2 - drag.x1) * 100}%`}
        style:height={`${Math.abs(drag.y2 - drag.y1) * 100}%`}
      ></span>
    {/if}
  {/if}
{/snippet}

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
      {#if baseUrl}
        {#if promptKind === 'points'}
          <button
            type="button"
            class="frame aim"
            style:aspect-ratio={`${base.width} / ${base.height}`}
            aria-label={boxTool ? 'Draw a region' : 'Add a point'}
            onclick={onPreviewClick}
            onpointerdown={onPointerDown}
            onpointermove={onPointerMove}
            onpointerup={onPointerUp}
            onpointercancel={onPointerUp}
          >
            {@render frameBody()}
          </button>
        {:else}
          <div class="frame" style:aspect-ratio={`${base.width} / ${base.height}`}>
            {@render frameBody()}
          </div>
        {/if}
      {:else if !problem}
        <p class="wait">Rendering preview…</p>
      {/if}
      {#if predBusy}
        <div class="predicting" role="status" aria-live="polite">
          <span class="spin" aria-hidden="true"></span>
          Running prediction
        </div>
      {/if}
    </div>
    <label class="slider">
      Labels
      <input type="range" min="0" max="1" step="0.05" bind:value={labelOpacity} />
    </label>
    <label class="slider">
      Prediction
      <input type="range" min="0" max="1" step="0.05" bind:value={predOpacity} disabled={!overlayUrl} />
    </label>
  </div>

  <div class="controls">
    <label class="field">
      Model
      <select value={modelId} disabled={busy || !models.length} onchange={(e) => (modelId = e.currentTarget.value)}>
        {#each models as item (item.id)}
          <option value={item.id}>{item.label}</option>
        {/each}
      </select>
    </label>

    {#if promptKind === 'text'}
      <p class="step-hint">Pick a phrase. A custom phrase runs when you press Enter or leave the field.</p>
      <div class="chips">
        {#each presets as preset (preset.stats_name)}
          <button
            type="button"
            class:on={selected.includes(preset.stats_name)}
            title={preset.name}
            disabled={busy}
            onclick={() => togglePreset(preset.stats_name)}
          >
            <i class="swatch" style:--swatch={preset.color}></i>
            {preset.prompt}
          </button>
        {/each}
      </div>
      <label class="field">
        Another phrase
        <input
          type="text"
          maxlength="80"
          placeholder="flowering shrub"
          bind:value={extra}
          disabled={busy}
          onkeydown={onPhraseKey}
          onblur={commitPhrase}
        />
      </label>
    {:else}
      <p class="step-hint">
        {#if boxTool}
          Drag a rectangle, then run. Include keeps that area. Exclude marks it as background.
        {:else}
          Click the tile, then run. Include keeps that spot. Exclude marks it as background.
        {/if}
      </p>
      <div class="tools" role="group" aria-label="SAM prompt">
        <button type="button" class:on={tool === 'include-point'} disabled={busy} onclick={() => (tool = 'include-point')}>Include point</button>
        <button type="button" class:on={tool === 'exclude-point'} disabled={busy} onclick={() => (tool = 'exclude-point')}>Exclude point</button>
        <button type="button" class:on={tool === 'include-box'} disabled={busy} onclick={() => (tool = 'include-box')}>Include region</button>
        <button type="button" class:on={tool === 'exclude-box'} disabled={busy} onclick={() => (tool = 'exclude-box')}>Exclude region</button>
      </div>
      {#if points.length || boxes.length}
        <ul class="clicks">
          {#each points as point (point.id)}
            <li>
              <span class="dot inline" class:neg={point.label === 0}></span>
              {point.label === 1 ? 'Include point' : 'Exclude point'}
              <button type="button" disabled={busy} onclick={() => removePoint(point.id)}>Remove</button>
            </li>
          {/each}
          {#each boxes as box (box.id)}
            <li>
              <span class="region-key" class:neg={box.label === 0}></span>
              {box.label === 1 ? 'Include region' : 'Exclude region'}
              <button type="button" disabled={busy} onclick={() => removeBox(box.id)}>Remove</button>
            </li>
          {/each}
        </ul>
      {/if}
      <div class="actions">
        <button type="button" class="run" onclick={runSam} disabled={busy || !hasInclude}>Run</button>
        <button type="button" onclick={clearMarks} disabled={busy || (!points.length && !boxes.length)}>Clear</button>
      </div>
    {/if}

    <label class="slider">
      Confidence
      <input type="range" min={confMin} max={confMax} step={confStep} bind:value={conf} disabled={busy} />
      <span class="readout">{Number(conf).toFixed(2)}</span>
    </label>

    <div class="hist-head">
      <h3>Masks</h3>
      {#if predBusy}<span class="busy">Running…</span>{/if}
    </div>
    {#if instances.length}
      <ul class="hits">
        {#each instances as item, index (`${item.label}:${index}`)}
          <li>
            <i class="swatch" style:--swatch={item.color}></i>
            <span>{item.label}</span>
            <span class="num">{score(item.confidence)}</span>
            <span class="num">{percent(item.share)}</span>
          </li>
        {/each}
      </ul>
    {:else if pred && predFor === activeKey && !predBusy}
      <p class="unsaved">No mask above this confidence.</p>
    {:else if promptKind === 'text' && !texts.length}
      <p class="unsaved">Choose a species or type a phrase.</p>
    {/if}

    {#if problem}
      <p class="problem" role="alert">{problem}</p>
    {/if}
    <p class="unsaved">This preview is not saved. The first run downloads the model.</p>
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

  .field input[type='text'] {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 5px 8px;
    background: white;
  }

  .stage { display: flex; align-items: center; justify-content: center; }

  .frame {
    position: relative;
    width: 100%;
  }

  button.frame {
    display: block;
    padding: 0;
    border: 0;
    background: transparent;
    color: inherit;
    text-align: left;
  }

  button.frame.aim { cursor: crosshair; touch-action: none; }

  .region {
    position: absolute;
    z-index: 2;
    border: 1.5px solid #1b7a45;
    background: rgba(27, 122, 69, 0.16);
    pointer-events: none;
    box-sizing: border-box;
  }

  .region.neg {
    border-color: #b42318;
    border-style: dashed;
    background: rgba(180, 35, 24, 0.16);
  }

  .region-key {
    width: 12px;
    height: 8px;
    border: 1.5px solid #1b7a45;
    background: rgba(27, 122, 69, 0.16);
    box-sizing: border-box;
    flex: none;
  }

  .region-key.neg {
    border-color: #b42318;
    border-style: dashed;
    background: rgba(180, 35, 24, 0.16);
  }

  .base, .mask {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  .mask, .dot { pointer-events: none; }

  .pred { z-index: 1; }

  .dot {
    position: absolute;
    z-index: 2;
    width: 10px;
    height: 10px;
    margin: -5px 0 0 -5px;
    border: 1.5px solid white;
    border-radius: 50%;
    background: #1b7a45;
    box-shadow: 0 0 0 1px rgba(18, 32, 36, 0.45);
  }

  .dot.neg { background: #b42318; }

  .dot.inline {
    position: static;
    display: inline-block;
    margin: 0;
  }

  .wait {
    position: relative;
    z-index: 1;
    margin: 0;
    padding: 16px;
    color: var(--sheet);
    font-size: 13px;
  }

  .step-hint {
    margin: 0 0 8px;
    color: var(--quiet);
    font-size: 11px;
    line-height: 1.35;
  }

  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }

  .chips button {
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

  .chips button.on,
  .tools button.on {
    background: color-mix(in srgb, var(--canopy) 12%, white);
    border-color: var(--canopy);
  }

  .tools {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }

  .tools button {
    border: 1px solid var(--line);
    background: transparent;
    border-radius: 2px;
    padding: 3px 8px;
    font-size: 12px;
    color: var(--ink);
  }

  .clicks, .hits {
    list-style: none;
    margin: 0 0 8px;
    padding: 0;
  }

  .clicks li, .hits li {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
    padding: 4px 0;
    border-bottom: 1px solid var(--line);
    font-size: 12px;
  }

  .clicks button, .actions button {
    border: 1px solid var(--line);
    background: transparent;
    border-radius: 2px;
    padding: 2px 6px;
    font-size: 11px;
    color: var(--mist);
  }

  .clicks button { margin-left: auto; }

  .actions {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
  }

  .actions .run {
    background: var(--canopy);
    border-color: var(--canopy);
    color: var(--sheet);
    padding: 4px 12px;
    font-size: 12px;
  }

  .slider .readout {
    min-width: 32px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
  }

  .hist-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
  }

  h3 {
    margin: 8px 0 6px;
    font-size: 13px;
  }

  .hits .num {
    margin-left: auto;
    font-variant-numeric: tabular-nums;
    color: var(--mist);
  }

  .hits li .num + .num { margin-left: 0; min-width: 48px; text-align: right; }

  .predicting {
    position: absolute;
    inset: 0;
    z-index: 4;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    background: rgba(18, 32, 36, 0.4);
    color: var(--sheet);
    font-size: 13px;
  }

  .spin {
    width: 28px;
    height: 28px;
    border: 2px solid rgba(255, 255, 255, 0.35);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  @media (prefers-reduced-motion: reduce) {
    .spin { animation: none; opacity: 0.85; }
  }

  .busy {
    color: var(--quiet);
    font-size: 11px;
  }

  .problem {
    margin: 0 0 10px;
    color: #8a2a22;
    font-size: 12px;
    line-height: 1.4;
  }

  .unsaved {
    margin: 8px 0;
    color: var(--quiet);
    font-size: 11px;
  }

  button:disabled,
  select:disabled,
  input:disabled {
    opacity: 0.45;
    cursor: default;
  }

  @media (max-width: 720px) {
    .lab { grid-template-columns: 1fr; }
  }
</style>
