<script>
  let { plan, live = null, onflight, onedit, onrun, onmove = null, busy = false, queueing = false } = $props();

  function f1(value) {
    if (value == null || Number.isNaN(Number(value))) return '';
    return Number(value).toFixed(2);
  }

  let scores = $derived.by(() => {
    const map = {};
    for (const row of live?.metrics?.per_flight ?? []) map[row.flight] = row.f1;
    return map;
  });

  let active = $derived.by(() => {
    if (!live) return '';
    if (live.metrics) return 'done';
    const steps = live.steps ?? [];
    const hit = [...steps].reverse().find((step) => step.progress);
    if (hit) return hit.kind;
    return live.job?.status === 'running' ? (steps[0]?.kind ?? '') : '';
  });

  function progressOf(kind) {
    return (live?.steps ?? []).find((step) => step.kind === kind)?.progress ?? null;
  }

  function epochLine(kind) {
    const progress = progressOf(kind);
    if (!progress) return live ? 'waiting' : '';
    const epoch = `${progress.epoch ?? 0}/${progress.max_epochs ?? '?'}`;
    if (kind === 'base') return `epoch ${epoch}`;
    const score = f1(progress.novel_f1);
    return score ? `epoch ${epoch} · new species F1 ${score}` : `epoch ${epoch}`;
  }

  function subsetLine(metrics) {
    const scored = metrics?.eval_tiles;
    const full = metrics?.eval_tiles_full;
    if (scored == null || full == null || scored === full) return '';
    return ` · ${scored} of ${full} tiles`;
  }

  function finetuneLine(ft) {
    const label = plan.species_label;
    const parts = [];
    parts.push('the saved checkpoint is the best score of the new species on the validation tiles that contain it');
    if (ft.head_lr_mult && ft.head_lr_mult !== 1) parts.push(`the decoder learns ${ft.head_lr_mult}× faster than the encoder`);
    if (ft.head_from === 'random') parts.push(`the ${label} output starts from random weights with bias 0`);
    else parts.push(`the ${label} output starts from the average of the other species`);
    if (ft.novel_fraction) parts.push(`${Math.round(ft.novel_fraction * 100)}% of the samples are tiles with ${label}`);
    parts.push('validation runs every epoch');
    return `Fine-tune: ${parts.join(', ')}.`;
  }

  let settings = $derived(plan?.settings ?? {});
  let two = $derived(plan?.schedule === 'two_stage');
  let editable = $derived(!live || (live.job?.status !== 'running' && live.job?.status !== 'queued'));
  let stages = $derived([
    ['replay', two ? 'Base' : 'Other species'],
    ['novel_train', two ? 'Fine-tune' : 'New species'],
    ['val', 'Validation'],
    ['test', 'Test'],
    ['same_site', 'Same site'],
    ['excluded', 'Left out'],
  ]);
  let open = $state('');
  let mapOpen = $state(false);

  function toggle(folder) {
    open = open === folder ? '' : folder;
    mapOpen = false;
  }

  let opened = $derived.by(() => {
    if (!open || !plan?.columns) return null;
    for (const list of Object.values(plan.columns)) {
      const hit = list.find((flight) => flight.folder === open);
      if (hit) return hit;
    }
    return null;
  });

  function overview(flight, mask) {
    const kind = mask ? 'overview-mask.webp' : 'overview.webp';
    return `/api/mosaic/${encodeURIComponent(flight.site_name)}/${encodeURIComponent(flight.flight)}/${kind}`;
  }
</script>

{#snippet chip(flight)}
  <button
    type="button"
    class:novel={flight.role === 'novel_train'}
    class:out={open === flight.folder}
    aria-expanded={open === flight.folder}
    onclick={() => toggle(flight.folder)}
  >
    <span class="name">{flight.folder}</span>
    <small>
      {flight.tiles} tiles{flight.date ? ` · ${flight.date}` : ''}
      {#if scores[flight.folder] != null} · F1 {f1(scores[flight.folder])}{/if}
    </small>
    <span class="bar" aria-hidden="true">
      {#each flight.species ?? [] as spec (spec.stats_name)}
        {#if spec.share > 0}
          <i style:width="{spec.share * 100}%" style:background={spec.color} title={spec.name}></i>
        {/if}
      {/each}
    </span>
  </button>
{/snippet}

{#snippet mover(flight)}
  {#if editable && onmove}
    <div class="controls">
      <select
        aria-label="Stage for {flight.folder}"
        value={flight.role}
        disabled={busy || queueing}
        onchange={(event) => onmove(flight.folder, event.currentTarget.value)}
      >
        {#each stages as [role, label] (role)}
          <option value={role}>{label}</option>
        {/each}
      </select>
      {#if flight.role !== 'excluded'}
        <button type="button" class="x" aria-label="Leave {flight.folder} out" disabled={busy || queueing} onclick={() => onmove(flight.folder, 'excluded')}>×</button>
      {/if}
    </div>
  {/if}
{/snippet}

{#snippet entry(flight)}
  <div class="flight-row">
    {@render chip(flight)}
    {@render mover(flight)}
  </div>
{/snippet}

<div class="plan">
  <header>
    <h2>{plan.species_label}{#if plan.flight} · {plan.flight}{/if}</h2>
    <p>
      {settings.loss} loss · {settings.model} · {settings.crop ?? '?'} px · seed {settings.seed}
      · lr {settings.lr}
      {#if two && plan.base_exp}
        · fine-tune {settings.ft_epochs} epochs at lr {settings.ft_lr} from the saved base
      {:else if two}
        · base {settings.epochs} epochs, then fine-tune {settings.ft_epochs} epochs at lr {settings.ft_lr}
      {:else}
        · {settings.epochs} epochs
      {/if}
    </p>
    {#if two && settings.finetune}
      <p class="sub">{finetuneLine(settings.finetune)}</p>
    {/if}
  </header>

  {#if plan.warnings?.length}
    <ul class="warn">
      {#each plan.warnings as warning}
        <li>{warning}</li>
      {/each}
    </ul>
  {/if}

  <div class="flow">
    {#if two}
      <section class:on={active === 'base'}>
        {#if plan.base_exp}
          <h3>1. Base, already trained</h3>
          <p class="sub base-name">{plan.base_exp}</p>
          <p>Not trained again. The fine-tune loads its best checkpoint.</p>
        {:else}
          <h3>1. Base, five species</h3>
          <p>{epochLine('base') || `${plan.columns.replay.length} flights. The new species is left out.`}</p>
        {/if}
        <div class="flights">
          {#each plan.columns.replay as flight (flight.folder)}
            {@render entry(flight)}
          {/each}
        </div>
      </section>
      <div class="arrow" aria-hidden="true">→</div>
    {/if}

    <section class:on={active === 'finetune' || active === 'joint'}>
      <h3>{two ? '2. Fine-tune' : 'Train'}</h3>
      <p>{epochLine(two ? 'finetune' : 'joint') || 'One flight of the new species, plus the five species.'}</p>
      <div class="flights">
        {#each plan.columns.novel_train as flight (flight.folder)}
          {@render entry(flight)}
        {/each}
      </div>
      {#if two}
        <p class="sub">Replay keeps the {plan.columns.replay.length} base flights in, so the old species are not forgotten.</p>
      {:else}
        <div class="flights">
          {#each plan.columns.replay as flight (flight.folder)}
            {@render entry(flight)}
          {/each}
        </div>
      {/if}
    </section>

    <div class="arrow" aria-hidden="true">→</div>

    <section>
      <h3>Validation</h3>
      <p class="sub">The checkpoint is the epoch with the highest F1 of {plan.species_label} on the val tiles that contain it. The other species do not raise that score.</p>
      {#if plan.columns.val?.length}
        <p>Flights moved here are validated as a whole. The other training flights still use their own val folder.</p>
        <div class="flights">
          {#each plan.columns.val as flight (flight.folder)}
            {@render entry(flight)}
          {/each}
        </div>
      {:else}
        <p>The val folder of each training flight. Only the new flight has labels of {plan.species_label}.</p>
      {/if}
    </section>

    <div class="arrow" aria-hidden="true">→</div>

    <section class:on={active === 'done'}>
      <h3>Test · {plan.test === 'fold' ? `${plan.fold} held-out sites` : 'other flights'}</h3>
      <p class="sub">Every tile of {plan.species_label}, plus 20 other tiles from each flight.</p>
      {#if live?.metrics?.novel_f1 != null}
        <p>New species F1 {f1(live.metrics.novel_f1)}{subsetLine(live.metrics)}</p>
      {/if}
      <div class="flights">
        {#each plan.columns.test as flight (flight.folder)}
          {@render entry(flight)}
        {/each}
      </div>
    </section>

    {#if plan.columns.same_site.length}
      <section>
        <h3>Same site, other date</h3>
        <p>Scored on its own. Not mixed into the test F1.</p>
        <div class="flights">
          {#each plan.columns.same_site as flight (flight.folder)}
            {@render entry(flight)}
          {/each}
        </div>
      </section>
    {/if}

    {#if plan.columns.excluded?.length}
      <section>
        <h3>Left out</h3>
        <p>Not trained on, not validated on, and not tested on.</p>
        <div class="flights">
          {#each plan.columns.excluded as flight (flight.folder)}
            {@render entry(flight)}
          {/each}
        </div>
      </section>
    {/if}
  </div>

  {#if opened}
    <article class="pulled">
      <header>
        <h3>{opened.folder}</h3>
        <p>{opened.tiles} tiles{opened.date ? ` · ${opened.date}` : ''} · {opened.novel_tiles} with {plan.species_label}{opened.role === 'novel_train' ? '. This is the training flight.' : ''}</p>
        <button type="button" class="ghost" onclick={() => toggle(opened.folder)}>Close</button>
        {@render mover(opened)}
      </header>
      <ul class="mix">
        {#each opened.species ?? [] as spec (spec.stats_name)}
          <li class:none={spec.tiles === 0}><i style:background={spec.color}></i><span class="spec">{spec.name}</span><span class="count">{spec.tiles} tiles · {(spec.share * 100).toFixed(0)}%</span><span class="track"><b style:width="{spec.share * 100}%" style:background={spec.color}></b></span></li>
        {/each}
      </ul>
      <div class="map-toggle">
        <button type="button" class="ghost" aria-expanded={mapOpen} onclick={() => (mapOpen = !mapOpen)}>
          {mapOpen ? 'Hide map' : 'Show map'}
        </button>
        <button type="button" class="ghost" onclick={() => onflight(opened)}>Open full map</button>
      </div>
      {#if mapOpen}
        <div class="preview">
          <img src={overview(opened, false)} alt="Flight {opened.folder}" />
          <img class="mask" src={overview(opened, true)} alt="" />
        </div>
      {/if}
    </article>
  {/if}

  <footer>
    {#if live}
      <p class="status" data-status={live.job?.status}>{live.job?.status} · {live.job?.exp_name}</p>
      {#if live.job?.error}<p class="problem">{live.job.error}</p>{/if}
      <button type="button" class="ghost" onclick={onedit}>New plan</button>
    {:else}
      <button type="button" class="ghost" onclick={onedit}>Edit</button>
    {/if}
    {#if editable}
      <button type="button" class="go" disabled={busy || queueing || (!plan.sweep && !plan.columns.novel_train?.length)} onclick={onrun}>
        {queueing ? 'Queuing…' : plan.sweep ? `Accept and run ${plan.candidates.length} flights` : 'Accept and run'}
      </button>
    {/if}
  </footer>
</div>

<style>
  .plan { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  header h2 { margin: 0 0 4px; font-size: 18px; }
  header p, .sub { margin: 0; color: var(--mist); font-size: 13px; }
  .base-name { overflow-wrap: anywhere; }
  .warn { margin: 0; padding-left: 18px; color: #8c2f2f; font-size: 13px; }
  .flow {
    display: flex;
    gap: 10px;
    align-items: stretch;
    overflow-x: auto;
    padding-bottom: 6px;
  }
  section {
    flex: 1 0 220px;
    max-width: 320px;
    border: 1px solid var(--line);
    background: var(--sheet);
    padding: 10px;
  }
  section.on { border-color: var(--canopy); box-shadow: inset 3px 0 0 var(--canopy); }
  section h3 { margin: 0 0 6px; font-size: 14px; }
  section p { margin: 0 0 8px; font-size: 13px; }
  .arrow { align-self: center; color: var(--mist); font-size: 20px; }
  .flights { display: flex; flex-direction: column; gap: 4px; max-height: 360px; overflow: auto; }
  .flight-row { display: flex; flex-direction: column; gap: 4px; align-items: stretch; }
  .flight-row > button:first-child { width: 100%; }
  .controls { display: flex; gap: 4px; }
  .controls select {
    flex: 1;
    min-width: 0;
    font-size: 12px;
    border: 1px solid var(--line);
    background: white;
    color: var(--ink);
  }
  .controls .x {
    width: 28px;
    flex: none;
    background: white;
    color: #8c2f2f;
    border: 1px solid var(--line);
    font-size: 16px;
    line-height: 1;
    padding: 0;
    text-align: center;
  }
  .controls .x:disabled, .controls select:disabled { opacity: 0.5; }
  .flights button {
    text-align: left;
    background: #f4f7f8;
    color: var(--ink);
    border: 1px solid var(--line);
    padding: 6px 8px;
    font-size: 13px;
  }
  .flights button.novel { background: color-mix(in srgb, var(--canopy) 18%, white); }
  .flights button.out { border-color: var(--canopy); box-shadow: 0 0 0 1px var(--canopy); }
  .flights button .name { display: block; }
  .flights button small { display: block; color: var(--mist); font-size: 11px; }
  .bar { display: flex; height: 6px; margin-top: 4px; background: #e6ebed; overflow: hidden; }
  .bar i { display: block; height: 100%; }
  .pulled {
    border: 1px solid var(--canopy);
    background: var(--sheet);
    padding: 12px;
  }
  .pulled header { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; }
  .pulled h3 { margin: 0; font-size: 16px; }
  .pulled header p { margin: 0; color: var(--mist); font-size: 13px; flex: 1; }
  .mix { list-style: none; margin: 12px 0; padding: 0; display: grid; gap: 8px; }
  .mix li { display: grid; grid-template-columns: 12px 11em 9em minmax(80px, 1fr); gap: 8px; align-items: center; font-size: 13px; }
  .mix li.none { color: var(--mist); }
  .mix i { width: 12px; height: 12px; }
  .mix .track { height: 6px; background: #e6ebed; }
  .mix .track b { display: block; height: 100%; }
  .mix .count { color: var(--mist); font-variant-numeric: tabular-nums; }
  .map-toggle { display: flex; gap: 8px; }
  .preview { position: relative; margin-top: 10px; max-width: 520px; background: #d5dbde; }
  .preview img { display: block; width: 100%; height: auto; }
  .preview .mask { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0.85; }
  footer { display: flex; gap: 8px; align-items: center; }
  .ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
  .go { background: var(--canopy); color: white; border: 0; padding: 8px 14px; }
  .status { margin: 0; font-size: 13px; }
  .problem { color: #8c2f2f; font-size: 13px; margin: 0; }
</style>
