<script>
  import { onMount } from 'svelte';
  import PlanDiagram from './PlanDiagram.svelte';

  let { options = null, onopenflight = () => {}, jobs = [], resume = null } = $props();

  const SPECIES = [
    ['A_altissima', 'A. altissima'],
    ['R_typhina', 'R. typhina'],
    ['B_davidii', 'B. davidii'],
    ['R_japonica', 'R. japonica'],
    ['B_orientalis', 'B. orientalis'],
    ['S_inaequidens', 'S. inaequidens'],
  ];

  const NOTES = {
    species: 'Pretend this species is new. Its labels are kept on one flight only.',
    flight: 'The only flight where the model sees this species. Its other tiles are left out, not called background.',
    joint: 'One run on the five species and this flight.',
    two: 'First a model on the five species. Then a short fine-tune on this flight, with the five species kept in. The new output starts from the average of the other species, or from random weights with bias 0. Val every applies to the base. The fine-tune validates every epoch.',
    head: 'Average copies the output row of the other five species, so the new class starts like a typical neophyte. Random draws small weights and sets the bias to 0. Neither needs a lookalike.',
    share: 'Share of training samples drawn from tiles that contain the new species. The rest stay on the five known species.',
    base: 'A finished five-species run. The newest match for this data, model, seed, and species is selected. Train a new base only when you want another.',
    flights: 'Score this species on all its other sites. Those sites are never trained on.',
    fold: "Score this species on the fold's held-out sites.",
    sweep: 'Repeat the run with each site as the training site. One flight per site, the one with the most labels.',
    score: 'The saved checkpoint is the epoch where this one species scores highest on the validation tiles that contain it. Tiles of the other five do not raise that score. The test keeps every tile of this species and 20 other tiles from each flight, so the number is not from the whole flight.',
  };

  let dataName = $state('neophytes_split_cv3_train_512');
  let modelName = $state('model_neophytes_mit_b2');
  let seed = $state(42);
  let maxEpochs = $state(30);
  let every = $state(10);
  let lr = $state(0.0001);
  let batchTrain = $state(8);
  let precision = $state('bf16-mixed');
  let logger = $state('wandb');
  let species = $state('R_typhina');
  let flight = $state('');
  let schedule = $state('two_stage');
  let testMode = $state('flights');
  let sweep = $state(false);
  let baseExp = $state('');
  let ftEpochs = $state(10);
  let ftLr = $state(0.00001);
  let headFrom = $state('mean');
  let novelFraction = $state(0.25);
  let runTest = $state(true);
  let excluded = $state([]);
  let place = $state({});
  let note = $state('');
  let screen = $state('form');
  let plan = $state(null);
  let live = $state(null);
  let jobId = $state('');
  let error = $state('');
  let busy = $state(false);
  let queueing = $state(false);
  let labelled = $state([]);
  let bases = $state([]);
  let basePicked = $state(false);

  function pickBase() {
    if (basePicked && (baseExp === '__new__' || bases.includes(baseExp))) return;
    const needle = `_${dataName}_${modelName}_s${Number(seed)}_nv${species}_base`;
    baseExp = bases.find((name) => name.endsWith(needle)) || '';
  }

  function finetuneSettings() {
    return {
      head_from: headFrom,
      novel_fraction: Number(novelFraction),
    };
  }

  function settings() {
    return {
      species,
      flight: sweep ? '' : flight,
      schedule,
      test: testMode,
      sweep,
      data: dataName,
      model: modelName,
      seed: Number(seed),
      max_epochs: Number(maxEpochs),
      ft_epochs: Number(ftEpochs),
      lr: Number(lr),
      ft_lr: Number(ftLr),
      batch_size_train: Number(batchTrain),
      precision,
      check_val_every_n_epoch: Number(every),
      base_exp: baseExp,
      exclude: excluded,
      place,
      finetune: schedule === 'two_stage' ? finetuneSettings() : null,
    };
  }

  function rememberPlace() {
    sessionStorage.setItem('fewshot-exclude', JSON.stringify(excluded));
    sessionStorage.setItem('fewshot-place', JSON.stringify(place));
  }

  function detachLive() {
    if (!live) return true;
    const status = live.job?.status;
    if (status === 'running' || status === 'queued') return false;
    live = null;
    jobId = '';
    screen = 'plan';
    sessionStorage.removeItem('fewshot-job');
    sessionStorage.setItem('fewshot-screen', 'plan');
    return true;
  }

  async function applyPlace(nextPlace, nextExcluded) {
    const previousPlace = place;
    const previousExcluded = excluded;
    place = nextPlace;
    excluded = nextExcluded;
    rememberPlace();
    error = '';
    busy = true;
    try {
      const res = await fetch('/api/fewshot/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings()),
      });
      if (!res.ok) throw new Error(await readError(res));
      plan = await res.json();
      place = plan.place ?? nextPlace;
      excluded = plan.exclude ?? nextExcluded;
      rememberPlace();
      sessionStorage.setItem('fewshot-plan', JSON.stringify(plan));
    } catch (err) {
      place = previousPlace;
      excluded = previousExcluded;
      rememberPlace();
      error = String(err.message || err);
    } finally {
      busy = false;
    }
  }

  function adoptPlan() {
    if (!plan) return;
    if (plan.species) species = plan.species;
    flight = plan.flight || '';
    if (plan.schedule) schedule = plan.schedule;
    if (plan.test) testMode = plan.test;
    sweep = !!plan.sweep;
    const saved = plan.settings || {};
    if (saved.data) dataName = saved.data;
    if (saved.model) modelName = saved.model;
    if (saved.seed != null) seed = saved.seed;
    if (saved.epochs != null) maxEpochs = saved.epochs;
    if (saved.ft_epochs != null) ftEpochs = saved.ft_epochs;
    if (saved.lr != null) lr = saved.lr;
    if (saved.ft_lr != null) ftLr = saved.ft_lr;
    const ft = saved.finetune;
    if (ft) {
      if (ft.head_from === 'mean' || ft.head_from === 'random') headFrom = ft.head_from;
      if (ft.novel_fraction != null) novelFraction = ft.novel_fraction;
    }
    if (saved.batch_size_train != null) batchTrain = saved.batch_size_train;
    if (saved.precision) precision = saved.precision;
    if (saved.check_val_every_n_epoch != null) every = saved.check_val_every_n_epoch;
  }

  function moveFlight(folder, role) {
    if (!detachLive()) return;
    adoptPlan();
    const nextPlace = { ...(plan?.place || {}), ...place, [folder]: role };
    const stillOut = new Set([...(plan?.exclude || []), ...excluded]);
    for (const [name, placed] of Object.entries(nextPlace)) {
      if (placed === 'excluded') stillOut.add(name);
      else stillOut.delete(name);
    }
    applyPlace(nextPlace, [...stillOut]);
  }

  async function readError(res) {
    try {
      const body = await res.json();
      return body.detail || res.statusText;
    } catch {
      return res.statusText;
    }
  }

  async function loadChoices() {
    const res = await fetch('/api/fewshot/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...settings(), flight: '', sweep: false }),
    });
    if (!res.ok) {
      error = await readError(res);
      return;
    }
    const body = await res.json();
    labelled = body.labelled ?? [];
    bases = body.bases ?? [];
    pickBase();
    if (flight && !labelled.some((row) => row.folder === flight)) flight = '';
    error = '';
  }

  async function showPlan() {
    error = '';
    if (!sweep && flight && (excluded.includes(flight) || (place[flight] && place[flight] !== 'novel_train'))) {
      excluded = excluded.filter((name) => name !== flight);
      const nextPlace = { ...place };
      delete nextPlace[flight];
      place = nextPlace;
      rememberPlace();
    }
    busy = true;
    try {
      const res = await fetch('/api/fewshot/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings()),
      });
      if (!res.ok) throw new Error(await readError(res));
      plan = await res.json();
      bases = plan.bases ?? bases;
      if (baseExp !== '__new__' && plan.base_exp) {
        baseExp = plan.base_exp;
        if (!bases.includes(baseExp)) bases = [baseExp, ...bases];
      }
      live = null;
      jobId = '';
      screen = 'plan';
      sessionStorage.setItem('fewshot-screen', 'plan');
      sessionStorage.setItem('fewshot-plan', JSON.stringify(plan));
    } catch (err) {
      error = String(err.message || err);
    } finally {
      busy = false;
    }
  }

  async function run() {
    error = '';
    queueing = true;
    try {
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataName,
          model: modelName,
          seed: Number(seed),
          max_epochs: Number(maxEpochs),
          check_val_every_n_epoch: Number(every),
          lr: Number(lr),
          batch_size_train: Number(batchTrain),
          batch_size_val: 2,
          precision,
          logger,
          run_test: runTest,
          novel: {
            species,
            flight: sweep ? '' : flight,
            schedule,
            test: testMode,
            sweep,
            base_exp: baseExp,
            ft_epochs: Number(ftEpochs),
            ft_lr: Number(ftLr),
            exclude: excluded,
            place,
            finetune: schedule === 'two_stage' ? finetuneSettings() : null,
          },
        }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const job = await res.json();
      jobId = job.id;
      screen = 'live';
      sessionStorage.setItem('fewshot-job', jobId);
      sessionStorage.setItem('fewshot-screen', 'live');
      await poll();
    } catch (err) {
      error = String(err.message || err);
    } finally {
      queueing = false;
    }
  }

  async function poll() {
    const id = jobId;
    if (!id) return;
    const res = await fetch(`/api/fewshot/state/${id}`);
    if (!res.ok || jobId !== id) return;
    live = await res.json();
    if (jobId !== id) return;
    if (live.plan) plan = live.plan;
  }

  function attach(id) {
    if (!id) return;
    jobId = id;
    screen = 'live';
    sessionStorage.setItem('fewshot-job', id);
    sessionStorage.setItem('fewshot-screen', 'live');
  }

  function shotLabel(job) {
    const shot = job.fewshot || {};
    const name = String(shot.species || '').replaceAll('_', '. ');
    const where = shot.flight || '';
    return [name, where].filter(Boolean).join(' · ');
  }

  let queuedPlans = $derived((jobs ?? []).filter((job) => job.fewshot));

  function edit() {
    screen = 'form';
    live = null;
    sessionStorage.setItem('fewshot-screen', 'form');
  }

  function openFlight(flightRow) {
    onopenflight({ ...flightRow, settings: settings() });
  }

  onMount(() => {
    const saved = sessionStorage.getItem('fewshot-screen');
    const savedPlan = sessionStorage.getItem('fewshot-plan');
    const savedJob = sessionStorage.getItem('fewshot-job');
    if (savedPlan) {
      try { plan = JSON.parse(savedPlan); } catch { /* ignore a bad draft */ }
    }
    if (plan?.place && typeof plan.place === 'object') {
      place = plan.place;
    } else {
      const savedPlace = sessionStorage.getItem('fewshot-place');
      if (savedPlace) {
        try { place = JSON.parse(savedPlace); } catch { /* ignore a bad draft */ }
      }
    }
    if (Array.isArray(plan?.exclude)) {
      excluded = plan.exclude;
    } else {
      const savedExclude = sessionStorage.getItem('fewshot-exclude');
      if (savedExclude) {
        try { excluded = JSON.parse(savedExclude); } catch { /* ignore a bad draft */ }
      }
    }
    adoptPlan();
    if (saved === 'plan' || saved === 'live') screen = saved;
    if (savedJob) jobId = savedJob;
  });

  $effect(() => {
    if (resume?.id) attach(resume.id);
  });

  $effect(() => {
    if (!options) return;
    if (species && dataName && modelName && seed != null && testMode) loadChoices();
  });

  $effect(() => {
    if (screen !== 'live' || !jobId) return;
    const timer = setInterval(poll, 4000);
    poll();
    return () => clearInterval(timer);
  });
</script>

<section class="fewshot card">
  <div class="head">
    <h2>Few-shot</h2>
    <span class="note-wrap">
      <button type="button" class="info" aria-label="What is scored" aria-expanded={note === 'score'} onclick={() => (note = note === 'score' ? '' : 'score')}>i</button>
      {#if note === 'score'}<div class="test-note" role="dialog"><p>{NOTES.score}</p></div>{/if}
    </span>
  </div>

  {#if error}<p class="problem" role="alert">{error}</p>{/if}

  {#if screen === 'form' && queuedPlans.length}
    <ul class="return">
      {#each queuedPlans as job (job.id)}
        <li>
          <span class="status" data-status={job.status}>{job.status}</span>
          <span>{shotLabel(job)}</span>
          <button type="button" onclick={() => attach(job.id)}>Open plan</button>
        </li>
      {/each}
    </ul>
  {/if}

  {#if screen === 'form'}
    <div class="grid">
      <label>
        Data
        <select bind:value={dataName}>
          {#each options?.data ?? [dataName] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <label>
        Model
        <select bind:value={modelName}>
          {#each options?.models ?? [modelName] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <label>
        Species
        <span class="row">
          <select bind:value={species}>
            {#each SPECIES as [name, label]}
              <option value={name}>{label}</option>
            {/each}
          </select>
          <button type="button" class="info" aria-label="Novel species" aria-expanded={note === 'species'} onclick={() => (note = note === 'species' ? '' : 'species')}>i</button>
        </span>
        {#if note === 'species'}<p class="hint">{NOTES.species}</p>{/if}
      </label>
      <label>
        Training flight
        <span class="row">
          <select bind:value={flight} disabled={sweep}>
            <option value="">{sweep ? 'Every site' : 'Pick a flight'}</option>
            {#each labelled as row (row.folder)}
              <option value={row.folder} disabled={testMode === 'fold' && !row.fold_train}>
                {row.folder} · {row.novel_tiles} tiles{row.date ? ` · ${row.date}` : ''}{row.fold_train ? '' : ' · not in this fold'}
              </option>
            {/each}
          </select>
          <button type="button" class="info" aria-label="Training flight" aria-expanded={note === 'flight'} onclick={() => (note = note === 'flight' ? '' : 'flight')}>i</button>
        </span>
        {#if note === 'flight'}<p class="hint">{NOTES.flight}</p>{/if}
      </label>
      <fieldset>
        <legend>
          Schedule
          <button type="button" class="info" aria-label="Schedule" aria-expanded={note === 'schedule'} onclick={() => (note = note === 'schedule' ? '' : 'schedule')}>i</button>
        </legend>
        <label><input type="radio" bind:group={schedule} value="joint" /> Joint</label>
        <label><input type="radio" bind:group={schedule} value="two_stage" /> Two-stage</label>
        {#if note === 'schedule'}<p class="hint">{schedule === 'joint' ? NOTES.joint : NOTES.two}</p>{/if}
      </fieldset>
      <fieldset>
        <legend>
          Test
          <button type="button" class="info" aria-label="Test set" aria-expanded={note === 'test'} onclick={() => (note = note === 'test' ? '' : 'test')}>i</button>
        </legend>
        <label><input type="radio" bind:group={testMode} value="flights" /> Other flights</label>
        <label><input type="radio" bind:group={testMode} value="fold" /> This fold</label>
        {#if note === 'test'}<p class="hint">{testMode === 'flights' ? NOTES.flights : NOTES.fold}</p>{/if}
      </fieldset>
      {#if schedule === 'two_stage'}
        <label>
          Base run
          <span class="row">
            <select bind:value={baseExp} onchange={() => (basePicked = true)}>
              {#each bases as name}
                <option value={name}>{name}</option>
              {/each}
              <option value="__new__">Train a new base</option>
            </select>
            <button type="button" class="info" aria-label="Base run" aria-expanded={note === 'base'} onclick={() => (note = note === 'base' ? '' : 'base')}>i</button>
          </span>
          {#if note === 'base'}<p class="hint">{NOTES.base}</p>{/if}
        </label>
        <label>Fine-tune epochs <input type="number" min="1" max="500" bind:value={ftEpochs} /></label>
        <label>Fine-tune lr <input type="number" min="0.00000001" step="0.00001" bind:value={ftLr} /></label>
        <fieldset class="wide">
          <legend>Fine-tune</legend>
          <div class="grid inner">
            <label>
              New species starts from
              <span class="row">
                <select bind:value={headFrom}>
                  <option value="mean">Average of the other species</option>
                  <option value="random">Random, bias 0</option>
                </select>
                <button type="button" class="info" aria-label="New output" aria-expanded={note === 'head'} onclick={() => (note = note === 'head' ? '' : 'head')}>i</button>
              </span>
              {#if note === 'head'}<p class="hint">{NOTES.head}</p>{/if}
            </label>
            <label>
              New species share
              <span class="row">
                <input type="number" min="0.01" max="0.9" step="0.05" bind:value={novelFraction} />
                <button type="button" class="info" aria-label="New species share" aria-expanded={note === 'share'} onclick={() => (note = note === 'share' ? '' : 'share')}>i</button>
              </span>
              {#if note === 'share'}<p class="hint">{NOTES.share}</p>{/if}
            </label>
          </div>
        </fieldset>
      {/if}
      <label>Epochs <input type="number" min="1" max="500" bind:value={maxEpochs} /></label>
      <label>Val every <input type="number" min="1" max="500" bind:value={every} /></label>
      <label>Learning rate <input type="number" min="0.00000001" step="0.0001" bind:value={lr} /></label>
      <label>Seed <input type="number" bind:value={seed} /></label>
      <label>Batch <input type="number" min="1" max="64" bind:value={batchTrain} /></label>
      <label>
        Precision
        <select bind:value={precision}>
          {#each options?.precisions ?? [precision] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <label>
        Logger
        <select bind:value={logger}>
          {#each options?.loggers ?? ['wandb', 'csv'] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <label class="check">
        <input type="checkbox" bind:checked={sweep} />
        Queue every site
        <button type="button" class="info" aria-label="Sweep" aria-expanded={note === 'sweep'} onclick={() => (note = note === 'sweep' ? '' : 'sweep')}>i</button>
      </label>
      {#if note === 'sweep'}<p class="hint">{NOTES.sweep}</p>{/if}
      <label class="check">
        <input type="checkbox" bind:checked={runTest} />
        Run the test when training finishes
      </label>
    </div>
    <button type="button" class="go" disabled={busy || (!sweep && !flight)} onclick={showPlan}>
      {busy ? 'Building…' : 'Show plan'}
    </button>
  {:else if plan}
    <PlanDiagram {plan} {live} {busy} {queueing} onflight={openFlight} onedit={edit} onrun={run} onmove={moveFlight} />
  {/if}
</section>

<style>
  .fewshot { margin: 16px; padding: 16px; }
  .head { display: flex; align-items: center; gap: 8px; }
  .head h2 { margin: 0; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    margin: 12px 0;
  }
  label, fieldset { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--mist); }
  fieldset { border: 1px solid var(--line); padding: 8px; }
  legend { padding: 0 4px; }
  select, input { font-size: 13px; color: var(--ink); padding: 6px; }
  .row { display: flex; gap: 6px; align-items: center; }
  .row select { flex: 1; }
  .check { flex-direction: row; align-items: center; color: var(--ink); font-size: 13px; }
  .hint { margin: 4px 0 0; color: var(--mist); font-size: 12px; }
  .problem { color: #8c2f2f; font-size: 13px; }
  .return { list-style: none; margin: 12px 0 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
  .return li { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .return button { margin-left: auto; background: transparent; color: var(--canopy); border: 1px solid var(--line); padding: 4px 10px; }
  .status { text-transform: lowercase; font-size: 12px; color: var(--mist); }
  .wide { grid-column: 1 / -1; }
  .inner { margin: 0; }
  .go { background: var(--canopy); color: white; border: 0; padding: 8px 14px; }
  fieldset label { flex-direction: row; align-items: center; color: var(--ink); font-size: 13px; }
</style>
