<script>
  import { onMount } from 'svelte';
  import Filters from './Filters.svelte';
  import Stats from './Stats.svelte';
  import Grid from './Grid.svelte';
  import Map from './Map.svelte';
  import Detail from './Detail.svelte';
  import Runs from './Runs.svelte';
  import Model from './Model.svelte';
  import Batch from './Batch.svelte';

  let facets = $state(null);
  let stats = $state(null);
  let error = $state('');
  let showLabels = $state(true);
  let openId = $state(null);
  let detailTab = $state('tile');
  let detailCkpt = $state('');
  let detailKey = $state(0);
  let infoOpen = $state(false);
  let view = $state('tiles');
  let modelTileId = $state(null);
  let gpu = $state(null);
  let watchGpu = $state(storedWatchGpu());

  function defaultFilters() {
    return {
      fold: 'cv3',
      role: 'train',
      split: '',
      site: '',
      flight: '',
      year: '',
      month: '',
      species: [],
      speciesOp: 'or',
      labelled: false,
      backgroundOnly: false,
      nodataMin: '',
      nodataMax: '',
      minM2: '',
      maxM2: '',
      m2Species: 'label_m2',
      sort: 'label_m2',
    };
  }

  let filters = $state(defaultFilters());
  let filtersChanged = $derived.by(() => {
    const base = defaultFilters();
    for (const key of Object.keys(base)) {
      if (key === 'species') {
        if (filters.species.length > 0) return true;
        continue;
      }
      if (filters[key] !== base[key]) return true;
    }
    return false;
  });

  function resetFilters() {
    filters = defaultFilters();
  }

  function openModel(id) {
    if (id) modelTileId = id;
    openId = null;
    view = 'model';
  }

  function openTile(id) {
    detailTab = 'tile';
    detailCkpt = '';
    openId = id;
    detailKey += 1;
  }

  function openBatchTile(id, ckpt) {
    detailTab = 'predict';
    detailCkpt = ckpt || '';
    openId = id;
    detailKey += 1;
  }

  function setNumber(params, key, value) {
    if (value == null || value === '' || Number.isNaN(Number(value))) return;
    params.set(key, String(value));
  }

  let query = $derived.by(() => {
    const p = new URLSearchParams();
    p.set('fold', filters.fold);
    p.set('role', filters.role || 'all');
    if (filters.split) p.set('split', filters.split);
    if (filters.site) p.set('site', filters.site);
    if (filters.flight) p.set('flight', filters.flight);
    if (filters.year) p.set('year', filters.year);
    if (filters.month) p.set('month', filters.month);
    for (const name of filters.species) p.append('species', name);
    if (filters.species.length) p.set('species_op', filters.speciesOp || 'or');
    if (filters.labelled) p.set('labelled', 'true');
    if (filters.backgroundOnly) p.set('background_only', 'true');
    setNumber(p, 'nodata_min', filters.nodataMin);
    setNumber(p, 'nodata_max', filters.nodataMax);
    setNumber(p, 'min_m2', filters.minM2);
    setNumber(p, 'max_m2', filters.maxM2);
    if (p.has('min_m2') || p.has('max_m2')) p.set('m2_species', filters.m2Species);
    p.set('sort', filters.sort);
    return p.toString();
  });

  const ROLE = {
    all: 'all roles',
    train: 'train',
    val: 'validation',
    test: 'test',
    unused: 'unused',
  };

  let heldOut = $derived(facets?.folds?.[filters.fold]?.test_sites ?? []);
  let colors = $derived(Object.fromEntries((facets?.species ?? []).map((s) => [s.stats_name, s.color])));

  function storedWatchGpu() {
    try {
      return localStorage.getItem('explorer.watchGpu') !== '0';
    } catch {
      return true;
    }
  }

  function gb(mib) {
    if (mib == null) return '—';
    const value = mib / 1024;
    return value >= 10 ? value.toFixed(0) : value.toFixed(1);
  }

  let gpuText = $derived.by(() => {
    if (!gpu) return watchGpu ? 'GPU …' : 'GPU';
    if (!gpu.available || !gpu.gpus?.length) return 'GPU n/a';
    return gpu.gpus.map((card) => {
      const util = card.util == null ? '—' : `${Math.round(card.util)}%`;
      return `GPU ${util} · ${gb(card.memory_used)}/${gb(card.memory_total)} GB`;
    }).join('   ');
  });

  onMount(async () => {
    const res = await fetch('/api/facets');
    if (!res.ok) {
      error = await res.text();
      return;
    }
    facets = await res.json();
  });

  $effect(() => {
    try {
      localStorage.setItem('explorer.watchGpu', watchGpu ? '1' : '0');
    } catch {
      /* storage unavailable */
    }
  });

  $effect(() => {
    if (!watchGpu) return;
    let dead = false;
    async function poll() {
      while (!dead) {
        try {
          const res = await fetch('/api/gpu');
          if (res.ok && !dead) gpu = await res.json();
        } catch {
          /* keep the last readout */
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }
    poll();
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    const q = query;
    let dead = false;
    fetch('/api/stats?' + q)
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      })
      .then((data) => {
        if (!dead) {
          stats = data;
          error = '';
        }
      })
      .catch((err) => {
        if (!dead) error = String(err.message || err);
      });
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    if (!infoOpen) return;
    function onKey(event) {
      if (event.key === 'Escape') infoOpen = false;
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });
</script>

<div class="app" class:map-mode={view === 'map'} class:runs-mode={view === 'runs'} class:model-mode={view === 'model'} class:batch-mode={view === 'batch'}>
  <header>
    <div class="title-wrap">
      <div class="title-row">
        <h1>
          {#if view === 'runs'}
            Runs
          {:else if view === 'model'}
            Model
          {:else if view === 'batch'}
            Batch
          {:else}
            {filters.fold}<span class="role-name">{ROLE[filters.role] ?? filters.role}</span>
          {/if}
        </h1>
        {#if view !== 'runs' && view !== 'model' && view !== 'batch'}
          <button
            type="button"
            class="info"
            aria-label="How this fold is split"
            aria-expanded={infoOpen}
            aria-controls="fold-note"
            onclick={() => (infoOpen = !infoOpen)}
          >i</button>
        {/if}
      </div>
      {#if infoOpen && view !== 'runs' && view !== 'model' && view !== 'batch'}
        <div class="note" id="fold-note" role="dialog" aria-label="How this fold is split">
          <p>
            The default baseline is a U-Net with a MiT-b2 encoder. It trains on the {filters.fold} split below.
          </p>
          <p>
            Each site on disk is already cut into <em>train</em>, <em>val</em>, and <em>test</em> tiles.
            That folder split is not what {filters.fold} trains on.
          </p>
          {#if heldOut.length}
            <p>Held out in {filters.fold}: {heldOut.join(', ')}. The model does not train on these sites.</p>
          {/if}
          <ul>
            <li><strong>Train</strong> is the train and test folders of every site that is not held out.</li>
            <li><strong>Validation</strong> is the val folders of those same sites.</li>
            <li><strong>Test</strong> is only the train folders of the held-out sites.</li>
            <li><strong>Unused</strong> is the val and test folders of the held-out sites. They stay out of the score, so a later run can use a few labelled tiles from a site the model has never been tested on.</li>
          </ul>
          <p>CV role shows one of those four groups. Folder is the on-disk split, which is a different cut.</p>
        </div>
      {/if}
    </div>
    <div class="header-tools">
      <label class="gpu-watch" title="Keep reading GPU state">
        <input type="checkbox" bind:checked={watchGpu} aria-label="Live GPU refresh" />
        <span class="gpu-chip" class:paused={!watchGpu}>{gpuText}</span>
      </label>
    <div class="view-switch" role="tablist" aria-label="Explorer view">
      <button type="button" role="tab" aria-selected={view === 'tiles'} class:on={view === 'tiles'} onclick={() => (view = 'tiles')}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <rect x="1.2" y="1.2" width="5.2" height="5.2" />
          <rect x="9.6" y="1.2" width="5.2" height="5.2" />
          <rect x="1.2" y="9.6" width="5.2" height="5.2" />
          <rect x="9.6" y="9.6" width="5.2" height="5.2" />
        </svg>
        Tiles
      </button>
      <button type="button" role="tab" aria-selected={view === 'map'} class:on={view === 'map'} onclick={() => (view = 'map')}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M1.5 3.2 5.5 1.5l5 1.8 4-1.6v10.1l-4 1.7-5-1.8-4 1.5V3.2z" />
          <path d="M5.5 1.5v10.2M10.5 3.3v10.2" />
        </svg>
        Map
      </button>
      <button type="button" role="tab" aria-selected={view === 'model'} class:on={view === 'model'} onclick={() => openModel(openId)}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M8 1.6 14.2 4.8 8 8 1.8 4.8Z" />
          <path d="M1.8 7.2 8 10.4 14.2 7.2" />
          <path d="M1.8 9.8 8 13 14.2 9.8" />
        </svg>
        Model
      </button>
      <button type="button" role="tab" aria-selected={view === 'batch'} class:on={view === 'batch'} onclick={() => (view = 'batch')}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <rect x="1.4" y="1.4" width="8.2" height="8.2" />
          <rect x="6.4" y="6.4" width="8.2" height="8.2" />
        </svg>
        Batch
      </button>
      <button type="button" role="tab" aria-selected={view === 'runs'} class:on={view === 'runs'} onclick={() => (view = 'runs')}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M2 12.5 5.2 8.4l2.4 2.2 3-4.2L14 11" />
          <path d="M2 3.5h12" />
        </svg>
        Runs
      </button>
    </div>
    </div>
  </header>

  {#if view === 'tiles'}
    <Filters {facets} bind:filters bind:showLabels changed={filtersChanged} onreset={resetFilters} />
  {/if}

  <div class="workspace" class:map-mode={view === 'map'} class:runs-mode={view === 'runs'} class:model-mode={view === 'model'} class:batch-mode={view === 'batch'}>
    {#if view === 'model'}
      <Model {facets} initialTileId={modelTileId} fold={filters.fold} />
    {:else if view === 'batch'}
      <Batch {facets} fold={filters.fold} onopen={openBatchTile} />
    {:else if view === 'runs'}
      <Runs {watchGpu} {gpu} />
    {:else if view === 'map'}
      <Map {facets} bind:filters bind:showLabels onopen={openTile} />
    {:else}
      {#if error}
        <p class="error">{error}</p>
      {:else}
        <Stats {stats} bind:filters />
      {/if}
      <div class="grid-wrap">
        <Grid {query} {showLabels} {colors} onopen={openTile} />
      </div>
    {/if}
  </div>
</div>

{#if openId}
  {#key detailKey}
    <Detail tileId={openId} fold={filters.fold} initialTab={detailTab} initialCkpt={detailCkpt} onclose={() => (openId = null)} onopenmodel={openModel} />
  {/key}
{/if}
