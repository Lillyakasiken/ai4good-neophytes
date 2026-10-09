<script>
  import { onMount } from 'svelte';
  import Filters from './Filters.svelte';
  import Stats from './Stats.svelte';
  import Grid from './Grid.svelte';
  import Map from './Map.svelte';
  import Detail from './Detail.svelte';

  let facets = $state(null);
  let stats = $state(null);
  let error = $state('');
  let showLabels = $state(true);
  let openId = $state(null);
  let infoOpen = $state(false);
  let view = $state('tiles');
  let filters = $state({
    fold: 'cv3',
    role: 'train',
    split: '',
    site: '',
    flight: '',
    year: '',
    month: '',
    species: [],
    labelled: false,
    backgroundOnly: false,
    nodataMin: '',
    nodataMax: '',
    minM2: '',
    minM2Species: 'A_altissima',
    sort: 'label_m2',
  });

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
    if (filters.labelled) p.set('labelled', 'true');
    if (filters.backgroundOnly) p.set('background_only', 'true');
    setNumber(p, 'nodata_min', filters.nodataMin);
    setNumber(p, 'nodata_max', filters.nodataMax);
    setNumber(p, 'min_m2', filters.minM2);
    if (p.has('min_m2')) p.set('min_m2_species', filters.minM2Species);
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

  onMount(async () => {
    const res = await fetch('/api/facets');
    if (!res.ok) {
      error = await res.text();
      return;
    }
    facets = await res.json();
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

<div class="app" class:map-mode={view === 'map'}>
  <header>
    <div class="title-wrap">
      <div class="title-row">
        <h1>{filters.fold}<span class="role-name">{ROLE[filters.role] ?? filters.role}</span></h1>
        <button
          type="button"
          class="info"
          aria-label="How this fold is split"
          aria-expanded={infoOpen}
          aria-controls="fold-note"
          onclick={() => (infoOpen = !infoOpen)}
        >i</button>
      </div>
      {#if infoOpen}
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
    </div>
  </header>

  {#if view !== 'map'}
    <Filters {facets} bind:filters bind:showLabels />
  {/if}

  <div class="workspace" class:map-mode={view === 'map'}>
    {#if view === 'map'}
      <Map {facets} {filters} bind:showLabels onopen={(id) => (openId = id)} />
    {:else}
      {#if error}
        <p class="error">{error}</p>
      {:else}
        <Stats {stats} bind:filters />
      {/if}
      <div class="grid-wrap">
        <Grid {query} {showLabels} {colors} onopen={(id) => (openId = id)} />
      </div>
    {/if}
  </div>
</div>

{#if openId}
  <Detail tileId={openId} fold={filters.fold} onclose={() => (openId = null)} />
{/if}
