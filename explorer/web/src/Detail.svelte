<script>
  import { tick } from 'svelte';
  import CopyButton from './CopyButton.svelte';
  import Process from './Process.svelte';
  import Segment from './Segment.svelte';
  import Predict from './Predict.svelte';
  import Zoom from './Zoom.svelte';
  import { paintIsolate } from './isolate.js';

  let { tileId, fold, onclose, onopenmodel = null, initialTab = 'tile', initialCkpt = '' } = $props();

  let tile = $state(null);
  let activeId = $state(null);
  let error = $state('');
  let loading = $state(false);
  let opacity = $state(0.8);
  // Remounted with the tab to open. Later switches stay on this drawer.
  // svelte-ignore state_referenced_locally
  let tab = $state(initialTab || 'tile');
  let legendOpen = $state(false);
  let focusColor = $state('');
  let isolatedMask = $state('');
  let zoomOpen = $state(false);
  let sharp = $state('');
  let sharpFor = $state('');
  let sharpBusy = $state(false);
  // A failed season switch puts the selection back. Skip the fetch that write would start.
  let ignoreNext = false;

  const MONTHS = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  const IGNORE = {
    254: { kind: 'ignore', color: '#737373', label: 'unlabelled' },
    255: { kind: 'ignore', color: '#facc15', label: 'uncertain' },
  };

  const ROLE_LABELS = { train: 'train', val: 'val', test: 'test', unused: 'unused' };

  $effect(() => {
    const id = tileId;
    activeId = id;
    tile = null;
    error = '';
    loading = false;
    legendOpen = false;
    focusColor = '';
    isolatedMask = '';
    zoomOpen = false;
    sharp = '';
    sharpFor = '';
    sharpBusy = false;
  });

  $effect(() => {
    function onKey(event) {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      onclose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  $effect(() => {
    const id = activeId;
    if (!id) return;
    if (ignoreNext) {
      ignoreNext = false;
      return;
    }
    const ctrl = new AbortController();
    let dead = false;
    loading = true;
    error = '';
    fetch('/api/tiles/' + id, { priority: 'high', signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      })
      .then((data) => {
        if (!dead) tile = data;
      })
      .catch((err) => {
        if (dead || err.name === 'AbortError') return;
        error = String(err.message || err);
        if (tile?.tile_id && tile.tile_id !== id) {
          ignoreNext = true;
          activeId = tile.tile_id;
        }
      })
      .finally(() => {
        if (!dead) loading = false;
      });
    return () => {
      dead = true;
      ctrl.abort();
    };
  });

  function dayOf(acq) {
    if (!acq || String(acq).length < 10) return '';
    const day = Number(String(acq).slice(8, 10));
    return Number.isFinite(day) ? String(day) : '';
  }

  function seasonLabel(item) {
    const month = MONTHS[item.month] ?? item.month ?? '';
    const day = dayOf(item.acq_date);
    const when = [day, month, item.year].filter(Boolean).join(' ');
    return item.tile_id ? when : `No tile here on ${when}`;
  }

  function chooseSeason(id) {
    if (!id || id === activeId) return;
    activeId = id;
  }

  async function onSeasonKey(event) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const group = event.currentTarget.closest('.seasons');
    const buttons = [...group.querySelectorAll('button.season:not(:disabled)')];
    const index = buttons.indexOf(event.currentTarget);
    const next = buttons[index + (event.key === 'ArrowRight' ? 1 : -1)];
    if (!next) return;
    event.preventDefault();
    next.click();
    await tick();
    group.querySelector('button.season.on')?.focus();
  }

  let seasons = $derived(tile?.seasons ?? []);
  let seasonGroups = $derived.by(() => {
    const groups = [];
    for (const item of seasons) {
      const year = item.year ?? '';
      let group = groups.at(-1);
      if (!group || group.year !== year) {
        group = { year, items: [] };
        groups.push(group);
      }
      group.items.push(item);
    }
    return groups;
  });
  let switching = $derived(Boolean(loading && tile && tile.tile_id !== activeId));

  function area(value) {
    if (value == null) return '—';
    if (value === 0) return '0';
    if (value >= 100) return Math.round(value).toLocaleString();
    if (value >= 10) return value.toFixed(1);
    return value.toFixed(2);
  }

  let legend = $derived.by(() => {
    const species = tile?.species ?? [];
    const ids = tile?.label_ids;
    if (!ids) {
      return species
        .filter((spec) => spec.px > 0)
        .map((spec) => ({ kind: 'species', color: spec.color, label: spec.name }));
    }
    const present = new Set(ids);
    const items = species
      .filter((spec) => present.has(spec.id))
      .map((spec) => ({ kind: 'species', color: spec.color, label: spec.name }));
    for (const id of [254, 255]) {
      if (present.has(id)) items.push(IGNORE[id]);
    }
    return items;
  });
  let shownIgnore = $derived(legend.filter((item) => item.kind === 'ignore'));

  function px(value) {
    if (value == null || value === 0) return value === 0 ? '0' : '—';
    return Number(value).toLocaleString();
  }

  $effect(() => {
    const src = tile?.mask;
    const color = focusColor;
    if (!src || !color) {
      isolatedMask = '';
      return;
    }
    let dead = false;
    paintIsolate(src, color).then((url) => {
      if (!dead) isolatedMask = url;
    }).catch(() => {
      if (!dead) isolatedMask = '';
    });
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    if (!zoomOpen || !tile?.tile_id || sharpFor === tile.tile_id) return;
    const id = tile.tile_id;
    sharp = '';
    const ctrl = new AbortController();
    let dead = false;
    sharpBusy = true;
    fetch(`/api/tiles/${id}/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_edge: 1024, ops: [] }),
      signal: ctrl.signal,
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      })
      .then((data) => {
        if (dead) return;
        sharpFor = id;
        sharp = data?.image ? `data:image/jpeg;base64,${data.image}` : '';
        sharpBusy = false;
      })
      .catch((err) => {
        if (dead || err.name === 'AbortError') return;
        sharp = '';
        sharpBusy = false;
      });
    return () => {
      dead = true;
      ctrl.abort();
    };
  });
</script>

<button type="button" class="drawer-back" aria-label="Close details" onclick={onclose}></button>
<aside class="drawer" class:vision={tab === 'vision'} class:segment={tab === 'segment' || tab === 'predict'}>
  {#if !tile && !error}
    <p class="empty">Loading tile…</p>
  {:else if !tile}
    <p class="error">{error}</p>
  {:else}
    <header>
      <div>
        <h2>{tile.site_name}_{tile.flight}</h2>
        <p>{tile.acq_date ?? 'date unknown'} · folder {tile.split} · {fold} {tile.roles[fold]}</p>
      </div>
      <button type="button" class="close" onclick={onclose}>Close</button>
    </header>

    {#if seasons.length > 1}
      <div class="seasons" role="group" aria-label="Season of this tile" aria-busy={switching}>
        {#each seasonGroups as group (group.year)}
          <div class="season-year">
            <span class="season-year-label">{group.year}</span>
            <div class="season-months">
              {#each group.items as item (`${group.year}-${item.month}-${item.flight}`)}
                <button
                  type="button"
                  class="season"
                  class:on={item.tile_id === activeId}
                  class:off={!item.tile_id}
                  class:pending={switching && item.tile_id === activeId}
                  aria-pressed={item.tile_id ? item.tile_id === activeId : undefined}
                  aria-label={seasonLabel(item)}
                  title={seasonLabel(item)}
                  disabled={!item.tile_id}
                  onclick={() => chooseSeason(item.tile_id)}
                  onkeydown={onSeasonKey}
                >
                  <span class="season-name">{MONTHS[item.month] ?? item.month}</span>
                  <span class="season-day">{dayOf(item.acq_date)}</span>
                </button>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    {/if}
    {#if error}
      <div class="season-error">{error}</div>
    {/if}

    <div class="drawer-tabs" role="tablist" aria-label="Tile detail">
      <button type="button" role="tab" aria-selected={tab === 'tile'} class:on={tab === 'tile'} onclick={() => (tab = 'tile')}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <rect x="1.6" y="2.2" width="12.8" height="11.6" />
          <path d="M1.6 10.4 5.2 7.4l2.6 2.2 2.4-2 4.2 3.2" />
          <circle cx="5.2" cy="5.6" r="1" />
        </svg>
        Tile
      </button>
      <button type="button" role="tab" aria-selected={tab === 'vision'} class:on={tab === 'vision'} onclick={() => (tab = 'vision')}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="7.2" cy="7.2" r="4.4" />
          <path d="M10.4 10.4 14 14" />
          <path d="M7.2 4.6v1.3M7.2 8.5v1.3M4.6 7.2h1.3M8.5 7.2h1.3" />
        </svg>
        Vision
      </button>
      <button type="button" role="tab" aria-selected={tab === 'segment'} class:on={tab === 'segment'} onclick={() => (tab = 'segment')}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M3.1 5.6 6.2 2.7l4.1 1.1L13.2 6.4l-1.1 4.1-3.6 2.2-4.3-1.3L2.6 8.1Z" />
        </svg>
        Segment
      </button>
      <button type="button" role="tab" aria-selected={tab === 'predict'} class:on={tab === 'predict'} onclick={() => (tab = 'predict')}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M2.2 11.2 6.2 4.4l2.2 3.2 1.6-2.2 3.8 5.8" />
          <path d="M2 13.2h12" />
        </svg>
        Predict
      </button>
    </div>

    {#if tab === 'vision'}
      <Process tileId={tile.tile_id} mask={tile.mask} />
    {:else if tab === 'segment'}
      <Segment tileId={tile.tile_id} mask={tile.mask} />
    {:else if tab === 'predict'}
      <Predict tileId={tile.tile_id} thumb={tile.thumb} mask={tile.mask} bind:focusColor {onopenmodel} {initialCkpt} />
    {:else}
      <div class="shot">
        <button type="button" class="stage" class:busy={switching} aria-label="Enlarge orthophoto" onclick={() => (zoomOpen = true)}>
          <img class="photo" class:dim={!!focusColor} src={tile.thumb} alt="" fetchpriority="high" draggable="false" />
          <img class="mask" src={isolatedMask || tile.mask} alt="" style:opacity={opacity} fetchpriority="high" draggable="false" />
        </button>
        <CopyButton overlay />
      </div>
      <Zoom open={zoomOpen} onclose={() => (zoomOpen = false)} label="Orthophoto" note={sharpBusy ? 'Loading a sharper view…' : ''} bind:opacity>
        <img class="photo" class:dim={!!focusColor} src={sharp || tile.thumb} alt="Orthophoto" draggable="false" />
        <img class="mask" src={isolatedMask || tile.mask} alt="" style:opacity={opacity} draggable="false" />
      </Zoom>
      <label class="slider">
        Labels
        <input type="range" min="0" max="1" step="0.05" bind:value={opacity} />
      </label>
    {/if}
    <div class="legend">
      {#each legend as item}
        {#if item.kind === 'species'}
          <button
            type="button"
            class:on={focusColor === item.color}
            style:--swatch={item.color}
            aria-pressed={focusColor === item.color}
            title="Show only this class"
            onclick={() => (focusColor = focusColor === item.color ? '' : item.color)}
          >
            <i class="swatch" style:--swatch={item.color}></i>
            <span class="binomial">{item.label}</span>
          </button>
        {:else}
          <span>
            <i class="swatch" style:--swatch={item.color}></i>
            {item.label}
          </span>
        {/if}
      {/each}
      {#if shownIgnore.length}
        <button
          type="button"
          class="info"
          aria-label="What the grey and yellow marks mean"
          aria-expanded={legendOpen}
          onclick={() => (legendOpen = !legendOpen)}
        >i</button>
      {/if}
    </div>
    {#if legendOpen}
      <p class="legend-note">
        {#if shownIgnore.some((item) => item.label === 'unlabelled')}Grey was never labelled. {/if}
        {#if shownIgnore.some((item) => item.label === 'uncertain')}Yellow is a label the annotator marked as unsure. {/if}
        Training skips these pixels, so they count toward neither the loss nor the score.
      </p>
    {/if}

    <div class="meta">
      <div><span>Size</span><strong>{tile.width ?? '—'} × {tile.height ?? '—'}</strong></div>
      <div><span>GSD</span><strong>{tile.gsd_mm == null ? '—' : `${tile.gsd_mm.toFixed(2)} mm/px`}</strong></div>
      <div><span>Empty margin</span><strong>{(100 * (tile.nodata_frac || 0)).toFixed(1)}%</strong></div>
      <div><span>Labelled area</span><strong>{area(tile.label_m2)} m²</strong></div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Species</th>
          <th>m²</th>
          <th>non-fl.</th>
          <th>flower</th>
          <th>fruit</th>
          <th>unknown</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Background</td>
          <td>{area(tile.background.m2)}</td>
          <td colspan="4"></td>
        </tr>
        {#each tile.species as spec}
          <tr>
            <td><span class="binomial"><i class="swatch" style:--swatch={spec.color}></i> {spec.name}</span></td>
            <td>{area(spec.m2)}</td>
            {#each spec.phases as phase}
              <td>{area(phase.m2)}</td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>

    <h3>Role in each fold</h3>
    <div class="roles">
      {#each Object.entries(tile.roles) as [name, role]}
        <button type="button" class:on={name === fold} disabled>
          {name} {ROLE_LABELS[role]}
        </button>
      {/each}
    </div>

    <div class="paths">
      <div>{tile.img_path}</div>
      <div>{tile.mask_path}</div>
      <div>{px(tile.nodata_px)} empty px / {px(tile.total_px)} px</div>
    </div>
  {/if}
</aside>

<style>
  h3 { font-size: 13px; margin: 16px 0 6px; }
  p { margin: 2px 0 0; color: var(--muted); }
</style>
