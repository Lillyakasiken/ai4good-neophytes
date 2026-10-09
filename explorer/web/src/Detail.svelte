<script>
  import Process from './Process.svelte';
  import Segment from './Segment.svelte';
  import Zoom from './Zoom.svelte';

  let { tileId, fold, onclose } = $props();

  let tile = $state(null);
  let error = $state('');
  let opacity = $state(0.8);
  let tab = $state('tile');
  let legendOpen = $state(false);
  let maskLegend = $state(null);
  let zoomOpen = $state(false);
  let sharp = $state('');
  let sharpFor = $state('');
  let sharpBusy = $state(false);

  const IGNORE = [
    { kind: 'ignore', color: '#737373', label: 'unlabelled' },
    { kind: 'ignore', color: '#facc15', label: 'uncertain' },
  ];

  const ROLE_LABELS = { train: 'train', val: 'val', test: 'test', unused: 'unused' };

  $effect(() => {
    const id = tileId;
    tile = null;
    error = '';
    zoomOpen = false;
    sharp = '';
    sharpFor = '';
    sharpBusy = false;
    if (!id) return;
    let dead = false;
    fetch('/api/tiles/' + id, { priority: 'high' })
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      })
      .then((data) => {
        if (!dead) tile = data;
      })
      .catch((err) => {
        if (!dead) error = String(err.message || err);
      });
    legendOpen = false;
    maskLegend = null;
    function onKey(event) {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      onclose();
    }
    window.addEventListener('keydown', onKey);
    return () => {
      dead = true;
      window.removeEventListener('keydown', onKey);
    };
  });

  function area(value) {
    if (value == null) return '—';
    if (value === 0) return '0';
    if (value >= 100) return Math.round(value).toLocaleString();
    if (value >= 10) return value.toFixed(1);
    return value.toFixed(2);
  }

  function rgbOf(hex) {
    const n = Number.parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function scanMask(src, palette) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        const hits = new Array(palette.length).fill(0);
        const limit = 55 * 55;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i + 3] < 140) continue;
          let best = -1;
          let bestD = limit;
          for (let p = 0; p < palette.length; p++) {
            const rgb = palette[p].rgb;
            const d = (data[i] - rgb[0]) ** 2 + (data[i + 1] - rgb[1]) ** 2 + (data[i + 2] - rgb[2]) ** 2;
            if (d < bestD) {
              bestD = d;
              best = p;
            }
          }
          if (best >= 0) hits[best] += 1;
        }
        resolve(palette.filter((_, index) => hits[index] > 12));
      };
      img.onerror = () => resolve([]);
      img.src = src;
    });
  }

  $effect(() => {
    const src = tile?.mask;
    const species = tile?.species;
    if (!src || !species) return;
    const palette = [
      ...species.map((spec) => ({
        kind: 'species',
        color: spec.color,
        label: spec.name,
        rgb: rgbOf(spec.color),
      })),
      ...IGNORE.map((item) => ({ ...item, rgb: rgbOf(item.color) })),
    ];
    let dead = false;
    scanMask(src, palette).then((found) => {
      if (!dead) maskLegend = found;
    });
    return () => {
      dead = true;
    };
  });

  let legend = $derived(
    maskLegend ??
      (tile?.species ?? [])
        .filter((spec) => spec.px > 0)
        .map((spec) => ({ kind: 'species', color: spec.color, label: spec.name })),
  );
  let shownIgnore = $derived(legend.filter((item) => item.kind === 'ignore'));

  function px(value) {
    if (value == null || value === 0) return value === 0 ? '0' : '—';
    return Number(value).toLocaleString();
  }

  $effect(() => {
    if (!zoomOpen || !tile?.tile_id || sharpFor === tile.tile_id) return;
    const id = tile.tile_id;
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
<aside class="drawer" class:vision={tab === 'vision'} class:segment={tab === 'segment'}>
  {#if error}
    <p class="error">{error}</p>
  {:else if !tile}
    <p class="empty">Loading tile…</p>
  {:else}
    <header>
      <div>
        <h2>{tile.site_name}_{tile.flight}</h2>
        <p>{tile.acq_date ?? 'date unknown'} · folder {tile.split} · {fold} {tile.roles[fold]}</p>
      </div>
      <button type="button" class="close" onclick={onclose}>Close</button>
    </header>

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
    </div>

    {#if tab === 'vision'}
      <Process tileId={tile.tile_id} mask={tile.mask} />
    {:else if tab === 'segment'}
      <Segment tileId={tile.tile_id} mask={tile.mask} />
    {:else}
      <button type="button" class="stage" aria-label="Enlarge orthophoto" onclick={() => (zoomOpen = true)}>
        <img src={tile.thumb} alt="" fetchpriority="high" draggable="false" />
        <img class="mask" src={tile.mask} alt="" style:opacity={opacity} fetchpriority="high" draggable="false" />
      </button>
      <Zoom open={zoomOpen} onclose={() => (zoomOpen = false)} label="Orthophoto" note={sharpBusy ? 'Loading a sharper view…' : ''}>
        <img src={sharp || tile.thumb} alt="Orthophoto" draggable="false" />
        <img class="mask" src={tile.mask} alt="" style:opacity={opacity} draggable="false" />
      </Zoom>
      <label class="slider">
        Labels
        <input type="range" min="0" max="1" step="0.05" bind:value={opacity} />
      </label>
    {/if}
    <div class="legend">
      {#each legend as item}
        <span>
          <i class="swatch" style:--swatch={item.color}></i>
          {#if item.kind === 'species'}<span class="binomial">{item.label}</span>{:else}{item.label}{/if}
        </span>
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
