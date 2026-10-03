<script>
  import { untrack } from 'svelte';

  let { query, showLabels, colors, onopen } = $props();

  const PAGE = 60;
  const GAP = 10;
  const MIN = 168;

  let scroller = $state(null);
  let width = $state(0);
  let viewportH = $state(0);
  let scrollTop = $state(0);
  let total = $state(0);
  let ready = $state(false);
  let failed = $state('');
  let pages = $state({});
  let inflight = new Set();
  let generation = 0;

  // clientWidth includes the scroller's horizontal padding (18px each side).
  let inner = $derived(Math.max(0, width - 36));
  let cols = $derived(Math.max(1, Math.floor((Math.max(inner, MIN) + GAP) / (MIN + GAP))));
  let cellW = $derived(inner > 0 ? Math.max(1, (inner - GAP * (cols - 1)) / cols) : MIN);
  let cellH = $derived(cellW + 28);
  let stride = $derived(cellH + GAP);
  let rowCount = $derived(total > 0 ? Math.ceil(total / cols) : 0);
  let firstRow = $derived(stride > 0 ? Math.max(0, Math.floor(scrollTop / stride) - 1) : 0);
  let visRows = $derived(Math.max(1, Math.ceil(((viewportH || 640) / stride) + 3)));
  let lastRow = $derived(Math.min(rowCount, firstRow + visRows));
  let totalH = $derived(rowCount * stride);

  let cells = $derived.by(() => {
    const out = [];
    for (let row = firstRow; row < lastRow; row++) {
      for (let col = 0; col < cols; col++) {
        const index = row * cols + col;
        if (index >= total) break;
        const page = Math.floor(index / PAGE) * PAGE;
        out.push({
          index,
          tile: pages[page]?.[index - page] ?? null,
          x: col * (cellW + GAP),
          y: row * stride,
        });
      }
    }
    return out;
  });

  async function load(key, offset, gen) {
    const token = `${gen}:${offset}`;
    if (inflight.has(token)) return;
    inflight.add(token);
    try {
      const p = new URLSearchParams(key);
      p.set('limit', String(PAGE));
      p.set('offset', String(offset));
      const res = await fetch('/api/tiles?' + p);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (gen !== generation) return;
      total = data.total;
      ready = true;
      pages = { ...pages, [offset]: data.tiles };
    } catch (err) {
      if (gen === generation) failed = String(err.message || err);
    }
  }

  // Reset only when the filter query changes. Writes stay untracked so the
  // arriving page does not clear itself and refetch forever.
  $effect(() => {
    const key = query;
    generation += 1;
    const gen = generation;
    untrack(() => {
      inflight = new Set();
      pages = {};
      total = 0;
      ready = false;
      failed = '';
      scrollTop = 0;
      if (scroller) scroller.scrollTop = 0;
      load(key, 0, gen);
    });
  });

  $effect(() => {
    const key = query;
    const gen = generation;
    const offsets = new Set();
    for (const cell of cells) offsets.add(Math.floor(cell.index / PAGE) * PAGE);
    untrack(() => {
      for (const offset of offsets) {
        if (!pages[offset]) load(key, offset, gen);
      }
    });
  });
</script>

<div
  class="scroller"
  bind:this={scroller}
  bind:clientWidth={width}
  bind:clientHeight={viewportH}
  onscroll={() => (scrollTop = scroller.scrollTop)}
>
  {#if failed}
    <p class="error">{failed}</p>
  {:else if ready && total === 0}
    <p class="empty">No tiles match these filters.</p>
  {:else}
    <div class="spacer" style:height="{totalH}px">
      {#each cells as cell (cell.index)}
        {#if cell.tile}
          <button
            type="button"
            class="cell"
            style:width="{cellW}px"
            style:height="{cellH}px"
            style:transform="translate({cell.x}px, {cell.y}px)"
            onclick={() => onopen(cell.tile.tile_id)}
          >
            <span class="frame" style:height="{cellW}px">
              <img
                src={cell.tile.thumb}
                alt=""
                onerror={(e) => e.currentTarget.classList.add('missing')}
              />
              {#if showLabels}
                <img class="mask" src={cell.tile.mask} alt="" style:opacity="0.85" />
              {/if}
            </span>
            <span class="caption">
              <strong>{cell.tile.site_name}_{cell.tile.flight}</strong>
              <span class="dots">
                {#each cell.tile.species as name}
                  <i class="swatch" style:--swatch={colors[name] || '#999'}></i>
                {/each}
              </span>
            </span>
          </button>
        {:else}
          <div
            class="skel"
            style:width="{cellW}px"
            style:height="{cellH}px"
            style:transform="translate({cell.x}px, {cell.y}px)"
          ></div>
        {/if}
      {/each}
    </div>
  {/if}
</div>
