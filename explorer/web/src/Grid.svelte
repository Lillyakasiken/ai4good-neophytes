<script>
  import { untrack } from 'svelte';

  let { query, showLabels, colors, onopen } = $props();

  const PAGE = 48;

  let scroller = $state(null);
  let page = $state(0);
  let jump = $state('1');
  let total = $state(0);
  let tiles = $state([]);
  let ready = $state(false);
  let loading = $state(false);
  let failed = $state('');
  let generation = 0;
  let alive = true;

  let pageCount = $derived(total > 0 ? Math.ceil(total / PAGE) : 0);
  let rangeStart = $derived(total === 0 ? 0 : page * PAGE + 1);
  let rangeEnd = $derived(Math.min(total, (page + 1) * PAGE));
  let links = $derived.by(() => {
    if (pageCount < 1) return [];
    const current = page + 1;
    const wanted = [1, pageCount, current - 2, current - 1, current, current + 1, current + 2];
    const nums = [...new Set(wanted.filter((n) => n >= 1 && n <= pageCount))].sort((a, b) => a - b);
    const out = [];
    for (let i = 0; i < nums.length; i++) {
      if (i > 0 && nums[i] - nums[i - 1] > 1) out.push({ kind: 'gap', id: `gap-${nums[i]}` });
      out.push({ kind: 'page', n: nums[i], id: `page-${nums[i]}` });
    }
    return out;
  });

  async function load(key, pageIndex, gen, clear) {
    loading = true;
    if (clear) {
      tiles = [];
      total = 0;
      ready = false;
    }
    try {
      const params = new URLSearchParams(key);
      params.set('limit', String(PAGE));
      params.set('offset', String(pageIndex * PAGE));
      const res = await fetch('/api/tiles?' + params);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (!alive || gen !== generation) return;
      const pages = data.total > 0 ? Math.ceil(data.total / PAGE) : 1;
      if (data.total > 0 && pageIndex > pages - 1) {
        page = pages - 1;
        jump = String(page + 1);
        await load(key, page, gen, clear);
        return;
      }
      total = data.total;
      tiles = data.tiles;
      ready = true;
      loading = false;
      if (scroller) scroller.scrollTop = 0;
    } catch (err) {
      if (!alive || gen !== generation) return;
      failed = String(err.message || err);
      loading = false;
      ready = true;
    }
  }

  function show(next) {
    const pages = Math.max(1, pageCount || 1);
    const clamped = Math.min(Math.max(0, next), pages - 1);
    jump = String(clamped + 1);
    if (clamped === page && ready) return;
    page = clamped;
    generation += 1;
    load(query, clamped, generation, false);
  }

  function commitJump() {
    const n = Number.parseInt(String(jump).trim(), 10);
    if (!Number.isFinite(n)) {
      jump = String(page + 1);
      return;
    }
    show(n - 1);
  }

  $effect(() => {
    const key = query;
    generation += 1;
    const gen = generation;
    alive = true;
    untrack(() => {
      page = 0;
      jump = '1';
      failed = '';
      load(key, 0, gen, true);
    });
    return () => {
      alive = false;
    };
  });
</script>

<div class="tiles-pane">
  <div class="scroller" bind:this={scroller}>
    {#if failed}
      <p class="error">{failed}</p>
    {:else if ready && total === 0}
      <p class="empty">No tiles match these filters.</p>
    {:else if !ready}
      <div class="mosaic" aria-hidden="true">
        {#each { length: PAGE } as _, i (i)}
          <div class="skel"></div>
        {/each}
      </div>
    {:else}
      <div class="mosaic" class:loading aria-busy={loading}>
        {#each tiles as tile (tile.tile_id)}
          <button type="button" class="cell" onclick={() => onopen(tile.tile_id)}>
            <span class="frame">
              <img
                src={tile.thumb}
                alt=""
                onerror={(e) => e.currentTarget.classList.add('missing')}
              />
              {#if showLabels}
                <img class="mask" src={tile.mask} alt="" />
              {/if}
            </span>
            <span class="caption">
              <strong>{tile.site_name}_{tile.flight}</strong>
              <span class="dots">
                {#each tile.species as name}
                  <i class="swatch" style:--swatch={colors[name] || '#999'}></i>
                {/each}
              </span>
            </span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  {#if total > 0}
    <nav class="pager" aria-label="Tile pages">
      <button type="button" disabled={page === 0} onclick={() => show(page - 1)}>Previous</button>
      <div class="pages">
        {#each links as item (item.id)}
          {#if item.kind === 'gap'}
            <span class="gap" aria-hidden="true">…</span>
          {:else}
            <button
              type="button"
              aria-current={item.n === page + 1 ? 'page' : undefined}
              onclick={() => show(item.n - 1)}
            >{item.n}</button>
          {/if}
        {/each}
      </div>
      <button type="button" disabled={page >= pageCount - 1} onclick={() => show(page + 1)}>Next</button>
      <form
        class="jump"
        onsubmit={(event) => {
          event.preventDefault();
          commitJump();
        }}
      >
        <label>
          Page
          <input inputmode="numeric" bind:value={jump} aria-label="Page number" />
        </label>
      </form>
      <p class="count">{rangeStart.toLocaleString()}–{rangeEnd.toLocaleString()} of {total.toLocaleString()}</p>
    </nav>
  {/if}
</div>
