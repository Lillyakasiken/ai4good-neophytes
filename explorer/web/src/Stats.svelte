<script>
  let { stats, filters = $bindable() } = $props();
  let marginOpen = $state(false);

  const MONTHS = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const ROLE_LABELS = { train: 'Train', val: 'Val', test: 'Test', unused: 'Unused' };

  let totalM2 = $derived((stats?.species ?? []).reduce((sum, row) => sum + row.m2, 0) || 1);
  let totalMonth = $derived((stats?.by_month ?? []).reduce((sum, row) => sum + row.tiles, 0) || 1);
  let totalNodata = $derived((stats?.nodata ?? []).reduce((sum, row) => sum + row.tiles, 0) || 1);

  function area(value) {
    if (value >= 100) return Math.round(value).toLocaleString();
    if (value >= 10) return value.toFixed(0);
    return value.toFixed(1);
  }

  function pct(part, total) {
    if (!total) return '0%';
    const text = ((100 * part) / total).toFixed(1);
    return `${text.endsWith('.0') ? text.slice(0, -2) : text}%`;
  }
</script>

{#if stats}
  <section class="stats">
    <div class="stat-block">
      <h2>Labelled area</h2>
      <div class="roles">
        {#each Object.entries(stats.by_role) as [role, tiles]}
          <button
            type="button"
            class:on={filters.role === role}
            onclick={() => (filters.role = role)}
          >
            {ROLE_LABELS[role]} {tiles.toLocaleString()}
          </button>
        {/each}
      </div>
      <p class="balance">
        <span>{stats.labelled.toLocaleString()} labelled</span>
        <span>{stats.background_only.toLocaleString()} background-only</span>
      </p>
      <p class="hint">share of labelled area (total: {area(totalM2)} m²)</p>
      {#each stats.species as spec}
        <div class="bar-row">
          <span class="binomial">{spec.name}</span>
          <div class="bar" data-pct={pct(spec.m2, totalM2)}>
            <div class="track"><span style:width="{(100 * spec.m2) / totalM2}%" style:--swatch={spec.color}></span></div>
          </div>
          <span class="num">{area(spec.m2)} m²</span>
        </div>
      {/each}
    </div>
    <div class="stat-block">
      <h2>Month</h2>
      <p class="hint">share of tiles (total: {stats.total.toLocaleString()})</p>
      {#each stats.by_month as row}
        <div class="month-row">
          <span>{row.year} {MONTHS[row.month] ?? ''}</span>
          <div class="bar" data-pct={pct(row.tiles, totalMonth)}>
            <div class="track"><span style:width="{(100 * row.tiles) / totalMonth}%"></span></div>
          </div>
          <span class="num">{row.tiles.toLocaleString()}</span>
        </div>
      {/each}
    </div>
    <div class="stat-block">
      <h2>Site</h2>
      <p class="hint">labelled / tiles</p>
      {#each stats.by_site as row}
        <div class="site-row">
          <span>{row.site_name}</span>
          <div class="bar" data-pct={pct(row.labelled, row.tiles)}>
            <div class="track"><span style:width="{row.tiles ? (100 * row.labelled) / row.tiles : 0}%"></span></div>
          </div>
          <span class="num" title="{row.labelled.toLocaleString()} labelled of {row.tiles.toLocaleString()} tiles">{row.labelled.toLocaleString()} / {row.tiles.toLocaleString()}</span>
        </div>
      {/each}
    </div>
    <div class="stat-block">
      <div class="block-title">
        <h2>Empty margin</h2>
        <button
          type="button"
          class="info"
          aria-label="What empty margin means"
          aria-expanded={marginOpen}
          onclick={() => (marginOpen = !marginOpen)}
        >i</button>
      </div>
      {#if marginOpen}
        <p class="legend-note">
          Empty margin is the blank edge of a tile, where the orthophoto does not cover the ground. Flights are cut into a regular grid, so tiles along the outside are partly empty. The percentage is empty pixels divided by all pixels in the tile.
        </p>
      {/if}
      <p class="hint">share of tiles (total: {stats.total.toLocaleString()})</p>
      {#each stats.nodata as bin}
        <div class="month-row">
          <span>{bin.label}</span>
          <div class="bar" data-pct={pct(bin.tiles, totalNodata)}>
            <div class="track"><span style:width="{(100 * bin.tiles) / totalNodata}%"></span></div>
          </div>
          <span class="num">{bin.tiles.toLocaleString()}</span>
        </div>
      {/each}
    </div>
  </section>
{/if}

