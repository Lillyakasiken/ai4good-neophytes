<script>
  let { runs = [], selected = '', onselect } = $props();

  let kFilter = $state('all');
  let seedFilter = $state('all');
  let panel = $state(null);

  function groupName(run) {
    const keys = run.train_site || [];
    return keys.length ? keys.join('+') : 'All sites';
  }

  function f1(value) {
    return value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(2);
  }

  function scoreLine(run) {
    if (!run.tested) return 'not tested yet';
    return `same site F1 ${f1(run.same_site_f1)} · held-out F1 ${f1(run.held_out_f1)}`;
  }

  let kChoices = $derived(
    [...new Set(runs.map((run) => run.k).filter((value) => value != null))].sort((a, b) => a - b),
  );
  let seedChoices = $derived(
    [...new Set(runs.map((run) => run.seed).filter((value) => value != null))].sort((a, b) => a - b),
  );
  let visible = $derived(
    runs.filter((run) => {
      if (kFilter !== 'all' && String(run.k) !== kFilter) return false;
      if (seedFilter !== 'all' && String(run.seed) !== seedFilter) return false;
      return true;
    }),
  );
  let groups = $derived.by(() => {
    const buckets = new Map();
    for (const run of visible) {
      const name = groupName(run);
      if (!buckets.has(name)) buckets.set(name, []);
      buckets.get(name).push(run);
    }
    for (const rows of buckets.values()) {
      rows.sort((a, b) => (a.k ?? 1e9) - (b.k ?? 1e9)
        || (a.bg ?? 0) - (b.bg ?? 0)
        || (a.seed ?? 0) - (b.seed ?? 0)
        || (b.mtime || 0) - (a.mtime || 0));
    }
    return [...buckets.entries()].sort((a, b) => {
      if (a[0] === 'All sites') return 1;
      if (b[0] === 'All sites') return -1;
      return a[0].localeCompare(b[0]);
    });
  });
  let flat = $derived(groups.flatMap(([, rows]) => rows));
  let selectedRun = $derived(flat.find((run) => run.exp_name === selected) ?? null);
  let groupRuns = $derived.by(() => {
    if (!selectedRun) return flat;
    const name = groupName(selectedRun);
    return groups.find(([key]) => key === name)?.[1] ?? flat;
  });

  function step(delta) {
    const rows = groupRuns;
    if (!rows.length) return;
    const index = rows.findIndex((run) => run.exp_name === selected);
    const next = rows[(index + delta + rows.length) % rows.length];
    onselect?.(next);
  }

  function onKey(event) {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    step(event.key === 'ArrowDown' ? 1 : -1);
  }

  $effect(() => {
    const name = selected;
    if (!name || !panel) return;
    const row = panel.querySelector(`[data-run="${CSS.escape(name)}"]`);
    row?.scrollIntoView({ block: 'nearest' });
  });
</script>

<aside class="map-runs" bind:this={panel} tabindex="0" onkeydown={onKey} aria-label="Training runs">
  <div class="map-runs-head">
    <p class="map-runs-title">Runs in this fold</p>
    <div class="map-runs-chips" role="group" aria-label="Labelled tiles per species">
      <button type="button" class:on={kFilter === 'all'} onclick={() => (kFilter = 'all')}>any k</button>
      {#each kChoices as value (value)}
        <button type="button" class:on={kFilter === String(value)} onclick={() => (kFilter = String(value))}>
          k{value}
        </button>
      {/each}
    </div>
    <div class="map-runs-chips" role="group" aria-label="Seed">
      <button type="button" class:on={seedFilter === 'all'} onclick={() => (seedFilter = 'all')}>any seed</button>
      {#each seedChoices as value (value)}
        <button type="button" class:on={seedFilter === String(value)} onclick={() => (seedFilter = String(value))}>
          seed {value}
        </button>
      {/each}
    </div>
  </div>
  <div class="map-runs-list">
    {#if !runs.length}
      <p class="map-runs-empty">No runs for this fold yet.</p>
    {:else if !groups.length}
      <p class="map-runs-empty">No runs match these filters.</p>
    {:else}
      {#each groups as [name, rows] (name)}
        <h3>{name} <span>{rows.length}</span></h3>
        {#each rows as run (run.exp_name)}
          <button
            type="button"
            class="map-run"
            class:on={run.exp_name === selected}
            data-run={run.exp_name}
            aria-pressed={run.exp_name === selected}
            onclick={() => onselect?.(run)}
          >
            <span class="map-run-line">
              {run.k == null ? 'full split' : `k${run.k}`} · bg{run.bg ?? 0} · seed {run.seed ?? '?'}
            </span>
            <span class="map-run-score">{scoreLine(run)}</span>
          </button>
        {/each}
      {/each}
    {/if}
  </div>
  <div class="map-runs-nav">
    <button type="button" onclick={() => step(-1)} disabled={!groupRuns.length}>Previous</button>
    <button type="button" onclick={() => step(1)} disabled={!groupRuns.length}>Next</button>
  </div>
</aside>
