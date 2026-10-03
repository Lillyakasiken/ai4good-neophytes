<script>
  let { facets, filters = $bindable(), showLabels = $bindable() } = $props();

  const MONTHS = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const ROLES = [
    ['all', 'All roles'],
    ['train', 'Train'],
    ['val', 'Validation'],
    ['test', 'Test'],
    ['unused', 'Unused'],
  ];

  let flights = $derived(
    (facets?.flights ?? []).filter((row) => !filters.site || row.site_name === filters.site),
  );

  let sortOptions = $derived.by(() => {
    const base = [
      ['label_m2', 'Labelled area'],
      ['nodata_frac', 'Empty margin'],
      ['acq_date', 'Date'],
      ['site_name', 'Site'],
    ];
    for (const spec of facets?.species ?? []) {
      base.push([`${spec.stats_name}_m2`, `${spec.name} area`]);
    }
    return base;
  });

  function toggleSpecies(name) {
    filters.species = filters.species.includes(name)
      ? filters.species.filter((item) => item !== name)
      : [...filters.species, name];
  }
</script>

<div class="filters">
  <label class="field">
    Fold
    <select bind:value={filters.fold}>
      {#each ['cv1', 'cv2', 'cv3', 'cv4', 'cv5'] as fold}
        <option value={fold}>{fold}</option>
      {/each}
    </select>
  </label>
  <label class="field">
    CV role
    <select bind:value={filters.role}>
      {#each ROLES as [value, label]}
        <option {value}>{label}</option>
      {/each}
    </select>
  </label>
  <label class="field">
    Folder
    <select bind:value={filters.split}>
      <option value="">Any</option>
      <option value="train">train</option>
      <option value="val">val</option>
      <option value="test">test</option>
    </select>
  </label>
  <label class="field">
    Site
    <select
      value={filters.site}
      onchange={(e) => {
        filters.site = e.currentTarget.value;
        filters.flight = '';
      }}
    >
      <option value="">All sites</option>
      {#each facets?.sites ?? [] as site}
        <option value={site}>{site}</option>
      {/each}
    </select>
  </label>
  <label class="field">
    Flight
    <select bind:value={filters.flight}>
      <option value="">All</option>
      {#each flights as row}
        <option value={row.flight}>{filters.site ? row.flight : `${row.site_name} / ${row.flight}`}</option>
      {/each}
    </select>
  </label>
  <label class="field">
    Year
    <select bind:value={filters.year}>
      <option value="">Any</option>
      {#each facets?.years ?? [] as year}
        <option value={year}>{year}</option>
      {/each}
    </select>
  </label>
  <label class="field">
    Month
    <select bind:value={filters.month}>
      <option value="">Any</option>
      {#each facets?.months ?? [] as month}
        <option value={month}>{MONTHS[month] ?? month}</option>
      {/each}
    </select>
  </label>
  <label class="field">
    Sort
    <select bind:value={filters.sort}>
      {#each sortOptions as [value, label]}
        <option {value}>{label}</option>
      {/each}
    </select>
  </label>
  <label class="field">
    Empty margin
    <span>
      <input type="number" min="0" max="1" step="0.05" placeholder="min" bind:value={filters.nodataMin} />
      <input type="number" min="0" max="1" step="0.05" placeholder="max" bind:value={filters.nodataMax} />
    </span>
  </label>
  <label class="field">
    Min m²
    <span>
      <input type="number" min="0" step="1" placeholder="0" bind:value={filters.minM2} />
      <select bind:value={filters.minM2Species}>
        {#each facets?.species ?? [] as spec}
          <option value={spec.stats_name}>{spec.name}</option>
        {/each}
      </select>
    </span>
  </label>
  <label class="check">
    <input
      type="checkbox"
      checked={filters.labelled}
      onchange={(e) => {
        filters.labelled = e.currentTarget.checked;
        if (filters.labelled) filters.backgroundOnly = false;
      }}
    />
    Labelled only
  </label>
  <label class="check">
    <input
      type="checkbox"
      checked={filters.backgroundOnly}
      onchange={(e) => {
        filters.backgroundOnly = e.currentTarget.checked;
        if (filters.backgroundOnly) filters.labelled = false;
      }}
    />
    Background only
  </label>
  <label class="check">
    <input type="checkbox" bind:checked={showLabels} />
    Show labels
  </label>
  <div class="species">
    {#each facets?.species ?? [] as spec}
      <button
        type="button"
        class:on={filters.species.includes(spec.stats_name)}
        style:--swatch={spec.color}
        onclick={() => toggleSpecies(spec.stats_name)}
      >
        <i class="swatch" style:--swatch={spec.color}></i>
        <span class="binomial">{spec.name}</span>
      </button>
    {/each}
  </div>
</div>
