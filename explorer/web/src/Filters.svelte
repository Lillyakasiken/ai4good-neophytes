<script>
  import { tick } from 'svelte';

  let { facets, filters = $bindable(), showLabels = $bindable(), labelLayer = $bindable('species'), changed = false, onreset } = $props();

  // Same limits as explorer/api.py. Used until /api/facets returns.
  const FALLBACK = { nodata_min: 0, nodata_max: 1, m2_min: 0, m2_max: 40 };

  let bounds = $derived({ ...FALLBACK, ...(facets?.bounds ?? {}) });

  const MONTHS = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const ROLES = [
    ['all', 'All roles'],
    ['train', 'Train'],
    ['val', 'Validation'],
    ['test', 'Test'],
    ['unused', 'Unused'],
  ];

  let sortOptions = $derived.by(() => {
    const base = [
      ['label_m2', 'Labelled area'],
      ['nodata_frac', 'Empty margin'],
      ['acq_date', 'Date'],
      ['site_name', 'Site'],
      ['nonflower_m2', 'Non-flowering area'],
      ['flower_m2', 'Flowering area'],
      ['fruiting_m2', 'Fruiting area'],
    ];
    for (const spec of facets?.species ?? []) {
      base.push([`${spec.stats_name}_m2`, `${spec.name} area`]);
    }
    return base;
  });

  let years = $derived(facets?.years ?? []);
  let months = $derived(facets?.months ?? []);
  let flightsInScope = $derived(
    (facets?.flights ?? []).filter((row) => !filters.site || row.site_name === filters.site),
  );
  let flightAt = $derived.by(() => {
    const at = new Map();
    for (const row of flightsInScope) {
      if (row.year == null || row.month == null) continue;
      const key = `${row.year}-${row.month}`;
      const list = at.get(key);
      if (list) list.push(row);
      else at.set(key, [row]);
    }
    return at;
  });
  let timeline = $derived(
    years.map((year) => ({
      year,
      quiet: !months.some((month) => flightAt.has(`${year}-${month}`)),
      months: months.map((month) => ({
        month,
        rows: flightAt.get(`${year}-${month}`) ?? [],
      })),
    })),
  );

  function dayOf(acq) {
    if (!acq || String(acq).length < 10) return '';
    const day = Number(String(acq).slice(8, 10));
    return Number.isFinite(day) ? String(day) : '';
  }

  function sameDay(rows) {
    const days = [...new Set(rows.map((row) => dayOf(row.acq_date)).filter(Boolean))];
    return days.length === 1 ? days[0] : '';
  }

  function dateChosen(year, month) {
    return filters.year !== '' && String(filters.year) === String(year) && String(filters.month) === String(month);
  }

  function dateLabel(rows, year, month) {
    const day = sameDay(rows);
    const when = day ? `${day} ${MONTHS[month]} ${year}` : `${MONTHS[month] ?? month} ${year}`;
    return filters.site ? `${filters.site}, ${when}` : when;
  }

  function chooseDate(year, month) {
    if (dateChosen(year, month)) {
      filters.year = '';
      filters.month = '';
      filters.flight = '';
      return;
    }
    filters.year = year;
    filters.month = month;
    if (!filters.site) {
      filters.flight = '';
      return;
    }
    const row = (facets?.flights ?? []).find(
      (item) =>
        item.site_name === filters.site &&
        String(item.year) === String(year) &&
        String(item.month) === String(month),
    );
    filters.flight = row ? String(row.flight) : '';
  }

  async function onTimelineKey(event) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const group = event.currentTarget.closest('.map-time');
    const buttons = [...group.querySelectorAll('button.map-month')];
    const index = buttons.indexOf(event.currentTarget);
    const next = buttons[index + (event.key === 'ArrowRight' ? 1 : -1)];
    if (!next) return;
    event.preventDefault();
    next.click();
    await tick();
    group?.querySelector('.map-month.on')?.focus();
  }

  function chooseSite(value) {
    filters.site = value;
    if (!value || filters.year === '' || filters.month === '') {
      filters.flight = '';
      return;
    }
    const row = (facets?.flights ?? []).find(
      (item) =>
        item.site_name === value &&
        String(item.year) === String(filters.year) &&
        String(item.month) === String(filters.month),
    );
    if (!row) {
      filters.year = '';
      filters.month = '';
      filters.flight = '';
      return;
    }
    filters.flight = String(row.flight);
  }

  const SPECIES_OPS = [
    ['or', 'OR', 'Any selected species'],
    ['and', 'AND', 'Every selected species'],
    ['xor', 'XOR', 'An odd number of the selected species'],
    ['nor', 'NOR', 'None of the selected species'],
    ['nand', 'NAND', 'Not every selected species'],
  ];

  let opMenu = $state(false);
  let opEl = $state(null);
  let speciesOp = $derived(SPECIES_OPS.find((item) => item[0] === filters.speciesOp) ?? SPECIES_OPS[0]);

  function toggleSpecies(name) {
    filters.species = filters.species.includes(name)
      ? filters.species.filter((item) => item !== name)
      : [...filters.species, name];
  }

  const STAGE_FALLBACK = [
    { phase: 'nonflower', label: 'non-flowering', color: '#1b7f4e' },
    { phase: 'flower', label: 'flowering', color: '#f08c14' },
    { phase: 'fruiting', label: 'fruiting', color: '#5b2d8e' },
  ];

  let stageChoices = $derived(
    (facets?.phases ?? STAGE_FALLBACK).filter((phase) => phase.phase !== 'unknown'),
  );

  function toggleStage(name) {
    filters.stages = filters.stages.includes(name)
      ? filters.stages.filter((item) => item !== name)
      : [...filters.stages, name];
  }

  function setSpeciesOp(value) {
    filters.speciesOp = value;
    opMenu = false;
  }

  $effect(() => {
    if (!opMenu) return;
    function onPointer(event) {
      if (opEl?.contains(event.target)) return;
      opMenu = false;
    }
    function onKey(event) {
      if (event.key === 'Escape') opMenu = false;
    }
    window.addEventListener('pointerdown', onPointer);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('pointerdown', onPointer);
      window.removeEventListener('keydown', onKey);
    };
  });

  function pairLimit(raw) {
    if (raw === '' || raw == null) return null;
    if (typeof raw === 'string' && (raw === '.' || raw.endsWith('.'))) return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }

  function limit(raw, lo, hi) {
    if (raw === '') return '';
    if (raw.startsWith('-')) return String(lo);
    const numeric = Number(raw);
    if (!/^\d*\.?\d*$/.test(raw)) {
      if (!Number.isFinite(numeric)) return '';
      return String(Math.min(hi, Math.max(lo, numeric)));
    }
    if (raw === '.' || raw.endsWith('.')) {
      const head = raw === '.' ? 0 : Number(raw.slice(0, -1));
      if (head > hi) return String(hi);
      if (head < lo) return String(lo);
      return raw;
    }
    if (!Number.isFinite(numeric)) return '';
    if (numeric < lo) return String(lo);
    if (numeric > hi) return String(hi);
    return raw;
  }

  function writeBound(event, key, lo, hi) {
    const next = limit(event.currentTarget.value, lo, hi);
    if (event.currentTarget.value !== next) event.currentTarget.value = next;
    if (filters[key] === next) return;
    if (key === 'nodataMin') filters.nodataMin = next;
    else if (key === 'nodataMax') filters.nodataMax = next;
    else if (key === 'minM2') filters.minM2 = next;
    else filters.maxM2 = next;
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
    <select value={filters.site} onchange={(e) => chooseSite(e.currentTarget.value)}>
      <option value="">All sites</option>
      {#each facets?.sites ?? [] as site}
        <option value={site}>{site}</option>
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
      <input
        type="number"
        min={bounds.nodata_min}
        max={bounds.nodata_max}
        step="0.05"
        placeholder="min"
        title={`Share of the tile, from ${bounds.nodata_min} to ${bounds.nodata_max}`}
        value={filters.nodataMin}
        oninput={(e) => {
          const other = pairLimit(filters.nodataMax);
          const hi = other == null ? bounds.nodata_max : Math.min(bounds.nodata_max, other);
          writeBound(e, 'nodataMin', bounds.nodata_min, hi);
        }}
      />
      <input
        type="number"
        min={bounds.nodata_min}
        max={bounds.nodata_max}
        step="0.05"
        placeholder="max"
        title={`Share of the tile, from ${bounds.nodata_min} to ${bounds.nodata_max}`}
        value={filters.nodataMax}
        oninput={(e) => {
          const other = pairLimit(filters.nodataMin);
          const lo = other == null ? bounds.nodata_min : Math.max(bounds.nodata_min, other);
          writeBound(e, 'nodataMax', lo, bounds.nodata_max);
        }}
      />
    </span>
  </label>
  <label class="field">
    Min m²
    <input
      type="number"
      min={bounds.m2_min}
      max={bounds.m2_max}
      step="1"
      placeholder="0"
      title={`Square metres, from ${bounds.m2_min} to ${bounds.m2_max}`}
      value={filters.minM2}
      oninput={(e) => {
        const other = pairLimit(filters.maxM2);
        const hi = other == null ? bounds.m2_max : Math.min(bounds.m2_max, other);
        writeBound(e, 'minM2', bounds.m2_min, hi);
      }}
    />
  </label>
  <label class="field">
    Max m²
    <input
      type="number"
      min={bounds.m2_min}
      max={bounds.m2_max}
      step="1"
      placeholder="0"
      title={`Square metres, from ${bounds.m2_min} to ${bounds.m2_max}`}
      value={filters.maxM2}
      oninput={(e) => {
        const other = pairLimit(filters.minM2);
        const lo = other == null ? bounds.m2_min : Math.max(bounds.m2_min, other);
        writeBound(e, 'maxM2', lo, bounds.m2_max);
      }}
    />
  </label>
  <label class="field">
    Area
    <select bind:value={filters.m2Species}>
      <option value="label_m2">Labelled area</option>
      {#each facets?.species ?? [] as spec}
        <option value={spec.stats_name}>{spec.name}</option>
      {/each}
    </select>
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
  <div class="layer-switch" role="group" aria-label="Label colours">
    <button type="button" class:on={labelLayer === 'species'} aria-pressed={labelLayer === 'species'} onclick={() => (labelLayer = 'species')}>Species</button>
    <button type="button" class:on={labelLayer === 'stage'} aria-pressed={labelLayer === 'stage'} onclick={() => (labelLayer = 'stage')}>Stage</button>
  </div>
  <button type="button" class="reset" disabled={!changed} onclick={onreset}>Reset filters</button>
  <div class="species">
    <div class="species-op" bind:this={opEl}>
      <button
        type="button"
        class="op"
        class:set={filters.speciesOp !== 'or'}
        aria-haspopup="listbox"
        aria-expanded={opMenu}
        aria-label="Match selected species with {speciesOp[1]}: {speciesOp[2]}"
        title={speciesOp[2]}
        onclick={() => (opMenu = !opMenu)}
      >{speciesOp[1]}</button>
      {#if opMenu}
        <div class="op-menu" role="listbox" aria-label="How selected species combine">
          {#each SPECIES_OPS as [value, label, hint]}
            <button
              type="button"
              role="option"
              aria-selected={filters.speciesOp === value}
              class:on={filters.speciesOp === value}
              onclick={() => setSpeciesOp(value)}
            >
              <strong>{label}</strong>
              <span>{hint}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
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
  <div class="species stages">
    <span class="stage-label">Stage</span>
    {#each stageChoices as phase (phase.phase)}
      <button
        type="button"
        class:on={filters.stages.includes(phase.phase)}
        style:--swatch={phase.color}
        onclick={() => toggleStage(phase.phase)}
      >
        <i class="swatch" style:--swatch={phase.color}></i>
        {phase.label}
      </button>
    {/each}
  </div>
</div>

<div class="map-time" role="group" aria-label={filters.site ? `Flights of ${filters.site} by date` : 'Flights by date'}>
  {#each timeline as group (group.year)}
    <div class="map-year" class:quiet={group.quiet}>
      <span class="map-year-label">{group.year}</span>
      <div class="map-months">
        {#each group.months as cell (`${group.year}-${cell.month}`)}
          {#if cell.rows.length}
            <button
              type="button"
              class="map-month"
              class:on={dateChosen(group.year, cell.month)}
              aria-pressed={dateChosen(group.year, cell.month)}
              aria-label={dateLabel(cell.rows, group.year, cell.month)}
              onclick={() => chooseDate(group.year, cell.month)}
              onkeydown={onTimelineKey}
            >
              <span class="map-dot" aria-hidden="true"></span>
              <span class="map-month-name">{MONTHS[cell.month]}</span>
              <span class="map-month-day">{sameDay(cell.rows)}</span>
            </button>
          {:else}
            <span class="map-month off">
              <span class="map-dot" aria-hidden="true"></span>
              <span class="map-month-name">{MONTHS[cell.month]}</span>
              <span class="map-month-day"></span>
            </span>
          {/if}
        {/each}
      </div>
    </div>
  {/each}
</div>
