<script>
  let { facets = null, initialTileId = null, fold = 'cv3' } = $props();

  let picks = $state([]);
  let picksLoading = $state(true);
  let tile = $state(null);
  let tileId = $state(null);

  let checkpoints = $state([]);
  let checkpointsLoading = $state(true);
  let listError = $state('');
  let runs = $state([]);
  let scoresReady = $state(false);
  let scoresNote = $state('');
  let ckpt = $state('');
  let touched = $state(false);
  let pickedCkpt = false;

  let tech = $state(storedTech());
  let trace = $state(null);
  let traceFor = $state('');
  let busy = $state(false);
  let error = $state('');
  let attempt = $state(0);

  let stageId = $state('input');
  let hoverId = $state('');
  let featView = $state('pca');
  let channelAt = $state(0);
  let gridOn = $state(false);
  let classId = $state(1);
  let mapMode = $state('prediction');
  let probe = $state(null);
  let pinned = $state(false);
  let pickerOpen = $state(false);
  let browse = $state([]);
  let browseLoading = $state(false);
  let siteFilter = $state('');
  let roleFilter = $state('all');
  let speciesFilter = $state('');

  let speciesColor = $derived(Object.fromEntries((facets?.species ?? []).map((row) => [row.stats_name, row.color])));
  let ranked = $derived(rankCheckpoints(checkpoints, runs));
  let shown = $derived(trace && traceFor === `${tileId}:${ckpt}` ? trace : null);
  let photoUrl = $derived(shown?.image ? `data:image/jpeg;base64,${shown.image}` : '');
  let layout = $derived(shown ? layoutGraph(shown) : blankLayout());
  let node = $derived(layout.nodes.find((item) => item.id === stageId) ?? null);
  let stageKind = $derived(node?.kind || '');
  let feature = $derived(node && (node.kind === 'encoder' || node.kind === 'decoder') ? node : null);
  let mixParents = $derived.by(() => {
    if (!node || node.kind !== 'decoder') return [];
    const byId = new Map(layout.nodes.map((item) => [item.id, item]));
    return (node.parents || []).map((id) => byId.get(id)).filter(Boolean);
  });
  let featureUrl = $derived.by(() => {
    if (!feature) return '';
    if (featView === 'energy') return png(feature.energy);
    if (featView === 'channel') {
      const sample = feature.samples?.[channelAt] ?? feature.samples?.[0];
      return png(sample?.image);
    }
    return png(feature.pca);
  });
  let orderedProbs = $derived.by(() => orderProbs(shown));
  let selectedProb = $derived(orderedProbs.find((row) => row.id === classId) ?? orderedProbs[0] ?? null);
  let gridBytes = $derived.by(() => unpackGrid(shown?.prob_grid));
  let labelBytes = $derived.by(() => unpackGrid(shown?.label_grid));
  let classById = $derived(new Map((shown?.classes ?? []).map((row) => [row.id, row])));
  let readout = $derived.by(() => readProbe(probe, gridBytes, labelBytes, classById));
  let cell = $derived.by(() => {
    if (!shown || !feature?.width) return 1;
    return Math.max(1, Math.round(shown.width / feature.width));
  });
  let gridFine = $derived((feature?.width ?? 0) > 96);

  function storedTech() {
    try {
      return localStorage.getItem('explorer.modelTech') === '1';
    } catch {
      return false;
    }
  }

  function png(data) {
    return data ? `data:image/png;base64,${data}` : '';
  }

  function shapeOf(list) {
    if (!list?.length) return '';
    return `(${list.join(', ')})`;
  }

  function blankLayout() {
    return { nodes: [], edges: [], width: 0, height: 0 };
  }

  function wirePath(x1, y1, x2, y2, kind) {
    const mid = (x1 + x2) / 2;
    if (kind === 'skip' || kind === 'aux' || kind === 'fuse') {
      const bow = Math.min(48, Math.max(18, Math.abs(x2 - x1) * 0.2));
      const lift = kind === 'aux' ? bow : Math.min(y1, y2) - bow * 0.35;
      const y = kind === 'aux' ? Math.max(y1, y2) + 28 : lift;
      return `M ${x1} ${y1} Q ${mid} ${y} ${x2} ${y2}`;
    }
    const dx = Math.max(24, Math.abs(x2 - x1) * 0.45);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }

  function layoutGraph(data) {
    const rawNodes = data.graph?.nodes ?? [];
    const rawEdges = data.graph?.edges ?? [];
    if (!rawNodes.length) return blankLayout();
    const strides = [...new Set(rawNodes.map((item) => item.stride || 1))].sort((a, b) => a - b);
    const colW = 156;
    const rowH = 150;
    const padX = 8;
    const padY = 56;
    const groups = new Map();
    for (const item of rawNodes) {
      const col = item.column || 0;
      const row = Math.max(0, strides.indexOf(item.stride || 1));
      const key = `${col}:${row}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    const maxCol = Math.max(...rawNodes.map((item) => item.column || 0));
    const width = padX * 2 + (maxCol + 1) * colW;
    const height = padY * 2 + strides.length * rowH;
    const placed = new Map();
    for (const [key, group] of groups) {
      const [col, row] = key.split(':').map(Number);
      group.sort((a, b) => (a.slot || 0) - (b.slot || 0));
      group.forEach((item, index) => {
        const share = group.length;
        const thumb = share > 1
          ? 44
          : Math.round(Math.min(76, Math.max(42, 84 / Math.sqrt(item.stride || 1))));
        const boxW = colW - 16;
        const boxH = (rowH - 10) / share;
        const x = padX + col * colW + 6;
        const y = padY + row * rowH + index * boxH;
        placed.set(item.id, {
          ...item,
          x,
          y,
          boxW,
          boxH,
          thumb,
          cx: x + boxW / 2,
          cy: y + thumb / 2 + 2,
        });
      });
    }
    const edges = rawEdges.map((edge) => {
      const from = placed.get(edge.from);
      const to = placed.get(edge.to);
      if (!from || !to) return null;
      const x1 = from.x + from.boxW - 4;
      const y1 = from.cy;
      const x2 = to.x + 4;
      const y2 = to.cy;
      return { ...edge, d: wirePath(x1, y1, x2, y2, edge.kind) };
    }).filter(Boolean);
    return { nodes: [...placed.values()], edges, width, height };
  }

  function nodeThumb(item, data) {
    if (!item || !data) return '';
    if (item.kind === 'input') return data.image ? `data:image/jpeg;base64,${data.image}` : '';
    if (item.kind === 'ndsm') return png(data.ndsm);
    if (item.kind === 'scores') return png(topProb(data)?.image) || png(data.overlay);
    if (item.kind === 'decision') return png(data.overlay) || (data.image ? `data:image/jpeg;base64,${data.image}` : '');
    return png(item.pca);
  }

  function edgeHot(edge) {
    const id = hoverId || stageId;
    return edge.from === id || edge.to === id;
  }

  function diversify(tiles) {
    const rank = { test: 0, val: 1, train: 2, unused: 3 };
    const sorted = tiles.slice().sort((a, b) => {
      const byRole = (rank[a.cv_role] ?? 9) - (rank[b.cv_role] ?? 9);
      if (byRole) return byRole;
      return (b.label_m2 || 0) - (a.label_m2 || 0);
    });
    const picked = [];
    const sites = new Set();
    const seen = new Set();
    for (const row of sorted) {
      const names = Array.isArray(row.species) ? row.species.filter((name) => typeof name === 'string') : [];
      const fresh = names.some((name) => !seen.has(name));
      if (!fresh || sites.has(row.site_name)) continue;
      picked.push(row);
      sites.add(row.site_name);
      names.forEach((name) => seen.add(name));
      if (picked.length >= 6) break;
    }
    if (picked.length < 4) {
      for (const row of sorted) {
        if (picked.some((item) => item.tile_id === row.tile_id)) continue;
        picked.push(row);
        if (picked.length >= 6) break;
      }
    }
    return picked;
  }

  function presentSpecies(row) {
    const items = row?.species ?? [];
    return items.filter((spec) => {
      if (typeof spec === 'string') return true;
      return (spec?.px || 0) > 0 || (spec?.m2 || 0) > 0;
    }).slice(0, 6);
  }

  function speciesKey(spec) {
    return typeof spec === 'string' ? spec : spec.stats_name || spec.name;
  }

  function speciesSwatch(spec) {
    if (typeof spec === 'string') return speciesColor[spec] || '#5d6e74';
    return spec.color || speciesColor[spec.stats_name] || '#5d6e74';
  }

  function inputMode(data) {
    if (!data?.in_channels?.ndsm) return 'Color only';
    if (data.fusion === 'stem') return 'Color, height at each scale';
    return 'Color and height together';
  }

  function topProb(data) {
    const counts = data.counts ?? [];
    const top = counts.slice().sort((a, b) => (b.px || 0) - (a.px || 0))[0];
    const id = top?.id ?? 1;
    return (data.probabilities ?? []).find((row) => row.id === id) ?? data.probabilities?.[0];
  }

  function orderProbs(data) {
    const rows = data?.probabilities ?? [];
    const byId = new Map(rows.map((row) => [row.id, row]));
    const order = [0, ...(data?.display_order ?? []).filter((id) => id !== 0)];
    for (const row of rows) {
      if (!order.includes(row.id)) order.push(row.id);
    }
    return order.map((id) => byId.get(id)).filter(Boolean);
  }

  function decodeBytes(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  function unpackGrid(grid) {
    if (!grid?.data) return null;
    return {
      width: grid.width,
      height: grid.height,
      classes: grid.classes || 1,
      bytes: decodeBytes(grid.data),
    };
  }

  function swatchOf(color) {
    if (!color || color.toLowerCase() === '#000000') return '#122024';
    return color;
  }

  function readProbe(point, grid, labels, byId) {
    if (!point || !grid) return null;
    const gx = Math.min(grid.width - 1, Math.max(0, Math.floor(point.x * grid.width)));
    const gy = Math.min(grid.height - 1, Math.max(0, Math.floor(point.y * grid.height)));
    const plane = grid.width * grid.height;
    const items = [];
    let sum = 0;
    let winnerId = 0;
    let winnerValue = -1;
    for (let id = 0; id < grid.classes; id++) {
      const value = grid.bytes[id * plane + gy * grid.width + gx] / 255;
      sum += value;
      if (value > winnerValue) {
        winnerValue = value;
        winnerId = id;
      }
      const spec = byId.get(id);
      items.push({
        id,
        name: spec?.name || `Class ${id}`,
        color: swatchOf(spec?.color),
        percent: Math.round(value * 100),
      });
    }
    items.sort((a, b) => b.percent - a.percent || a.id - b.id);
    let labelId = null;
    if (labels) {
      labelId = labels.bytes[gy * labels.width + gx];
    }
    return {
      empty: sum <= 0,
      items,
      winnerId,
      winner: byId.get(winnerId),
      labelId,
    };
  }

  function labelPhrase(id, byId) {
    if (id == null || id === 253) return '';
    if (id === 254) return 'Unlabelled';
    if (id === 255) return 'Uncertain';
    return byId.get(id)?.name || '';
  }

  function inputSummary(data) {
    const size = `${data.width.toLocaleString()} × ${data.height.toLocaleString()} px`;
    if (data.native_width && (data.native_width !== data.width || data.native_height !== data.height)) {
      return `The file is ${data.native_width.toLocaleString()} × ${data.native_height.toLocaleString()} px. This model reads a ${size} copy. Nothing has been decided yet.`;
    }
    return `The model reads this tile at ${size}. Nothing has been decided yet.`;
  }

  function heightSummary(data) {
    if (!data?.ndsm) return 'This model looks at color only.';
    const stats = data.ndsm_stats;
    const range = stats ? ` Heights in this tile run from ${stats.min} m to ${stats.max} m.` : '';
    return `Beside the photo, the model sees how far the canopy stands above the ground.${range}`;
  }

  function fusionDetail(data) {
    if (!data?.in_channels?.ndsm) return 'The input is the photo only.';
    if (data.fusion === 'stem') {
      return 'Height is not stacked onto the photo. After the encoder looks at the color, height nudges those features at each scale.';
    }
    return 'Height is placed next to the color channels, so the first layer sees the photo and the height together.';
  }

  function featureSummary(stage, data) {
    if (!stage || !data) return '';
    const cover = Math.max(1, Math.round(data.width / stage.width));
    const cellText = `Each cell covers ${cover} × ${cover} pixels of the photo.`;
    if (stage.kind === 'decoder') {
      const incoming = stage.parents?.length || 0;
      if (incoming > 1) {
        return `This block mixes ${incoming} earlier maps into one. ${cellText}`;
      }
      return `This block reshapes what the model has learned so far. ${cellText}`;
    }
    if (stage.stride <= 2) {
      return `Fine detail is still here. ${cellText}`;
    }
    if (stage.stride >= 16) {
      return `Only the broad layout of the tile remains. ${cellText}`;
    }
    return `Plants start to group together. ${cellText}`;
  }

  function featureTech(stage, data) {
    if (!stage || !data) return '';
    const lines = [];
    if (stage.module) lines.push(stage.module);
    lines.push(`${stage.kicker || 'Stage'} at ${stage.scale || 'this scale'}.`);
    if (stage.shape) lines.push(`Tensor ${shapeOf(stage.shape)}.`);
    if (stage.elev_scale != null) {
      lines.push(`Height adjustment at this scale: ${stage.elev_scale}. A larger number means this scale leans more on height.`);
    }
    return lines.join(' ');
  }

  function headDetail(data) {
    const head = data?.head ?? {};
    const kernel = head.kernel ? `A ${head.kernel}×${head.kernel} layer` : 'The last layer';
    let text = `${kernel} turns ${head.in_channels ?? 'the'} feature channels into ${data.num_classes} scores, one per class.`;
    if (head.upsample > 1) text += ` Those scores are then scaled up ${head.upsample}× to the photo size.`;
    text += ' The scores are turned into confidences that add up to 100% for each pixel.';
    if (data.logit_min != null && data.logit_max != null) {
      text += ` Before that, the raw scores on this tile run from ${data.logit_min} to ${data.logit_max}.`;
    }
    return text;
  }

  function agreeText(data) {
    if (data?.agreement_species == null) return '';
    const pct = Math.round(data.agreement_species * 1000) / 10;
    return `On labelled plants, this map matches the label on ${pct}% of those pixels.`;
  }

  function pointIn(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (x < 0 || y < 0 || x > 1 || y > 1) return null;
    return { x, y };
  }

  function moveProbe(event) {
    if (pinned) return;
    const point = pointIn(event);
    if (point) probe = point;
  }

  function pinProbe(event) {
    const point = pointIn(event);
    if (!point) return;
    if (pinned && probe && Math.hypot(probe.x - point.x, probe.y - point.y) < 0.015) {
      pinned = false;
      return;
    }
    probe = point;
    pinned = true;
  }

  function onRailKey(event) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const tag = event.target?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    const ids = layout.nodes
      .slice()
      .sort((a, b) => (a.column - b.column) || ((a.stride || 1) - (b.stride || 1)) || ((a.slot || 0) - (b.slot || 0)))
      .map((item) => item.id);
    const index = ids.indexOf(stageId);
    const next = index + (event.key === 'ArrowRight' ? 1 : -1);
    if (!ids.length || next < 0 || next >= ids.length) return;
    event.preventDefault();
    stageId = ids[next];
  }

  function chooseTile(id) {
    if (!id || id === tileId) {
      pickerOpen = false;
      return;
    }
    tileId = id;
    pickerOpen = false;
  }

  function onWindowClick(event) {
    if (!pickerOpen) return;
    if (event.target?.closest?.('.sample')) return;
    pickerOpen = false;
  }

  function retry() {
    attempt += 1;
  }

  function readError(res) {
    return res.text().then((text) => {
      try {
        const data = JSON.parse(text);
        if (typeof data?.detail === 'string') return data.detail;
      } catch {
        /* plain text */
      }
      return text || res.statusText;
    });
  }

  function titleOf(item) {
    return item.exp_name.replace(/^\d{8}_[\d.]+_/, '') || item.exp_name;
  }

  function f1(item) {
    const value = item.metrics?.test_f1_species;
    if (value == null || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(3);
  }

  function optionLabel(item) {
    const val = item.metrics?.val_f1_species;
    const valText = val == null || Number.isNaN(Number(val)) ? '' : ` · val ${Number(val).toFixed(3)}`;
    return `${titleOf(item)}${item.ds2 ? ' · half res' : ''} · test ${f1(item)}${valText}`;
  }

  function rankCheckpoints(rows, wandbRuns) {
    const byName = new Map((wandbRuns ?? []).map((run) => [run.name, run.metrics || {}]));
    return rows.map((item) => ({
      ...item,
      metrics: byName.get(item.exp_name) || {},
    })).sort((a, b) => compareRows(a, b));
  }

  function compareRows(a, b) {
    const byTest = compareScore(a, b, 'test_f1_species');
    if (byTest) return byTest;
    const byVal = compareScore(a, b, 'val_f1_species');
    if (byVal) return byVal;
    if (a.kind !== b.kind) {
      if (a.kind === 'best') return -1;
      if (b.kind === 'best') return 1;
    }
    return (b.mtime || 0) - (a.mtime || 0);
  }

  function compareScore(a, b, key) {
    const left = a.metrics?.[key];
    const right = b.metrics?.[key];
    if (left == null && right == null) return 0;
    if (left == null) return 1;
    if (right == null) return -1;
    return right - left;
  }

  function share(part, item) {
    const total = (item.tp || 0) + (item.fp || 0) + (item.fn || 0);
    if (!total) return 0;
    return (100 * part) / total;
  }

  $effect(() => {
    try {
      localStorage.setItem('explorer.modelTech', tech ? '1' : '0');
    } catch {
      /* storage unavailable */
    }
  });

  $effect(() => {
    const incoming = initialTileId;
    if (incoming) tileId = incoming;
  });

  $effect(() => {
    const foldName = fold || 'cv3';
    const ctrl = new AbortController();
    picksLoading = true;
    const params = new URLSearchParams({
      fold: foldName,
      role: 'all',
      labelled: 'true',
      sort: 'label_m2',
      limit: '240',
    });
    fetch('/api/tiles?' + params.toString(), { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        picks = diversify(data.tiles ?? []);
        if (!tileId && picks.length) tileId = picks[0].tile_id;
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
      })
      .finally(() => {
        if (!ctrl.signal.aborted) picksLoading = false;
      });
    return () => ctrl.abort();
  });

  $effect(() => {
    if (!pickerOpen) return;
    const foldName = fold || 'cv3';
    const site = siteFilter;
    const role = roleFilter || 'all';
    const species = speciesFilter;
    const ctrl = new AbortController();
    browseLoading = true;
    const params = new URLSearchParams({
      fold: foldName,
      role,
      labelled: 'true',
      sort: 'label_m2',
      limit: '24',
    });
    if (site) params.set('site', site);
    if (species) params.append('species', species);
    fetch('/api/tiles?' + params.toString(), { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        browse = data.tiles ?? [];
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        browse = [];
      })
      .finally(() => {
        if (!ctrl.signal.aborted) browseLoading = false;
      });
    return () => ctrl.abort();
  });

  async function surprise() {
    const foldName = fold || 'cv3';
    const params = new URLSearchParams({
      fold: foldName,
      role: roleFilter || 'all',
      labelled: 'true',
      sort: 'label_m2',
      limit: '1',
      offset: '0',
    });
    if (siteFilter) params.set('site', siteFilter);
    if (speciesFilter) params.append('species', speciesFilter);
    try {
      const head = await fetch('/api/tiles?' + params.toString());
      if (!head.ok) return;
      const first = await head.json();
      const total = Math.max(1, Number(first.total) || 1);
      params.set('offset', String(Math.floor(Math.random() * total)));
      const res = await fetch('/api/tiles?' + params.toString());
      if (!res.ok) return;
      const data = await res.json();
      const id = data.tiles?.[0]?.tile_id;
      if (id) chooseTile(id);
    } catch {
      /* leave the current tile */
    }
  }

  $effect(() => {
    const id = tileId;
    if (!id) {
      tile = null;
      return;
    }
    const ctrl = new AbortController();
    fetch('/api/tiles/' + id, { signal: ctrl.signal })
      .then(async (res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.tile_id === id) tile = data;
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
      });
    return () => ctrl.abort();
  });

  $effect(() => {
    const ctrl = new AbortController();
    fetch('/api/checkpoints', { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        checkpoints = data.checkpoints ?? [];
        listError = '';
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        listError = String(err.message || err);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) checkpointsLoading = false;
      });
    return () => ctrl.abort();
  });

  $effect(() => {
    const ctrl = new AbortController();
    fetch('/api/wandb/runs', { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        runs = data.runs ?? [];
        scoresNote = data.available ? '' : (data.detail || '');
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        scoresNote = String(err.message || err);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) scoresReady = true;
      });
    return () => ctrl.abort();
  });

  $effect(() => {
    if (touched || pickedCkpt || !scoresReady || !checkpoints.length) return;
    pickedCkpt = true;
    const rows = rankCheckpoints(checkpoints, runs);
    ckpt = rows[0]?.path ?? '';
  });

  $effect(() => {
    const id = tileId;
    const path = ckpt;
    const n = attempt;
    if (!id || !path) return;
    const ctrl = new AbortController();
    let dead = false;
    busy = true;
    error = '';
    fetch(`/api/tiles/${id}/trace`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ckpt: path }),
      signal: ctrl.signal,
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        if (dead) return;
        trace = data;
        traceFor = `${id}:${path}`;
        const ids = (data.graph?.nodes ?? []).map((item) => item.id);
        if (ids.length && !ids.includes(stageId)) stageId = ids[0];
        const top = (data.counts ?? []).slice().sort((a, b) => (b.px || 0) - (a.px || 0))[0];
        if (top) classId = top.id;
        probe = null;
        pinned = false;
        error = '';
      })
      .catch((err) => {
        if (dead || err.name === 'AbortError') return;
        error = String(err.message || err);
      })
      .finally(() => {
        if (!dead) busy = false;
      });
    void n;
    return () => {
      dead = true;
      ctrl.abort();
    };
  });
</script>

<svelte:window onkeydown={onRailKey} onclick={onWindowClick} />

<div class="model-page">
  <div class="toolbar">
    <label class="field">
      <span class="field-label">Model</span>
      {#if checkpointsLoading || !scoresReady}
        <span class="wait">Loading checkpoints…</span>
      {:else if ranked.length}
        <select
          bind:value={ckpt}
          disabled={busy}
          onchange={() => (touched = true)}
        >
          {#each ranked as item (item.path)}
            <option value={item.path}>{optionLabel(item)}</option>
          {/each}
        </select>
      {:else}
        <span class="wait">No checkpoints yet</span>
      {/if}
      {#if shown}
        <span class="badge">{shown.architecture || 'Model'} · {shown.encoder || 'encoder'} · {inputMode(shown)}</span>
      {/if}
    </label>

    <div class="sample">
      <div class="sample-card">
        {#if tile}
          <img src={tile.thumb || `/thumbs/${tile.tile_id}.webp`} alt="" />
          <span>
            <strong>{tile.site_name}_{tile.flight}</strong>
            <em>{tile.acq_date ?? 'date unknown'}</em>
            {#if presentSpecies(tile).length}
              <span class="dots" aria-hidden="true">
                {#each presentSpecies(tile) as spec (speciesKey(spec))}
                  <i style:background={speciesSwatch(spec)}></i>
                {/each}
              </span>
            {/if}
          </span>
        {:else if picksLoading}
          <span class="wait">Loading a sample tile…</span>
        {:else}
          <span class="wait">Choose a tile to run through the model.</span>
        {/if}
      </div>
      <div class="sample-actions">
        <button type="button" class="text-btn" aria-expanded={pickerOpen} onclick={() => (pickerOpen = !pickerOpen)}>
          {pickerOpen ? 'Close tiles' : 'Change tile…'}
        </button>
        <button type="button" class="text-btn" onclick={surprise}>Random tile</button>
      </div>
      {#if pickerOpen}
        <div class="popover" role="dialog" aria-label="Choose a tile">
          {#if picks.length}
            <p class="pop-label">Different sites and species</p>
            <div class="pop-row">
              {#each picks as item (item.tile_id)}
                <button type="button" class="mini-tile" class:on={item.tile_id === tileId} onclick={() => chooseTile(item.tile_id)}>
                  <img src={item.thumb} alt="" />
                  <span>{item.site_name}</span>
                </button>
              {/each}
            </div>
          {/if}
          <div class="filters">
            <label>
              Site
              <select bind:value={siteFilter}>
                <option value="">All sites</option>
                {#each facets?.sites ?? [] as site}
                  <option value={site}>{site}</option>
                {/each}
              </select>
            </label>
            <label>
              Role
              <select bind:value={roleFilter}>
                <option value="all">All</option>
                <option value="test">Test</option>
                <option value="val">Val</option>
                <option value="train">Train</option>
              </select>
            </label>
            <label>
              Species
              <select bind:value={speciesFilter}>
                <option value="">Any</option>
                {#each facets?.species ?? [] as spec}
                  <option value={spec.stats_name}>{spec.name}</option>
                {/each}
              </select>
            </label>
          </div>
          {#if browseLoading}
            <p class="hint">Loading tiles…</p>
          {:else if browse.length}
            <div class="browse">
              {#each browse as item (item.tile_id)}
                <button type="button" class="mini-tile" class:on={item.tile_id === tileId} onclick={() => chooseTile(item.tile_id)}>
                  <img src={item.thumb} alt="" />
                  <span>{item.site_name}</span>
                </button>
              {/each}
            </div>
          {:else}
            <p class="hint">No labelled tiles match these filters.</p>
          {/if}
        </div>
      {/if}
    </div>

    <button
      type="button"
      class="tech-btn"
      aria-pressed={tech}
      onclick={() => (tech = !tech)}
    >{tech ? 'Hide technical details' : 'Show technical details'}</button>
  </div>

  {#if listError}
    <p class="problem" role="alert">{listError}</p>
  {:else if scoresNote && scoresReady}
    <p class="hint">{scoresNote}</p>
  {/if}

  {#if busy}
    <p class="status" role="status"><span class="spin" aria-hidden="true"></span> Running this tile through the model…</p>
  {/if}
  {#if error}
    <p class="problem" role="alert">{error}</p>
    <button type="button" class="retry" onclick={retry} disabled={busy}>Try again</button>
  {/if}

  {#if shown}
    <div class:dimmed={busy}>
      <p class="lede">
        This is the model itself, drawn left to right. Blocks sit lower when they look at less detail. Open one to see what it did to this tile.
      </p>
      {#if tech}
        <p class="tech-line">
          {shown.architecture || 'Model'} · {shown.encoder || 'encoder'}
          {#if shown.input_shape}· input {shapeOf(shown.input_shape)}{/if}
        </p>
      {/if}
      {#if shown.cached}
        <p class="hint">Loaded from a saved trace.</p>
      {/if}

      <div class="arch-scroll">
        <div
          class="arch"
          style:width="{layout.width}px"
          style:height="{layout.height}px"
          role="tablist"
          aria-label="Model architecture"
        >
          <svg class="wires" width={layout.width} height={layout.height} aria-hidden="true">
            {#each layout.edges as edge (`${edge.from}-${edge.to}-${edge.kind}`)}
              <path d={edge.d} class="wire {edge.kind}" class:hot={edgeHot(edge)} />
            {/each}
          </svg>
          {#each layout.nodes as item (item.id)}
            <button
              type="button"
              role="tab"
              class="node"
              class:on={stageId === item.id}
              aria-selected={stageId === item.id}
              style:left="{item.x}px"
              style:top="{item.y}px"
              style:width="{item.boxW}px"
              style:height="{item.boxH}px"
              onclick={() => (stageId = item.id)}
              onpointerenter={() => (hoverId = item.id)}
              onpointerleave={() => (hoverId = '')}
            >
              <span class="thumb" style:width="{item.thumb}px" style:height="{item.thumb}px">
                <img src={nodeThumb(item, shown)} alt="" />
              </span>
              <span class="kicker">{item.kicker}</span>
              <strong>{item.label}</strong>
              <span class="size">{item.scale}{#if tech && item.channels} · {item.channels} ch{/if}</span>
            </button>
          {/each}
        </div>
      </div>
      <p class="legend">
        <i class="key forward"></i> main path
        <i class="key skip"></i> detail carried across
        <i class="key fuse"></i> scales mixed together
        {#if shown.ndsm}<i class="key aux"></i> canopy height{/if}
      </p>

      <section class="inspect" aria-live="polite">
        {#if stageKind === 'input'}
          <h2>The photo</h2>
          <p>{inputSummary(shown)}</p>
          <div class="pair">
            <figure>
              <div class="frame" style:aspect-ratio={`${shown.width} / ${shown.height}`}>
                <img class="photo" src={photoUrl} alt="Tile the model reads" />
              </div>
              <figcaption>Color photo · {shown.width.toLocaleString()} × {shown.height.toLocaleString()} px</figcaption>
            </figure>
            {#if shown.ndsm}
              <figure>
                <div class="frame" style:aspect-ratio={`${shown.width} / ${shown.height}`}>
                  <img class="photo" src={png(shown.ndsm)} alt="Canopy height" />
                </div>
                <figcaption>Canopy height. Brighter is taller.</figcaption>
              </figure>
            {/if}
          </div>
          <p class="hint">{heightSummary(shown)}</p>
          {#if tech}
            <div class="pair tech-pair">
              <figure>
                <div class="frame" style:aspect-ratio={`${shown.width} / ${shown.height}`}>
                  <img class="photo" src={`data:image/jpeg;base64,${shown.normalized}`} alt="After the color shift" />
                </div>
                <figcaption>The same photo after the color shift the network expects.</figcaption>
              </figure>
            </div>
            <dl>
              <div><dt>Tensor</dt><dd>{shapeOf(shown.input_shape)}</dd></div>
              {#if shown.aux_shape}
                <div><dt>Height tensor</dt><dd>{shapeOf(shown.aux_shape)}</dd></div>
              {/if}
              <div>
                <dt>Color shift</dt>
                <dd>mean {shown.norm.rgb_mean.join(', ')} · spread {shown.norm.rgb_std.join(', ')}</dd>
              </div>
              {#if shown.in_channels?.ndsm}
                <div>
                  <dt>Height shift</dt>
                  <dd>mean {shown.norm.ndsm_mean} · spread {shown.norm.ndsm_std}</dd>
                </div>
              {/if}
              <div><dt>How height enters</dt><dd>{fusionDetail(shown)}</dd></div>
              {#if shown.trained_crop}
                <div><dt>Training size</dt><dd>Trained on {shown.trained_crop} px crops. Here the whole tile is read at once.</dd></div>
              {/if}
            </dl>
          {/if}
        {:else if stageKind === 'ndsm'}
          <h2>Canopy height</h2>
          <p>{heightSummary(shown)}</p>
          {#if shown.ndsm}
            <figure class="solo">
              <div class="frame" style:aspect-ratio={`${shown.width} / ${shown.height}`}>
                <img class="photo" src={png(shown.ndsm)} alt="Canopy height" />
              </div>
              <figcaption>Brighter means the canopy stands taller above the ground.</figcaption>
            </figure>
          {/if}
          {#if tech}
            <p class="tech-copy">{fusionDetail(shown)}</p>
          {/if}
        {:else if feature}
          <h2>{feature.label}</h2>
          <p>{featureSummary(feature, shown)}</p>
          {#if mixParents.length > 1}
            <div class="mix" aria-label="What this block combines">
              {#each mixParents as parent, index (parent.id)}
                {#if index}<span class="plus" aria-hidden="true">+</span>{/if}
                <figure>
                  <img src={nodeThumb(parent, shown)} alt="" />
                  <figcaption>{parent.label} · {parent.scale}</figcaption>
                </figure>
              {/each}
              <span class="plus" aria-hidden="true">→</span>
              <figure>
                <img src={featureUrl} alt="" />
                <figcaption>{feature.label} · {feature.scale}</figcaption>
              </figure>
            </div>
          {/if}
          <div class="views" role="group" aria-label="Feature picture">
            <button type="button" class:on={featView === 'pca'} aria-pressed={featView === 'pca'} onclick={() => (featView = 'pca')}>Grouping</button>
            <button type="button" class:on={featView === 'energy'} aria-pressed={featView === 'energy'} onclick={() => (featView = 'energy')}>Activity</button>
            {#if feature.samples?.length}
              <button type="button" class:on={featView === 'channel'} aria-pressed={featView === 'channel'} onclick={() => (featView = 'channel')}>One channel</button>
            {/if}
          </div>
          <div class="pair">
            <figure>
              <div
                class="frame"
                style:aspect-ratio={`${shown.width} / ${shown.height}`}
                style:--cols={feature.width}
                style:--rows={feature.height}
              >
                <img class="photo" src={photoUrl} alt="Tile" />
                {#if gridOn && !gridFine}<div class="cells"></div>{/if}
              </div>
              <figcaption>Photo</figcaption>
            </figure>
            <figure>
              <div class="frame" style:aspect-ratio={`${shown.width} / ${shown.height}`}>
                <img class="photo feat" src={featureUrl} alt="" />
              </div>
              <figcaption>
                {#if featView === 'energy'}
                  Where this scale is active. Brighter areas drove the features more.
                {:else if featView === 'channel'}
                  A single feature channel. Similar brightness means a similar response.
                {:else}
                  Pixels the model treats as alike share a color.
                {/if}
              </figcaption>
            </figure>
          </div>
          {#if featView === 'channel' && feature.samples?.length}
            <div class="channels">
              {#each feature.samples as sample, index (sample.index)}
                <button type="button" class:on={channelAt === index} aria-pressed={channelAt === index} onclick={() => { channelAt = index; featView = 'channel'; }}>
                  <img src={png(sample.image)} alt="" />
                  Channel {sample.index}
                </button>
              {/each}
            </div>
          {/if}
          <div class="grid-row">
            <button type="button" onclick={() => (gridOn = !gridOn)} disabled={gridFine} aria-pressed={gridOn}>
              {gridOn && !gridFine ? 'Hide the grid' : 'Show the grid'}
            </button>
            <p class="hint">One cell covers {cell} × {cell} px of the photo. This scale is {feature.width.toLocaleString()} × {feature.height.toLocaleString()}.</p>
          </div>
          {#if gridFine}
            <p class="hint">This grid is too fine to draw line by line. The picture on the right is still made of those cells.</p>
          {/if}
          {#if tech}
            <p class="tech-copy">{featureTech(feature, shown)}</p>
          {/if}
        {:else if stageKind === 'scores'}
          <h2>A score for every species</h2>
          <p>
            Every pixel gets a confidence for the background and for each plant. Stronger color means the model is more sure.
            These maps are the segmentation. The single label comes next.
          </p>
          <div class="class-grid">
            {#each orderedProbs as row (row.id)}
              <button type="button" class:on={selectedProb?.id === row.id} aria-pressed={selectedProb?.id === row.id} onclick={() => (classId = row.id)}>
                <span class="mini">
                  <img src={photoUrl} alt="" />
                  <img class="tint" src={png(row.image)} alt="" />
                </span>
                <span class="mini-name" class:binomial={row.id !== 0}>{row.name}</span>
              </button>
            {/each}
          </div>
          <div class="split">
            <div
              class="frame probe"
              role="application"
              aria-label="Species confidence. Move over the picture, or click to pin a pixel."
              style:aspect-ratio={`${shown.width} / ${shown.height}`}
              onpointermove={moveProbe}
              onpointerdown={pinProbe}
            >
              <img class="photo" src={photoUrl} alt="" />
              {#if selectedProb}<img class="tint" src={png(selectedProb.image)} alt="" />{/if}
              {#if probe}<span class="pin" style:left="{probe.x * 100}%" style:top="{probe.y * 100}%"></span>{/if}
            </div>
            {@render bars()}
          </div>
          {#if tech}
            <p class="tech-copy">{headDetail(shown)}</p>
            <figure class="entropy">
              <div class="frame" style:aspect-ratio={`${shown.width} / ${shown.height}`}>
                <img class="photo" src={photoUrl} alt="" />
                <img class="tint" src={png(shown.entropy)} alt="Where the model is unsure" />
              </div>
              <figcaption>Warm color marks pixels where the model is split between species.</figcaption>
            </figure>
          {/if}
        {:else if stageKind === 'decision'}
          <h2>The winning species</h2>
          <p>
            Each pixel keeps the species with the highest score. That choice is the classification.
            Together, those pixels are the segmentation map.
          </p>
          <div class="views" role="group" aria-label="Map display">
            <button type="button" class:on={mapMode === 'prediction'} aria-pressed={mapMode === 'prediction'} onclick={() => (mapMode = 'prediction')}>Prediction</button>
            <button type="button" class:on={mapMode === 'labels'} aria-pressed={mapMode === 'labels'} disabled={!shown.label} onclick={() => (mapMode = 'labels')}>Labels</button>
            <button type="button" class:on={mapMode === 'both'} aria-pressed={mapMode === 'both'} disabled={!shown.label} onclick={() => (mapMode = 'both')}>Both</button>
          </div>
          <div class="split">
            <div
              class="frame probe"
              role="application"
              aria-label="Final species map. Move over the picture, or click to pin a pixel."
              style:aspect-ratio={`${shown.width} / ${shown.height}`}
              onpointermove={moveProbe}
              onpointerdown={pinProbe}
            >
              <img class="photo" src={photoUrl} alt="" />
              {#if mapMode !== 'labels'}
                <img class="tint" src={png(shown.overlay)} alt="" style:opacity={mapMode === 'both' ? 0.72 : 1} />
              {/if}
              {#if mapMode !== 'prediction' && shown.label}
                <img class="tint" src={png(shown.label)} alt="" style:opacity={mapMode === 'both' ? 0.55 : 1} />
              {/if}
              {#if probe}<span class="pin" style:left="{probe.x * 100}%" style:top="{probe.y * 100}%"></span>{/if}
            </div>
            {@render bars()}
          </div>
          {#if agreeText(shown)}
            <p class="hint">{agreeText(shown)}</p>
          {/if}
          {#if shown.counts?.length}
            <ul class="counts">
              {#each shown.counts as item (item.id)}
                <li>
                  <i class="swatch" style:--swatch={item.color}></i>
                  <span class="binomial">{item.name}</span>
                  {#if tech && item.tp != null}
                    <span class="splitbar" title="{item.tp.toLocaleString()} correct, {item.fp.toLocaleString()} extra, {item.fn.toLocaleString()} missed">
                      <i style:width="{share(item.tp, item)}%" style:background={item.color}></i>
                      <i class="fp" style:width="{share(item.fp, item)}%"></i>
                      <i class="fn" style:width="{share(item.fn, item)}%"></i>
                    </span>
                  {/if}
                  <span class="num">{item.px.toLocaleString()} px</span>
                </li>
              {/each}
            </ul>
            {#if tech && shown.counts.some((item) => item.tp != null)}
              <p class="hint">Bar: correct, extra, missed.</p>
            {/if}
          {:else}
            <p class="hint">No species pixels in this map.</p>
          {/if}
          {#if tech}
            <p class="tech-copy">
              The winner at each pixel is the class with the highest score.
              {#if shown.agreement != null}
                Counting background as well, the map matches the label on {Math.round(shown.agreement * 1000) / 10}% of labelled pixels.
              {/if}
              Grey in a label was never labelled, and yellow was marked unsure. Those pixels are left out of the score.
            </p>
          {/if}
        {/if}
      </section>
    </div>
  {:else if !busy && !error && ckpt}
    <p class="hint">The trace will appear here once the model has read the tile.</p>
  {/if}
</div>

{#snippet bars()}
  <div class="readout">
    {#if !readout}
      <p class="hint">Move over the picture. Click to pin a pixel and see why that species won.</p>
    {:else if readout.empty}
      <p>This pixel is outside the photo, so the model ignores it.</p>
    {:else}
      <p class="verdict">
        Classified as
        <span class:binomial={readout.winnerId !== 0}>{readout.winner?.name || 'this class'}</span>
      </p>
      {#if readout.labelId != null && readout.labelId !== 253}
        <p class="hint">
          {#if readout.labelId === readout.winnerId}
            The label agrees.
          {:else}
            The label here is
            <span class:binomial={readout.labelId > 0 && readout.labelId < 254}>{labelPhrase(readout.labelId, classById)}</span>.
          {/if}
        </p>
      {/if}
      <ul class="bars">
        {#each readout.items as item (item.id)}
          <li class:win={item.id === readout.winnerId}>
            <span class:binomial={item.id !== 0}>{item.name}</span>
            <span class="track"><i style:width="{item.percent}%" style:background={item.color}></i></span>
            <span class="num">{item.percent}%</span>
          </li>
        {/each}
      </ul>
      <p class="hint">{pinned ? 'Pinned. Click the pixel again to follow the cursor.' : 'Click to pin this pixel.'}</p>
    {/if}
  </div>
{/snippet}

<style>
  .model-page {
    height: 100%;
    min-height: 0;
    overflow: auto;
    padding: 16px 18px 40px;
  }

  .toolbar {
    display: grid;
    grid-template-columns: minmax(240px, 1.15fr) minmax(260px, 1fr) auto;
    gap: 14px 18px;
    align-items: start;
    margin-bottom: 14px;
  }

  .field { margin: 0; }

  .field-label {
    display: block;
    margin-bottom: 4px;
    font-size: 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--mist);
  }

  .badge {
    display: block;
    margin-top: 6px;
    color: var(--mist);
    font-size: 13px;
  }

  .sample { position: relative; }

  .sample-card {
    display: flex;
    gap: 10px;
    align-items: center;
    min-height: 56px;
  }

  .sample-card img {
    width: 56px;
    height: 56px;
    object-fit: cover;
    background: #8ea09a;
    flex: none;
  }

  .sample-card strong, .sample-card em {
    display: block;
    font-style: normal;
  }

  .sample-card em, .sample-card .wait { color: var(--mist); font-size: 13px; }

  .sample-actions {
    display: flex;
    gap: 8px;
    margin-top: 6px;
  }

  .dots {
    display: flex;
    gap: 3px;
    margin-top: 4px;
  }

  .dots i {
    width: 7px;
    height: 7px;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.7);
  }

  .text-btn, .tech-btn, .retry, .views button, .grid-row button, .channels button, .class-grid button {
    border: 1px solid var(--line);
    background: transparent;
    color: var(--ink);
  }

  .text-btn {
    height: 28px;
    padding: 0 10px;
    font-size: 13px;
  }

  .popover {
    position: absolute;
    z-index: 5;
    top: calc(100% + 6px);
    left: 0;
    width: min(520px, 80vw);
    padding: 12px;
    background: var(--sheet);
    border: 1px solid var(--line);
    box-shadow: 0 12px 32px rgba(18, 32, 36, 0.12);
  }

  .pop-label {
    margin: 0 0 6px;
    font-size: 12px;
    color: var(--mist);
  }

  .pop-row, .browse {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .browse { margin-top: 10px; max-height: 280px; overflow: auto; }

  .mini-tile {
    width: 76px;
    padding: 0;
    border: 1px solid var(--line);
    background: #8ea09a;
    color: var(--ink);
    text-align: left;
  }

  .mini-tile.on { border-color: var(--canopy); }

  .mini-tile img {
    display: block;
    width: 100%;
    height: 56px;
    object-fit: cover;
  }

  .mini-tile span {
    display: block;
    padding: 3px 4px 4px;
    font-size: 10px;
    background: var(--sheet);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .filters {
    display: grid;
    grid-template-columns: 1.3fr 0.7fr 1fr;
    gap: 8px;
    margin-top: 10px;
  }

  .filters label {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 11px;
    color: var(--mist);
  }

  .wait {
    display: flex;
    align-items: center;
    min-height: 32px;
    color: var(--mist);
    font-size: 13px;
  }

  h2 {
    margin: 0;
    font-family: "Bricolage Grotesque", "Public Sans", sans-serif;
    font-size: 18px;
    font-weight: 560;
    letter-spacing: -0.03em;
  }

  .lede, .hint, .tech-line, .tech-copy, figcaption, .legend {
    color: var(--mist);
  }

  .tech-btn {
    height: 32px;
    padding: 0 12px;
    color: var(--canopy);
    white-space: nowrap;
  }

  .tech-btn[aria-pressed="true"] {
    background: var(--canopy);
    border-color: var(--canopy);
    color: var(--sheet);
  }

  .lede {
    margin: 0 0 12px;
    max-width: 72ch;
    color: var(--ink);
    font-size: 15px;
  }

  .tech-line, .hint, .tech-copy { margin: 0 0 8px; font-size: 13px; }

  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 10px;
    color: var(--ink);
  }

  .spin {
    width: 14px;
    height: 14px;
    border: 2px solid color-mix(in srgb, var(--mist) 45%, white);
    border-top-color: var(--canopy);
    border-radius: 50%;
    box-sizing: border-box;
    animation: spin 0.75s linear infinite;
  }

  .problem { color: #8c2f2f; font-size: 13px; }

  .retry {
    margin: 0 0 12px;
    padding: 6px 10px;
  }

  .dimmed { opacity: 0.55; }

  .arch-scroll {
    overflow-x: auto;
    padding-bottom: 8px;
  }

  .arch {
    position: relative;
    min-width: 100%;
  }

  .wires {
    position: absolute;
    inset: 0;
    overflow: visible;
    pointer-events: none;
  }

  .wire {
    fill: none;
    stroke: color-mix(in srgb, var(--quiet) 80%, var(--ink));
    stroke-width: 1.4;
    opacity: 0.45;
  }

  .wire.skip, .wire.fuse {
    stroke-dasharray: 4 4;
  }

  .wire.aux { stroke: #2f6f62; }

  .wire.hot {
    opacity: 1;
    stroke: var(--canopy);
    stroke-width: 2;
  }

  .node {
    position: absolute;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 4px 4px 0;
    border: 1px solid transparent;
    background: transparent;
    color: var(--ink);
    text-align: center;
    overflow: hidden;
  }

  .node.on .thumb {
    box-shadow: 0 0 0 2px var(--canopy);
  }

  .node .thumb {
    display: block;
    overflow: hidden;
    background: #8ea09a;
    border: 1px solid var(--line);
    box-shadow: 2px 2px 0 color-mix(in srgb, var(--ink) 12%, transparent);
  }

  .node .kicker {
    font-size: 10px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--mist);
  }

  .node strong {
    font-size: 12px;
    font-weight: 620;
    letter-spacing: -0.01em;
    line-height: 1.15;
  }

  .node .size {
    font-size: 10px;
    color: var(--mist);
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    align-items: center;
    margin: 0 0 14px;
    font-size: 12px;
  }

  .key {
    width: 22px;
    height: 0;
    border-top: 2px solid var(--quiet);
  }

  .key.skip, .key.fuse { border-top-style: dashed; }
  .key.aux { border-top-color: #2f6f62; }
  .key.forward { border-top-color: var(--ink); }

  .mix {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: flex-end;
    margin-top: 12px;
  }

  .mix figure { width: 92px; }

  .mix img {
    width: 92px;
    height: 92px;
    object-fit: cover;
    background: #8ea09a;
    image-rendering: pixelated;
  }

  .plus {
    align-self: center;
    color: var(--mist);
    font-size: 18px;
  }

  .solo { max-width: 520px; margin-top: 12px; }

  .thumb img, .channels img, .node img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .shape { font-family: ui-monospace, monospace; }

  .inspect {
    padding: 14px 16px 18px;
    background: var(--sheet);
    border: 1px solid var(--line);
  }

  .inspect h2 { margin-bottom: 6px; }

  .inspect > p:first-of-type { margin-top: 0; max-width: 72ch; }

  .pair, .split {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(240px, 0.85fr);
    gap: 14px;
    align-items: start;
    margin-top: 12px;
  }

  .tech-pair { max-width: 520px; }

  figure { margin: 0; }

  figcaption { margin-top: 6px; font-size: 12px; }

  .frame {
    position: relative;
    width: 100%;
    background: #8ea09a;
    overflow: hidden;
    box-shadow: inset 0 0 0 1px rgba(18, 32, 36, 0.28);
  }

  .frame img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  img.feat { image-rendering: pixelated; }

  .probe { cursor: crosshair; touch-action: none; }

  .cells {
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(to right, rgba(244, 248, 249, 0.7) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(244, 248, 249, 0.7) 1px, transparent 1px);
    background-size: calc(100% / var(--cols)) calc(100% / var(--rows));
    pointer-events: none;
  }

  .pin {
    position: absolute;
    z-index: 2;
    width: 12px;
    height: 12px;
    margin: -6px 0 0 -6px;
    border: 2px solid white;
    border-radius: 50%;
    box-shadow: 0 0 0 1px rgba(18, 32, 36, 0.75);
    pointer-events: none;
  }

  .views, .channels, .grid-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: 10px;
  }

  .views button, .channels button, .grid-row button {
    padding: 3px 8px;
    font-size: 12px;
  }

  .views button.on, .channels button.on, .class-grid button.on {
    background: color-mix(in srgb, var(--canopy) 12%, white);
    border-color: var(--canopy);
  }

  .views button:disabled, .grid-row button:disabled { opacity: 0.45; }

  .channels button {
    width: 92px;
    padding: 4px;
    text-align: left;
    font-size: 11px;
    color: var(--mist);
  }

  .channels img {
    aspect-ratio: 1;
    margin-bottom: 4px;
    image-rendering: pixelated;
  }

  .grid-row .hint { margin: 0; }

  .class-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(108px, 1fr));
    gap: 8px;
    margin-top: 12px;
  }

  .class-grid button { padding: 4px; text-align: left; }

  .mini {
    position: relative;
    display: block;
    aspect-ratio: 1;
    background: #8ea09a;
    overflow: hidden;
  }

  .mini img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }

  .mini-name {
    display: block;
    margin-top: 4px;
    font-size: 12px;
  }

  .readout { min-width: 0; }

  .verdict { margin: 0 0 8px; font-size: 15px; }

  .bars, .counts {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .bars li {
    display: grid;
    grid-template-columns: minmax(88px, 0.8fr) minmax(0, 1fr) auto;
    gap: 8px;
    align-items: center;
    margin: 5px 0;
    font-size: 13px;
  }

  .bars li.win { font-weight: 650; }

  .track, .splitbar {
    display: flex;
    height: 8px;
    background: color-mix(in srgb, var(--line) 70%, white);
    overflow: hidden;
  }

  .track i, .splitbar i { display: block; height: 100%; }

  .splitbar { width: 72px; }
  .splitbar .fp { background: #f4f8f9; box-shadow: inset 0 0 0 1px #b42318; }
  .splitbar .fn { background: repeating-linear-gradient(-45deg, #8a5a00, #8a5a00 1px, transparent 1px, transparent 3px); }

  .num {
    font-variant-numeric: tabular-nums;
    color: var(--mist);
    font-size: 12px;
  }

  .counts { margin-top: 14px; }

  .counts li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
  }

  .counts .num { margin-left: auto; }

  .entropy { max-width: 420px; margin-top: 12px; }

  dl { margin: 12px 0 0; }

  dl div {
    display: grid;
    grid-template-columns: 140px minmax(0, 1fr);
    gap: 8px;
    padding: 6px 0;
    border-top: 1px solid var(--line);
    font-size: 13px;
  }

  dt { color: var(--mist); }
  dd { margin: 0; }

  @keyframes spin { to { transform: rotate(360deg); } }

  @media (prefers-reduced-motion: reduce) {
    .spin { animation: none; }
  }

  @media (max-width: 860px) {
    .toolbar, .pair, .split, .filters, dl div { grid-template-columns: 1fr; }
  }
</style>
