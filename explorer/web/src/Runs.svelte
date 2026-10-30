<script>
  import { untrack } from 'svelte';

  let { watchGpu = true, gpu = null } = $props();

  let options = $state(null);
  let optionsError = $state('');
  let dataName = $state('neophytes_split_cv3_train_1024');
  let modelName = $state('model_neophytes_mit_b2');
  let seed = $state(42);
  let maxEpochs = $state(50);
  let checkVal = $state(1);
  let lr = $state(0.0001);
  let batchTrain = $state(8);
  let batchVal = $state(2);
  let precision = $state('bf16-mixed');
  let logger = $state('wandb');
  let nameStamp = $state(stampNow());
  let expName = $state(untrack(() => `${nameStamp}_neophytes_split_cv3_train_1024_model_neophytes_mit_b2_s42`));
  let expNameCustom = $state(false);
  let subsetOn = $state(false);
  let fewShotK = $state(null);
  let fewShotBg = $state(10);
  let fewShotMode = $state('max_px_site_pheno');
  let valFewShotK = $state(null);
  let valFewShotBg = $state(0);
  let valFewShotMode = $state('all');
  let trainFlights = $state([]);
  let flightsOpen = $state(false);
  let flightsRoot = $state(null);
  let stageOn = $state({ nonflower: true, flower: true, fruiting: true });
  let speciesOn = $state({});
  let speciesOpen = $state(false);
  let speciesRoot = $state(null);
  let siteNote = $state(false);
  let stageNote = $state(false);
  let subsetNote = $state(false);
  let runTest = $state(false);
  let testNote = $state(false);
  let extraNote = $state(false);
  let confirmJob = $state(null);
  let confirmClear = $state(false);
  let confirmEl = $state(null);
  let showCommand = $state(false);
  let extra = $state('');
  let formError = $state('');
  let submitting = $state(false);

  let jobs = $state([]);
  let openId = $state('');
  let logText = $state('');
  let logOffset = $state(0);
  let logEl = $state(null);
  let logWindow = $state('200');
  let stick = true;

  let samples = $state([]);
  const history = [];

  let wandb = $state(null);
  let wandbError = $state('');
  let now = $state(Date.now());

  // Last model whose schedule was copied into the fields. Plain lets so editing
  // a field does not run this again and put the config's values back.
  let appliedModel = '';
  let pinnedData = '';
  let dataBeforePin = '';

  let modelNote = $derived(options?.model_forms?.[modelName]?.note ?? '');

  let command = $derived.by(() => {
    const lines = [
      '.venv/bin/python train.py',
      `data=${dataName}`,
      `model=${modelName}`,
      `pl_seed=${seed}`,
      `model.max_epochs=${maxEpochs}`,
      `model.check_val_every_n_epoch=${checkVal}`,
      `model.lr=${lr}`,
      `data.batch_size_train=${batchTrain}`,
      `data.batch_size_val=${batchVal}`,
      `model.precision=${precision}`,
      `logger=${logger}`,
      ...siteArgs(),
      ...stageArgs(),
      ...subsetArgs(),
      ...overrideLines(extra),
      `exp_name=${runName()}`,
    ];
    let text = lines.join(' ');
    if (runTest) {
      let test = `\n.venv/bin/python test.py exp_name=${runName()}`;
      if (trainFlights.length === 1) test += ` train_site=${trainFlights[0]}`;
      else if (trainFlights.length) test += ` train_site=[${trainFlights.join(',')}]`;
      text += test;
    }
    return text;
  });

  let running = $derived(jobs.find((job) => job.status === 'running') ?? null);
  let trace = $derived(polyline(samples));

  const modeLabel = {
    max_px_site_pheno: 'largest, spread over sites and stages',
    max_px_site: 'largest, spread over sites',
    max_px_pheno: 'largest, spread over stages',
    max_px: 'largest tiles',
    random: 'random',
    all: 'any tile with that species',
  };

  function stampNow() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }

  function countText(value) {
    const n = Number(value);
    return Number.isFinite(n) ? String(n) : '0';
  }

  const STAGE_COLOR = {
    nonflower: '#1b7f4e',
    flower: '#f08c14',
    fruiting: '#5b2d8e',
  };

  // Matches configs/data/neophytes_names_colors.yaml when the options payload has no color.
  const SPECIES_COLOR = {
    B_davidii: '#dce14c',
    R_typhina: '#b76e44',
    R_japonica: '#305dc6',
    B_orientalis: '#e2256a',
    S_inaequidens: '#50e3c2',
    A_altissima: '#dd60f6',
  };
  const STAGE_NAMES = ['nonflower', 'flower', 'fruiting'];

  let flights = $derived(options?.flights_by_data?.[dataName] ?? []);
  let flightGroups = $derived.by(() => {
    const groups = [];
    for (const row of flights) {
      let group = groups.at(-1);
      if (!group || group.site !== row.site) {
        group = { site: row.site, flights: [] };
        groups.push(group);
      }
      group.flights.push(row);
    }
    return groups;
  });
  let stageReady = $derived.by(() => {
    const chosen = flights.filter((row) => trainFlights.includes(row.flight));
    const pool = chosen.length ? chosen : flights;
    const ready = { nonflower: false, flower: false, fruiting: false };
    if (!pool.length) return { nonflower: true, flower: true, fruiting: true };
    for (const row of pool) {
      for (const name of STAGE_NAMES) {
        if ((row.stage_px?.[name] || 0) > 0) ready[name] = true;
      }
    }
    return ready;
  });

  let coveredSpecies = $derived.by(() => {
    const chosen = flights.filter((row) => trainFlights.includes(row.flight));
    if (!chosen.length) return null;
    const names = new Set();
    for (const row of chosen) {
      for (const [name, count] of Object.entries(row.species || {})) {
        if (count > 0) names.add(name);
      }
    }
    return names;
  });

  $effect(() => {
    if (!flights.length) return;
    const allowed = new Set(flights.map((row) => row.flight));
    if (trainFlights.some((key) => !allowed.has(key))) {
      trainFlights = trainFlights.filter((key) => allowed.has(key));
    }
  });

  $effect(() => {
    if (!flightsOpen) return;
    const close = (event) => {
      if (flightsRoot && !flightsRoot.contains(event.target)) flightsOpen = false;
    };
    const onKey = (event) => {
      if (event.key === 'Escape') flightsOpen = false;
    };
    document.addEventListener('click', close);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', close);
      document.removeEventListener('keydown', onKey);
    };
  });

  let speciesLocked = $derived(fewShotK == null || fewShotK === '');

  $effect(() => {
    if (!speciesOpen) return;
    const close = (event) => {
      if (speciesRoot && !speciesRoot.contains(event.target)) speciesOpen = false;
    };
    const onKey = (event) => {
      if (event.key === 'Escape') speciesOpen = false;
    };
    document.addEventListener('click', close);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', close);
      document.removeEventListener('keydown', onKey);
    };
  });

  $effect(() => {
    if (speciesLocked) speciesOpen = false;
  });

  function speciesChosen(name) {
    return speciesOn[name] !== false;
  }

  function selectedSpecies() {
    const all = options?.species ?? [];
    return all.filter((sp) => speciesChosen(sp.stats_name)).map((sp) => sp.stats_name);
  }

  function speciesRestricted() {
    const all = options?.species ?? [];
    if (!subsetOn || !all.length) return false;
    if (fewShotK == null || fewShotK === '') return false;
    return selectedSpecies().length < all.length;
  }

  // siteAarau_1_t10_tbg10_spB_davidii: site, then tile counts, then a species list
  // only when it is not every species.
  function readyStageCount() {
    return STAGE_NAMES.filter((name) => stageReady[name]).length;
  }

  function selectedStages() {
    return STAGE_NAMES.filter((name) => stageOn[name] && stageReady[name]);
  }

  function stagesRestricted() {
    const chosen = selectedStages();
    const ready = readyStageCount();
    return chosen.length > 0 && chosen.length < ready;
  }

  function toggleFlight(key) {
    trainFlights = trainFlights.includes(key)
      ? trainFlights.filter((item) => item !== key)
      : [...trainFlights, key];
  }

  function toggleSiteFlights(group) {
    const keys = group.flights.map((row) => row.flight);
    const allOn = keys.every((key) => trainFlights.includes(key));
    trainFlights = allOn
      ? trainFlights.filter((key) => !keys.includes(key))
      : [...trainFlights, ...keys.filter((key) => !trainFlights.includes(key))];
  }

  function flightWhen(row) {
    return row.acq_date || (row.year ? String(row.year) : '');
  }

  const STAGE_WORD = {
    nonflower: 'non-flowering',
    flower: 'flowering',
    fruiting: 'fruiting',
  };

  function stageWords(row) {
    return STAGE_NAMES
      .filter((name) => (row.stage_px?.[name] || 0) > 0)
      .map((name) => STAGE_WORD[name])
      .join(' · ');
  }

  function flightSpecies(row) {
    return (options?.species ?? []).filter((sp) => (row.species?.[sp.stats_name] || 0) > 0);
  }

  function chipColor(sp) {
    return sp.color || SPECIES_COLOR[sp.stats_name] || '#8d8d8d';
  }

  function nameTag() {
    const parts = [];
    if (trainFlights.length === 1) parts.push(`site${trainFlights[0]}`);
    else if (trainFlights.length > 3) parts.push(`fl${trainFlights.length}`);
    else if (trainFlights.length) parts.push(`fl_${trainFlights.join('-')}`);
    if (stagesRestricted()) parts.push(`st${selectedStages().join('-')}`);
    if (subsetOn) {
      const trainSet = fewShotK != null && fewShotK !== '';
      const valSet = valFewShotK != null && valFewShotK !== '';
      if (trainSet) parts.push(`t${countText(fewShotK)}`);
      if (valSet) parts.push(`v${countText(valFewShotK)}`);
      if (trainSet) parts.push(`tbg${countText(fewShotBg)}`);
      if (valSet) parts.push(`vbg${countText(valFewShotBg)}`);
      if (speciesRestricted()) parts.push(`sp${selectedSpecies().join('-')}`);
    }
    return parts.join('_');
  }

  function defaultRunName() {
    const base = `${nameStamp}_${dataName}_${modelName}_s${seed}`;
    const tag = nameTag();
    return tag ? `${base}_${tag}` : base;
  }

  function runName() {
    const typed = expName.trim();
    if (typed) return typed;
    return defaultRunName();
  }

  // Keep the prefilled name in step with data, model, seed, and the subset
  // counts until it is edited. The timestamp changes only with data, model, or seed.
  let trackedNameKey = '';
  $effect(() => {
    const key = `${dataName}\0${modelName}\0${seed}`;
    if (expNameCustom) return;
    const tag = nameTag();
    if (key !== trackedNameKey) {
      trackedNameKey = key;
      nameStamp = stampNow();
    }
    const base = `${nameStamp}_${dataName}_${modelName}_s${seed}`;
    expName = tag ? `${base}_${tag}` : base;
  });

  function siteArgs() {
    if (!trainFlights.length) return [];
    if (trainFlights.length === 1) return [`data.train_site=${trainFlights[0]}`];
    return [`data.train_site=[${trainFlights.join(',')}]`];
  }

  function stageArgs() {
    if (!stagesRestricted()) return [];
    return [`data.train_stages=[${selectedStages().join(',')}]`];
  }

  function subsetArgs() {
    const args = [];
    if (!subsetOn) return args;
    if (speciesRestricted()) {
      args.push(`data.few_shot_species=[${selectedSpecies().join(',')}]`);
    }
    if (fewShotK != null && fewShotK !== '') {
      args.push(`data.few_shot_k=${Number(fewShotK)}`);
      args.push(`data.few_shot_bg_k=${Number(fewShotBg)}`);
      args.push(`data.few_shot_mode=${fewShotMode}`);
    }
    if (valFewShotK != null && valFewShotK !== '') {
      args.push(`data.val_few_shot_k=${Number(valFewShotK)}`);
      args.push(`data.val_few_shot_bg_k=${Number(valFewShotBg)}`);
      args.push(`data.val_few_shot_mode=${valFewShotMode}`);
    }
    return args;
  }

  function countOrNull(value) {
    if (value == null || value === '') return null;
    return Number(value);
  }

  function overrideLines(text) {
    return text.split('\n').map((line) => line.trim()).filter((line) => line && !line.startsWith('#'));
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

  function mem(mib) {
    if (mib == null) return '—';
    const gb = mib / 1024;
    return gb >= 10 ? gb.toFixed(0) : gb.toFixed(1);
  }

  function polyline(rows) {
    if (rows.length < 2) return { util: '', mem: '' };
    const last = rows.slice(-450);
    const util = [];
    const memory = [];
    last.forEach((row, index) => {
      const x = (index / (last.length - 1)) * 100;
      const u = row.util == null ? 0 : Math.max(0, Math.min(100, row.util));
      const total = row.memoryTotal || 1;
      const m = row.memoryUsed == null ? 0 : Math.max(0, Math.min(100, (row.memoryUsed / total) * 100));
      util.push(`${x.toFixed(2)},${(28 - (u / 100) * 26).toFixed(2)}`);
      memory.push(`${x.toFixed(2)},${(28 - (m / 100) * 26).toFixed(2)}`);
    });
    return { util: util.join(' '), mem: memory.join(' ') };
  }

  function onLogScroll(event) {
    const el = event.currentTarget;
    stick = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }

  $effect(() => {
    logText;
    if (!stick || !logEl) return;
    logEl.scrollTop = logEl.scrollHeight;
  });

  $effect(() => {
    const ctrl = new AbortController();
    fetch('/api/train/options', { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res));
        return res.json();
      })
      .then((data) => {
        options = data;
        optionsError = '';
        const d = data.defaults ?? {};
        if (d.data) dataName = d.data;
        if (d.model) modelName = d.model;
        if (d.seed != null) seed = d.seed;
        if (d.max_epochs != null) maxEpochs = d.max_epochs;
        if (d.check_val_every_n_epoch != null) checkVal = d.check_val_every_n_epoch;
        if (d.lr != null) lr = d.lr;
        if (d.batch_size_train != null) batchTrain = d.batch_size_train;
        if (d.batch_size_val != null) batchVal = d.batch_size_val;
        if (d.precision) precision = d.precision;
        if (d.logger) logger = d.logger;
        if (d.few_shot_bg_k != null) fewShotBg = d.few_shot_bg_k;
        if (d.few_shot_mode) fewShotMode = d.few_shot_mode;
        if (d.val_few_shot_bg_k != null) valFewShotBg = d.val_few_shot_bg_k;
        if (d.val_few_shot_mode) valFewShotMode = d.val_few_shot_mode;
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        optionsError = String(err.message || err);
      });
    return () => ctrl.abort();
  });

  function applyModelForm(name, form) {
    if (!form || name === appliedModel) return;
    const currentData = untrack(() => dataName);
    const previousPin = pinnedData;
    appliedModel = name;
    maxEpochs = form.max_epochs;
    checkVal = form.check_val_every_n_epoch;
    lr = form.lr;
    if (form.precision) precision = form.precision;
    if (form.data) {
      if (!previousPin) dataBeforePin = currentData;
      dataName = form.data;
      pinnedData = form.data;
    } else {
      if (previousPin && currentData === previousPin && dataBeforePin) dataName = dataBeforePin;
      pinnedData = '';
      dataBeforePin = '';
    }
  }

  $effect(() => {
    const forms = options?.model_forms;
    const name = modelName;
    if (!forms?.[name]) return;
    applyModelForm(name, forms[name]);
  });

  $effect(() => {
    if (!watchGpu) return;
    const card = gpu?.gpus?.[0];
    if (!card) return;
    history.push({
      util: card.util,
      memoryUsed: card.memory_used,
      memoryTotal: card.memory_total,
    });
    if (history.length > 450) history.splice(0, history.length - 450);
    samples = history.slice();
  });

  $effect(() => {
    let dead = false;
    async function poll() {
      while (!dead) {
        try {
          const res = await fetch('/api/jobs');
          if (res.ok) {
            const body = await res.json();
            if (!dead) {
              jobs = body.jobs ?? [];
              if (openId && !jobs.some((job) => job.id === openId)) {
                openId = '';
                logText = '';
              }
              if (!openId) {
                const current = jobs.find((job) => job.status === 'running') ?? jobs.at(-1);
                if (current) openId = current.id;
              }
            }
          }
        } catch {
          /* keep the last snapshot */
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }
    poll();
    return () => {
      dead = true;
    };
  });

  $effect(() => {
    const id = openId;
    const window = logWindow;
    if (!id) return;
    let dead = false;
    let offset = 0;
    let text = '';
    const tail = window === 'all' ? 0 : Number(window);
    logText = '';
    logOffset = 0;
    async function poll() {
      while (!dead) {
        try {
          const url = tail
            ? `/api/jobs/${id}/log?tail=${tail}`
            : `/api/jobs/${id}/log?offset=${offset}`;
          const res = await fetch(url);
          if (res.ok) {
            const body = await res.json();
            if (!dead && tail) {
              if (body.text !== text) {
                text = body.text || '';
                logText = text;
              }
            } else if (!dead && body.text) {
              text += body.text;
              logText = text;
              offset = body.offset;
              logOffset = offset;
            } else if (!dead) {
              offset = body.offset ?? offset;
            }
          }
        } catch {
          /* the next tick retries */
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
    poll();
    return () => {
      dead = true;
    };
  });

  async function loadWandb(refresh) {
    try {
      const res = await fetch('/api/wandb/runs' + (refresh ? '?refresh=true' : ''));
      if (!res.ok) throw new Error(await readError(res));
      wandb = await res.json();
      wandbError = wandb.available ? '' : (wandb.detail || 'W&B is unavailable');
    } catch (err) {
      wandbError = String(err.message || err);
    }
  }

  $effect(() => {
    let dead = false;
    loadWandb(false);
    const timer = setInterval(() => {
      if (!dead) loadWandb(false);
    }, 60000);
    return () => {
      dead = true;
      clearInterval(timer);
    };
  });

  $effect(() => {
    const timer = setInterval(() => {
      now = Date.now();
    }, 30000);
    return () => clearInterval(timer);
  });

  async function enqueue() {
    formError = '';
    if (subsetOn && fewShotK != null && fewShotK !== '' && selectedSpecies().length === 0) {
      formError = 'Pick at least one species.';
      return;
    }
    if (selectedStages().length === 0) {
      formError = 'Pick at least one stage.';
      return;
    }
    submitting = true;
    try {
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataName,
          model: modelName,
          seed: Number(seed),
          max_epochs: Number(maxEpochs),
          check_val_every_n_epoch: Number(checkVal),
          lr: Number(lr),
          batch_size_train: Number(batchTrain),
          batch_size_val: Number(batchVal),
          precision,
          logger,
          exp_name: expName.trim(),
          run_test: runTest,
          train_site: trainFlights.length ? trainFlights : null,
          train_stages: stagesRestricted() ? selectedStages() : null,
          few_shot_species: speciesRestricted() ? selectedSpecies() : null,
          few_shot_k: subsetOn ? countOrNull(fewShotK) : null,
          few_shot_bg_k: subsetOn ? countOrNull(fewShotBg) : null,
          few_shot_mode: subsetOn ? fewShotMode : null,
          val_few_shot_k: subsetOn ? countOrNull(valFewShotK) : null,
          val_few_shot_bg_k: subsetOn ? countOrNull(valFewShotBg) : null,
          val_few_shot_mode: subsetOn ? valFewShotMode : null,
          overrides: overrideLines(extra),
        }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const job = await res.json();
      openId = job.id;
      logText = '';
      logOffset = 0;
    } catch (err) {
      formError = String(err.message || err);
    } finally {
      submitting = false;
    }
  }

  function finished(job) {
    return job.status === 'done' || job.status === 'failed' || job.status === 'cancelled';
  }

  let canClear = $derived(jobs.some((job) => job.status === 'queued' || finished(job)));

  function askCancel(job) {
    confirmClear = false;
    confirmJob = job;
  }

  function askClear() {
    confirmJob = null;
    confirmClear = true;
  }

  function closeConfirm() {
    confirmJob = null;
    confirmClear = false;
  }

  async function cancel(id) {
    confirmJob = null;
    const res = await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' });
    if (!res.ok) formError = await readError(res);
  }

  async function hide(id) {
    const res = await fetch(`/api/jobs/${id}/hide`, { method: 'POST' });
    if (!res.ok) {
      formError = await readError(res);
      return;
    }
    jobs = jobs.filter((job) => job.id !== id);
    if (openId === id) {
      openId = '';
      logText = '';
    }
  }

  async function clearQueue() {
    confirmClear = false;
    const res = await fetch('/api/jobs/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ waiting: true, finished: true }),
    });
    if (!res.ok) {
      formError = await readError(res);
      return;
    }
    const staying = jobs.filter((job) => job.status === 'running');
    jobs = staying;
    if (openId && !staying.some((job) => job.id === openId)) {
      openId = '';
      logText = '';
    }
  }

  $effect(() => {
    const el = confirmEl;
    const open = confirmJob || confirmClear;
    if (!el) return;
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
  });

  function metric(run, key) {
    const value = run.metrics?.[key];
    if (value == null) return '—';
    return Number(value).toFixed(3);
  }

  function when(value) {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
  }

  function formatDuration(ms) {
    const total = Math.max(0, Math.floor(ms / 60000));
    const days = Math.floor(total / 1440);
    const hours = Math.floor((total % 1440) / 60);
    const minutes = total % 60;
    const parts = [];
    if (days) parts.push(`${days}d`);
    if (days || hours) parts.push(`${hours}h`);
    parts.push(`${minutes}m`);
    return parts.join(' ');
  }

  function duration(run) {
    const start = Date.parse(run.created_at);
    if (Number.isNaN(start)) return '—';
    let end = NaN;
    if (run.finished_at) end = Date.parse(run.finished_at);
    else if (run.state === 'running') end = now;
    if (Number.isNaN(end) || end < start) return '—';
    return formatDuration(end - start);
  }

  function stateTone(state) {
    if (state === 'running') return 'running';
    if (state === 'failed' || state === 'crashed' || state === 'killed') return 'failed';
    return state || '';
  }
</script>

<div class="runs">
  <div class="top">
    <form class="card" onsubmit={(event) => { event.preventDefault(); enqueue(); }}>
      <h2>New run</h2>
      {#if optionsError}
        <p class="problem" role="alert">{optionsError}</p>
      {/if}
      <label>
        Data
        <select bind:value={dataName}>
          {#each options?.data ?? [dataName] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <label>
        Model
        <select bind:value={modelName}>
          {#each options?.models ?? [modelName] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      {#if modelNote}
        <p class="hint">{modelNote}</p>
      {/if}
      <div class="grid">
        <label>
          Seed
          <input type="number" min="0" bind:value={seed} />
        </label>
        <label>
          Epochs
          <input type="number" min="1" max="500" bind:value={maxEpochs} />
        </label>
        <label>
          Val every n epochs
          <input type="number" min="1" max="500" bind:value={checkVal} />
        </label>
        <label>
          Learning rate
          <input type="number" min="0.00000001" max="1" step="any" bind:value={lr} />
        </label>
        <label>
          Train batch
          <input type="number" min="1" max="64" bind:value={batchTrain} />
        </label>
        <label>
          Val batch
          <input type="number" min="1" max="64" bind:value={batchVal} />
        </label>
        <label>
          Precision
          <select bind:value={precision}>
            {#each options?.precisions ?? [precision] as name}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </label>
      </div>
      <label>
        Logger
        <select bind:value={logger}>
          {#each options?.loggers ?? [logger] as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <label>
        W&B name
        <input type="text" spellcheck="false" bind:value={expName} oninput={() => (expNameCustom = true)} />
      </label>
      <div class="override-field">
        <div class="check-row">
          <span class="override-label" id="flights-label">Train on flights</span>
          <span class="note-wrap">
            <button
              type="button"
              class="info"
              aria-label="What training on chosen flights does"
              aria-expanded={siteNote}
              onclick={() => (siteNote = !siteNote)}
            >i</button>
            {#if siteNote}
              <div class="test-note" role="dialog" aria-label="What training on chosen flights does">
                <p>
                  Leave this empty to train on every flight in the split. Pick flights to keep only those.
                  A new plant often has a single flight, so this is how that case is trained.
                </p>
                <p>
                  Each row is one flight. The names are the species labelled there, and the bar is the share of
                  non-flowering, flowering, and fruiting pixels. A site name selects every flight of that site.
                </p>
                <p>
                  Validation uses the same flights. The test split stays the other sites in the fold.
                  Val F1 is same-flight. Test F1 is held-out. test.py writes both to metrics_site_split.json.
                </p>
              </div>
            {/if}
          </span>
        </div>
        <div class="flight-pick" bind:this={flightsRoot}>
          <div class="multi" class:open={flightsOpen}>
            <div class="multi-value">
              {#if trainFlights.length === 0}
                <span class="multi-empty">All flights in this split</span>
              {:else}
                {#each trainFlights as key (key)}
                  <span class="chip">
                    {key}
                    <button type="button" aria-label={`Remove ${key}`} onclick={(event) => { event.stopPropagation(); toggleFlight(key); }}>×</button>
                  </span>
                {/each}
              {/if}
            </div>
            <button
              type="button"
              class="multi-toggle"
              aria-haspopup="listbox"
              aria-expanded={flightsOpen}
              aria-labelledby="flights-label"
              onclick={() => (flightsOpen = !flightsOpen)}
            >▾</button>
            {#if flightsOpen}
              <div class="flight-menu" role="listbox" aria-multiselectable="true" aria-labelledby="flights-label">
                {#each flightGroups as group (group.site)}
                  <div class="flight-site">
                    <button type="button" onclick={() => toggleSiteFlights(group)}>{group.site}</button>
                  </div>
                  {#each group.flights as row (row.flight)}
                    <label class="flight-row">
                      <input
                        type="checkbox"
                        checked={trainFlights.includes(row.flight)}
                        onchange={() => toggleFlight(row.flight)}
                      />
                      <span class="flight-name">
                        <strong>{row.flight}</strong>
                        <span>{flightWhen(row)} · {row.labelled} labelled</span>
                      </span>
                      <span class="flight-detail">
                        {#if flightSpecies(row).length}
                          <span class="flight-species">
                            {#each flightSpecies(row) as sp (sp.stats_name)}
                              <span class="sp-chip">
                                <i class="swatch" style:background={chipColor(sp)} style:--swatch={chipColor(sp)}></i>
                                {sp.name}
                              </span>
                            {/each}
                          </span>
                        {/if}
                        {#if stageWords(row)}
                          <span class="stage-line">
                            <span class="mini-stages" aria-hidden="true">
                              {#each STAGE_NAMES as name}
                                <i style:width="{(100 * (row.stages?.[name] || 0)).toFixed(2)}%" style:background={STAGE_COLOR[name]}></i>
                              {/each}
                            </span>
                            <span class="stage-words">{stageWords(row)}</span>
                          </span>
                        {/if}
                      </span>
                    </label>
                  {/each}
                {/each}
              </div>
            {/if}
          </div>
          {#if coveredSpecies}
            <p class="hint">
              Species in the chosen flights:
              {#each options?.species ?? [] as sp (sp.stats_name)}
                <span class:missing={!coveredSpecies.has(sp.stats_name)}>{sp.name}</span>
              {/each}
            </p>
            {#if (options?.species ?? []).some((sp) => !coveredSpecies.has(sp.stats_name))}
              <p class="hint">A flight that lacks a species cannot teach it. Add a flight that has the missing species.</p>
            {/if}
          {/if}
        </div>
      </div>
      <div class="override-field">
        <div class="check-row">
          <span class="override-label">Train stages</span>
          <span class="note-wrap">
            <button
              type="button"
              class="info"
              aria-label="What training on chosen stages does"
              aria-expanded={stageNote}
              onclick={() => (stageNote = !stageNote)}
            >i</button>
            {#if stageNote}
              <div class="test-note" role="dialog" aria-label="What training on chosen stages does">
                <p>
                  All three stages are the normal setting. Turn one off to ignore that stage.
                  Pixels of a skipped stage become ignore, so the model is not taught that they are background.
                  Tiles with none of the chosen stages are dropped. Background-only tiles stay.
                </p>
                <p>
                  Validation uses the same stages. The test split does not, so test scores still include every stage.
                  That is the stage-transfer check. Sampling weights still use the full pixel counts of a tile.
                </p>
              </div>
            {/if}
          </span>
        </div>
        <div class="stage-toggles">
          {#each options?.stages ?? STAGE_NAMES.map((name) => ({ phase: name, label: name })) as stage (stage.phase)}
            <label class="check" class:locked={!stageReady[stage.phase]}>
              <input
                type="checkbox"
                checked={stageOn[stage.phase] && stageReady[stage.phase]}
                disabled={!stageReady[stage.phase]}
                onchange={(event) => (stageOn[stage.phase] = event.currentTarget.checked)}
              />
              <i class="swatch" style:--swatch={STAGE_COLOR[stage.phase]}></i>
              {stage.label}
            </label>
          {/each}
        </div>
      </div>
      <div class="override-field">
        <div class="check-row">
          <label class="check">
            <input type="checkbox" bind:checked={subsetOn} />
            Subset
          </label>
          <span class="note-wrap">
            <button
              type="button"
              class="info"
              aria-label="What the subset fields do"
              aria-expanded={subsetNote}
              onclick={() => (subsetNote = !subsetNote)}
            >i</button>
            {#if subsetNote}
              <div class="test-note" role="dialog" aria-label="What the subset fields do">
                <p>
                  Leave images per species empty to train or validate on the whole split.
                  A number keeps that many tiles of each species. Background tiles are added on top.
                </p>
                <p>
                  The training choice prefers the largest single-species tiles and spreads them over
                  sites and stages. Remove a species to skip its tiles. A validation subset drops empty
                  tiles, so false positives on bare ground stay out of the score.
                </p>
              </div>
            {/if}
          </span>
        </div>
        {#if subsetOn}
        <div class="grid">
          <label>
            Train images per species
            <input type="number" min="1" max="5000" placeholder="all" bind:value={fewShotK} />
          </label>
          <label>
            Val images per species
            <input type="number" min="1" max="5000" placeholder="all" bind:value={valFewShotK} />
          </label>
          <label>
            Train background tiles
            <input type="number" min="0" max="5000" bind:value={fewShotBg} disabled={fewShotK == null || fewShotK === ''} />
          </label>
          <label>
            Val background tiles
            <input type="number" min="0" max="5000" bind:value={valFewShotBg} disabled={valFewShotK == null || valFewShotK === ''} />
          </label>
          <label>
            Train tile choice
            <select bind:value={fewShotMode} disabled={fewShotK == null || fewShotK === ''}>
              {#each options?.few_shot_modes ?? [fewShotMode] as name}
                <option value={name}>{modeLabel[name] ?? name}</option>
              {/each}
            </select>
          </label>
          <label>
            Val tile choice
            <select bind:value={valFewShotMode} disabled={valFewShotK == null || valFewShotK === ''}>
              {#each options?.few_shot_modes ?? [valFewShotMode] as name}
                <option value={name}>{modeLabel[name] ?? name}</option>
              {/each}
            </select>
          </label>
        </div>
        <div class="species-pick" bind:this={speciesRoot}>
          <span class="override-label" id="species-label">Species</span>
          <div class="multi" class:locked={speciesLocked} class:open={speciesOpen}>
            <div class="multi-value">
              {#each options?.species ?? [] as sp (sp.stats_name)}
                {#if speciesChosen(sp.stats_name)}
                  <span class="chip">
                    {sp.name}
                    <button
                      type="button"
                      disabled={speciesLocked}
                      aria-label={`Remove ${sp.name}`}
                      onclick={(event) => {
                        event.stopPropagation();
                        speciesOn[sp.stats_name] = false;
                      }}
                    >×</button>
                  </span>
                {/if}
              {/each}
              {#if selectedSpecies().length === 0}
                <span class="multi-empty">None selected</span>
              {/if}
            </div>
            <button
              type="button"
              class="multi-toggle"
              aria-haspopup="listbox"
              aria-expanded={speciesOpen}
              aria-labelledby="species-label"
              disabled={speciesLocked}
              onclick={() => (speciesOpen = !speciesOpen)}
            >▾</button>
            {#if speciesOpen}
              <ul class="multi-menu" role="listbox" aria-multiselectable="true" aria-labelledby="species-label">
                {#each options?.species ?? [] as sp (sp.stats_name)}
                  <li>
                    <button
                      type="button"
                      role="option"
                      aria-selected={speciesChosen(sp.stats_name)}
                      class:on={speciesChosen(sp.stats_name)}
                      onclick={(event) => {
                        event.stopPropagation();
                        speciesOn[sp.stats_name] = !speciesChosen(sp.stats_name);
                      }}
                    >{sp.name}</button>
                  </li>
                {/each}
              </ul>
            {/if}
          </div>
        </div>
        {/if}
      </div>
      <div class="check-row">
        <label class="check">
          <input type="checkbox" bind:checked={runTest} />
          Run test.py after training
        </label>
        <span class="note-wrap">
          <button
            type="button"
            class="info"
            aria-label="What test.py does"
            aria-expanded={testNote}
            onclick={() => (testNote = !testNote)}
          >i</button>
          {#if testNote}
            <div class="test-note" role="dialog" aria-label="What test.py does">
              <p>
                test.py scores the best checkpoint on this split's held-out sites.
                A name like neophytes_split_cv3_train_1024 is matched to neophytes_split_cv3_test.
                With flights selected it also scores those flights' validation tiles.
              </p>
              <p>Training already does a short test pass. This checkbox adds that full evaluation in the same tmux session.</p>
            </div>
          {/if}
        </span>
      </div>
      <div class="override-field">
        <div class="check-row">
          <span class="override-label">Extra overrides</span>
          <span class="note-wrap">
            <button
              type="button"
              class="info"
              aria-label="What extra overrides are"
              aria-expanded={extraNote}
              onclick={() => (extraNote = !extraNote)}
            >i</button>
            {#if extraNote}
              <div class="test-note" role="dialog" aria-label="What extra overrides are">
                <p>
                  Each line is one Hydra setting, written as key=value with no spaces.
                  A line starting with # is skipped.
                </p>
                <p>
                  These are added to the train.py command after the fields above.
                  A later key replaces an earlier one. The W&B name is added after these lines.
                </p>
              </div>
            {/if}
          </span>
        </div>
        <textarea rows="4" bind:value={extra} placeholder={'model.lr=0.0002\ndata.num_workers=4'}></textarea>
      </div>
      <button type="button" class="ghost" onclick={() => (showCommand = !showCommand)} aria-expanded={showCommand}>
        {showCommand ? 'Hide command' : 'Show command'}
      </button>
      {#if showCommand}
        <pre class="command">{command}</pre>
      {/if}
      {#if formError}
        <p class="problem" role="alert">{formError}</p>
      {/if}
      <button type="submit" class="run" disabled={submitting}>
        {submitting ? 'Queuing…' : 'Queue run'}
      </button>
    </form>

    <div class="side">
      <section class="card" aria-label="GPU">
        <h2>GPU</h2>
        {#if !watchGpu && !gpu}
          <p class="hint">Live refresh is off.</p>
        {:else if gpu && !gpu.available}
          <p class="hint">{gpu.detail || 'nvidia-smi is not available'}</p>
        {:else if gpu?.gpus?.length}
          {#if !watchGpu}
            <p class="hint">Paused.</p>
          {/if}
          {#each gpu.gpus as card, index (card.index ?? index)}
            <p class="readout">
              <strong>{card.name}</strong>
              {card.util == null ? '—' : `${Math.round(card.util)}%`}
              · {mem(card.memory_used)}/{mem(card.memory_total)} GB
              {#if card.temperature != null}· {Math.round(card.temperature)}°C{/if}
              {#if card.power != null}· {Math.round(card.power)} W{/if}
            </p>
            {#if trace.util}
              <svg class="trace" viewBox="0 0 100 28" preserveAspectRatio="none" aria-hidden="true">
                <polyline class="mem" points={trace.mem} />
                <polyline class="util" points={trace.util} />
              </svg>
              <p class="legend-line"><span class="util-key">utilization</span><span class="mem-key">memory</span></p>
            {/if}
            {#if card.processes?.length}
              <ul class="procs">
                {#each card.processes as proc (`${proc.pid}-${proc.name}`)}
                  <li>{proc.name} · pid {proc.pid}{#if proc.memory != null} · {mem(proc.memory)} GB{/if}</li>
                {/each}
              </ul>
            {:else}
              <p class="hint">No compute process.</p>
            {/if}
          {/each}
        {:else}
          <p class="hint">Reading nvidia-smi…</p>
        {/if}
      </section>

      <section class="card">
        <div class="queue-head">
          <h2>Queue</h2>
          {#if canClear}
            <button type="button" class="cancel" onclick={askClear}>Clear</button>
          {/if}
        </div>
        {#if jobs.length}
          <ul class="queue">
            {#each jobs as job (job.id)}
              <li class:open={job.id === openId}>
                <div class="job-row">
                  <button type="button" class="job" onclick={() => (openId = job.id)}>
                    <span class="status" data-status={job.status}>{job.status}</span>
                    <span class="exp">{job.exp_name}</span>
                  </button>
                  {#if job.status === 'queued' || job.status === 'running'}
                    <button type="button" class="cancel" onclick={() => askCancel(job)}>Cancel</button>
                  {:else if finished(job)}
                    <button type="button" class="cancel" onclick={() => hide(job.id)}>Hide</button>
                  {/if}
                </div>
                {#if job.progress}
                  <p class="now">{job.progress}</p>
                {/if}
                {#if job.wait_reason}
                  <p class="hint">{job.wait_reason}</p>
                {/if}
                {#if job.error}
                  <p class="problem">{job.error}</p>
                {/if}
              </li>
            {/each}
          </ul>
        {:else}
          <p class="hint">No jobs yet.</p>
        {/if}
        <dialog
          class="confirm"
          bind:this={confirmEl}
          aria-labelledby="cancel-title"
          oncancel={(event) => {
            event.preventDefault();
            closeConfirm();
          }}
          onclick={(event) => {
            if (event.target === event.currentTarget) closeConfirm();
          }}
        >
          {#if confirmJob}
            <p id="cancel-title">Cancel this run?</p>
            <p class="hint">{confirmJob.exp_name}</p>
            <div class="confirm-actions">
              <button type="button" class="ghost" onclick={closeConfirm}>Keep it</button>
              <button type="button" class="cancel" onclick={() => cancel(confirmJob.id)}>Cancel run</button>
            </div>
          {:else if confirmClear}
            <p id="cancel-title">Clear the queue?</p>
            <p class="hint">Waiting jobs are cancelled. Finished jobs are hidden. A running job stays.</p>
            <div class="confirm-actions">
              <button type="button" class="ghost" onclick={closeConfirm}>Keep them</button>
              <button type="button" class="cancel" onclick={clearQueue}>Clear</button>
            </div>
          {/if}
        </dialog>
        {#if running?.wait_reason}
          <p class="hint">{running.wait_reason}</p>
        {/if}
      </section>

      <section class="card log-card">
        <div class="log-head">
          <h2>Log {#if openId}<span class="hint">{openId}</span>{/if}</h2>
          <label class="log-window">
            Show
            <select bind:value={logWindow}>
              <option value="100">100 recent</option>
              <option value="200">200 recent</option>
              <option value="500">500 recent</option>
              <option value="1000">1,000 recent</option>
              <option value="all">All</option>
            </select>
          </label>
        </div>
        <pre class="log" bind:this={logEl} onscroll={onLogScroll}>{logText || 'Log output appears here once the job starts.'}</pre>
      </section>
    </div>
  </div>

  <section class="card">
    <div class="wandb-head">
      <h2>Weights & Biases</h2>
      <button type="button" onclick={() => loadWandb(true)}>Refresh</button>
    </div>
    {#if wandbError}
      <p class="problem" role="alert">{wandbError}</p>
    {/if}
    {#if wandb?.project}
      <p class="hint">{wandb.project}</p>
    {/if}
    {#if wandb?.runs?.length}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>State</th>
              <th>Created</th>
              <th>Finished</th>
              <th>Duration</th>
              <th>val F1 species</th>
              <th>test F1 species</th>
              <th>Encoder</th>
            </tr>
          </thead>
          <tbody>
            {#each wandb.runs as run (run.id)}
              <tr>
                <td>
                  {#if run.url}
                    <a href={run.url} target="_blank" rel="noreferrer">{run.name}</a>
                  {:else}
                    {run.name}
                  {/if}
                </td>
                <td><span class="status" data-status={stateTone(run.state)}>{run.state}</span></td>
                <td>{when(run.created_at)}</td>
                <td>{when(run.finished_at) || '—'}</td>
                <td class="num">{duration(run)}</td>
                <td class="num">{metric(run, 'val_f1_species')}</td>
                <td class="num">{metric(run, 'test_f1_species')}</td>
                <td>{run.encoder ?? '—'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else if wandb?.available}
      <p class="hint">No runs in this project yet.</p>
    {/if}
  </section>
</div>

<style>
  .runs {
    min-height: 0;
    overflow: auto;
    padding: 16px 18px 32px;
  }

  .top {
    display: grid;
    grid-template-columns: minmax(280px, 420px) minmax(0, 1fr);
    gap: 16px;
    align-items: start;
  }

  .side { display: grid; gap: 16px; }

  .card {
    background: var(--sheet);
    border: 1px solid var(--line);
    padding: 14px 16px 16px;
    margin-bottom: 16px;
  }

  .side .card { margin-bottom: 0; }

  h2 {
    margin: 0 0 10px;
    font-family: "Bricolage Grotesque", "Public Sans", sans-serif;
    font-size: 18px;
    font-weight: 560;
    letter-spacing: -0.03em;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--mist);
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0 10px;
  }

  input, select, textarea {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: #fff;
    color: var(--ink);
    padding: 6px 8px;
  }

  .check {
    flex-direction: row;
    align-items: center;
    gap: 8px;
    color: var(--ink);
    font-size: 13px;
  }

  .check input { width: auto; }

  .check-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .check-row .check { margin-bottom: 0; }

  .override-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 8px;
  }

  .override-field .check-row { margin-bottom: 0; }

  .flight-pick { position: relative; }

  .flight-menu {
    position: absolute;
    z-index: 6;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    max-height: 320px;
    overflow: auto;
    margin: 0;
    padding: 4px;
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 2px;
    box-shadow: var(--shadow);
  }

  .flight-site {
    position: sticky;
    top: 0;
    background: var(--sheet);
    padding: 4px 4px 2px;
  }

  .flight-site button {
    width: auto;
    border: 0;
    background: transparent;
    color: var(--canopy);
    padding: 2px 4px;
    font-size: 12px;
    font-weight: 600;
  }

  .flight-row {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 2px 6px;
    align-items: start;
    margin: 0;
    padding: 6px 4px;
    color: var(--ink);
    font-size: 12px;
  }

  .flight-row input { width: auto; margin-top: 2px; }

  .flight-name {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .flight-name span { color: var(--mist); }

  .flight-detail {
    grid-column: 2;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }

  .flight-species {
    display: flex;
    flex-wrap: wrap;
    gap: 3px 8px;
  }

  .sp-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--ink);
    font-size: 11px;
  }

  .stage-line {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }

  .stage-words {
    color: var(--mist);
    font-size: 11px;
  }

  .mini-stages {
    display: flex;
    width: 64px;
    flex: none;
    height: 8px;
    background: #eef3f4;
    border: 1px solid var(--line);
  }

  .mini-stages i { display: block; height: 100%; }

  .stage-toggles {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 12px;
  }

  .stage-toggles .check { margin: 0; }

  .stage-toggles .locked { opacity: 0.45; }

  .hint .missing { color: #8c2f2f; }

  .hint span + span::before { content: " · "; color: var(--mist); }

  .species-pick {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 8px;
  }

  .multi {
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: 4px;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: #fff;
    padding: 4px 4px 4px 6px;
  }

  .multi.locked { opacity: 0.55; }

  .multi-value {
    flex: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    min-height: 26px;
    min-width: 0;
  }

  .chip {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: #f4f7f8;
    color: var(--ink);
    font-size: 12px;
    line-height: 1;
    padding: 3px 2px 3px 6px;
  }

  .chip button {
    width: auto;
    border: 0;
    background: transparent;
    color: var(--mist);
    padding: 0 4px;
    font-size: 14px;
    line-height: 1;
  }

  .chip button:hover:not(:disabled) { color: var(--ink); }

  .multi-empty {
    color: var(--mist);
    font-size: 12px;
    padding: 4px 2px;
  }

  .multi-toggle {
    width: auto;
    flex: none;
    border: 0;
    background: transparent;
    color: var(--ink);
    padding: 2px 6px;
    line-height: 1;
  }

  .multi-menu {
    position: absolute;
    z-index: 6;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    margin: 0;
    padding: 4px;
    list-style: none;
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 2px;
    box-shadow: var(--shadow);
  }

  .multi-menu button {
    width: 100%;
    border: 0;
    background: transparent;
    border-radius: 2px;
    text-align: left;
    color: var(--ink);
    padding: 6px 8px;
    font-size: 13px;
  }

  .multi-menu button.on { background: color-mix(in srgb, var(--canopy) 16%, white); }

  .multi-menu button:hover,
  .multi-menu button:focus-visible { background: #eef3f4; }

  .override-label {
    font-size: 12px;
    color: var(--mist);
  }

  .note-wrap { position: relative; }

  .test-note {
    position: absolute;
    z-index: 5;
    top: calc(100% + 8px);
    left: 0;
    width: min(320px, 70vw);
    padding: 12px 14px;
    background: var(--sheet);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    color: var(--ink);
    font-size: 13px;
  }

  .test-note p { margin: 0 0 8px; }
  .test-note p:last-child { margin: 0; }

  .ghost {
    margin: 0 0 10px;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: transparent;
    color: var(--canopy);
    padding: 6px 10px;
  }

  .command, .log {
    margin: 0 0 10px;
    padding: 8px 10px;
    background: #122024;
    color: #e7eef0;
    font-size: 12px;
    line-height: 1.45;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .log { max-height: 280px; margin: 0; }

  .log-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
  }

  .log-head h2, .queue-head h2 { margin: 0; }

  .queue-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
  }

  .log-window {
    flex-direction: row;
    align-items: center;
    gap: 8px;
    margin: 0;
    flex: none;
    color: var(--mist);
  }

  .log-window select { width: auto; }

  .run, .wandb-head button, .cancel {
    border: 0;
    border-radius: 2px;
    background: var(--canopy);
    color: var(--sheet);
    padding: 7px 12px;
  }

  .job-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }

  .job-row .job { flex: 1; min-width: 0; }

  .cancel {
    flex: none;
    align-self: center;
    background: transparent;
    color: var(--canopy);
    border: 1px solid var(--line);
  }

  .run:disabled { opacity: 0.55; }

  .readout { margin: 0 0 8px; font-variant-numeric: tabular-nums; }

  .trace { width: 100%; height: 72px; background: #eef3f4; border: 1px solid var(--line); }

  .trace .util { fill: none; stroke: var(--canopy); stroke-width: 1.2; }
  .trace .mem { fill: none; stroke: #b76e44; stroke-width: 1.2; }

  .legend-line { display: flex; gap: 12px; margin: 4px 0 8px; font-size: 12px; color: var(--mist); }
  .util-key { color: var(--canopy); }
  .mem-key { color: #b76e44; }

  .procs, .queue { list-style: none; margin: 0; padding: 0; }
  .procs li, .queue li { padding: 6px 0; border-bottom: 1px solid var(--line); font-size: 13px; }

  .job {
    display: flex;
    gap: 8px;
    align-items: baseline;
    width: 100%;
    padding: 0;
    border: 0;
    background: transparent;
    text-align: left;
  }

  .queue li.open .job { font-weight: 600; }

  .status {
    flex: none;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--mist);
  }

  .status[data-status='running'] { color: var(--canopy); }
  .status[data-status='failed'] { color: #8c2f2f; }

  .now {
    margin: 2px 0 0;
    color: var(--canopy);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }

  .exp { word-break: break-all; }

  .confirm {
    width: min(420px, 100%);
    background: var(--sheet);
    border: 1px solid var(--line);
    padding: 16px;
    box-shadow: var(--shadow);
  }

  .confirm::backdrop {
    background: rgba(18, 32, 36, 0.28);
  }

  .confirm p { margin: 0 0 8px; color: var(--ink); }

  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 14px;
  }

  .confirm .cancel { align-self: auto; }

  .hint { color: var(--mist); font-size: 12px; margin: 4px 0; }
  .problem { color: #8c2f2f; font-size: 13px; }

  .wandb-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
  .wandb-head h2 { margin: 0; }
  .wandb-head button { background: transparent; color: var(--canopy); border: 1px solid var(--line); }

  .table-wrap { overflow: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--mist); font-weight: 600; font-size: 12px; }
  td.num { font-variant-numeric: tabular-nums; }
  a { color: var(--canopy); }

  @media (max-width: 980px) {
    .top { grid-template-columns: 1fr; }
  }
</style>
