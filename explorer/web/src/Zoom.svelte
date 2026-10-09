<script>
  let { open = false, onclose, label = 'Image', note = '', children } = $props();

  const MIN_SCALE = 1;
  const MAX_SCALE = 8;

  let dialog = $state(null);
  let viewport = $state(null);
  let frameEl = $state(null);
  let scale = $state(1);
  let tx = $state(0);
  let ty = $state(0);
  let frameSize = $state(320);
  let dragging = $state(false);
  let drag = null;

  function center() {
    if (!viewport) return;
    const rect = viewport.getBoundingClientRect();
    const size = Math.max(160, Math.floor(Math.min(rect.width, rect.height) * 0.92));
    frameSize = size;
    scale = 1;
    tx = (rect.width - size) / 2;
    ty = (rect.height - size) / 2;
  }

  function clampPan() {
    if (!viewport || scale <= 1) return;
    const rect = viewport.getBoundingClientRect();
    const size = frameSize * scale;
    const keep = Math.min(size * 0.25, 96);
    tx = Math.min(rect.width - keep, Math.max(keep - size, tx));
    ty = Math.min(rect.height - keep, Math.max(keep - size, ty));
  }

  function zoomAt(px, py, factor) {
    const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
    if (next === scale) return;
    const ratio = next / scale;
    tx = px - ratio * (px - tx);
    ty = py - ratio * (py - ty);
    scale = next;
    if (next === 1) center();
    else clampPan();
  }

  function zoomBy(factor) {
    if (!viewport) return;
    const rect = viewport.getBoundingClientRect();
    zoomAt(rect.width / 2, rect.height / 2, factor);
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    dragging = true;
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event) {
    if (!drag || event.pointerId !== drag.id) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    drag.moved = true;
    drag.x = event.clientX;
    drag.y = event.clientY;
    tx += dx;
    ty += dy;
    clampPan();
  }

  function onPointerUp(event) {
    if (!drag || event.pointerId !== drag.id) return;
    const moved = drag.moved;
    drag = null;
    dragging = false;
    if (moved || !frameEl) return;
    const box = frameEl.getBoundingClientRect();
    const inside =
      event.clientX >= box.left &&
      event.clientX <= box.right &&
      event.clientY >= box.top &&
      event.clientY <= box.bottom;
    if (!inside) onclose();
  }

  $effect(() => {
    if (!open) return;
    function onKey(event) {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        onclose();
        return;
      }
      if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        zoomBy(1.25);
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        zoomBy(1 / 1.25);
      } else if (event.key === '0') {
        event.preventDefault();
        center();
      }
    }
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  });

  $effect(() => {
    if (!open || !dialog || !viewport) return;
    const box = dialog;
    const node = viewport;
    const frame = requestAnimationFrame(() => center());
    box.focus();
    function onWheel(event) {
      event.preventDefault();
      if (event.target !== node && !node.contains(event.target)) return;
      const rect = node.getBoundingClientRect();
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      zoomAt(event.clientX - rect.left, event.clientY - rect.top, factor);
    }
    function onResize() {
      center();
    }
    box.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(frame);
      box.removeEventListener('wheel', onWheel);
      window.removeEventListener('resize', onResize);
    };
  });
</script>

{#if open}
  <div class="lightbox" role="dialog" aria-modal="true" aria-label={label} tabindex="-1" bind:this={dialog}>
    <div class="bar">
      <button type="button" onclick={() => zoomBy(1 / 1.25)} aria-label="Zoom out">−</button>
      <span class="pct">{Math.round(scale * 100)}%</span>
      <button type="button" onclick={() => zoomBy(1.25)} aria-label="Zoom in">+</button>
      <button type="button" onclick={center}>Reset</button>
      <span class="hint">Scroll to zoom, drag to move</span>
      {#if note}<span class="note">{note}</span>{/if}
      <button type="button" class="done" onclick={onclose}>Close</button>
    </div>
    <div
      class="viewport"
      class:dragging
      role="application"
      aria-label="Zoomable image. Scroll to zoom, drag to move."
      tabindex="-1"
      bind:this={viewport}
      onpointerdown={onPointerDown}
      onpointermove={onPointerMove}
      onpointerup={onPointerUp}
      onpointercancel={onPointerUp}
      ondblclick={center}
    >
      <div
        class="stage frame"
        bind:this={frameEl}
        style:width="{frameSize}px"
        style:height="{frameSize}px"
        style:transform="translate({tx}px, {ty}px) scale({scale})"
      >
        {#if children}
          {@render children()}
        {/if}
      </div>
    </div>
  </div>
{/if} 

<style>
  .lightbox {
    position: fixed;
    inset: 0;
    z-index: 40;
    display: flex;
    flex-direction: column;
    background: rgba(18, 32, 36, 0.9);
    color: var(--sheet);
  }

  .bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: #122024;
  }

  .bar button {
    border: 1px solid rgba(244, 248, 249, 0.35);
    background: transparent;
    color: var(--sheet);
    border-radius: 2px;
    padding: 4px 10px;
  }

  .pct {
    min-width: 3.2em;
    font-variant-numeric: tabular-nums;
    font-size: 13px;
  }

  .hint,
  .note {
    color: rgba(244, 248, 249, 0.72);
    font-size: 12px;
  }

  .done { margin-left: auto; }

  .viewport {
    position: relative;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    touch-action: none;
    cursor: grab;
    user-select: none;
  }

  .viewport.dragging { cursor: grabbing; }

  .frame {
    position: absolute;
    left: 0;
    top: 0;
    margin: 0;
    transform-origin: 0 0;
    pointer-events: none;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
  }
</style>
