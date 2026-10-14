<script>
  import { copyDisplayed } from './copyImage.js';

  let { root = null, overlay = false } = $props();

  let label = $state('Copy');
  let timer = 0;

  async function onCopy(event) {
    event.preventDefault();
    event.stopPropagation();
    const node = root ?? event.currentTarget.closest('.shot')?.querySelector('.stage');
    try {
      await copyDisplayed(node);
      label = 'Copied';
    } catch {
      label = 'Failed';
    }
    clearTimeout(timer);
    timer = setTimeout(() => {
      label = 'Copy';
    }, 1400);
  }
</script>

<button
  type="button"
  class="copy-shot"
  class:overlay
  aria-live="polite"
  onclick={onCopy}
>
  {label}
</button>

<style>
  .copy-shot {
    border: 1px solid rgba(244, 248, 249, 0.35);
    background: transparent;
    color: var(--sheet);
    border-radius: 2px;
    padding: 4px 10px;
    line-height: 1.2;
  }

  .copy-shot.overlay {
    position: absolute;
    top: 8px;
    right: 8px;
    z-index: 6;
    border-color: rgba(244, 248, 249, 0.45);
    background: rgba(18, 32, 36, 0.78);
    padding: 3px 8px;
    font-size: 12px;
  }
</style>
