function clipFractions(node) {
  const raw = node.style.clipPath;
  if (!raw || raw === 'none') return null;
  const match = raw.match(/inset\(\s*([^)]+)\)/i);
  if (!match) return null;
  const parts = match[1].trim().split(/\s+/).filter(Boolean);
  let top;
  let right;
  let bottom;
  let left;
  if (parts.length === 1) {
    top = right = bottom = left = parts[0];
  } else if (parts.length === 2) {
    top = bottom = parts[0];
    right = left = parts[1];
  } else if (parts.length === 3) {
    top = parts[0];
    right = left = parts[1];
    bottom = parts[2];
  } else {
    [top, right, bottom, left] = parts;
  }
  const boxW = node.clientWidth || 1;
  const boxH = node.clientHeight || 1;
  const frac = (token, size) => {
    if (token.endsWith('%')) return Math.min(1, Math.max(0, parseFloat(token) / 100));
    if (token.endsWith('px')) return Math.min(1, Math.max(0, parseFloat(token) / size));
    const value = parseFloat(token);
    return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
  };
  return {
    top: frac(top, boxH),
    right: frac(right, boxW),
    bottom: frac(bottom, boxH),
    left: frac(left, boxW),
  };
}

function layerOrder(node) {
  const z = Number(getComputedStyle(node).zIndex);
  return Number.isFinite(z) ? z : 0;
}

function containClip(node, width, height) {
  const frac = clipFractions(node);
  if (!frac) return null;
  const boxW = node.clientWidth || 1;
  const boxH = node.clientHeight || 1;
  const scale = Math.min(boxW / width, boxH / height);
  if (!scale) return null;
  const ox = (boxW - width * scale) / 2;
  const oy = (boxH - height * scale) / 2;
  const x = (frac.left * boxW - ox) / scale;
  const y = (frac.top * boxH - oy) / scale;
  const right = (boxW - frac.right * boxW - ox) / scale;
  const bottom = (boxH - frac.bottom * boxH - oy) / scale;
  const left = Math.max(0, x);
  const top = Math.max(0, y);
  return {
    x: left,
    y: top,
    w: Math.max(0, Math.min(width, right) - left),
    h: Math.max(0, Math.min(height, bottom) - top),
  };
}

export async function copyDisplayed(root) {
  if (!root) throw new Error('No image to copy');
  const nodes = [...root.querySelectorAll('img, canvas')].filter((node) => {
    const style = getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    if (Number(style.opacity) === 0) return false;
    if (node.tagName === 'IMG') return node.complete && node.naturalWidth > 0;
    return node.width > 0 && node.height > 0;
  });
  nodes.sort((a, b) => layerOrder(a) - layerOrder(b));
  const base = nodes.find((node) => node.tagName === 'IMG' && !node.classList.contains('mask'))
    ?? nodes.find((node) => node.tagName === 'IMG');
  if (!base) throw new Error('No image to copy');

  const width = base.naturalWidth;
  const height = base.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  for (const node of nodes) {
    const alpha = Number(getComputedStyle(node).opacity);
    ctx.save();
    ctx.globalAlpha = Number.isFinite(alpha) ? alpha : 1;
    const clip = containClip(node, width, height);
    if (clip) {
      ctx.beginPath();
      ctx.rect(clip.x, clip.y, clip.w, clip.h);
      ctx.clip();
    }
    ctx.drawImage(node, 0, 0, width, height);
    ctx.restore();
  }

  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
    throw new Error('Clipboard is unavailable');
  }
  const png = new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Could not encode image'))), 'image/png');
  });
  await navigator.clipboard.write([new ClipboardItem({ 'image/png': png })]);
}
