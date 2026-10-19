function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('image failed to load'));
    img.src = src;
  });
}

function draw(img) {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return { canvas, ctx, image: ctx.getImageData(0, 0, canvas.width, canvas.height) };
}

function hexRgb(hex) {
  const h = String(hex).replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

export function paintIsolate(src, hex) {
  return loadImage(src).then((img) => {
    const { canvas, ctx, image } = draw(img);
    const [tr, tg, tb] = hexRgb(hex);
    const data = image.data;
    const limit = 36 * 36;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] === 0) continue;
      const dr = data[i] - tr;
      const dg = data[i + 1] - tg;
      const db = data[i + 2] - tb;
      if (dr * dr + dg * dg + db * db > limit) data[i + 3] = Math.round(data[i + 3] * 0.15);
    }
    ctx.putImageData(image, 0, 0);
    return canvas.toDataURL('image/png');
  });
}

export function paintImportance(src, style) {
  if (style === 'heat') return Promise.resolve(src);
  return loadImage(src).then((img) => {
    const { canvas, ctx, image } = draw(img);
    const data = image.data;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];
      const gray = r === g && g === b;
      const empty = a === 0 && (gray || (r === 0 && g === 0 && b === 0));
      const heat = gray ? r : a;
      if (empty) {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
        continue;
      }
      if (style === 'grayscale') {
        data[i] = heat;
        data[i + 1] = heat;
        data[i + 2] = heat;
        data[i + 3] = 255;
      } else {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 255 - heat;
      }
    }
    ctx.putImageData(image, 0, 0);
    return canvas.toDataURL('image/png');
  });
}
