// Click-to-pick canvas shared by the HW7 pages: draws an image, reports clicks
// in ORIGINAL image pixels however much CSS scales the canvas, and shows a
// magnifier that follows the cursor (same behaviour as the HW5 SfM page).
const hw7Loupe = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 180;
  c.style.cssText = 'position:fixed;width:180px;height:180px;border:2px solid #1c2b36;border-radius:50%;' +
    'pointer-events:none;display:none;z-index:50;box-shadow:0 4px 14px rgba(0,0,0,.35);background:#000;';
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(c));
  return c;
})();

function makePicker(canvas, onPick, drawOverlay) {
  const ZOOM = 4;
  const ctx = canvas.getContext('2d');
  const img = new Image();
  const p = {
    canvas, ctx, img,
    unit() { return canvas.width / (canvas.getBoundingClientRect().width || canvas.width); },
    load(src) { img.src = src; },
    redraw() {
      if (!img.naturalWidth) return;
      ctx.drawImage(img, 0, 0);
      drawOverlay(ctx, p.unit());
    },
  };
  function toImage(e) {
    const r = canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(img.naturalWidth - 1, (e.clientX - r.left) / r.width * img.naturalWidth)),
            Math.max(0, Math.min(img.naturalHeight - 1, (e.clientY - r.top) / r.height * img.naturalHeight))];
  }
  function loupe(e) {
    const [x, y] = toImage(e);
    const l = hw7Loupe.getContext('2d'), w = hw7Loupe.width, src = w / ZOOM;
    l.imageSmoothingEnabled = false;
    l.fillStyle = '#000';
    l.fillRect(0, 0, w, w);
    l.drawImage(canvas, x - src / 2, y - src / 2, src, src, 0, 0, w, w);
    l.strokeStyle = 'rgba(255,255,0,.9)';
    l.beginPath();
    l.moveTo(w / 2, 0); l.lineTo(w / 2, w);
    l.moveTo(0, w / 2); l.lineTo(w, w / 2);
    l.stroke();
    hw7Loupe.style.left = (e.clientX + 24) + 'px';
    hw7Loupe.style.top = (e.clientY - 200 < 0 ? e.clientY + 24 : e.clientY - 200) + 'px';
    hw7Loupe.style.display = 'block';
  }
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    p.redraw();
  };
  canvas.addEventListener('click', e => { if (img.naturalWidth) { onPick(...toImage(e)); loupe(e); } });
  canvas.addEventListener('mousemove', e => { if (img.naturalWidth) loupe(e); });
  canvas.addEventListener('mouseleave', () => { hw7Loupe.style.display = 'none'; });
  window.addEventListener('resize', p.redraw);
  return p;
}

function drawMark(ctx, u, x, y, label, colour) {
  ctx.lineWidth = 1.5 * u;
  ctx.strokeStyle = colour;
  ctx.beginPath();
  ctx.moveTo(x - 9 * u, y); ctx.lineTo(x + 9 * u, y);
  ctx.moveTo(x, y - 9 * u); ctx.lineTo(x, y + 9 * u);
  ctx.stroke();
  ctx.font = `bold ${13 * u}px sans-serif`;
  ctx.lineWidth = 3 * u;
  ctx.strokeStyle = 'black';
  ctx.strokeText(label, x + 8 * u, y - 8 * u);
  ctx.fillStyle = colour;
  ctx.fillText(label, x + 8 * u, y - 8 * u);
}
