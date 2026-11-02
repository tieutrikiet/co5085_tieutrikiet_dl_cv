/* CO5085 — render a Jupyter notebook (.ipynb) into [data-notebook] containers.
 * Depends on marked, highlight.js and KaTeX auto-render (loaded with `defer` before this file). */
(function () {
  'use strict';

  var ANSI = /\x1b\[[0-9;]*[A-Za-z]/g;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function joinSrc(src) { return Array.isArray(src) ? src.join('') : (src || ''); }

  // Keep only what a terminal would show after carriage returns (tqdm-style progress bars).
  function cleanStream(text) {
    return text.replace(ANSI, '').split('\n').map(function (line) {
      var parts = line.split('\r');
      return parts[parts.length - 1];
    }).join('\n');
  }

  // Shield math from the markdown parser, then put it back verbatim for KaTeX.
  function renderMarkdown(src, base) {
    var stash = [];
    var guarded = src.replace(/\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g, function (m) {
      stash.push(m);
      return '@@MATH' + (stash.length - 1) + '@@';
    });
    var html = window.marked.parse(guarded);
    html = html.replace(/@@MATH(\d+)@@/g, function (_, i) {
      return stash[+i].replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    });
    var div = el('div', 'nb-md');
    div.innerHTML = html;
    // Resolve relative image paths against the notebook's folder.
    div.querySelectorAll('img').forEach(function (img) {
      var raw = img.getAttribute('src') || '';
      if (!/^(https?:|data:|\/)/.test(raw)) img.src = new URL(raw, base).href;
      img.loading = 'lazy';
    });
    div.querySelectorAll('table').forEach(function (t) {
      var wrap = el('div', 'table-wrap');
      t.parentNode.insertBefore(wrap, t);
      wrap.appendChild(t);
    });
    return div;
  }

  function renderCode(cell) {
    var wrap = el('div', 'nb-code');
    var n = cell.execution_count;
    var src = el('div', 'nb-input');
    src.appendChild(el('span', 'nb-prompt', '[' + (n == null ? ' ' : n) + ']'));
    var pre = el('pre');
    var code = el('code', 'language-python', joinSrc(cell.source));
    pre.appendChild(code);
    src.appendChild(pre);
    wrap.appendChild(src);
    if (window.hljs) window.hljs.highlightElement(code);

    (cell.outputs || []).forEach(function (out) {
      var node = renderOutput(out);
      if (node) wrap.appendChild(node);
    });
    return wrap;
  }

  function renderOutput(out) {
    var box;
    if (out.output_type === 'stream') {
      box = el('pre', 'nb-output nb-stream' + (out.name === 'stderr' ? ' nb-stderr' : ''),
               cleanStream(joinSrc(out.text)));
      return box;
    }
    if (out.output_type === 'error') {
      return el('pre', 'nb-output nb-stderr', (out.traceback || []).join('\n').replace(ANSI, ''));
    }
    var data = out.data || {};
    if (data['image/png'] || data['image/jpeg']) {
      var mime = data['image/png'] ? 'image/png' : 'image/jpeg';
      box = el('figure', 'nb-output nb-image');
      var img = el('img');
      img.src = 'data:' + mime + ';base64,' + joinSrc(data[mime]).replace(/\s/g, '');
      img.alt = 'Notebook output';
      img.loading = 'lazy';
      box.appendChild(img);
      return box;
    }
    if (data['image/svg+xml']) {
      box = el('div', 'nb-output nb-image');
      box.innerHTML = joinSrc(data['image/svg+xml']);
      return box;
    }
    if (data['text/html']) {
      box = el('div', 'nb-output nb-html table-wrap');
      box.innerHTML = joinSrc(data['text/html']);
      return box;
    }
    if (data['text/plain']) {
      return el('pre', 'nb-output nb-stream', joinSrc(data['text/plain']).replace(ANSI, ''));
    }
    return null;
  }

  function render(container) {
    var url = new URL(container.getAttribute('data-notebook'), document.baseURI);
    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (nb) {
        container.textContent = '';
        (nb.cells || []).forEach(function (cell) {
          var src = joinSrc(cell.source);
          if (cell.cell_type === 'markdown') container.appendChild(renderMarkdown(src, url));
          else if (cell.cell_type === 'code') container.appendChild(renderCode(cell));
        });
        if (window.renderMathInElement) {
          window.renderMathInElement(container, {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: false }
            ],
            ignoredClasses: ['nb-code'],
            throwOnError: false
          });
        }
      })
      .catch(function (err) {
        container.textContent = '';
        var p = el('p', 'nb-status');
        p.append('Không tải được notebook (' + err.message + '). ');
        var a = el('a', null, 'Mở notebook.ipynb');
        a.href = url.href;
        p.appendChild(a);
        container.appendChild(p);
      });
  }

  document.querySelectorAll('[data-notebook]').forEach(render);
})();
