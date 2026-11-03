(() => {
  const csrf = document.querySelector('meta[name="csrf-token"]').content;

  async function post(url, body) {
    let r;
    try {
      r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf}, body: JSON.stringify(body)});
    } catch (e) {
      return {ok: false, error: 'Could not reach the server. Check your connection.'};
    }
    let data = {};
    try { data = await r.json(); } catch (e) { /* non-JSON error page */ }
    if (r.ok) return {ok: true, data};
    const fallback = r.status === 503 ? 'Not ready yet, try again in a moment.' : 'Something went wrong.';
    return {ok: false, error: data.error || fallback};
  }

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  document.querySelectorAll('form[data-confirm]').forEach(f =>
    f.addEventListener('submit', ev => { if (!confirm(f.dataset.confirm)) ev.preventDefault(); }));

  const root = document.getElementById('analysis');
  if (!root) return;
  const base = root.dataset.base;

  document.querySelectorAll('input[name=choice]').forEach(radio => radio.addEventListener('change', async () => {
    const status = document.getElementById('selection-status');
    status.textContent = 'Saving...';
    const r = await post(base + '/selection', {label: radio.value});
    status.textContent = r.ok ? 'Saved.' : 'Could not save your selection: ' + r.error;
  }));

  function passageNode(p) {
    const box = el('div', undefined, 'passage');
    const body = el('div');
    body.innerHTML = p.html;  // rendered and sanitised by the server
    const src = el('p', undefined, 'src');
    if (p.url) {
      const a = el('a', p.title || p.source);
      a.href = p.url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      src.append('Source: ', a);
    } else {
      src.append('Source: ' + (p.title || p.source));
    }
    src.append(', page ' + p.page);
    box.append(body, src);
    return box;
  }

  const form = document.getElementById('chat-form');
  if (form) {
    const input = document.getElementById('chat-message');
    const send = document.getElementById('chat-send');
    const log = document.getElementById('chat-log');
    let busy = false;

    async function ask(message, reply) {
      if (busy) return;
      busy = true;
      send.disabled = true;
      reply.replaceChildren(el('p', 'Searching the documents...'));
      const r = await post(base + '/chat', {message});
      busy = false;
      send.disabled = false;
      if (!r.ok) {
        const retry = el('button', 'Try again');
        retry.type = 'button';
        retry.addEventListener('click', () => ask(message, reply));
        reply.replaceChildren(el('p', 'Could not get an answer: ' + r.error, 'error'), retry);
        return;
      }
      const nodes = [];
      if (!r.data.passages.length) nodes.push(el('p', 'No supporting document found.', 'nosrc'));
      nodes.push(el('p', r.data.note, 'note'));
      r.data.passages.forEach(p => nodes.push(passageNode(p)));
      reply.replaceChildren(...nodes);
    }

    form.addEventListener('submit', ev => {
      ev.preventDefault();
      const message = input.value.trim();
      if (busy || !message) return;
      input.value = '';
      const reply = el('div', undefined, 'reply');
      log.append(el('div', message, 'msg-user'), reply);  // user text is always text
      ask(message, reply);
    });
  }

  const reportForm = document.getElementById('report-form');
  const reportStatus = document.getElementById('report-status');
  document.getElementById('report-open').addEventListener('click', () => { reportForm.hidden = false; });
  reportForm.addEventListener('submit', async ev => {
    ev.preventDefault();
    const button = document.getElementById('report-send');
    if (button.disabled) return;
    button.disabled = true;
    reportStatus.textContent = 'Sending...';
    const r = await post(base + '/report', {note: document.getElementById('report-note').value});
    button.disabled = false;
    if (r.ok) {
      reportStatus.textContent = 'Report stored. Thank you.';
      reportForm.hidden = true;
    } else {
      reportStatus.textContent = 'Could not store the report: ' + r.error;
    }
  });
})();
