(() => {
  const csrf = document.querySelector('meta[name="csrf-token"]').content;

  async function post(url, body) {
    let r;
    try {
      r = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'Accept': 'application/json', 'X-CSRF-Token': csrf},
        body: JSON.stringify(body),
      });
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

  // upload page: click or drop to choose a file, preview it, send once
  const dropArea = document.getElementById('dropArea');
  if (dropArea) {
    const fileInput = document.getElementById('file');
    const uploadBtn = document.getElementById('uploadBtn');
    const previewBox = document.getElementById('previewContainer');
    dropArea.addEventListener('click', () => fileInput.click());
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name =>
      dropArea.addEventListener(name, ev => { ev.preventDefault(); ev.stopPropagation(); }));
    ['dragenter', 'dragover'].forEach(name => dropArea.addEventListener(name, () => dropArea.classList.add('drag')));
    ['dragleave', 'drop'].forEach(name => dropArea.addEventListener(name, () => dropArea.classList.remove('drag')));
    dropArea.addEventListener('drop', ev => { fileInput.files = ev.dataTransfer.files; showPreview(); });
    fileInput.addEventListener('change', showPreview);
    function showPreview() {
      const file = fileInput.files[0];
      if (!file) return;
      const img = el('img', undefined, 'preview mx-auto d-block');
      img.alt = 'Preview of the chosen photo';
      img.src = URL.createObjectURL(file);
      previewBox.replaceChildren(img);
      uploadBtn.disabled = false;
    }
    document.getElementById('uploadForm').addEventListener('submit', ev => {
      if (uploadBtn.disabled) { ev.preventDefault(); return; }
      uploadBtn.disabled = true;  // a second click cannot send the photo twice
      uploadBtn.textContent = 'Analysing...';
    });
  }

  // camera page: capture a frame and send it through the normal upload route
  const cam = document.getElementById('camera-root');
  if (cam) {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const photo = document.getElementById('photo');
    const message = document.getElementById('camera-message');
    const capture = document.getElementById('captureBtn');
    const retake = document.getElementById('retakeBtn');
    const analyze = document.getElementById('analyzeBtn');
    const flip = document.getElementById('switchCamera');
    let stream = null;
    let facing = 'environment';

    function say(text) { message.textContent = text; message.hidden = !text; }
    function showLive(live) {
      video.hidden = !live;
      photo.hidden = live;
      capture.hidden = !live;
      retake.hidden = live;
      analyze.hidden = live;
    }

    async function start() {
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        say('This browser or connection cannot use a camera. Upload a photo instead.');
        capture.disabled = flip.disabled = true;
        return;
      }
      say('Loading camera...');
      try {
        stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: facing}, audio: false});
      } catch (e) {
        say('Camera access was denied or no camera is available. Upload a photo instead.');
        capture.disabled = true;
        return;
      }
      video.srcObject = stream;
      say('');
      capture.disabled = false;
      showLive(true);
    }

    flip.addEventListener('click', () => { facing = facing === 'user' ? 'environment' : 'user'; start(); });
    capture.addEventListener('click', () => {
      if (!video.videoWidth) { say('The camera is not ready yet.'); return; }
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      photo.src = canvas.toDataURL('image/jpeg', 0.9);
      showLive(false);
    });
    retake.addEventListener('click', () => showLive(true));

    analyze.addEventListener('click', () => {
      if (analyze.disabled) return;
      analyze.disabled = true;
      say('Analysing...');
      canvas.toBlob(async blob => {
        const form = new FormData();
        form.append('file', blob, 'camera-capture.jpg');
        let r;
        try {
          r = await fetch(cam.dataset.upload, {method: 'POST', headers: {'Accept': 'application/json', 'X-CSRF-Token': csrf}, body: form});
        } catch (e) {
          say('Could not reach the server. Try again.');
          analyze.disabled = false;
          return;
        }
        if (r.redirected) { window.location.href = r.url; return; }
        let data = {};
        try { data = await r.json(); } catch (e) { /* not JSON */ }
        say(data.error || 'The photo could not be analysed. Try again.');
        analyze.disabled = false;
      }, 'image/jpeg', 0.9);
    });
    start();
  }

  // result page
  const root = document.getElementById('analysis');
  if (!root) return;
  const base = root.dataset.base;

  document.querySelectorAll('input[name=choice]').forEach(radio => radio.addEventListener('change', async () => {
    const status = document.getElementById('selection-status');
    status.textContent = 'Saving...';
    const r = await post(base + '/selection', {label: radio.value});
    status.textContent = r.ok ? 'Saved.' : 'Could not save your selection: ' + r.error;
    // the healthy caution is drawn by the server, so a change between healthy and not reloads the page
    if (r.ok && String(radio.value.toLowerCase().includes('healthy') ? 1 : 0) !== root.dataset.healthy) window.location.reload();
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
      reply.className = 'reply';
      reply.replaceChildren(el('p', 'Searching the documents and writing an answer...'));
      const r = await post(base + '/chat', {message});
      busy = false;
      send.disabled = false;
      if (!r.ok) {
        const retry = el('button', 'Try again', 'btn btn-sm btn-outline-secondary');
        retry.type = 'button';
        retry.addEventListener('click', () => ask(message, reply));
        reply.replaceChildren(el('p', 'Could not get an answer: ' + r.error, 'error'), retry);
        return;
      }
      const nodes = [];
      if (r.data.text === undefined) {  // a server with no writer: the cited passages only
        if (!r.data.passages.length) nodes.push(el('p', 'No supporting document found.', 'nosrc'));
        nodes.push(el('p', r.data.note, 'note'));
        r.data.passages.forEach(p => nodes.push(passageNode(p)));
      } else {
        const state = r.data.state;
        reply.className = 'reply reply-' + state;
        const text = el('div', undefined, 'answer-text');
        text.innerHTML = r.data.html;  // rendered and sanitised by the server
        nodes.push(text);
        if (state === 'clarify') nodes.push(el('p', 'Answer in your next message.', 'note'));
        if (r.data.note) nodes.push(el('p', r.data.note, 'note'));
        r.data.passages.forEach(p => nodes.push(passageNode(p)));
        if (state === 'unavailable' || state === 'retrieval_failed') {
          const retry = el('button', 'Try again', 'btn btn-sm btn-outline-secondary');
          retry.type = 'button';
          retry.addEventListener('click', () => ask(message, reply));
          nodes.push(retry);
        }
      }
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
  const reportModal = document.getElementById('report-modal');
  reportModal.addEventListener('show.bs.modal', () => { reportForm.hidden = false; reportStatus.textContent = ''; });
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
      reportForm.reset();
    } else {
      reportStatus.textContent = 'Could not store the report: ' + r.error;
    }
  });
})();
