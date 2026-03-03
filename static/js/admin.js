/* ═══════════════════════════════════════════════════════════════════════════
   PolicyBot Admin JS — v6  (Dashboard + Policy Knowledge Base + all sections)
   ═══════════════════════════════════════════════════════════════════════════ */

let charts = {}, userPage = 0;
const PAGE_SIZE = 20;

/* ── cached KB data for client-side filtering ─────────────────────────── */
let _kbAllPlans = [];

/* ── Boot ─────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const p = new URLSearchParams(location.search);
  showSection(p.get('section') || 'dashboard');
  loadAll();
  setInterval(loadAll, 30000);
});

/* ── Section router ───────────────────────────────────────────────────── */
function showSection(name) {
  document.querySelectorAll('.admin-section').forEach(s => s.classList.add('hidden'));
  document.querySelectorAll('.nav-link').forEach(l =>
    l.classList.toggle('active', l.dataset.section === name));
  const s = document.getElementById(`sec-${name}`);
  if (s) s.classList.remove('hidden');
  const loaders = {
    dashboard: loadAnalytics,
    kb:        loadKB,
    users:     loadUsers,
    chats:     loadChats,
    leads:     loadLeads,
    ratings:   loadRatings,
    documents: loadDocs,
    fraud:     loadFraud,
    apikeys:   loadApiKeys,
  };
  if (loaders[name]) loaders[name]();
}

function loadAll() { loadAnalytics(); }

/* ════════════════════════════════════════════════════════════════════════
   DASHBOARD / ANALYTICS
   ════════════════════════════════════════════════════════════════════════ */
async function loadAnalytics() {
  try {
    const r = await fetch('/api/admin/analytics');
    const d = await r.json();
    const a = d.analytics || {};
    set('s-users',      a.total_users ?? '—');
    set('s-verified',   `${a.verified_users ?? 0} (${a.verified_pct ?? 0}%)`);
    set('s-leads',      a.total_leads ?? '—');
    set('s-chats',      a.total_chats ?? '—');
    set('s-rating',     a.avg_rating ? `${a.avg_rating} ⭐` : '—');
    set('s-escalations',a.total_escalations ?? '—');
    set('s-conversion', `${a.conversion_rate ?? 0}%`);
    set('s-diabetes',   a.diabetes_users ?? '—');
    const C = ['#5a72ff','#9d4edd','#06b6d4','#f59e0b','#10b981','#ef4444'];
    mkChart('ch-ins',   'doughnut',
      (a.popular_insurance_types||[]).map(d => d.insurance_type||'?'),
      (a.popular_insurance_types||[]).map(d => d.c), C);
    mkChart('ch-plans', 'bar',
      (a.popular_plans||[]).map(d => (d.plan_name||'?').slice(0,16)),
      (a.popular_plans||[]).map(d => d.c), C, 'Plans');
    mkChart('ch-lang',  'pie',
      (a.language_stats||[]).map(d => d.language||'?'),
      (a.language_stats||[]).map(d => d.c), C);
    mkChart('ch-rating','bar',
      (a.rating_distribution||[]).map(d => `${d.score}★`),
      (a.rating_distribution||[]).map(d => d.c), C, 'Count');
  } catch(e) { console.error('Analytics', e); }
}

function mkChart(id, type, labels, data, colors, label = '') {
  const canvas = document.getElementById(id); if (!canvas) return;
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(canvas.getContext('2d'), {
    type,
    data: { labels, datasets: [{ label, data, backgroundColor: colors, borderWidth: 0, borderRadius: 6 }] },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { labels: { color: '#7c8db5', font: { size: 11 }, boxWidth: 12 } } },
      scales: type === 'bar'
        ? { x: { ticks: { color: '#7c8db5' }, grid: { color: 'rgba(255,255,255,.04)' } },
            y: { ticks: { color: '#7c8db5' }, grid: { color: 'rgba(255,255,255,.04)' } } }
        : undefined,
    },
  });
}

/* ════════════════════════════════════════════════════════════════════════
   POLICY KNOWLEDGE BASE
   ════════════════════════════════════════════════════════════════════════ */

/* ── Tab switcher ─────────────────────────────────────────────────────── */
function kbTab(name) {
  ['docs', 'plans', 'analytics'].forEach(t => {
    const el = document.getElementById(`kb-tab-${t}`);
    if (el) el.classList.toggle('hidden', t !== name);
  });
  document.querySelectorAll('.kb-tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  if (name === 'plans')     { _renderKBPlans(); }
  if (name === 'analytics') { loadKBAnalytics(); }
}

/* ── Main KB loader ───────────────────────────────────────────────────── */
async function loadKB() {
  try {
    /* load documents */
    const r  = await fetch('/api/admin/kb/documents');
    const d  = await r.json();
    const docs = d.documents || [];
    renderKBDocs(docs);
    set('kb-total-docs', docs.length);

    /* load analytics for stats */
    const ra = await fetch('/api/admin/kb/analytics');
    const da = await ra.json();
    const an = da.analytics || {};
    set('kb-active-plans', an.active_plans ?? '—');
    set('kb-missing',      (an.missing_info || []).length || 0);
    const top = (an.top_recommended || [])[0];
    set('kb-top-plan', top ? (top.plan_name || '—').slice(0, 22) : '—');

    /* pre-load plans for filtering */
    const rp = await fetch('/api/admin/kb/documents/0').catch(() => null);
    /* use dedicated plans endpoint instead */
    await _fetchAllPlans();
  } catch(e) { console.error('KB load', e); }
}

async function _fetchAllPlans() {
  try {
    /* We'll get plans from the first doc listing which returns all */
    const r = await fetch('/api/admin/kb/documents');
    const d = await r.json();
    const docs = d.documents || [];
    let plans = [];
    /* batch-fetch plans per doc */
    for (const doc of docs) {
      const rd = await fetch(`/api/admin/kb/documents/${doc.id}`);
      const dd = await rd.json();
      (dd.plans || []).forEach(p => { p._docname = doc.filename; plans.push(p); });
    }
    _kbAllPlans = plans;
  } catch(e) { console.error('KB plans fetch', e); }
}

/* ── Render documents table ───────────────────────────────────────────── */
function renderKBDocs(docs) {
  const tbody = document.getElementById('kb-docs-tbody');
  if (!tbody) return;
  if (!docs.length) {
    tbody.innerHTML = emptyRow(8, '📂 No policy documents uploaded yet. Click "Upload Policy Document" to get started.');
    return;
  }
  tbody.innerHTML = docs.map((doc, i) => {
    const status = doc.status || 'unknown';
    const statusClass = status === 'active' ? 'kb-status-active'
                      : status === 'processing' ? 'kb-status-processing'
                      : status.startsWith('failed') ? 'kb-status-failed' : 'kb-status-pending';
    const isMaster = doc.filename === 'master_policy.docx' || doc.uploaded_by === 'system';
    return `<tr>
      <td><span style="color:var(--tx3);font-size:11px">#${doc.id}</span></td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <span class="kb-file-icon">${fileIcon(doc.filename)}</span>
          <div>
            <div style="font-weight:600;color:var(--tx);font-size:12px">${esc(doc.filename)}</div>
            <div style="font-size:10px;color:var(--tx3)">${fmtBytes(doc.file_size||0)}</div>
          </div>
        </div>
      </td>
      <td><span class="kb-status ${statusClass}">${statusLabel(status)}</span>
          ${isMaster ? '<span class="kb-master-badge">MASTER</span>' : ''}</td>
      <td><span class="kb-plan-count">${doc.plan_count ?? 0}</span></td>
      <td><span style="font-size:11px;color:var(--tx3)">v${doc.version || 1}</span></td>
      <td style="font-size:11px">${doc.uploaded_by || 'admin'}</td>
      <td style="font-size:11px">${(doc.uploaded_at || '').slice(0, 10)}</td>
      <td>
        <div style="display:flex;gap:5px;align-items:center">
          <button class="kb-action-btn kb-btn-view"   onclick="kbViewDoc(${doc.id})"    title="View plans"><i class="fas fa-eye"></i></button>
          <button class="kb-action-btn" style="background:rgba(99,179,237,0.15);color:#63b3ed" onclick="kbDownloadDoc(${doc.id},'${esc(doc.filename)}')" title="Download original file"><i class="fas fa-download"></i></button>
          ${!isMaster ? `<button class="kb-action-btn" style="background:rgba(251,191,36,0.15);color:#f6ad55" onclick="kbReextractDoc(${doc.id})" title="Re-extract plans from file"><i class="fas fa-wand-magic-sparkles"></i></button>` : ''}
          ${!isMaster ? `<button class="kb-action-btn kb-btn-update" onclick="kbOpenUpdate(${doc.id})" title="Update / re-upload"><i class="fas fa-rotate"></i></button>` : ''}
          ${!isMaster ? `<button class="kb-action-btn kb-btn-delete" onclick="kbDeleteDoc(${doc.id},'${esc(doc.filename)}')" title="Delete"><i class="fas fa-trash"></i></button>` : ''}
        </div>
      </td>
    </tr>`;
  }).join('');
}

/* ── Render plans table (with client-side filter) ─────────────────────── */
function _renderKBPlans() {
  const q    = (document.getElementById('kb-plan-search')?.value || '').toLowerCase();
  const type = document.getElementById('kb-type-filter')?.value || '';
  const plans = _kbAllPlans.filter(p => {
    const matchQ = !q || (p.plan_name||'').toLowerCase().includes(q)
                       || (p.company_name||'').toLowerCase().includes(q)
                       || (p.insurance_type||'').toLowerCase().includes(q);
    const matchT = !type || (p.insurance_type||'') === type;
    return matchQ && matchT;
  });
  const tbody = document.getElementById('kb-plans-tbody');
  if (!tbody) return;
  if (!plans.length) {
    tbody.innerHTML = emptyRow(10, 'No plans match your filter.');
    return;
  }
  tbody.innerHTML = plans.map(p => {
    const isMaster = p.is_master;
    const active   = p.active;
    return `<tr>
      <td><strong style="color:var(--tx)">${esc(p.company_name||'—')}</strong></td>
      <td>
        <span class="kb-plan-name" onclick="kbViewPlan(${p.id})" title="View full details">${esc(p.plan_name||'—')}</span>
        ${isMaster ? '<span class="kb-master-badge" style="margin-left:4px">MASTER</span>' : ''}
      </td>
      <td><span class="kb-type-chip">${typeIcon(p.insurance_type)} ${esc(p.insurance_type||'—')}</span></td>
      <td style="font-size:11px">${esc(p.coverage_amount||'—')}</td>
      <td style="font-size:11px">${esc(p.premium_range||'—')}</td>
      <td style="font-size:11px">${esc(p.waiting_period||'—')}</td>
      <td><span class="kb-rec-badge">${p.recommend_count||0}</span></td>
      <td style="font-size:10px;color:var(--tx3)">${esc((p._docname||p.filename||'master').slice(0,20))}</td>
      <td><span class="kb-status ${active ? 'kb-status-active' : 'kb-status-failed'}">${active ? '● Active' : '○ Off'}</span></td>
      <td>
        <div style="display:flex;gap:5px">
          <button class="kb-action-btn kb-btn-view"   onclick="kbViewPlan(${p.id})"   title="View details"><i class="fas fa-eye"></i></button>
          <button class="kb-action-btn ${active ? 'kb-btn-toggle-on' : 'kb-btn-toggle-off'}" onclick="kbTogglePlan(${p.id})" title="${active ? 'Deactivate' : 'Activate'}">
            <i class="fas fa-toggle-${active ? 'on' : 'off'}"></i></button>
          ${!isMaster ? `<button class="kb-action-btn kb-btn-delete" onclick="kbDeletePlan(${p.id},'${esc(p.plan_name||'')}')"><i class="fas fa-trash"></i></button>` : ''}
        </div>
      </td>
    </tr>`;
  }).join('');
}

function kbFilterPlans() { _renderKBPlans(); }

/* ── View document → show plans modal ────────────────────────────────── */
async function kbViewDoc(docId) {
  try {
    const r = await fetch(`/api/admin/kb/documents/${docId}`);
    const d = await r.json();
    const doc   = d.document || {};
    const plans = d.plans || [];
    const vers  = d.versions || [];

    let html = `
      <div class="kb-modal-doc-info">
        <div class="kb-modal-info-row"><span>File</span><strong>${esc(doc.filename||'—')}</strong></div>
        <div class="kb-modal-info-row"><span>Status</span><span class="kb-status ${doc.status==='active'?'kb-status-active':'kb-status-processing'}">${statusLabel(doc.status)}</span></div>
        <div class="kb-modal-info-row"><span>Version</span><strong>v${doc.version||1}</strong></div>
        <div class="kb-modal-info-row"><span>Uploaded</span><strong>${(doc.uploaded_at||'').slice(0,16).replace('T',' ')}</strong></div>
        <div class="kb-modal-info-row"><span>By</span><strong>${doc.uploaded_by||'admin'}</strong></div>
        <div class="kb-modal-info-row"><span>Plans extracted</span><strong>${plans.length}</strong></div>
      </div>`;

    if (plans.length) {
      html += `<div class="kb-modal-section-title">Extracted Plans (${plans.length})</div>`;
      plans.forEach(p => {
        html += planCard(p);
      });
    } else {
      html += `<div class="kb-empty-state"><i class="fas fa-circle-info"></i> No plans extracted from this document yet.</div>`;
    }

    if (vers.length) {
      html += `<div class="kb-modal-section-title">Version History</div>
        <div class="kb-version-list">`;
      vers.forEach(v => {
        html += `<div class="kb-version-row">
          <span class="kb-ver-num">v${v.version}</span>
          <span class="kb-ver-file">${esc(v.filename)}</span>
          <span class="kb-ver-date">${(v.changed_at||'').slice(0,10)}</span>
          <span class="kb-ver-note">${esc(v.change_note||'')}</span>
        </div>`;
      });
      html += `</div>`;
    }

    document.getElementById('kb-modal-title').textContent = `📄 ${doc.filename}`;
    document.getElementById('kb-modal-body').innerHTML = html;
    document.getElementById('kb-modal').classList.remove('hidden');
  } catch(e) { console.error('View doc', e); }
}

/* ── View single plan modal ───────────────────────────────────────────── */
async function kbViewPlan(planId) {
  const plan = _kbAllPlans.find(p => p.id === planId);
  if (!plan) { alert('Plan not found in cache — please refresh.'); return; }
  document.getElementById('kb-modal-title').textContent =
    `📋 ${plan.plan_name} — ${plan.company_name}`;
  document.getElementById('kb-modal-body').innerHTML = planCard(plan, true);
  document.getElementById('kb-modal').classList.remove('hidden');
}

function planCard(p, full = false) {
  const fields = [
    ['Insurance Type',      p.insurance_type],
    ['Coverage Amount',     p.coverage_amount],
    ['Premium Range',       p.premium_range],
    ['Waiting Period',      p.waiting_period],
    ['Eligibility Age',     p.eligibility_age],
    ['Network Hospitals',   p.network_hospitals],
    ['Conditions Covered',  p.conditions_covered],
    ['Exclusions',          p.exclusions],
    ['Claim Process',       p.claim_process],
    ['Special Benefits',    p.special_benefits],
  ];
  const rows = fields.filter(([, v]) => v && v !== 'Not specified')
    .map(([k, v]) => `<div class="kb-plan-field"><span class="kb-field-label">${k}</span><span class="kb-field-val">${esc(v)}</span></div>`)
    .join('');
  const summary = p.raw_summary ? `<div class="kb-plan-summary">${esc(p.raw_summary)}</div>` : '';
  return `
    <div class="kb-plan-card ${p.is_master ? 'kb-plan-master' : ''}">
      <div class="kb-plan-card-header">
        <div>
          <div class="kb-plan-card-name">${esc(p.plan_name||'—')}</div>
          <div class="kb-plan-card-company">${esc(p.company_name||'—')}</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
          <span class="kb-type-chip">${typeIcon(p.insurance_type)} ${esc(p.insurance_type||'—')}</span>
          ${p.is_master ? '<span class="kb-master-badge">MASTER</span>' : ''}
          <span class="kb-rec-badge">🏆 ${p.recommend_count||0} recs</span>
        </div>
      </div>
      ${summary}
      <div class="kb-plan-fields">${rows}</div>
    </div>`;
}

/* ── Upload pipeline ──────────────────────────────────────────────────── */
async function kbUpload(input) {
  const file = input.files[0];
  if (!file) return;
  input.value = '';

  const bar    = document.getElementById('kb-upload-bar');
  const step   = document.getElementById('kb-upload-step');
  const prog   = document.getElementById('kb-upload-progress');
  const pct    = document.getElementById('kb-upload-pct');
  const result = document.getElementById('kb-result');

  result.className = 'upload-result hidden';
  bar.classList.remove('hidden');
  bar.style.display = 'flex';

  const steps = [
    [10,  '🔒 Security scan…'],
    [25,  '📤 Uploading file…'],
    [50,  '📖 Extracting text…'],
    [75,  '🤖 AI extracting policy data…'],
    [90,  '💾 Storing in knowledge base…'],
  ];

  let si = 0;
  const ticker = setInterval(() => {
    if (si < steps.length) {
      const [p, s] = steps[si++];
      prog.style.width = p + '%';
      pct.textContent  = p + '%';
      step.textContent = s;
    }
  }, 800);

  const form = new FormData();
  form.append('file', file);

  try {
    const r = await fetch('/api/admin/kb/upload', { method: 'POST', body: form });
    const d = await r.json();
    clearInterval(ticker);

    if (d.success) {
      prog.style.width = '100%'; pct.textContent = '100%';
      step.textContent = '✅ Done!';
      setTimeout(() => { bar.classList.add('hidden'); }, 600);
      result.textContent = d.message || '✅ Document processed successfully!';
      result.className   = 'upload-result success';
      loadKB();
    } else {
      clearInterval(ticker);
      bar.classList.add('hidden');
      result.textContent = d.message || d.error || '❌ Processing failed.';
      result.className   = 'upload-result error';
    }
  } catch(e) {
    clearInterval(ticker);
    bar.classList.add('hidden');
    result.textContent = '❌ Upload failed — check your connection.';
    result.className   = 'upload-result error';
    console.error('KB upload', e);
  }

  result.classList.remove('hidden');
  setTimeout(() => result.classList.add('hidden'), 8000);
}

/* ── Delete document ──────────────────────────────────────────────────── */
async function kbDeleteDoc(docId, filename) {
  if (!confirm(`Delete "${filename}" and all its extracted plans?\n\nThis cannot be undone.`)) return;
  try {
    const r = await fetch(`/api/admin/kb/documents/${docId}`, { method: 'DELETE' });
    const d = await r.json();
    kbToast(d.message || '🗑️ Document deleted.', 'success');
    loadKB();
  } catch(e) { kbToast('❌ Delete failed.', 'error'); }
}

/* ── Download original document ───────────────────────────────────────── */
function kbDownloadDoc(docId, filename) {
  const a = document.createElement('a');
  a.href = `/api/admin/kb/documents/${docId}/download`;
  a.download = filename || `policy_doc_${docId}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  kbToast('📥 Downloading…', 'success');
}

/* ── Re-extract plans from existing document ──────────────────────────── */
async function kbReextractDoc(docId) {
  if (!confirm('Re-run AI extraction on this document?\n\nThis will replace existing extracted plans.')) return;
  kbToast('🔄 Re-extracting plans… this may take a moment.', 'success');
  try {
    const r = await fetch(`/api/admin/kb/documents/${docId}/reextract`, { method: 'POST' });
    const d = await r.json();
    if (d.success) {
      kbToast(d.message || '✅ Plans re-extracted!', 'success');
      loadKB();
    } else {
      kbToast(d.message || d.error || '❌ Re-extraction failed.', 'error');
    }
  } catch(e) { kbToast('❌ Re-extraction error.', 'error'); }
}

/* ── Toggle plan active/inactive ──────────────────────────────────────── */
async function kbTogglePlan(planId) {
  try {
    await fetch(`/api/admin/kb/plans/${planId}/toggle`, { method: 'POST' });
    const plan = _kbAllPlans.find(p => p.id === planId);
    if (plan) plan.active = plan.active ? 0 : 1;
    _renderKBPlans();
  } catch(e) { kbToast('❌ Toggle failed.', 'error'); }
}

/* ── Delete plan ──────────────────────────────────────────────────────── */
async function kbDeletePlan(planId, planName) {
  if (!confirm(`Delete plan "${planName}"?`)) return;
  try {
    await fetch(`/api/admin/kb/plans/${planId}`, { method: 'DELETE' });
    _kbAllPlans = _kbAllPlans.filter(p => p.id !== planId);
    _renderKBPlans();
    kbToast('🗑️ Plan deleted.', 'success');
  } catch(e) { kbToast('❌ Delete failed.', 'error'); }
}

/* ── Update modal ─────────────────────────────────────────────────────── */
function kbOpenUpdate(docId) {
  document.getElementById('update-doc-id').value = docId;
  document.getElementById('update-file-name').textContent = '';
  document.getElementById('update-note').value = '';
  document.getElementById('update-result').className = 'hidden upload-result';
  document.getElementById('update-file-input').value = '';
  document.getElementById('kb-update-modal').classList.remove('hidden');
}
function closeUpdateModal() {
  document.getElementById('kb-update-modal').classList.add('hidden');
}
function showUpdateFile(input) {
  const f = input.files[0];
  document.getElementById('update-file-name').textContent = f ? `📄 ${f.name}` : '';
}
async function submitUpdate() {
  const docId = document.getElementById('update-doc-id').value;
  const file  = document.getElementById('update-file-input').files[0];
  const note  = document.getElementById('update-note').value;
  const res   = document.getElementById('update-result');
  if (!file) { res.textContent = '⚠️ Please choose a file first.'; res.className = 'upload-result error'; return; }
  res.textContent = '⏳ Processing update…'; res.className = 'upload-result'; res.classList.remove('hidden');
  const form = new FormData();
  form.append('file', file);
  form.append('note', note);
  try {
    const r = await fetch(`/api/admin/kb/documents/${docId}/update`, { method: 'POST', body: form });
    const d = await r.json();
    res.textContent = d.message || (d.success ? '✅ Updated!' : '❌ Failed');
    res.className   = `upload-result ${d.success ? 'success' : 'error'}`;
    if (d.success) { setTimeout(closeUpdateModal, 1500); loadKB(); }
  } catch(e) { res.textContent = '❌ Update failed.'; res.className = 'upload-result error'; }
}

/* ── KB Analytics ─────────────────────────────────────────────────────── */
async function loadKBAnalytics() {
  try {
    const r  = await fetch('/api/admin/kb/analytics');
    const d  = await r.json();
    const an = d.analytics || {};

    renderKBList('kb-top-rec-list', an.top_recommended || [], item =>
      `<div class="kb-analy-row"><span class="kb-analy-name">${esc(item.plan_name||'—')}</span><span class="kb-analy-count">${item.rc||0} recs</span></div>`);

    renderKBList('kb-missing-list', an.missing_info || [], item =>
      `<div class="kb-analy-row kb-analy-warn"><i class="fas fa-triangle-exclamation"></i><span>${esc(item.company_name||'')} — ${esc(item.plan_name||'—')}</span></div>`);

    renderKBList('kb-failed-list', an.failed_searches || [], item =>
      `<div class="kb-analy-row kb-analy-fail"><i class="fas fa-magnifying-glass"></i><span class="kb-analy-name">${esc(item.detail||'—')}</span><span class="kb-analy-count">${item.c}×</span></div>`);

    renderKBList('kb-viewed-list', an.top_viewed || [], item =>
      `<div class="kb-analy-row"><span class="kb-analy-name">${esc(item.plan_name||'—')}</span><span class="kb-analy-count">${item.view_count} views</span></div>`);
  } catch(e) { console.error('KB analytics', e); }
}

function renderKBList(id, items, fn) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = items.length
    ? items.map(fn).join('')
    : `<div style="color:var(--tx3);font-size:12px;padding:8px 0">No data yet</div>`;
}

/* ── Modals ───────────────────────────────────────────────────────────── */
function closeKBModal() {
  document.getElementById('kb-modal').classList.add('hidden');
}

/* ── Toast helper ─────────────────────────────────────────────────────── */
function kbToast(msg, type = 'success') {
  const el = document.getElementById('kb-result');
  if (!el) return;
  el.textContent = msg;
  el.className   = `upload-result ${type}`;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 5000);
}

/* ════════════════════════════════════════════════════════════════════════
   USERS
   ════════════════════════════════════════════════════════════════════════ */
async function loadUsers(page = 0) {
  userPage = page;
  const q = document.getElementById('user-search')?.value || '';
  try {
    const r = await fetch(`/api/admin/users?q=${encodeURIComponent(q)}&limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`);
    const d = await r.json();
    const tbody = document.getElementById('users-tbody'); if (!tbody) return;
    const total = document.getElementById('user-count'); if (total) total.textContent = `(${d.total || 0})`;
    tbody.innerHTML = '';
    (d.users || []).forEach(u => {
      const v = u.gov_id_verified;
      tbody.innerHTML += `<tr>
        <td><strong>${u.name || '—'}</strong></td><td>${u.age || '—'}</td><td>${u.city || '—'}</td>
        <td>${u.insurance_type || '—'}</td><td>${u.family_members || '—'}</td>
        <td>${u.medical_conditions || '—'}</td><td>${u.budget_range || '—'}</td>
        <td>${v ? '<span class="verif-badge verified">Verified ✅</span>' : '<span class="verif-badge pending">Pending</span>'}</td>
        <td>${(u.created_at || '').slice(0, 10)}</td>
      </tr>`;
    });
    if (!d.users?.length) tbody.innerHTML = emptyRow(9);
    mkPages('user-pagination', d.total || 0, page, p => loadUsers(p));
  } catch(e) { console.error('Users', e); }
}
function searchUsers() { loadUsers(0); }

/* ════════════════════════════════════════════════════════════════════════
   CHATS
   ════════════════════════════════════════════════════════════════════════ */
async function loadChats() {
  const q = document.getElementById('chat-search')?.value || '';
  try {
    const r = await fetch(`/api/admin/chats?q=${encodeURIComponent(q)}&limit=50`);
    const d = await r.json();
    const tbody = document.getElementById('chats-tbody'); if (!tbody) return;
    tbody.innerHTML = '';
    (d.chats || []).forEach(c => {
      tbody.innerHTML += `<tr>
        <td title="${c.user_id || ''}">${(c.user_id || '').slice(0, 14)}…</td>
        <td title="${c.message || ''}">${(c.message || '').slice(0, 50)}${(c.message || '').length > 50 ? '…' : ''}</td>
        <td title="${c.bot_reply || ''}">${(c.bot_reply || '').slice(0, 50)}${(c.bot_reply || '').length > 50 ? '…' : ''}</td>
        <td>${c.module || 'general'}</td><td>${c.language || 'English'}</td>
        <td>${(c.timestamp || '').slice(0, 16).replace('T', ' ')}</td>
      </tr>`;
    });
    if (!d.chats?.length) tbody.innerHTML = emptyRow(6);
  } catch(e) { console.error('Chats', e); }
}
function searchChats() { loadChats(); }

/* ════════════════════════════════════════════════════════════════════════
   LEADS
   ════════════════════════════════════════════════════════════════════════ */
async function loadLeads() {
  try {
    const r = await fetch('/api/admin/leads');
    const d = await r.json();
    const tbody = document.getElementById('leads-tbody'); if (!tbody) return;
    tbody.innerHTML = '';
    (d.leads || []).forEach(l => {
      const ic = { high: '#10b981', medium: '#f59e0b', low: '#94a3b8' }[l.interest_level] || '#94a3b8';
      tbody.innerHTML += `<tr>
        <td>${(l.user_id || '').slice(0, 14)}…</td><td>${l.plan_name || '—'}</td>
        <td><span style="color:${ic};font-weight:600">${l.interest_level || '—'}</span></td>
        <td>${l.lead_status || '—'}</td>
        <td>${(l.timestamp || '').slice(0, 16).replace('T', ' ')}</td>
      </tr>`;
    });
    if (!d.leads?.length) tbody.innerHTML = emptyRow(5);
  } catch(e) { console.error('Leads', e); }
}

/* ════════════════════════════════════════════════════════════════════════
   RATINGS
   ════════════════════════════════════════════════════════════════════════ */
async function loadRatings() {
  try {
    const r = await fetch('/api/admin/ratings');
    const d = await r.json();
    const tbody = document.getElementById('ratings-tbody'); if (!tbody) return;
    tbody.innerHTML = '';
    (d.ratings || []).forEach(r => {
      const filled = '★'.repeat(r.score); const empty = '☆'.repeat(5 - r.score);
      tbody.innerHTML += `<tr>
        <td>${r.name || r.user_id?.slice(0, 12) || '—'}</td>
        <td><span class="stars-display">${filled}${empty}</span></td>
        <td><strong>${r.score}/5</strong></td>
        <td>${r.comment || '—'}</td>
        <td>${(r.timestamp || '').slice(0, 16).replace('T', ' ')}</td>
      </tr>`;
    });
    if (!d.ratings?.length) tbody.innerHTML = emptyRow(5, 'No ratings yet');
  } catch(e) { console.error('Ratings', e); }
}

/* ════════════════════════════════════════════════════════════════════════
   USER DOCUMENTS
   ════════════════════════════════════════════════════════════════════════ */
async function loadDocs() {
  try {
    const r = await fetch('/api/admin/documents');
    const d = await r.json();
    const tbody = document.getElementById('docs-tbody'); if (!tbody) return;
    tbody.innerHTML = '';
    (d.documents || []).forEach(doc => {
      tbody.innerHTML += `<tr>
        <td>${doc.filename || '—'}</td><td>${doc.doc_type || '—'}</td>
        <td>${(doc.user_id || 'admin').slice(0, 12)}</td>
        <td><span class="verif-badge ${doc.active ? 'verified' : 'pending'}">${doc.active ? 'Active' : 'Inactive'}</span></td>
        <td>${(doc.uploaded_at || '').slice(0, 10)}</td>
        <td>
          <button class="page-btn" onclick="toggleDoc(${doc.id})" title="${doc.active ? 'Deactivate' : 'Activate'}">
            <i class="fas fa-toggle-${doc.active ? 'on' : 'off'}"></i>
          </button>
          <button class="page-btn" onclick="deleteDoc(${doc.id})" style="color:#ef4444" title="Delete">
            <i class="fas fa-trash"></i>
          </button>
        </td>
      </tr>`;
    });
    if (!d.documents?.length) tbody.innerHTML = emptyRow(6, 'No user documents uploaded yet.');
  } catch(e) { console.error('Docs', e); }
}
async function toggleDoc(id) { await fetch(`/api/admin/documents/${id}/toggle`, { method: 'POST' }); loadDocs(); }
async function deleteDoc(id) { if (!confirm('Delete this document?')) return; await fetch(`/api/admin/documents/${id}`, { method: 'DELETE' }); loadDocs(); }

/* ════════════════════════════════════════════════════════════════════════
   FRAUD
   ════════════════════════════════════════════════════════════════════════ */
async function loadFraud() {
  try {
    const r = await fetch('/api/admin/fraud-alerts');
    const d = await r.json();
    const tbody = document.getElementById('fraud-tbody'); if (!tbody) return;
    tbody.innerHTML = '';
    (d.alerts || []).forEach(a => {
      const cls = { HIGH: 'risk-high', MEDIUM: 'risk-medium', LOW: 'risk-low' }[a.risk_level] || 'risk-low';
      tbody.innerHTML += `<tr>
        <td>${a.user_name || a.user_id?.slice(0, 12) || '—'}</td>
        <td><span class="risk-badge ${cls}">${a.risk_level}</span></td>
        <td>${(a.flags || []).join(', ') || 'None'}</td>
        <td>${a.recommendation || '—'}</td>
      </tr>`;
    });
    if (!d.alerts?.length) tbody.innerHTML = emptyRow(4, '✅ No fraud alerts detected');
  } catch(e) { console.error('Fraud', e); }
}

/* ════════════════════════════════════════════════════════════════════════
   API KEYS
   ════════════════════════════════════════════════════════════════════════ */
async function loadApiKeys() {
  try {
    const r = await fetch('/api/admin/gemini/status');
    const d = await r.json();
    const grid = document.getElementById('api-keys-grid'); if (!grid) return;
    grid.innerHTML = '';
    const usage = d.usage || [];

    if (!usage.length) {
      grid.innerHTML = `<div style="grid-column:1/-1;padding:24px;text-align:center;color:var(--tx3)">
        <i class="fas fa-triangle-exclamation" style="font-size:28px;margin-bottom:8px"></i>
        <div style="font-weight:600">No Gemini API keys configured!</div>
        <div style="font-size:12px;margin-top:6px">Add GEMINI_API_KEY_1 … GEMINI_API_KEY_4 to your .env file</div>
        <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:var(--a1);font-size:12px;margin-top:8px;display:inline-block">
          Get free keys at Google AI Studio →
        </a>
      </div>`;
      return;
    }

    usage.forEach(k => {
      const isCooling = k.status === 'cooling';
      const isReady   = k.status === 'ready';
      const statusColor = isReady ? '#4ade80' : isCooling ? '#f59e0b' : '#ef4444';
      const statusIcon  = isReady ? 'fa-circle-check' : isCooling ? 'fa-clock' : 'fa-circle-xmark';
      const statusText  = isReady ? 'Ready' : isCooling ? `Cooling ${k.cooldown_remaining}s` : 'Error';
      grid.innerHTML += `<div class="api-key-card" style="border:1px solid ${statusColor}22;position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:${statusColor}"></div>
        <div class="ak-label" style="display:flex;justify-content:space-between;align-items:center">
          <span><i class="fas fa-key"></i> ${k.label}</span>
          <span style="font-size:11px;color:${statusColor};display:flex;align-items:center;gap:4px">
            <i class="fas ${statusIcon}"></i> ${statusText}
          </span>
        </div>
        <div class="ak-masked" style="letter-spacing:2px">${k.masked}</div>
        <div style="display:flex;gap:16px;margin-top:8px">
          <div><div class="ak-count" style="color:var(--a1)">${k.requests}</div><div class="ak-sub">requests</div></div>
          <div><div class="ak-count" style="color:#f87171">${k.errors || 0}</div><div class="ak-sub">errors</div></div>
        </div>
        ${isCooling ? `<div style="margin-top:8px;background:rgba(245,158,11,0.1);border-radius:6px;padding:6px 8px;font-size:11px;color:#f59e0b">
          <i class="fas fa-clock"></i> Auto-retry in ${k.cooldown_remaining}s
        </div>` : ''}
      </div>`;
    });

    // Add test button
    grid.innerHTML += `<div style="grid-column:1/-1;display:flex;gap:10px;align-items:center;padding-top:4px">
      <button onclick="testGeminiKeys()" style="background:var(--a1);color:#0d0f1a;border:none;padding:8px 18px;border-radius:8px;cursor:pointer;font-weight:600;font-size:13px">
        <i class="fas fa-flask"></i> Test All Keys
      </button>
      <span id="gemini-test-result" style="font-size:12px;color:var(--tx2)"></span>
    </div>`;

  } catch(e) { console.error('ApiKeys', e); }
}

async function testGeminiKeys() {
  const res = document.getElementById('gemini-test-result');
  if (res) { res.textContent = '⏳ Testing keys…'; res.style.color = 'var(--tx2)'; }
  try {
    const r = await fetch('/api/admin/gemini/health');
    const d = await r.json();
    const health = d.health || {};
    const statuses = Object.entries(health).map(([k, v]) =>
      `${k}: ${v.status === 'ok' ? '✅' : v.status === 'cooling' ? '⏸' : '❌'}`
    ).join('  ');
    if (res) { res.textContent = statuses || 'Done'; res.style.color = 'var(--tx)'; }
    loadApiKeys(); // refresh cards
  } catch(e) {
    if (res) { res.textContent = '❌ Test failed'; res.style.color = '#ef4444'; }
  }
}

/* ════════════════════════════════════════════════════════════════════════
   SHARED HELPERS
   ════════════════════════════════════════════════════════════════════════ */
function mkPages(id, total, page, onClick) {
  const c = document.getElementById(id); if (!c) return;
  const pages = Math.ceil(total / PAGE_SIZE); c.innerHTML = '';
  for (let i = 0; i < pages; i++) {
    const btn = document.createElement('button');
    btn.className = `page-btn${i === page ? ' active-page' : ''}`;
    btn.textContent = i + 1; btn.onclick = () => onClick(i); c.appendChild(btn);
  }
}

function set(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function emptyRow(cols, msg = 'No data yet') {
  return `<tr><td colspan="${cols}" style="text-align:center;color:var(--tx3);padding:24px">${msg}</td></tr>`;
}
function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmtBytes(b) {
  if (!b) return '—';
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}
function fileIcon(name) {
  const ext = (name || '').split('.').pop().toLowerCase();
  const icons = { pdf: '📕', txt: '📄', docx: '📘', doc: '📘', jpg: '🖼️', jpeg: '🖼️', png: '🖼️' };
  return icons[ext] || '📎';
}
function statusLabel(s) {
  return { active: '● Active', processing: '⏳ Processing', failed_extraction: '❌ Failed', pending: '○ Pending' }[s] || s || '—';
}
function typeIcon(t) {
  const map = {
    'Health Insurance': '🏥', 'Life Insurance': '👪', 'Term Life Insurance': '📋',
    'Vehicle Insurance': '🚗', 'Travel Insurance': '✈️', 'Property Insurance': '🏠',
    'Accident Insurance': '⚡',
  };
  return map[t] || '🛡️';
}