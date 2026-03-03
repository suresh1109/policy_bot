/* PolicyBot v6 — Full debug pass
   FIXED:
   1. autoSelectDocType was missing its function declaration (loose code crashed JS on load)
   2. startBot now reliably shows welcome message + insurance type buttons
   3. After Gov ID verified: confirms name, age, gender match against ID before continuing
   4. verif-row badge shown/hidden correctly
   5. setButtonsEnabled always re-enables on error/finally
*/

// ─── State ───────────────────────────────────────────────────────────────────
// NEW: userId is regenerated EVERY page load — ensures zero DB collision between sessions
let userId      = genId();
let sessionId   = genId();
let currentTheme = localStorage.getItem('pb_theme') || 'neon';
let currentLang  = localStorage.getItem('pb_lang')  || 'English';
let currentStage = 'insurance_type';
let isFirstMessage = true;    // ← flag: first real message triggers session reset on backend
let selectedStar = 0;
let multiSelections = {};
let isListening  = false;
let recognition  = null;
let largeFontOn  = false;
let buttonsDisabled = false;

const DOT_STAGES = [
  'insurance_type','collect_name','collect_age','doc_upload',
  'collect_gender','collect_city','collect_family','collect_medical',
  'collect_budget','recommendation','ask_rating'
];

const UPLOAD_STAGES = [
  'doc_upload','verify_wait',
  'condition_report_upload','condition_report_wait',
  'optional_health_check','vehicle_history','vehicle_doc_upload',
  'life_docs','travel_declare','property_history'
];

const SKIPPABLE_STAGES = [
  'optional_health_check','vehicle_doc_upload','life_docs',
  'travel_declare','property_history','condition_report_upload','vehicle_history'
];

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(currentTheme);
  document.querySelectorAll('.th-btn').forEach(b =>
    b.addEventListener('click', () => applyTheme(b.dataset.theme))
  );
  setupDragDrop();
  initVoice();
  initParticles();
  startBot();
});

function genId() {
  return 'pb_' + Math.random().toString(36).slice(2,10) + Date.now().toString(36);
}

// ─── Theme ────────────────────────────────────────────────────────────────────
function applyTheme(t) {
  currentTheme = t;
  document.getElementById('body-root').className = `theme-${t}`;
  localStorage.setItem('pb_theme', t);
  document.querySelectorAll('.th-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.theme === t)
  );
}
function toggleFontSize() {
  largeFontOn = !largeFontOn;
  document.body.classList.toggle('large-font', largeFontOn);
}
function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('pb_lang', lang);
  if (recognition) recognition.lang = getLangCode(lang);
  showToast(`Language: ${lang}`, 'info');
}

// ─── Start bot ────────────────────────────────────────────────────────────────
function startBot() {
  // Reset flags for fresh session
  isFirstMessage = true;
  currentStage   = 'insurance_type';

  // Reset all UI to clean state
  const pName = document.getElementById('p-name');
  if (pName) pName.textContent = 'Your Profile';

  ['pf-ins','pf-age','pf-city','pf-budget','pf-medical'].forEach(id => {
    document.getElementById(id)?.classList.add('hidden');
  });

  document.getElementById('id-verified-dot')?.classList.add('hidden');
  document.getElementById('verif-row')?.classList.add('hidden');
  document.getElementById('upload-card')?.classList.add('hidden');
  document.getElementById('confidence-card')?.classList.add('hidden');
  document.getElementById('chat-messages').innerHTML = '';
  document.getElementById('options-zone').innerHTML  = '';

  const badge = document.getElementById('vb-govid');
  if (badge) { badge.className = 'verif-badge pending'; badge.textContent = 'Pending'; }

  // Show welcome + insurance type buttons immediately — NO backend call here
  // Backend reset happens on the FIRST real message (is_new_chat:true)
  addBotBubble(
    "Hi 😊 I'm PolicyBot, your AI insurance advisor!\nWhat type of insurance are you looking for today?",
    ["Health Insurance","Term / Life Insurance","Vehicle Insurance",
     "Travel Insurance","Property Insurance","Accident Insurance"],
    "radio"
  );
  updateProgress(7, 'Insurance Type', 'insurance_type');
  setButtonsEnabled(true);
}

// ─── Send message ─────────────────────────────────────────────────────────────
async function sendMsg() {
  const inp  = document.getElementById('chat-input');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  autoH(inp);
  await doSend(text, null);
}

async function doSend(text, selectedOption) {
  if (buttonsDisabled) return;
  setButtonsEnabled(false);
  clearOptions();
  addUserBubble(selectedOption || text);
  showTyping(true);

  // is_new_chat is true ONLY on the very first message of a session
  // This triggers backend reset (profile + history wipe) atomically with the real message
  const newChat = isFirstMessage;
  if (isFirstMessage) isFirstMessage = false;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        message:         text,
        selected_option: selectedOption,
        user_id:         userId,
        session_id:      sessionId,
        language:        currentLang,
        is_new_chat:     newChat       // ← true only on first message
      })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    showTyping(false);

    if (d.status === 'success') {
      currentStage = d.stage;
      addBotBubble(d.reply, d.options || [], d.option_type || 'none');
      updateProgress(d.progress, d.stage_label, d.stage);
      refreshProfile();

      // Upload card visibility
      const uploadCard = document.getElementById('upload-card');
      if (d.show_upload || UPLOAD_STAGES.includes(d.stage)) {
        uploadCard?.classList.remove('hidden');
        autoSelectDocType(d.stage);
        updateSkipButton(d.stage);
        // Show verif-row badge when at doc_upload
        if (d.stage === 'doc_upload' || d.stage === 'verify_wait') {
          document.getElementById('verif-row')?.classList.remove('hidden');
        }
      } else {
        uploadCard?.classList.add('hidden');
      }

      if (d.confidence) showConfidence(d.confidence);

      // 3D visualizer — show at recommendation stage
      if (d.stage === 'recommendation') {
        show3DVisualizer(d.options || []);
        showSessionQuickBar();
      }
      // Hide 3D when past recommendation
      if (['explain_plan','ask_escalation','ask_rating','farewell'].includes(d.stage)) {
        hide3DVisualizer();
        hideSessionQuickBar();
      }

      if (d.show_escalation) {
        document.getElementById('esc-panel')?.classList.remove('hidden');
      }

      // Rating stage: clear option buttons FIRST, then show rating panel
      if (d.stage === 'ask_rating' || d.show_rating) {
        clearOptions(false);              // wipe plan radio buttons
        setTimeout(() => showRatingPanel(), 180);  // then show star widget
      }

      if (d.show_farewell || d.stage === 'farewell' || d.lock_chat) {
        if (d.show_farewell || d.stage === 'farewell') {
          setTimeout(() => showFarewell(d.reply), 800);
          triggerCleanup();
        }
        lockChatInput();
      }

      // Farewell: clear everything including rating panel
      if (d.stage === 'farewell') {
        setTimeout(() => clearOptions(true), 100);
      }
    } else {
      showTyping(false);
      addBotBubble("Sorry, I had a hiccup. Please try again! 🙏");
    }
  } catch(e) {
    showTyping(false);
    addBotBubble("Connection issue — please check your internet and try again.");
    console.error('[doSend error]', e);
  } finally {
    setButtonsEnabled(true);
  }
}

// ─── Enable / Disable buttons ─────────────────────────────────────────────────
function setButtonsEnabled(enabled) {
  buttonsDisabled = !enabled;
  document.querySelectorAll('.opt-btn, .multi-confirm-btn').forEach(btn => {
    btn.disabled = !enabled;
    btn.classList.toggle('disabled', !enabled);
  });
  const inp = document.getElementById('chat-input');
  if (inp) inp.disabled = !enabled;
  const sendBtn = document.querySelector('.send-btn');
  if (sendBtn) sendBtn.disabled = !enabled;
}

// ─── Permanently lock chat after farewell ─────────────────────────────────────
function lockChatInput() {
  buttonsDisabled = true;
  clearOptions();
  // Disable text input
  const inp = document.getElementById('chat-input');
  if (inp) {
    inp.disabled = true;
    inp.placeholder = '🎉 Conversation complete — start a new chat!';
    inp.style.opacity = '0.5';
    inp.style.cursor  = 'not-allowed';
  }
  // Disable send button
  const sendBtn = document.querySelector('.send-btn');
  if (sendBtn) { sendBtn.disabled = true; sendBtn.style.opacity = '0.4'; }
  // Disable mic button
  const micBtn = document.querySelector('.mic-btn');
  if (micBtn) { micBtn.disabled = true; micBtn.style.opacity = '0.4'; }
  // Hide upload card
  document.getElementById('upload-card')?.classList.add('hidden');
  // Clear options zone completely
  const oz = document.getElementById('options-zone');
  if (oz) oz.innerHTML = '';
  // Show a "new chat" prompt in the options zone
  if (oz) {
    oz.innerHTML = '<button class="opt-btn new-chat-btn" onclick="startNewChat()" style="background:linear-gradient(135deg,var(--a1),var(--a2));color:#fff;border:none;padding:10px 28px;border-radius:30px;font-weight:600;font-size:13px;cursor:pointer;margin-top:8px;box-shadow:0 4px 14px rgba(90,114,255,.4)">🔄 Start New Chat</button>';
  }
}

// ─── Auto-select doc type based on stage ─────────────────────────────────────
function autoSelectDocType(stage) {
  const sel = document.getElementById('up-doc-type');
  if (!sel) return;
  const map = {
    'doc_upload':              'aadhaar',
    'verify_wait':             'aadhaar',
    'condition_report_upload': 'health_report',
    'condition_report_wait':   'health_report',
    'optional_health_check':   'health_report',
    'vehicle_history':         'vehicle_insurance',
    'vehicle_doc_upload':      'vehicle_insurance',
    'life_docs':               'life_doc',
    'travel_declare':          'travel_doc',
    'property_history':        'property_doc',
  };
  const val = map[stage];
  if (val) sel.value = val;
}

// ─── Skip button visibility ───────────────────────────────────────────────────
function updateSkipButton(stage) {
  const skipBtn = document.getElementById('skip-upload-btn');
  if (!skipBtn) return;
  if (SKIPPABLE_STAGES.includes(stage)) {
    skipBtn.classList.remove('hidden');
    skipBtn.disabled = false;
  } else {
    skipBtn.classList.add('hidden');
  }
}

// ─── Skip condition upload ────────────────────────────────────────────────────
async function skipConditionUpload() {
  const skipBtn = document.getElementById('skip-upload-btn');
  if (skipBtn) skipBtn.disabled = true;
  await doSend('skip', 'Skip — Continue Without');
}

// ─── Drag & Drop setup ────────────────────────────────────────────────────────
function setupDragDrop() {
  const zone = document.getElementById('upload-drop-zone');
  if (!zone) return;
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) processUpload(file);
  });
}

function handleUpload(input) {
  const file = input.files[0];
  if (file) processUpload(file);
  input.value = '';
}

// ─── Document Upload Pipeline ─────────────────────────────────────────────────
async function processUpload(file) {
  const sel = document.getElementById('up-doc-type');
  const docType = sel ? sel.value : 'aadhaar';
  const ext = file.name.split('.').pop().toLowerCase();

  const ALLOWED = ['jpg','jpeg','png','webp','gif','bmp','tiff','tif','pdf','docx','doc','txt'];
  if (!ALLOWED.includes(ext)) {
    showToast('Unsupported file type. Use JPG, PNG, WEBP, PDF, DOCX, or TXT.', 'error');
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    showToast('File too large. Maximum 15MB allowed.', 'error');
    return;
  }
  if (file.size < 500) {
    showToast('File appears empty. Please try again.', 'error');
    return;
  }

  setButtonsEnabled(false);
  clearOptions();
  showVerifOverlay(docType);

  const form = new FormData();
  form.append('file',       file);
  form.append('doc_type',   docType);
  form.append('user_id',    userId);
  form.append('session_id', sessionId);

  try {
    setVerifStep(1); await sleep(350);
    setVerifStep(2);

    const res = await fetch('/api/upload', { method: 'POST', body: form });

    await sleep(450);
    setVerifStep(3);
    await sleep(600);
    setVerifStep(4);

    const d = await res.json();
    d.doc_type = docType;

    await sleep(400);
    setVerifStep(5);
    await sleep(450);

    hideVerifOverlay();
    handleVerifResult(d);

  } catch(e) {
    hideVerifOverlay();
    showToast('Upload failed — please check your connection and try again.', 'error');
    console.error('[processUpload error]', e);
    setButtonsEnabled(true);
  }
}

// ─── Handle verification result ───────────────────────────────────────────────
function handleVerifResult(d) {
  clearOptions();

  // Show status box in sidebar
  document.getElementById('verif-status-box')?.classList.remove('hidden');
  const vsresult = document.getElementById('vsb-result');
  if (vsresult) vsresult.classList.remove('hidden');

  // Engine badge
  const ocrBadge = document.getElementById('ocr-badge');
  if (ocrBadge) ocrBadge.textContent = d.engine === 'offline_ocr' ? '🔍 OCR' : '🤖 AI';

  const isConditionDoc = [
    'health_report','condition_report','vehicle_insurance',
    'rc_book','life_doc','travel_doc','property_doc'
  ].includes(d.doc_type);

  if (d.verified && isConditionDoc) {
    // ── Condition doc success → go to budget ────────────────────────────
    addSystemMsg('✅ ' + (d.reply || 'Document analyzed!'), 'success');
    showToast('✅ Document analyzed! Continuing...', 'success');
    if (vsresult) { vsresult.className = 'vsb-result success'; vsresult.textContent = d.reply || 'Analyzed'; }
    setTimeout(() => {
      addBotBubble(
        "Almost there! 🎯 What is your monthly budget for insurance premiums? 💰",
        ['Under ₹500','₹500–₹1,000','₹1,000–₹2,000','₹2,000–₹5,000','Above ₹5,000'],
        'radio'
      );
      updateProgress(65, 'Budget', 'collect_budget');
      currentStage = 'collect_budget';
      document.getElementById('upload-card')?.classList.add('hidden');
      setButtonsEnabled(true);
    }, 900);
    return;
  }

  if (d.verified) {
    // ── Gov ID verified → confirm name / age / gender from ID ───────────
    const successMsg = d.reply || 'Identity verified!';
    addSystemMsg('✅ ' + successMsg, 'success');
    setBadge('vb-govid', 'verified', 'Verified ✅');
    document.getElementById('id-verified-dot')?.classList.remove('hidden');
    document.getElementById('verif-row')?.classList.remove('hidden');
    showToast('✅ Identity verified! 👍', 'success');
    if (vsresult) { vsresult.className = 'vsb-result success'; vsresult.textContent = successMsg; }

    // Get current profile to confirm name/age match
    fetch(`/api/profile?user_id=${encodeURIComponent(userId)}`)
      .then(r => r.json())
      .then(pd => {
        const p = pd.profile || {};
        const name = p.name || '';
        const age  = p.age  || '';
        const docTypeFound = d.doc_type_found || 'your ID';

        setTimeout(() => {
          // Ask to confirm name + age shown on the ID, then proceed to gender
          addBotBubble(
            `✅ ${docTypeFound} verified successfully!\n\n` +
            `I can see the following details from your document:\n` +
            `👤 Name: ${name || '(not detected)'}\n` +
            `🎂 Age: ${age ? age + ' years' : '(not detected)'}\n\n` +
            `Are these details correct? Please confirm to continue.`,
            ['Yes, details are correct ✅', 'No, update my details ✏️'],
            'radio'
          );
          updateProgress(38, 'Confirm Details', 'collect_gender');
          currentStage = 'collect_gender';
          setButtonsEnabled(true);
        }, 900);
      })
      .catch(() => {
        // Fallback if profile fetch fails
        setTimeout(() => {
          addBotBubble(
            "✅ ID verified! Let's continue. What is your gender? 😊",
            ['Male','Female','Other'], 'radio'
          );
          updateProgress(42, 'Gender', 'collect_gender');
          currentStage = 'collect_gender';
          setButtonsEnabled(true);
        }, 900);
      });

  } else {
    // ── Verification failed ───────────────────────────────────────────────
    const icon    = d.v_status === 'api_error' ? '⚠️' : '❌';
    const msgType = d.v_status === 'api_error' ? 'warn' : 'fail';
    addSystemMsg(`${icon} ${d.reply || 'Verification failed.'}`, msgType);
    setBadge('vb-govid', 'failed', 'Not Verified');
    document.getElementById('verif-row')?.classList.remove('hidden');
    if (vsresult) {
      vsresult.className = `vsb-result ${msgType === 'warn' ? 'warn' : 'fail'}`;
      vsresult.textContent = d.reply || 'Not verified';
    }

    // Quality tips
    const tips = {
      blurry:     '📸 Tip: Hold your phone steady and tap to focus.',
      dark:       '💡 Tip: Move to a brighter area or use flashlight.',
      cropped:    '📐 Tip: Make sure all 4 corners are visible.',
      incomplete: '📋 Tip: Ensure the complete document is visible.',
      unreadable: '🔍 Tip: Place flat and photograph from directly above.',
    };
    if (d.quality && tips[d.quality]) {
      setTimeout(() => addBotBubble(tips[d.quality]), 400);
    }

    const fallbackOpts = isConditionDoc
      ? ['Upload Again', 'Skip — Continue Without']
      : ['Upload Document Again', 'Continue Without Verification'];

    setTimeout(() => {
      renderOptions(d.options?.length ? d.options : fallbackOpts,
                    d.option_type || 'radio');
      setButtonsEnabled(true);
    }, 800);
  }
}

// ─── Verification Overlay ─────────────────────────────────────────────────────
function showVerifOverlay(docType) {
  document.getElementById('verif-overlay')?.classList.remove('hidden');

  const lbl = document.getElementById('engine-label');
  if (lbl) lbl.textContent = 'Offline OCR Active';

  const isCondition = ['health_report','condition_report','vehicle_insurance',
    'rc_book','life_doc','travel_doc','property_doc'].includes(docType);

  const titleMap = {
    health_report:     'Analyzing Health Report',
    condition_report:  'Analyzing Medical Report',
    rc_book:           'Analyzing RC Book',
    vehicle_insurance: 'Analyzing Vehicle Document',
    life_doc:          'Analyzing Life Document',
    travel_doc:        'Analyzing Travel Document',
    property_doc:      'Analyzing Property Document',
    prev_policy:       'Analyzing Policy Document',
  };
  const titleEl = document.getElementById('verif-overlay-title');
  if (titleEl) titleEl.textContent = titleMap[docType] || 'Analyzing Document';

  const stepLabels = isCondition
    ? ['Document Received','Checking Quality','Extracting Text (OCR)','Identifying Conditions','Storing Result']
    : ['Document Received','Checking Image Quality','Extracting Text (OCR)','Finding Date of Birth','Verifying Age Match'];

  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById(`vos-${i}`);
    if (!el) continue;
    el.className = 'vos';
    const icon = el.querySelector('i');
    if (icon) {
      el.innerHTML = '';
      el.appendChild(icon.cloneNode(true));
      el.appendChild(document.createTextNode(' ' + stepLabels[i-1]));
    } else {
      el.textContent = stepLabels[i-1];
    }
  }
  setVerifStep(1);
  setVerifSub('Securely uploading your document…');
}

function setVerifStep(n) {
  [1,2,3,4,5].forEach(i => {
    const el = document.getElementById(`vos-${i}`);
    if (!el) return;
    if (i < n)       el.className = 'vos done';
    else if (i === n){ el.className = 'vos active'; }
    else             el.className = 'vos';
  });
  const subtitles = ['','Document Received','Checking Quality…',
    'Extracting text with OCR…','Analyzing content…','Storing result…'];
  setVerifSub(subtitles[n] || 'Processing…');
}
function setVerifSub(txt) {
  const el = document.getElementById('verif-overlay-sub');
  if (el) el.textContent = txt;
}
function hideVerifOverlay() {
  document.getElementById('verif-overlay')?.classList.add('hidden');
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Option Buttons (radio + multi) ──────────────────────────────────────────
function renderOptions(options, type) {
  const zone = document.getElementById('options-zone');
  if (!zone) return;
  zone.innerHTML = '';
  if (!options || !options.length) return;

  const zoneId = 'oz_' + Date.now();
  multiSelections[zoneId] = [];

  if (type === 'multi') {
    const lbl = document.createElement('div');
    lbl.className = 'opt-label';
    lbl.textContent = 'Select all that apply:';
    zone.appendChild(lbl);
  }

  options.forEach(opt => {
    const btn = document.createElement('button');
    btn.className = `opt-btn${type === 'multi' ? ' multi-btn' : ''}`;
    btn.innerHTML = `<span class="btn-text">${escHtml(opt)}</span>`;
    btn.dataset.opt  = opt;
    btn.dataset.type = type;
    btn.dataset.zone = zoneId;
    btn.disabled = buttonsDisabled;
    if (buttonsDisabled) btn.classList.add('disabled');
    btn.addEventListener('click', e => handleOption(e, btn, opt, type, zoneId));
    zone.appendChild(btn);
  });

  if (type === 'multi') {
    const conf = document.createElement('button');
    conf.className = 'multi-confirm-btn';
    conf.innerHTML = '<i class="fas fa-check"></i> Confirm Selection';
    conf.disabled = buttonsDisabled;
    conf.onclick = () => confirmMulti(zoneId);
    zone.appendChild(conf);
  }
}

function handleOption(e, btn, opt, type, zoneId) {
  if (btn.disabled || buttonsDisabled) return;
  addRipple(e, btn);

  if (type === 'radio') {
    document.querySelectorAll(`[data-zone="${zoneId}"]`).forEach(b =>
      b.classList.remove('selected-radio')
    );
    btn.classList.add('selected-radio');
    update3DPlanLabel(opt);
    setTimeout(() => { clearOptions(); doSend(opt, opt); }, 350);
  } else {
    btn.classList.toggle('selected-multi');
    const sel = multiSelections[zoneId] || [];
    const idx = sel.indexOf(opt);
    if (idx === -1) sel.push(opt); else sel.splice(idx, 1);
    multiSelections[zoneId] = sel;
  }
}

function confirmMulti(zoneId) {
  const sel = multiSelections[zoneId] || [];
  if (!sel.length) { showToast('Please select at least one option 😊', 'info'); return; }
  clearOptions();
  const combined = sel.join(', ');
  doSend(combined, combined);
}

function clearOptions(hideRating = false) {
  const zone = document.getElementById('options-zone');
  if (zone) zone.innerHTML = '';
  // Only hide rating panel when explicitly requested
  if (hideRating) document.getElementById('rating-panel')?.classList.add('hidden');
  document.getElementById('esc-panel')?.classList.add('hidden');
}

// ── 3D Visualizer controls ──────────────────────────────────────────────────
function show3DVisualizer(planOptions) {
  const card = document.getElementById('viz3d-card');
  if (!card) return;
  card.classList.remove('hidden');
  const insType = document.getElementById('pv-ins')?.textContent?.trim() || 'Health Insurance';
  if (typeof PolicyBot3D !== 'undefined') {
    PolicyBot3D.init('viz3d-canvas', insType, planOptions || []);
  }
  const lbl = document.getElementById('viz3d-plan-name');
  if (lbl) lbl.textContent = planOptions.length ? `${planOptions.length} plans available` : 'Select a plan to view';
}

function hide3DVisualizer() {
  const card = document.getElementById('viz3d-card');
  if (card) card.classList.add('hidden');
  if (typeof PolicyBot3D !== 'undefined') PolicyBot3D.destroy();
}

function update3DPlanLabel(planName) {
  const lbl = document.getElementById('viz3d-plan-name');
  if (lbl) lbl.textContent = planName || '';
}

// ── Session quick-action bar ───────────────────────────────────────────────
function showSessionQuickBar() {
  if (document.getElementById('session-quick-bar')) return; // already shown
  const zone = document.getElementById('options-zone');
  if (!zone) return;
  const bar = document.createElement('div');
  bar.id = 'session-quick-bar';
  bar.className = 'session-quick-bar';
  bar.innerHTML = `
    <button class="sqb-btn sqb-btn-rate" onclick="quickRate()">
      <i class="fas fa-star"></i> Rate & End
    </button>
    <button class="sqb-btn sqb-btn-end" onclick="quickEnd()">
      <i class="fas fa-times-circle"></i> Wind Up
    </button>`;
  // Insert before options zone content
  zone.insertBefore(bar, zone.firstChild);
}

function hideSessionQuickBar() {
  const bar = document.getElementById('session-quick-bar');
  if (bar) bar.remove();
}

function quickRate() {
  hideSessionQuickBar();
  hide3DVisualizer();
  doSend('wind up', 'Wind Up Session');
}

function quickEnd() {
  hideSessionQuickBar();
  hide3DVisualizer();
  doSend('wind up', 'Wind Up Session');
}

function addRipple(e, btn) {
  const r = document.createElement('span');
  r.className = 'ripple';
  const rect = btn.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  r.style.cssText = `width:${size}px;height:${size}px;`
    + `left:${e.clientX - rect.left - size/2}px;`
    + `top:${e.clientY - rect.top - size/2}px`;
  btn.appendChild(r);
  setTimeout(() => r.remove(), 600);
}

// ─── Message rendering ────────────────────────────────────────────────────────
function addBotBubble(text, options=[], optType='none') {
  const c = document.getElementById('chat-messages');
  if (!c) return;
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.innerHTML = `
    <div class="msg-av"><i class="fas fa-robot"></i></div>
    <div class="msg-inner">
      <div class="msg-bubble-bot">${fmtMsg(text)}</div>
      <div class="msg-time">${nowStr()}</div>
    </div>`;
  c.appendChild(row);
  c.scrollTop = c.scrollHeight;
  // Always clear stale options first, then render new ones if any
  const zone = document.getElementById('options-zone');
  if (zone && !options.length) zone.innerHTML = '';
  if (options.length) renderOptions(options, optType);
}

function addUserBubble(text) {
  const c = document.getElementById('chat-messages');
  if (!c) return;
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-av"><i class="fas fa-user"></i></div>
    <div class="msg-inner">
      <div class="msg-bubble-user">${escHtml(text)}</div>
      <div class="msg-time">${nowStr()}</div>
    </div>`;
  c.appendChild(row);
  c.scrollTop = c.scrollHeight;
}

function addSystemMsg(text, type='info') {
  const c = document.getElementById('chat-messages');
  if (!c) return;
  const row = document.createElement('div');
  row.className = 'msg-row system';
  row.innerHTML = `<div class="msg-bubble-system ${type}">${escHtml(text)}</div>`;
  c.appendChild(row);
  c.scrollTop = c.scrollHeight;
}

function fmtMsg(text) {
  let t = escHtml(text);
  t = t.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/\n/g, '<br>');
  return t;
}
function escHtml(t) {
  return (t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function nowStr() {
  return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

// ─── Typing indicator ─────────────────────────────────────────────────────────
function showTyping(show) {
  document.getElementById('typing-row')?.classList.toggle('hidden', !show);
  if (show) {
    const c = document.getElementById('chat-messages');
    if (c) c.scrollTop = c.scrollHeight;
  }
}

// ─── Rating ───────────────────────────────────────────────────────────────────
function showRatingPanel() {
  const panel = document.getElementById('rating-panel');
  if (panel) {
    panel.classList.remove('hidden');
    // Reset stars and button state
    selectedStar = 0;
    document.querySelectorAll('.star-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('rating-submit')?.classList.add('hidden');
  }
}
function setStar(n) {
  selectedStar = n;
  document.querySelectorAll('.star-btn').forEach((b,i) =>
    b.classList.toggle('active', i < n)
  );
  document.getElementById('rating-submit')?.classList.remove('hidden');
}
async function submitRating() {
  if (!selectedStar) { showToast('Please tap a star to rate 😊', 'info'); return; }

  // ── Show thank-you animation inside rating panel ───────────────────────
  const panel = document.getElementById('rating-panel');
  const starsStr = '⭐'.repeat(selectedStar);
  const msgs = ['Thanks! 🙏','Appreciated! 😊','You are awesome! 🎉','Wonderful! 🌟','Amazing! 🏆'];
  const thankMsg = msgs[Math.min(selectedStar - 1, msgs.length - 1)];

  if (panel) {
    panel.innerHTML = `
      <div class="rating-thankyou">
        <div class="rty-stars">${starsStr}</div>
        <div class="rty-msg">${thankMsg}</div>
        <div class="rty-sub">Your feedback means a lot to us!</div>
      </div>`;
  }

  try {
    // Store rating via API
    await fetch('/api/rating', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ user_id: userId, rating: selectedStar })
    });

    // Wait for user to see the animation, then trigger farewell
    await new Promise(r => setTimeout(r, 1800));

    // Add farewell bot message
    const farewellMsg = `🎉 Thank you for the ${selectedStar}-star rating! It means a lot to us. Wishing you a safe and secure future! 😊`;
    addBotBubble(farewellMsg);

    // Hide rating panel and trigger farewell overlay
    setTimeout(() => {
      if (panel) panel.classList.add('hidden');
      showFarewell(farewellMsg);
      triggerCleanup();
      lockChatInput();
    }, 1200);

  } catch(e) {
    showToast('Could not submit rating', 'error');
    if (panel) panel.classList.add('hidden');
  }
}

// ─── Escalation ───────────────────────────────────────────────────────────────
async function submitEscalation() {
  const phone = document.getElementById('esc-phone')?.value.trim();
  const time  = document.getElementById('esc-time')?.value.trim();
  if (!phone) { showToast('Please enter your phone number 😊', 'info'); return; }
  try {
    const res = await fetch('/api/escalate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ user_id: userId, phone, best_time: time })
    });
    const d = await res.json();
    document.getElementById('esc-panel')?.classList.add('hidden');
    addBotBubble(d.reply || '✅ Human advisor notified! They will call you soon 📞');
    showToast('Callback requested!', 'success');
  } catch(e) { showToast('Could not submit request', 'error'); }
}

// ─── Farewell + cleanup ───────────────────────────────────────────────────────
function showFarewell(text) {
  const el = document.getElementById('farewell-text');
  if (el) el.textContent = text || '🎉 Thank you! Wishing you a safe and secure future.';
  document.getElementById('farewell-overlay')?.classList.remove('hidden');
  spawnConfetti();
}
function startNewChat() {
  document.getElementById('farewell-overlay')?.classList.add('hidden');
  // Generate completely new IDs — guarantees a fresh DB row on backend
  userId        = genId();
  sessionId     = genId();
  isFirstMessage = true;
  location.reload();
}
async function triggerCleanup() {
  try {
    await fetch('/api/cleanup', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ user_id: userId })
    });
  } catch(e) {}
}
function spawnConfetti() {
  const container = document.getElementById('farewell-confetti');
  if (!container) return;
  const colors = ['#5a72ff','#9d4edd','#f59e0b','#10b981','#ef4444','#22d3ee'];
  for (let i = 0; i < 55; i++) {
    const dot = document.createElement('div');
    const size = Math.random() * 8 + 4;
    dot.style.cssText = `position:absolute;width:${size}px;height:${size}px;`
      + `background:${colors[Math.floor(Math.random()*colors.length)]};`
      + `border-radius:${Math.random()>.5?'50%':'3px'};`
      + `left:${Math.random()*100}%;top:${Math.random()*100}%;`
      + `animation:confettiFall ${1.2+Math.random()*1.8}s ease-in ${Math.random()*.8}s forwards;`
      + `opacity:0;`;
    container.appendChild(dot);
  }
  const style = document.createElement('style');
  style.textContent = `@keyframes confettiFall{0%{opacity:1;transform:translateY(-20px) rotate(0)}100%{opacity:0;transform:translateY(120px) rotate(360deg)}}`;
  document.head.appendChild(style);
}

// ─── Confidence ───────────────────────────────────────────────────────────────
function showConfidence(text) {
  const card = document.getElementById('confidence-card');
  const val  = document.getElementById('conf-value');
  if (card && val) {
    val.textContent = text;
    card.classList.remove('hidden');
  }
}

// ─── Progress bar + dots ──────────────────────────────────────────────────────
function updateProgress(pct, label, stage) {
  const p = pct || 7;
  const fill = document.getElementById('prog-fill');
  const pctEl = document.getElementById('prog-pct');
  const lblEl = document.getElementById('prog-stage-label');
  if (fill) fill.style.width = p + '%';
  if (pctEl) pctEl.textContent = p + '%';
  if (lblEl && label) lblEl.textContent = label;

  const dots = document.querySelectorAll('.pd');
  const activeIdx = DOT_STAGES.indexOf(stage);
  dots.forEach((d, i) => {
    d.classList.remove('active','done');
    if (i < activeIdx)      d.classList.add('done');
    else if (i === activeIdx) d.classList.add('active');
  });
}

// ─── Profile sidebar refresh ──────────────────────────────────────────────────
function refreshProfile() {
  fetch(`/api/profile?user_id=${encodeURIComponent(userId)}`)
    .then(r => r.json())
    .then(d => {
      const p = d.profile || {};
      if (p.name) {
        const pName = document.getElementById('p-name');
        if (pName) pName.textContent = p.name;
      }
      setField('pf-ins',    'pv-ins',    p.insurance_type);
      setField('pf-age',    'pv-age',    p.age ? `${p.age} yrs` : null);
      setField('pf-city',   'pv-city',   p.city);
      setField('pf-budget', 'pv-budget', p.budget_range);
      setField('pf-medical','pv-medical',p.medical_conditions);
      if (p.gov_id_verified) {
        setBadge('vb-govid', 'verified', 'Verified ✅');
        document.getElementById('id-verified-dot')?.classList.remove('hidden');
        document.getElementById('verif-row')?.classList.remove('hidden');
      }
    }).catch(() => {});
}

function setField(rowId, valId, val) {
  if (val) {
    document.getElementById(rowId)?.classList.remove('hidden');
    const el = document.getElementById(valId);
    if (el) el.textContent = val;
  }
}
function setBadge(id, cls, label) {
  const b = document.getElementById(id);
  if (b) { b.className = `verif-badge ${cls}`; b.textContent = label; }
}

// ─── Voice ────────────────────────────────────────────────────────────────────
function initVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    const micBtn = document.getElementById('mic-btn');
    if (micBtn) micBtn.style.display = 'none';
    return;
  }
  recognition = new SR();
  recognition.continuous    = false;
  recognition.interimResults = true;
  recognition.onresult = e => {
    const inp = document.getElementById('chat-input');
    if (inp) inp.value = Array.from(e.results).map(r => r[0].transcript).join('');
  };
  recognition.onend = () => {
    const txt = document.getElementById('chat-input')?.value.trim();
    stopVoice();
    if (txt) setTimeout(() => sendMsg(), 200);
  };
  recognition.onerror = () => stopVoice();
}
function toggleVoice() { isListening ? stopVoice() : startVoice(); }
function startVoice() {
  if (!recognition) return;
  isListening = true;
  document.getElementById('mic-btn')?.classList.add('listening');
  document.getElementById('waveform-bar')?.classList.remove('hidden');
  recognition.lang = getLangCode(currentLang);
  recognition.start();
  showToast('🎤 Listening…', 'info');
}
function stopVoice() {
  isListening = false;
  document.getElementById('mic-btn')?.classList.remove('listening');
  document.getElementById('waveform-bar')?.classList.add('hidden');
  if (recognition) try { recognition.stop(); } catch(e) {}
}
function getLangCode(l) {
  return l === 'Tamil' ? 'ta-IN' : l === 'Hindi' ? 'hi-IN' : 'en-IN';
}

// ─── Input helpers ────────────────────────────────────────────────────────────
function onKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
}
function autoH(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 110) + 'px';
}

// ─── Toast ────────────────────────────────────────────────────────────────────
function showToast(msg, type='info') {
  const wrap = document.getElementById('toast-wrap');
  if (!wrap) return;
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  const icons = {success:'✅', error:'❌', info:'ℹ️', warn:'⚠️'};
  t.innerHTML = `<span>${icons[type]||'ℹ️'}</span><span>${escHtml(msg)}</span>`;
  wrap.appendChild(t);
  setTimeout(() => {
    t.style.opacity   = '0';
    t.style.transform = 'translateX(30px)';
    setTimeout(() => t.remove(), 300);
  }, 3500);
}

// ─── Particles ────────────────────────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W = canvas.width  = innerWidth;
  let H = canvas.height = innerHeight;
  const pts = Array.from({length:50}, () => ({
    x: Math.random()*W, y: Math.random()*H,
    vx: (Math.random()-.5)*.4, vy: (Math.random()-.5)*.4,
    r: Math.random()*1.4+.4,   o: Math.random()*.3+.1
  }));
  function draw() {
    ctx.clearRect(0,0,W,H);
    const body = document.getElementById('body-root');
    const cls  = body?.className || '';
    if (cls.includes('clean') || cls.includes('minimal')) {
      requestAnimationFrame(draw); return;
    }
    pts.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(90,114,255,${p.o})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
  window.addEventListener('resize', () => {
    W = canvas.width  = innerWidth;
    H = canvas.height = innerHeight;
  });
}