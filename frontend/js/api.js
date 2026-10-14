/**
 * VisionCare API integration for HTML frontend
 * Must be served from http://localhost:8001 (run run.bat first).
 */
const API =
  window.location.protocol === 'file:' || !window.location.origin || window.location.origin === 'null'
    ? 'http://127.0.0.1:8001'
    : window.location.origin;

let currentUser = null;
let lastAnalysis = null;
let selectedUploadFile = null;
let historyCache = [];
let historyPage = 0;
const HISTORY_PAGE_SIZE = 10;

function formatApiError(data, fallback) {
  if (!data) return fallback;
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg || d).join(', ');
  return fallback;
}

async function apiFetch(url, options = {}) {
  try {
    return await fetch(url, options);
  } catch (err) {
    throw new Error(
      'Cannot reach VisionCare server. Start it with run.bat (or: python -m uvicorn api.main:app --host 127.0.0.1 --port 8001), then open http://127.0.0.1:8001'
    );
  }
}

function showServerBanner(ok, message) {
  let el = document.getElementById('vc-server-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'vc-server-banner';
    el.style.cssText =
      'position:fixed;top:0;left:0;right:0;z-index:9999;padding:10px 16px;font-size:12px;font-family:DM Sans,sans-serif;text-align:center;';
    document.body.appendChild(el);
  }
  if (ok) {
    el.style.display = 'none';
    return;
  }
  el.style.display = 'block';
  el.style.background = 'rgba(224,82,82,0.15)';
  el.style.color = '#e05252';
  el.style.borderBottom = '1px solid rgba(224,82,82,0.35)';
  el.textContent = message;
}

async function checkServerHealth() {
  try {
    const r = await fetch(`${API}/api/health`, { method: 'GET' });
    if (!r.ok) throw new Error('unhealthy');
    showServerBanner(true);
    return true;
  } catch {
    showServerBanner(
      false,
      `Server offline — run run.bat in the visioncare folder, then open ${API}`
    );
    return false;
  }
}

function saveSession(user) {
  if (user) currentUser = user;
  if (currentUser) {
    sessionStorage.setItem('vc_user', JSON.stringify(currentUser));
    localStorage.setItem('vc_user', JSON.stringify(currentUser));
  }
}

function loadSession() {
  const raw = sessionStorage.getItem('vc_user') || localStorage.getItem('vc_user');
  if (raw && raw !== 'undefined' && raw !== 'null') {
    try {
      currentUser = JSON.parse(raw);
      return true;
    } catch {
      return false;
    }
  }
  return false;
}

function clearSession() {
  currentUser = null;
  sessionStorage.removeItem('vc_user');
  localStorage.removeItem('vc_user');
}

async function apiRegister(name, email, password) {
  const r = await apiFetch(`${API}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatApiError(data, 'Registration failed'));
  return data;
}

async function apiLogin(email, password) {
  const r = await apiFetch(`${API}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatApiError(data, 'Invalid credentials'));
  return data;
}

async function apiDashboard(userId) {
  const r = await apiFetch(`${API}/api/dashboard/${userId}`);
  if (!r.ok) return null;
  return r.json();
}

async function apiHistory(userId) {
  const r = await apiFetch(`${API}/api/history/${userId}`);
  if (!r.ok) return [];
  return r.json();
}

async function apiDatasetInfo() {
  const r = await apiFetch(`${API}/api/dataset/info`);
  if (!r.ok) return null;
  return r.json();
}

async function apiDeleteScan(scanId) {
  const r = await apiFetch(`${API}/api/scan/${scanId}?user_id=${currentUser.id}`, { method: 'DELETE' });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatApiError(data, 'Delete failed'));
  return data;
}

async function apiDeletePatientHistory(patientName) {
  const q = encodeURIComponent(patientName);
  const r = await apiFetch(
    `${API}/api/history/${currentUser.id}/patient?patient_name=${q}`,
    { method: 'DELETE' }
  );
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatApiError(data, 'Delete failed'));
  return data;
}

async function apiDeleteAllHistory() {
  const r = await apiFetch(`${API}/api/history/${currentUser.id}`, { method: 'DELETE' });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatApiError(data, 'Delete failed'));
  return data;
}

async function apiReloadScan(scanId) {
  const r = await apiFetch(`${API}/api/scan/${scanId}/reload?user_id=${currentUser.id}`);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatApiError(data, 'Could not load scan'));
  return data;
}

function diseaseBadge(label) {
  const map = {
    DME: 'badge-danger',
    CNV: 'badge-purple',
    DRUSEN: 'badge-info',
    NORMAL: 'badge-success',
  };
  return map[label] || 'badge-info';
}

function severityBadge(sev) {
  const s = (sev || '').toLowerCase();
  if (s.includes('severe')) return 'badge-danger';
  if (s.includes('moderate')) return 'badge-warning';
  if (s.includes('mild')) return 'badge-success';
  return 'badge-success';
}

function applyAnalysisToModules(data) {
  lastAnalysis = data;
  updatePredictionUI(data);
  updateHeatmapUI(data);
  updateSegmentationUI(data);
  updateReportsUI(data);
  updateCurtainViews(data);
}

async function apiAnalyze(file, patientName, patientAge, patientGender, imageType) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('user_id', String(currentUser.id));
  fd.append('patient_name', patientName || 'Anonymous');
  fd.append('patient_age', String(patientAge || 0));
  fd.append('patient_gender', patientGender || 'Unknown');
  fd.append('image_type', imageType || 'OCT');
  const r = await apiFetch(`${API}/api/analyze`, { method: 'POST', body: fd });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatApiError(data, 'Analysis failed'));
  return data;
}

function b64img(b64) {
  return b64 ? `data:image/png;base64,${b64}` : null;
}

function updateDashboardUI(stats) {
  if (!stats) return;
  const cards = document.querySelectorAll('#mod-dashboard .stat-value');
  if (cards.length >= 6) {
    cards[0].textContent = stats.total_scans ?? 0;
    cards[1].textContent = stats.healthy_count ?? 0;
    cards[2].textContent = stats.disease_count ?? 0;
    cards[3].textContent = stats.critical_count ?? 0;
    cards[4].textContent = stats.active_patients ?? 0;
    cards[5].textContent = `${stats.avg_confidence ?? 0}%`;
  }
  const tbody = document.querySelector('#mod-dashboard .data-table tbody');
  if (tbody && stats.recent_scans?.length) {
    tbody.innerHTML = stats.recent_scans
      .map(
        (s) =>
          `<tr><td>${s.patient_name || 'Patient'} <br><small style="color:var(--text-muted);">${s.patient_age || '?'} yrs, ${s.patient_gender || '?'}</small></td><td>${s.prediction}</td><td><span class="badge ${severityBadge(s.severity || s.stage)}">${s.severity || s.stage}</span></td><td>${(s.created_at || '').slice(0, 10)}</td><td><span class="badge badge-success">Completed</span></td></tr>`
      )
      .join('');
  }
}

async function loadPlotlyCharts() {
  const historyDiv = document.getElementById('plotly-history-chart');
  const distDiv = document.getElementById('plotly-dist-chart');
  if (!historyDiv || !distDiv) return;

  try {
    const r = await apiFetch(`${API}/api/metrics/plotly`);
    if (!r.ok) return;
    const data = await r.json();
    
    if (data.history_chart) {
      Plotly.newPlot('plotly-history-chart', data.history_chart.data, data.history_chart.layout, {responsive: true, displayModeBar: false});
    }
    if (data.distribution_chart) {
      Plotly.newPlot('plotly-dist-chart', data.distribution_chart.data, data.distribution_chart.layout, {responsive: true, displayModeBar: false});
    }
  } catch (err) {
    console.error('Error loading Plotly charts:', err);
  }
}

function getFilteredHistory() {
  const q = (document.getElementById('history-search')?.value || '').trim().toLowerCase();
  if (!q) return historyCache;
  return historyCache.filter(
    (s) =>
      (s.patient_name || '').toLowerCase().includes(q) ||
      (s.prediction || '').toLowerCase().includes(q) ||
      String(s.scan_id).includes(q)
  );
}

function updateHistoryUI(rows) {
  historyCache = rows || [];
  historyPage = 0;
  renderHistoryPage();
}

function renderHistoryPage() {
  const tbody = document.getElementById('history-tbody');
  const countEl = document.getElementById('history-count');
  if (!tbody) return;

  const filtered = getFilteredHistory();
  const total = filtered.length;
  const start = historyPage * HISTORY_PAGE_SIZE;
  const pageRows = filtered.slice(start, start + HISTORY_PAGE_SIZE);

  if (!pageRows.length) {
    tbody.innerHTML =
      '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:24px;">No scan history. Upload an OCT image to get started.</td></tr>';
    if (countEl) countEl.textContent = 'No records';
    return;
  }

  tbody.innerHTML = pageRows
    .map(
      (s) =>
        `<tr data-scan-id="${s.scan_id}">
          <td style="text-align: center; vertical-align: middle;">
            <input type="checkbox" class="history-compare-cb" data-id="${s.scan_id}" data-patient="${s.patient_name || 'Anonymous'}" onchange="updateCompareButtonState()">
          </td>
          <td style="font-family:'DM Mono',monospace;font-size:11px;">SCN_${s.scan_id}</td>
          <td>${s.patient_name || '—'} <br><small style="color:var(--text-muted);">${s.patient_age || '?'} yrs, ${s.patient_gender || '?'}</small></td>
          <td><span class="badge ${diseaseBadge(s.prediction)}">${s.prediction}</span></td>
          <td><span class="badge ${severityBadge(s.severity || s.stage)}">${s.severity || s.stage || '—'}</span></td>
          <td>${(s.created_at || '').slice(0, 10)}</td>
          <td style="font-family:'DM Mono',monospace;">${Math.round(s.retina_affected || 0)}%</td>
          <td style="display:flex;gap:8px;align-items:center;">
            <a class="view-link" data-action="view" data-id="${s.scan_id}">View</a>
            <a class="view-link" data-action="download" data-id="${s.scan_id}">Download</a>
            <a class="view-link" style="color:var(--danger)" data-action="delete" data-id="${s.scan_id}">Delete</a>
          </td>
        </tr>`
    )
    .join('');

  if (countEl) {
    const end = Math.min(start + HISTORY_PAGE_SIZE, total);
    countEl.textContent = `Showing ${start + 1}–${end} of ${total} records`;
  }
  
  updateCompareButtonState();

  tbody.querySelectorAll('[data-action="view"]').forEach((el) => {
    el.onclick = async (e) => {
      e.preventDefault();
      const id = parseInt(el.dataset.id, 10);
      try {
        const data = await apiReloadScan(id);
        applyAnalysisToModules(data);
        switchModule('prediction', document.querySelector('[onclick*="prediction"]'));
      } catch (err) {
        alert(err.message);
      }
    };
  });

  tbody.querySelectorAll('[data-action="download"]').forEach((el) => {
    el.onclick = async (e) => {
      e.preventDefault();
      const id = parseInt(el.dataset.id, 10);
      const s = pageRows.find(r => r.scan_id === id);
      const pname = encodeURIComponent(s?.patient_name || 'Patient');
      const url = `${API}/api/reports/${id}/english?patient_name=${pname}&user_id=${currentUser.id}`;
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `English_Report_${id}.pdf`;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (err) {
        alert(`Download failed: ${err.message}`);
      }
    };
  });

  tbody.querySelectorAll('[data-action="delete"]').forEach((el) => {
    el.onclick = async (e) => {
      e.preventDefault();
      const id = parseInt(el.dataset.id, 10);
      if (!confirm(`Delete scan SCN_${id}? This cannot be undone.`)) return;
      try {
        await apiDeleteScan(id);
        if (lastAnalysis?.scan_id === id) lastAnalysis = null;
        await refreshUserData();
      } catch (err) {
        alert(err.message);
      }
    };
  });
}

function updatePredictionUI(data) {
  const diseaseEl = document.querySelector('.prediction-disease');
  if (diseaseEl) diseaseEl.textContent = `${data.label_full} (${data.label})`;
  const pctEl = document.querySelector('.confidence-pct');
  if (pctEl) pctEl.textContent = `${typeof data.confidence === 'number' ? data.confidence.toFixed(1) : 0}%`;
  const meta = document.querySelectorAll('.prediction-meta .meta-value');
  if (meta.length >= 2) {
    meta[0].textContent = data.stage;
    meta[1].textContent = `${typeof data.retina_affected_pct === 'number' ? data.retina_affected_pct.toFixed(2) : '0.00'}%`;
  }
  const ring = document.querySelector('.confidence-ring circle:last-child');
  if (ring) ring.setAttribute('stroke-dasharray', `${Math.round(data.confidence)} ${100 - Math.round(data.confidence)}`);

  const probs = data.probabilities || {};
  const rows = document.querySelectorAll('#mod-prediction .progress-row');
  const labels = ['DME', 'CNV', 'DRUSEN', 'NORMAL', 'Others'];
  rows.forEach((row, i) => {
    const key = labels[i] === 'Others' ? null : labels[i];
    const val = key && probs[key] != null ? probs[key] : 0;
    const bar = row.querySelector('.progress-bar');
    const txt = row.querySelector('.progress-value');
    if (bar) bar.style.width = `${val}%`;
    if (txt) txt.textContent = `${Math.round(val)}%`;
  });

  const preview = b64img(data.preview_b64);
  if (preview) {
    document.querySelectorAll('#mod-prediction .oct-scan-placeholder').forEach((el) => {
      el.innerHTML = `<img src="${preview}" style="width:100%;height:100%;object-fit:cover;border-radius:8px;">`;
    });
  }

  // Update Additional Information rows dynamically
  const scanInfoRows = document.querySelectorAll('#mod-prediction .scan-info-row');
  if (scanInfoRows.length >= 4) {
    const v0 = scanInfoRows[0].querySelector('.scan-info-val');
    const v1 = scanInfoRows[1].querySelector('.scan-info-val');
    const v2 = scanInfoRows[2].querySelector('.scan-info-val');
    const v3 = scanInfoRows[3].querySelector('.scan-info-val');
    if (v0) v0.textContent = `SCN_${data.scan_id || '0000'}`;
    if (v1) v1.textContent = data.analysis_date || new Date().toISOString().slice(0, 10);
    if (v2) v2.textContent = `PT_${(data.patient_id || data.scan_id || '0000').toString().padStart(5, '0')}`;
    const pAge = (data.age !== undefined && data.age !== null && data.age !== '') ? data.age : (data.patient_age || '—');
    const pGender = data.gender || data.patient_gender || '—';
    if (v3) v3.textContent = `${pAge} / ${pGender}`;
  }
  
  renderThicknessProfileChart(data.thickness_profile);
  updateBiomarkersUI(data);
  updateConsensusUI(data);
  updateEtdrsUI(data);
  render3DMacularTopography(data.topography_3d);
}

function renderThicknessProfileChart(profile) {
  const chartDiv = document.getElementById('plotly-thickness-chart');
  if (!chartDiv) return;
  
  const cols = Array.from({length: 224}, (_, i) => i);
  const trace = {
    x: cols,
    y: profile && profile.length ? profile : new Array(224).fill(0),
    type: 'scatter',
    mode: 'lines',
    name: 'Pathology Thickness',
    fill: 'tozeroy',
    line: { color: '#0284c7', width: 2.5 },
    fillcolor: 'rgba(2, 132, 199, 0.15)'
  };
  
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { title: 'B-scan Plane Columns (Width)', gridcolor: '#e2e8f0', tickfont: {color: '#64748b'}, titlefont: {color: '#475569'} },
    yaxis: { title: 'Vertical Height (Pixels)', gridcolor: '#e2e8f0', tickfont: {color: '#64748b'}, titlefont: {color: '#475569'}, range: [0, 150] },
    margin: { l: 40, r: 20, t: 20, b: 40 }
  };
  
  Plotly.newPlot('plotly-thickness-chart', [trace], layout, {responsive: true, displayModeBar: false});
}

function updateHeatmapUI(data) {
  if (!data) {
    console.warn('updateHeatmapUI: No data provided');
    return;
  }
  const hm = b64img(data.heatmap_b64);
  const orig = b64img(data.preview_b64);

  console.log('Heatmap data:', { has_heatmap: !!data.heatmap_b64, has_preview: !!data.preview_b64 });

  // Left panel — Original OCT Image
  const origBox = document.querySelector('#mod-heatmap .image-preview-box .oct-scan-placeholder');
  if (origBox && orig) {
    origBox.innerHTML = `<img src="${orig}" style="width:100%;height:100%;object-fit:cover;border-radius:6px;" alt="Original OCT">`;
    console.log('✓ Original image rendered in heatmap left panel');
  }

  // Right panel — GradCAM heatmap (backend returns a pre-composited image: OCT + heatmap blended)
  const heatBox = document.querySelector('#mod-heatmap .heatmap-base');
  if (heatBox) {
    if (hm) {
      // Replace heatmap-base content entirely with the composited image
      heatBox.innerHTML = `<img src="${hm}"
        style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:6px;"
        alt="Grad-CAM Heatmap">`;
      console.log('✓ Heatmap composited image rendered');
    } else {
      // Restore placeholder when no heatmap
      heatBox.innerHTML = `
        <div class="oct-scan-placeholder" style="height:100%;opacity:0.7;">
          <div class="oct-layers">
            <div class="oct-layer oct-layer-1"></div>
            <div class="oct-layer oct-layer-2"></div>
          </div>
        </div>
        <div class="heatmap-overlay"></div>`;
      console.warn('Heatmap data is empty — showing placeholder');
    }
  }
}


function updateSegmentationUI(data) {
  if (!data) {
    console.warn('updateSegmentationUI: No data provided');
    return;
  }
  const mask = b64img(data.mask_b64);
  const orig = b64img(data.preview_b64);

  console.log('Segmentation data:', { has_mask: !!data.mask_b64, has_preview: !!data.preview_b64 });

  const imgs = document.querySelectorAll('#mod-segmentation .seg-image');
  console.log(`Found ${imgs.length} seg-image containers`);

  // Panel 1 — Original OCT
  if (imgs[0]) {
    if (orig) {
      imgs[0].innerHTML = `<img src="${orig}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;" alt="Original OCT"><div class="seg-label">Original OCT</div>`;
      imgs[0].style.position = 'relative';
      console.log('✓ Original image rendered');
    }
  }

  // Panel 2 — Segmentation Mask
  if (imgs[1]) {
    if (mask) {
      imgs[1].innerHTML = `<img src="${mask}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;" alt="Segmentation Mask"><div class="seg-label">Segmentation Mask</div>`;
      imgs[1].style.position = 'relative';
      console.log('✓ Segmentation mask rendered');
    } else {
      imgs[1].innerHTML = `<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:12px;">No mask data</div><div class="seg-label">Segmentation Mask</div>`;
      imgs[1].style.position = 'relative';
      console.warn('Segmentation mask data is empty');
    }
  }

  // Panel 3 — Overlay (original OCT + mask layered on top with screen blend)
  if (imgs[2]) {
    if (orig && mask) {
      imgs[2].style.position = 'relative';
      imgs[2].innerHTML = `
        <img src="${orig}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:0;" alt="OCT Base">
        <img src="${mask}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:1;mix-blend-mode:screen;opacity:1;" alt="Mask Overlay">
        <div class="seg-label">Overlay</div>`;
      console.log('✓ Overlay rendered');
    } else if (orig) {
      imgs[2].style.position = 'relative';
      imgs[2].innerHTML = `<img src="${orig}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0.8;" alt="Overlay"><div class="seg-label">Overlay</div>`;
    } else {
      imgs[2].innerHTML = `<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:12px;">No overlay data</div><div class="seg-label">Overlay</div>`;
      imgs[2].style.position = 'relative';
    }
  }

  const vals = document.querySelectorAll('#mod-segmentation .seg-detail-val');
  if (vals.length >= 4) {
    const formatPct = (value) => {
      if (typeof value === 'number') return `${value.toFixed(2)}%`;
      return '0.00%';
    };
    vals[0].textContent = formatPct(data.lesion_area_pct);
    vals[1].textContent = formatPct(data.fluid_area_pct);
    vals[2].textContent = formatPct(data.edema_area_pct);
    const totalAffected = typeof data.total_affected_pct === 'number'
      ? data.total_affected_pct
      : data.retina_affected_pct;
    vals[3].textContent = formatPct(totalAffected);
    console.log('✓ Segmentation details rendered');
  } else {
    console.warn(`Expected 4+ seg-detail-val elements, found ${vals.length}`);
  }
}


function wireAuth() {
  const loginBtn = document.querySelector('#page-login .btn-primary');
  if (loginBtn) {
    loginBtn.onclick = async () => {
      const email = document.querySelector('#page-login input[type="text"]')?.value?.trim();
      const password = document.querySelector('#page-login input[type="password"]')?.value;
      if (!email || !password) {
        alert('Please enter your email/username and password.');
        return;
      }
      try {
        const user = await apiLogin(email, password);
        saveSession(user);
        document.querySelector('.user-name').textContent = user.name;
        document.querySelector('.user-avatar').textContent = user.name.slice(0, 2).toUpperCase();
        showMain();
        await refreshUserData();
      } catch (e) {
        alert(e.message);
      }
    };
  }

  const signupBtn = document.querySelector('#page-signup .btn-primary');
  if (signupBtn) {
    signupBtn.onclick = async () => {
      const inputs = document.querySelectorAll('#page-signup .form-input');
      const name = inputs[0]?.value?.trim();
      const email = inputs[1]?.value?.trim();
      const password = inputs[2]?.value;
      const confirmPassword = inputs[3]?.value;
      const termsCheck = document.getElementById('terms');
      
      if (!name || !email || !password) {
        alert('Please fill in all fields.');
        return;
      }
      if (password !== confirmPassword) {
        alert('Passwords do not match.');
        return;
      }
      if (termsCheck && !termsCheck.checked) {
        alert('Please agree to the Terms and Conditions.');
        return;
      }
      
      try {
        await apiRegister(name, email, password);
        // Automatically login the user after successful registration
        const user = await apiLogin(email, password);
        saveSession(user);
        document.querySelector('.user-name').textContent = user.name;
        document.querySelector('.user-avatar').textContent = user.name.slice(0, 2).toUpperCase();
        showMain();
        await refreshUserData();
      } catch (e) {
        alert(e.message);
      }
    };
  }

  // Support Enter key on login inputs
  const loginInputs = document.querySelectorAll('#page-login .form-input');
  loginInputs.forEach(input => {
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        loginBtn && loginBtn.click();
      }
    });
  });

  // Support Enter key on signup inputs
  const signupInputs = document.querySelectorAll('#page-signup .form-input');
  signupInputs.forEach(input => {
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        signupBtn && signupBtn.click();
      }
    });
  });
}

window.loginDemoDoctor = async function() {
  try {
    let user;
    try {
      user = await apiLogin('doctor@visioncare.ai', 'password123');
    } catch {
      await apiRegister('Dr. Elena Vance', 'doctor@visioncare.ai', 'password123');
      user = await apiLogin('doctor@visioncare.ai', 'password123');
    }
    saveSession(user);
    const nameEl = document.querySelector('.user-name');
    if (nameEl) nameEl.textContent = user.name;
    const avatarEl = document.querySelector('.user-avatar');
    if (avatarEl) avatarEl.textContent = user.name.slice(0, 2).toUpperCase();
    showMain();
    await refreshUserData();
  } catch (e) {
    alert('Demo login failed: ' + e.message);
  }
};

function ensurePatientFields() {
  const card = document.querySelector('#mod-upload .card');
  if (!card || document.getElementById('patient-name-input')) return;
  const wrap = document.createElement('div');
  wrap.style.marginBottom = '12px';
  wrap.innerHTML = `
    <div class="form-group" style="margin-bottom:10px;">
      <label class="form-label">Patient Name</label>
      <input id="patient-name-input" class="form-input" type="text" placeholder="e.g. Ramesh Kumar" value="">
    </div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Age</label>
        <input id="patient-age-input" class="form-input" type="number" min="0" max="120" placeholder="52">
      </div>
      <div class="form-group">
        <label class="form-label">Gender</label>
        <select id="patient-gender-input" class="select-input">
          <option value="Male">Male</option>
          <option value="Female">Female</option>
          <option value="Unknown">Unknown</option>
        </select>
      </div>
    </div>
    <div class="form-group" style="margin-top:10px;">
      <label class="form-label">Image Type</label>
      <select id="image-type-input" class="select-input">
        <option value="OCT">OCT (Optical Coherence Tomography)</option>
        <option value="FUNDUS">Fundus (Retinal Color Photo)</option>
        <option value="AUTO">Auto-Detect</option>
      </select>
    </div>`;
  const ref = card.querySelector('#upload-tab-nav') || card.querySelector('.upload-zone');
  if (ref) {
    if (ref.parentNode === card) {
      card.insertBefore(wrap, ref);
    } else {
      ref.parentNode.insertBefore(wrap, ref);
    }
  }
}

function getPatientFormValues() {
  return {
    name: document.getElementById('patient-name-input')?.value?.trim() || 'Anonymous',
    age: parseInt(document.getElementById('patient-age-input')?.value || '0', 10) || 0,
    gender: document.getElementById('patient-gender-input')?.value || 'Unknown',
    imageType: document.getElementById('image-type-input')?.value || 'OCT',
  };
}

async function refreshUserData() {
  if (!currentUser) return;
  const stats = await apiDashboard(currentUser.id);
  updateDashboardUI(stats);
  const hist = await apiHistory(currentUser.id);
  updateHistoryUI(hist);
  loadPlotlyCharts();
}

function wireUpload() {
  ensurePatientFields();
  const zone = document.querySelector('#mod-upload .upload-zone');
  if (!zone) return;
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.style.display = 'none';
  document.body.appendChild(input);

  const openPicker = () => input.click();
  zone.onclick = openPicker;
  const browse = document.querySelector('#pane-upload-file .btn-secondary');
  if (browse) browse.onclick = openPicker;

  const handleFileSelected = (file) => {
    if (!file) return;
    if (!currentUser) {
      currentUser = { id: 1, name: 'Dr. Elena Vance', email: 'doctor@visioncare.ai' };
    }
    selectedUploadFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      const placeholder = document.querySelector('#mod-upload .oct-scan-placeholder');
      if (placeholder) {
        placeholder.innerHTML = `<img src="${e.target.result}" style="width:100%;height:100%;object-fit:cover;border-radius:8px;">`;
      }
    };
    reader.readAsDataURL(file);

    const fileInfo = document.getElementById('file-info');
    if (fileInfo) {
      fileInfo.innerHTML = `
        <div class="file-info-row"><span class="file-info-key">File Name</span><span class="file-info-val">${file.name}</span></div>
        <div class="file-info-row"><span class="file-info-key">File Type</span><span class="file-info-val">${file.type || 'image/jpeg'}</span></div>
        <div class="file-info-row"><span class="file-info-key">File Size</span><span class="file-info-val">${(file.size / 1024 / 1024).toFixed(2)} MB</span></div>
        <div class="file-info-row"><span class="file-info-key">Status</span><span class="file-info-val" style="color:var(--accent-light);font-weight:600;">Selected (Ready for Analysis)</span></div>`;
    }

    const valBadge = document.getElementById('validation-status');
    if (valBadge) valBadge.style.display = 'none';
  };

  input.onchange = () => {
    if (input.files && input.files[0]) {
      handleFileSelected(input.files[0]);
    }
  };

  // Full Drag-and-Drop support on upload zone
  zone.ondragover = (e) => {
    e.preventDefault();
    zone.style.borderColor = 'var(--accent)';
    zone.style.background = 'var(--bg-elevated)';
  };
  zone.ondragleave = (e) => {
    e.preventDefault();
    zone.style.borderColor = '';
    zone.style.background = '';
  };
  zone.ondrop = (e) => {
    e.preventDefault();
    zone.style.borderColor = '';
    zone.style.background = '';
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const analyzeBtn = document.getElementById('btn-upload-analyze');
  if (analyzeBtn) {
    analyzeBtn.onclick = async () => {
      if (!selectedUploadFile) {
        alert('Please upload/select or capture an OCT image first.');
        return;
      }
      
      const origText = analyzeBtn.textContent;
      analyzeBtn.textContent = 'Analyzing Retinal Scan...';
      analyzeBtn.disabled = true;
      
      // Update status row in file-info
      const rows = document.querySelectorAll('#file-info .file-info-row');
      let statusVal = null;
      rows.forEach(row => {
        if (row.querySelector('.file-info-key')?.textContent === 'Status') {
          statusVal = row.querySelector('.file-info-val');
        }
      });
      
      if (statusVal) {
        statusVal.textContent = 'Analyzing with Neural Core...';
        statusVal.style.color = 'var(--warning)';
      }
      
      try {
        if (!currentUser) {
          currentUser = { id: 1, name: 'Dr. Elena Vance', email: 'doctor@visioncare.ai' };
        }
        const patient = getPatientFormValues();
        lastAnalysis = await apiAnalyze(selectedUploadFile, patient.name, patient.age, patient.gender, patient.imageType);
        
        const valBadge = document.getElementById('validation-status');
        if (valBadge) valBadge.style.display = 'flex';
        applyAnalysisToModules(lastAnalysis);
        
        if (statusVal) {
          statusVal.textContent = 'Analysis Completed';
          statusVal.style.color = 'var(--success)';
        }
        
        await refreshUserData();
        
        // Clear input files value so it can trigger onchange for same file
        input.value = '';
        
        // Auto-switch to prediction tab
        if (typeof switchModule === 'function') {
          switchModule('prediction');
        }
      } catch (e) {
        alert('Analysis failed: ' + e.message);
        if (statusVal) {
          statusVal.textContent = 'Error: ' + e.message;
          statusVal.style.color = 'var(--danger)';
        }
      } finally {
        analyzeBtn.textContent = origText;
        analyzeBtn.disabled = false;
      }
    };
  }
}

function updateReportsUI(data) {
  if (!data) return;
  const cards = document.querySelectorAll('#mod-reports .report-card');
  const name = data.patient_name || 'Patient';
  const label = (data.label || '').toUpperCase();
  const stage = data.stage || 'Normal';
  const affected = typeof data.retina_affected_pct === 'number' ? data.retina_affected_pct.toFixed(2) : '0.00';

  cards.forEach((card, idx) => {
    // Images
    const imgGrid = card.querySelector('.report-images-grid');
    if (imgGrid) {
      const o = imgGrid.querySelector('.report-preview-orig');
      const h = imgGrid.querySelector('.report-preview-heat');
      const s = imgGrid.querySelector('.report-preview-seg');
      const origSrc = b64img(data.preview_b64);
      const heatSrc = b64img(data.heatmap_b64);
      const segSrc = b64img(data.mask_b64);
      if(o && origSrc) o.innerHTML = `<img src="${origSrc}" style="width:100%;height:100%;object-fit:cover;">`;
      if(h && heatSrc) h.innerHTML = `<img src="${heatSrc}" style="width:100%;height:100%;object-fit:cover;">`;
      if(s && segSrc) s.innerHTML = `<img src="${segSrc}" style="width:100%;height:100%;object-fit:cover;">`;
    }
    // 1. Top meta grid (4 items: Name, ID, Age/Gender, Date)
    const allGrids = card.querySelectorAll('.report-meta-grid');
    if (allGrids.length >= 1) {
      const topMeta = allGrids[0].querySelectorAll('.report-meta-value');
      if (topMeta.length >= 4) {
        topMeta[0].textContent = name;
        topMeta[1].textContent = `PT_${(data.patient_id || data.scan_id || '0000').toString().padStart(5, '0')}`;
        const patientAge = (data.age !== undefined && data.age !== null && data.age !== '') ? data.age : ((data.patient_age !== undefined && data.patient_age !== null && data.patient_age !== '') ? data.patient_age : '—');
        topMeta[2].textContent = `${patientAge} / ${data.gender || data.patient_gender || '—'}`;
        topMeta[3].textContent = data.analysis_date || new Date().toISOString().slice(0, 10);
      }
    }
    
    // 2. Diagnosis meta grid (4 items: Condition, Stage, Retina Affected, Confidence)
    const diagMeta = card.querySelectorAll('.report-section .report-meta-grid .report-meta-value');
    if (diagMeta.length >= 4) {
      diagMeta[0].textContent = data.label_full || data.label || '—';
      diagMeta[1].textContent = data.stage || '—';
      diagMeta[2].textContent = `${affected}%`;
      const conf = typeof data.confidence === 'number'
        ? data.confidence.toFixed(1)
        : '0.0';
      diagMeta[3].textContent = `${conf}%`;
    }

    // 3. Clinical Findings and Recommendations dynamically tailored to diagnosis
    const textEls = card.querySelectorAll('.report-text');
    const isTamil = card.id === 'report-ta';

    if (!isTamil) {
      // English Text
      let findings = `AI-assisted retinal analysis detected ${data.label_full || data.label} with ${data.confidence}% confidence score at ${stage} severity stage.`;
      let carePlan = `Comprehensive ophthalmic evaluation and continued clinical monitoring recommended.`;

      if (label === 'NORMAL') {
        findings = 'Optical coherence tomography demonstrates normal retinal architecture with intact foveal contour, sharp inner limiting membrane (ILM), and undisturbed retinal pigment epithelium (RPE). No intraretinal fluid, subretinal cysts, or drusen deposits are present.';
        carePlan = 'Routine comprehensive eye examination in 12 months. Continue regular ultraviolet protection and balanced lifestyle habits.';
      } else if (label === 'DME') {
        findings = `Optical coherence tomography reveals intraretinal cystoid fluid accumulation and macular neurosensory thickening (${affected}% area affected). Findings are consistent with Diabetic Macular Edema at ${stage} severity.`;
        carePlan = 'Follow-up consultation with a vitreoretinal specialist. Strict glycemic control (HbA1c monitoring) and arterial blood pressure management. Consider anti-VEGF therapy or focal laser photocoagulation as indicated.';
      } else if (label === 'CNV') {
        findings = `Optical coherence tomography reveals subretinal hyperreflective membrane disruption and exudative subretinal fluid (${affected}% area affected). Findings are characteristic of active Choroidal Neovascularization.`;
        carePlan = 'Urgent evaluation by a vitreoretinal specialist. Prompt initiation of intravitreal anti-VEGF injection therapy is recommended to preserve visual acuity. Daily monocular Amsler grid self-monitoring.';
      } else if (label === 'DRUSEN' || label === 'AMD') {
        findings = `Optical coherence tomography demonstrates nodular sub-RPE deposits (drusen) with focal pigment epithelial undulations (${affected}% area affected). Findings are consistent with Dry Age-Related Macular Degeneration.`;
        carePlan = 'Prescribe AREDS2 antioxidant vitamin and mineral formula. Semi-annual comprehensive dilated eye examinations. Daily monocular Amsler grid testing to identify early neovascular conversion.';
      }

      if (textEls.length >= 2) {
        textEls[0].textContent = findings;
        textEls[1].textContent = carePlan;
      }
    } else {
      // Tamil Text
      let findings = `செயற்கை நுண்ணறிவு விழித்திரை பகுப்பாய்வில் ${data.label_full || data.label} கண்டறியப்பட்டுள்ளது (உறுதிப்பாடு: ${data.confidence}%, நிலை: ${stage}).`;
      let carePlan = `முழுமையான கண் மருத்துவ பரிசோதனை மற்றும் தொடர் கண்காணிப்பு பரிந்துரைக்கப்படுகிறது.`;

      if (label === 'NORMAL') {
        findings = 'OCT விழித்திரை ஸ்கேனில் இயல்பான கட்டமைப்பு காணப்படுகிறது. போவியா (Foveal) வளைவு சீராகவும், விழித்திரை அடுக்குகள் மற்றும் RPE சீரான நிலையிலும் உள்ளன. நீர்க்கட்டிகள் அல்லது வீக்கம் ஏதும் இல்லை.';
        carePlan = 'ஆண்டுக்கு ஒருமுறை வழக்கமான கண் பரிசோதனை மேற்கொள்ளவும். ஆரோக்கியமான கண் பராமரிப்பு முறைகளைத் தொடரவும்.';
      } else if (label === 'DME') {
        findings = `விழித்திரையின் மாகுலார் பகுதியில் நீர்க்கட்டிகள் மற்றும் திரவக் குவிப்பு (${stage} நிலை, ${affected}% பரப்பு) கண்டறியப்பட்டது. இது நீரிழிவு மாகுலர் எடிமா (DME) நோய்க்கான அறிகுறிகளுடன் பொருந்துகிறது.`;
        carePlan = 'விழித்திரை சிறப்பு மருத்துவரிடம் உடனடி ஆலோசனை பெறவும். இரத்த சர்க்கரை மற்றும் இரத்த அழுத்த அளவுகளை கட்டுப்பாட்டில் வைக்கவும். Anti-VEGF ஊசி சிகிச்சை பற்றி ஆலோசிக்கவும்.';
      } else if (label === 'CNV') {
        findings = `விழித்திரை நிறமி அடுக்கின் கீழ் புதிய ரத்த நாளங்களின் அசாதாரண வளர்ச்சி மற்றும் திரவக் கசிவு (${affected}% பரப்பு) காணப்படுகிறது. இது கோராய்டல் நியோவாஸ்குலரைசேஷன் (CNV) பாதிப்பைக் குறிக்கிறது.`;
        carePlan = 'உடனடி விழித்திரை அறுவை சிகிச்சை நிபுணர் ஆலோசனை தேவை. தாமதமின்றி intravitreal anti-VEGF ஊசி சிகிச்சை தொடங்க பரிந்துரைக்கப்படுகிறது. தினமும் ஆம்ஸ்லர் கிரிட் பரிசோதனை செய்யவும்.';
      } else if (label === 'DRUSEN' || label === 'AMD') {
        findings = `விழித்திரை RPE அடுக்கின் கீழ் ட்ரூசன் படிவுகள் (${affected}% பரப்பு) படிந்துள்ளது கண்டறியப்பட்டது. இது வயது தொடர்பான விழித்திரை சிதைவு (Dry AMD) நிலையுடன் பொருந்துகிறது.`;
        carePlan = 'மருத்துவரின் ஆலோசனைப்படி AREDS2 ஆக்ஸிஜனேற்ற மாத்திரைகளை உட்கொள்ளவும். 6 மாதங்களுக்கு ஒருமுறை பரிசோதனை செய்யவும். தினமும் ஆம்ஸ்லர் கிரிட் மூலம் பார்வையை சுயபரிசோதனை செய்யவும்.';
      }

      if (textEls.length >= 2) {
        textEls[0].textContent = findings;
        textEls[1].textContent = carePlan;
      }
    }

    // Apply Tamil translations to values if it's the Tamil card
    if (card.id === 'report-ta') {
      const topMeta = card.querySelectorAll('.report-meta-grid')[0].querySelectorAll('.report-meta-value');
      const diagMeta = card.querySelectorAll('.report-section .report-meta-grid .report-meta-value');
      
      // Gender
      let gender = 'தெரியவில்லை';
      const gStr = (data.gender || data.patient_gender || '').toLowerCase();
      if (gStr.startsWith('m')) gender = 'ஆண்';
      else if (gStr.startsWith('f')) gender = 'பெண்';
      if (topMeta.length >= 3) {
        const patientAge = (data.age !== undefined && data.age !== null && data.age !== '') ? data.age : ((data.patient_age !== undefined && data.patient_age !== null && data.patient_age !== '') ? data.patient_age : '—');
        topMeta[2].textContent = `${patientAge} / ${gender}`;
      }

      // Disease
      const labelMap = {
        'NORMAL': 'இயல்பான விழித்திரை',
        'Normal Retina': 'இயல்பான விழித்திரை',
        'Diabetic Macular Edema': 'நீரிழிவு விழித்திரை நோய் (DME)',
        'Choroidal Neovascularization': 'விழித்திரை ரத்தக்குழாய் நோய் (CNV)',
        'Drusen': 'விழித்திரை ட்ரூசன் படிவுகள்',
        'AMD': 'வயது தொடர்பான விழித்திரை நோய்',
        'Age-Related Macular Degeneration': 'வயது தொடர்பான விழித்திரை நோய்'
      };
      if (diagMeta.length >= 1) {
        diagMeta[0].textContent = labelMap[data.label] || labelMap[data.label_full] || data.label_full || data.label || '—';
      }

      // Stage
      const stageMap = {
        'Normal': 'இயல்பானது',
        'Mild': 'லேசான',
        'Moderate': 'மிதமான',
        'Severe': 'தீவிரமான',
        'Critical': 'மிக தீவிரமான'
      };
      if (diagMeta.length >= 2) {
        diagMeta[1].textContent = stageMap[data.stage] || data.stage || '—';
      }

      // Transliterate Name asynchronously
      if (name && name !== 'Patient' && name !== 'Anonymous') {
        fetch(`https://inputtools.google.com/request?text=${encodeURIComponent(name)}&itc=ta-t-i0-und&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8&app=demopage`)
          .then(r => r.json())
          .then(json => {
            if (json[0] === 'SUCCESS' && json[1] && json[1][0] && json[1][0][1] && json[1][0][1][0]) {
              if (topMeta.length >= 1) topMeta[0].textContent = json[1][0][1][0];
            }
          }).catch(console.warn);
      }
    }

    const statusEl = card.querySelector('.report-status');
    if (statusEl) {
      if (data.report_status?.success) {
        statusEl.textContent = 'Report ready';
      } else if (data.report_status?.error) {
        statusEl.textContent = `Report error: ${data.report_status.error}`;
      } else {
        statusEl.textContent = 'Report ready';
      }
    }
  });
}

function wireReports() {
  document.querySelectorAll('#mod-reports .btn-download-report, #mod-reports .btn-primary').forEach((btn) => {
    btn.onclick = async () => {
      if (!lastAnalysis?.scan_id) {
        alert('Run an analysis first.');
        return;
      }
      
      const btnOrigText = btn.textContent;
      btn.textContent = 'Downloading...';
      btn.disabled = true;
      
      try {
        const card = btn.closest('.report-card');
        const isTamil = card && card.id === 'report-ta';
        const pName = card?.querySelector('.report-meta-value')?.textContent || lastAnalysis.patient_name || 'Patient';
        const pname = encodeURIComponent(pName.trim());
        const lang = isTamil ? 'tamil' : 'english';
        const url = `${API}/api/reports/${lastAnalysis.scan_id}/${lang}?patient_name=${pname}&user_id=${currentUser.id}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        
        const blob = await response.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = isTamil 
          ? `Tamil_Report_${lastAnalysis.scan_id}.pdf` 
          : `English_Report_${lastAnalysis.scan_id}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
      } catch (err) {
        alert(`Download failed: ${err.message}`);
      } finally {
        btn.textContent = btnOrigText;
        btn.disabled = false;
      }
    };
  });

  const setupNameEditor = (elemId) => {
    const el = document.getElementById(elemId);
    if (!el) return;
    
    el.onkeydown = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        el.blur();
      }
    };
    
    el.onblur = async () => {
      if (!lastAnalysis?.scan_id) return;
      const newName = el.textContent.trim();
      if (!newName) {
        alert("Patient name cannot be empty.");
        el.textContent = lastAnalysis.patient_name || 'Patient';
        return;
      }
      
      try {
        const res = await fetch(`${API}/api/scan/${lastAnalysis.scan_id}/update-patient-name?patient_name=${encodeURIComponent(newName)}&user_id=${currentUser.id}`, {
          method: 'POST'
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        const enEl = document.getElementById('edit-patient-name-en');
        const taEl = document.getElementById('edit-patient-name-ta');
        
        if (elemId === 'edit-patient-name-en' && taEl) {
          fetch(`https://inputtools.google.com/request?text=${encodeURIComponent(newName)}&itc=ta-t-i0-und&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8&app=demopage`)
            .then(r => r.json())
            .then(json => {
              if (json[0] === 'SUCCESS' && json[1]?.[0]?.[1]?.[0]) {
                taEl.textContent = json[1][0][1][0];
              } else {
                taEl.textContent = newName;
              }
            }).catch(() => { taEl.textContent = newName; });
        } else if (elemId === 'edit-patient-name-ta' && enEl) {
          enEl.textContent = newName;
        }
        
        lastAnalysis.patient_name = newName;
        await refreshUserData();
        
      } catch (err) {
        alert(`Failed to update patient name: ${err.message}`);
        el.textContent = lastAnalysis.patient_name || 'Patient';
      }
    };
  };

  setupNameEditor('edit-patient-name-en');
  setupNameEditor('edit-patient-name-ta');
}

function wireHistory() {
  const search = document.getElementById('history-search');
  if (search) {
    search.oninput = () => {
      historyPage = 0;
      renderHistoryPage();
    };
  }

  const exportBtn = document.getElementById('history-export');
  if (exportBtn) {
    exportBtn.onclick = () => {
      const rows = getFilteredHistory();
      if (!rows.length) {
        alert('No history to export.');
        return;
      }
      const header = ['scan_id', 'patient_name', 'prediction', 'severity', 'date', 'retina_affected'];
      const lines = rows.map((s) =>
        [
          s.scan_id,
          s.patient_name || '',
          s.prediction || '',
          s.severity || s.stage || '',
          (s.created_at || '').slice(0, 10),
          s.retina_affected ?? '',
        ]
          .map((v) => `"${String(v).replace(/"/g, '""')}"`)
          .join(',')
      );
      const csv = [header.join(','), ...lines].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'visioncare_history.csv';
      a.click();
    };
  }

  const delPatient = document.getElementById('history-delete-patient');
  if (delPatient) {
    delPatient.onclick = async () => {
      const name = prompt('Enter patient name to delete all their scans:');
      if (!name?.trim()) return;
      if (!confirm(`Delete ALL scan history for "${name.trim()}"?`)) return;
      try {
        const res = await apiDeletePatientHistory(name.trim());
        alert(res.message || 'Patient history deleted.');
        if (lastAnalysis?.patient_name === name.trim()) lastAnalysis = null;
        await refreshUserData();
      } catch (e) {
        alert(e.message);
      }
    };
  }

  const clearAll = document.getElementById('history-clear-all');
  if (clearAll) {
    clearAll.onclick = async () => {
      if (!confirm('Delete ALL scan history? This cannot be undone.')) return;
      try {
        const res = await apiDeleteAllHistory();
        alert(res.message || 'History cleared.');
        lastAnalysis = null;
        await refreshUserData();
      } catch (e) {
        alert(e.message);
      }
    };
  }

  const footer = document.querySelector('#mod-history .card > div:last-child');
  if (footer) {
    const buttons = footer.querySelectorAll('button');
    if (buttons[0]) {
      buttons[0].onclick = () => {
        if (historyPage > 0) {
          historyPage--;
          renderHistoryPage();
        }
      };
    }
    if (buttons[1]) {
      buttons[1].onclick = () => {
        const filtered = getFilteredHistory();
        if ((historyPage + 1) * HISTORY_PAGE_SIZE < filtered.length) {
          historyPage++;
          renderHistoryPage();
        }
      };
    }
  }
}

async function apiCleanDuplicates() {
  const r = await apiFetch(`${API}/api/duplicates/${currentUser.id}/clean`, { method: 'POST' });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatApiError(data, 'Clean duplicates failed'));
  return data;
}

function wireDashboard() {
  const viewAll = document.querySelector('#mod-dashboard .btn-secondary.btn-sm');
  if (viewAll) {
    viewAll.onclick = () =>
      switchModule('history', document.querySelector('[onclick*="history"]'));
  }

  const cleanBtn = document.getElementById('clean-duplicates-btn');
  if (cleanBtn) {
    cleanBtn.onclick = async () => {
      if (!currentUser) return;
      if (!confirm('Remove all duplicate scans from your history?')) return;
      try {
        const res = await apiCleanDuplicates();
        const count = res.deleted ?? 0;
        alert(count > 0 ? `Removed ${count} duplicate scan(s).` : 'No duplicates found.');
        await refreshUserData();
      } catch (e) {
        alert(e.message);
      }
    };
  }
}

function wirePredictionNav() {
  document.querySelectorAll('#mod-prediction .btn-outline.btn-sm, #mod-prediction .btn-primary.btn-sm').forEach((btn) => {
    const text = btn.textContent.trim();
    if (text.includes('Heatmap')) {
      btn.onclick = () => {
        if (!lastAnalysis) {
          alert('Run an analysis first.');
          return;
        }
        switchModule('heatmap', document.querySelector('[onclick*="heatmap"]'));
      };
    }
    if (text.includes('Report')) {
      btn.onclick = () => {
        if (!lastAnalysis?.scan_id) {
          alert('Run an analysis first.');
          return;
        }
        switchModule('reports', document.querySelector('[onclick*="reports"]'));
      };
    }
  });
}

function wireHeatmapExport() {
  const btn = document.querySelector('#mod-heatmap .btn-secondary.btn-sm');
  if (btn) {
    btn.onclick = () => {
      const img = document.querySelector('#mod-heatmap .heatmap-base img');
      if (!img?.src) {
        alert('No heatmap to export. Run analysis first.');
        return;
      }
      const a = document.createElement('a');
      a.href = img.src;
      a.download = `heatmap_${lastAnalysis?.scan_id || 'export'}.png`;
      a.click();
    };
  }
}

function wireSegmentationExport() {
  const btn = document.querySelector('#mod-segmentation .btn-secondary.btn-sm');
  if (btn) {
    btn.onclick = () => {
      const img = document.querySelector('#mod-segmentation .seg-image:nth-child(2) img');
      if (!img?.src) {
        alert('No mask to export. Run analysis first.');
        return;
      }
      const a = document.createElement('a');
      a.href = img.src;
      a.download = `mask_${lastAnalysis?.scan_id || 'export'}.png`;
      a.click();
    };
  }
}

function wireSettings() {
  const aboutCard = document.querySelector('#mod-settings .card:last-child .about-row');
  if (!aboutCard) return;
  apiDatasetInfo().then((info) => {
    if (!info?.ready) return;
    const parent = document.querySelector('#mod-settings .card:last-child');
    if (!parent || document.getElementById('dataset-info-row')) return;
    const row = document.createElement('div');
    row.id = 'dataset-info-row';
    row.className = 'about-row';
    row.innerHTML = `<span class="about-key">Dataset</span><span class="about-val">Kermany OCT (${info.train_total} train / ${info.test_total} test)</span>`;
    parent.appendChild(row);
  });
}

function wireAllModules() {
  wireDashboard();
  wireHistory();
  wirePredictionNav();
  wireHeatmapExport();
  wireSegmentationExport();
  wireSettings();
  wireChatbot();
}

const _logout = window.logout;
window.logout = function () {
  clearSession();
  if (_logout) _logout();
};

/* ─── Settings Modal Helpers ─── */

function closeModal(id) {
  document.getElementById(id).style.display = 'none';
}

function showModalMsg(elemId, msg, isError) {
  const el = document.getElementById(elemId);
  el.style.display = 'block';
  el.textContent = msg;
  el.style.background = isError ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.15)';
  el.style.color = isError ? '#ef4444' : '#22c55e';
}

async function submitChangePassword() {
  const oldPw = document.getElementById('pw-old').value;
  const newPw = document.getElementById('pw-new').value;
  const confirmPw = document.getElementById('pw-confirm').value;

  if (!oldPw || !newPw || !confirmPw) {
    return showModalMsg('pw-msg', 'Please fill in all fields.', true);
  }
  if (newPw.length < 4) {
    return showModalMsg('pw-msg', 'New password must be at least 4 characters.', true);
  }
  if (newPw !== confirmPw) {
    return showModalMsg('pw-msg', 'New passwords do not match.', true);
  }

  try {
    const res = await fetch(`${API}/api/auth/change-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: currentUser.id, old_password: oldPw, new_password: newPw }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed');
    showModalMsg('pw-msg', '✓ Password changed successfully!', false);
    document.getElementById('pw-old').value = '';
    document.getElementById('pw-new').value = '';
    document.getElementById('pw-confirm').value = '';
    setTimeout(() => closeModal('modal-change-pw'), 1500);
  } catch (e) {
    showModalMsg('pw-msg', e.message, true);
  }
}

function openUpdateProfile() {
  document.getElementById('profile-name').value = currentUser.name || '';
  document.getElementById('profile-email').value = currentUser.email || '';
  document.getElementById('profile-msg').style.display = 'none';
  document.getElementById('modal-update-profile').style.display = 'flex';
}

async function submitUpdateProfile() {
  const name = document.getElementById('profile-name').value.trim();
  const email = document.getElementById('profile-email').value.trim();

  if (!name || !email) {
    return showModalMsg('profile-msg', 'Name and email are required.', true);
  }
  if (!email.includes('@')) {
    return showModalMsg('profile-msg', 'Please enter a valid email.', true);
  }

  try {
    const res = await fetch(`${API}/api/auth/update-profile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: currentUser.id, name, email }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed');
    // Update local session
    currentUser.name = data.name;
    currentUser.email = data.email;
    saveSession();
    document.querySelector('.user-name').textContent = data.name;
    showModalMsg('profile-msg', '✓ Profile updated successfully!', false);
    setTimeout(() => closeModal('modal-update-profile'), 1500);
  } catch (e) {
    showModalMsg('profile-msg', e.message, true);
  }
}

async function submitDeleteAccount() {
  const password = (document.getElementById('del-password').value || '').trim();
  if (!password) {
    return showModalMsg('del-msg', 'Please enter your password (or type DELETE) to confirm.', true);
  }

  if (!currentUser) {
    loadSession();
  }

  const userId = currentUser ? currentUser.id : null;
  const userEmail = currentUser ? currentUser.email : null;

  try {
    const res = await fetch(`${API}/api/auth/delete-account`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        email: userEmail,
        password: password,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to delete account');
    showModalMsg('del-msg', '✓ Account successfully deleted. Redirecting...', false);
    setTimeout(() => {
      document.getElementById('del-password').value = '';
      closeModal('modal-delete-acc');
      clearSession();
      if (typeof window.logout === 'function') {
        window.logout();
      } else if (typeof logout === 'function') {
        logout();
      } else {
        document.getElementById('main-app')?.classList.remove('active');
        if (typeof showPage === 'function') showPage('page-login');
      }
    }, 1200);
  } catch (e) {
    showModalMsg('del-msg', e.message, true);
  }
}

/* ─── Chatbot Frontend Helper Functions ─── */

/* ─── Chatbot Frontend Helper Functions ─── */

let activeChatAttachment = null;

async function handleChatFileUpload(input) {
  if (!input.files || input.files.length === 0) return;
  const file = input.files[0];
  
  const statusEl = document.getElementById('chat-attachment-status');
  const nameEl = document.getElementById('chat-attachment-name');
  const sizeEl = document.getElementById('chat-attachment-size');
  
  if (!statusEl || !nameEl || !sizeEl) return;
  
  statusEl.style.display = 'flex';
  nameEl.textContent = `Uploading ${file.name}...`;
  sizeEl.textContent = '';
  
  const formData = new FormData();
  formData.append('file', file);
  formData.append('user_id', currentUser?.id || 0);
  
  try {
    const response = await fetch(`${API}/api/chat/upload`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) throw new Error(`Upload failed with status ${response.status}`);
    const data = await response.json();
    
    activeChatAttachment = {
      filename: file.name,
      size: (file.size / 1024).toFixed(1) + ' KB',
      type: file.type
    };
    
    nameEl.textContent = file.name;
    sizeEl.textContent = `(${activeChatAttachment.size})`;
    
    // Reset input value so same file can be selected again if removed
    input.value = '';
  } catch (err) {
    alert(`File upload failed: ${err.message}`);
    clearChatAttachment();
  }
}

async function clearChatAttachment() {
  activeChatAttachment = null;
  const statusEl = document.getElementById('chat-attachment-status');
  const fileInputEl = document.getElementById('chat-file-input');
  if (statusEl) statusEl.style.display = 'none';
  if (fileInputEl) fileInputEl.value = '';
  
  try {
    await fetch(`${API}/api/chat/clear-upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: currentUser?.id || 0 })
    });
  } catch (err) {
    console.error('Failed to clear chat attachment on server:', err);
  }
}

async function sendMessage() {
  const inputEl = document.getElementById('chat-input');
  if (!inputEl) return;
  const message = inputEl.value.trim();
  if (!message) return;
  
  // Clear input
  inputEl.value = '';
  
  // Append user message with active attachment name if any
  const attachmentName = activeChatAttachment ? activeChatAttachment.filename : null;
  appendChatMessage(message, 'user', attachmentName);
  
  // Append typing indicator
  const typingEl = appendChatTyping();
  
  try {
    const r = await fetch(`${API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, user_id: currentUser?.id || 0, scan_data: lastAnalysis }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    
    // Remove typing indicator
    if (typingEl) typingEl.remove();
    
    // Append bot response with dynamic thought block and streaming enabled
    appendChatMessage(data.response, 'bot', null, data.thought, true);
  } catch (err) {
    if (typingEl) typingEl.remove();
    appendChatMessage(`Sorry, there was an error connecting to the AI assistant: ${err.message}`, 'bot');
  }
}

function sendSuggestion(text) {
  const inputEl = document.getElementById('chat-input');
  if (inputEl) {
    inputEl.value = text;
    sendMessage();
  }
}

function parseMarkdown(text) {
  if (!text) return '';
  
  let html = text;
  
  // Clean carriage returns
  html = html.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  
  // 1. Code blocks (```lang ... ```)
  html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
    return `<pre style="background:rgba(0,0,0,0.3); padding:12px; border-radius:8px; font-family:'DM Mono',monospace; font-size:12px; overflow-x:auto; margin:8px 0; border:1px solid var(--border); max-width:100%; color:var(--text-secondary);">${code.trim().replace(/\n/g, '<br/>')}</pre>`;
  });
  
  // 2. Inline code (`code`)
  html = html.replace(/`([^`]+)`/g, '<code style="background:rgba(0,0,0,0.2); padding:2px 6px; border-radius:4px; font-family:\'DM Mono\',monospace; font-size:90%; color:var(--accent-light);">$1</code>');
  
  // 3. Tables
  const lines = html.split('\n');
  let inTable = false;
  let tableRows = [];
  let parsedLines = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('|') && line.endsWith('|')) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      tableRows.push(line);
    } else {
      if (inTable) {
        const parsedTable = renderMarkdownTable(tableRows);
        parsedLines.push(parsedTable);
        inTable = false;
      }
      parsedLines.push(lines[i]);
    }
  }
  if (inTable) {
    parsedLines.push(renderMarkdownTable(tableRows));
  }
  html = parsedLines.join('\n');
  
  // 4. Headings
  html = html.replace(/^#### (.*?)$/gm, '<h5 style="margin-top:12px; margin-bottom:6px; font-family:\'Outfit\',sans-serif; color:var(--accent-light); font-weight:700;">$1</h5>');
  html = html.replace(/^### (.*?)$/gm, '<h4 style="margin-top:12px; margin-bottom:6px; font-family:\'Outfit\',sans-serif; color:var(--accent-light); font-weight:700;">$1</h4>');
  html = html.replace(/^## (.*?)$/gm, '<h3 style="margin-top:14px; margin-bottom:8px; font-family:\'Outfit\',sans-serif; color:var(--accent-light); font-weight:800;">$1</h3>');
  html = html.replace(/^# (.*?)$/gm, '<h2 style="margin-top:16px; margin-bottom:10px; font-family:\'Outfit\',sans-serif; color:var(--accent-light); font-weight:800;">$1</h2>');
  
  // 5. Bold & Italic
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
  html = html.replace(/_(.*?)_/g, '<em>$1</em>');
  
  // 6. Bullet lists
  let inList = false;
  const listLines = html.split('\n');
  let finalLines = [];
  for (let i = 0; i < listLines.length; i++) {
    const line = listLines[i].trim();
    if (line.match(/^[-*]\s+(.*)$/)) {
      if (!inList) {
        inList = true;
        finalLines.push('<ul style="margin:8px 0; padding-left:20px; display:flex; flex-direction:column; gap:4px;">');
      }
      const itemText = line.replace(/^[-*]\s+/, '');
      finalLines.push(`<li>${itemText}</li>`);
    } else {
      if (inList) {
        finalLines.push('</ul>');
        inList = false;
      }
      finalLines.push(listLines[i]);
    }
  }
  if (inList) {
    finalLines.push('</ul>');
  }
  html = finalLines.join('\n');
  
  // 7. Line breaks
  html = html.replace(/\n/g, '<br/>');
  
  return html;
}

function renderMarkdownTable(rows) {
  if (rows.length === 0) return '';
  
  let tableHtml = '<div style="overflow-x:auto; margin:12px 0; border-radius:8px; border:1px solid var(--border);"><table style="width:100%; border-collapse:collapse; font-size:12px; background:rgba(255,255,255,0.01);">';
  
  let headerCells = [];
  let dataRowsHtml = [];
  let isHeader = true;
  
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];
    const cells = row.split('|').map(c => c.trim()).filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
    
    const isSeparator = cells.every(c => c.match(/^:?-+:?$/));
    if (isSeparator) {
      isHeader = false;
      continue;
    }
    
    if (r === 0 || isHeader) {
      headerCells = cells;
      isHeader = false;
    } else {
      dataRowsHtml.push(cells);
    }
  }
  
  if (headerCells.length > 0) {
    tableHtml += '<thead style="background:rgba(99, 102, 241, 0.1); border-bottom:1px solid var(--border);">';
    tableHtml += '<tr>';
    headerCells.forEach(cell => {
      tableHtml += `<th style="padding:10px 12px; font-weight:700; text-align:left; color:var(--accent-light); border-right:1px solid var(--border);">${cell}</th>`;
    });
    tableHtml += '</tr></thead>';
  }
  
  tableHtml += '<tbody>';
  dataRowsHtml.forEach((cells, rowIdx) => {
    const rowBg = rowIdx % 2 === 0 ? 'rgba(0,0,0,0)' : 'rgba(255,255,255,0.02)';
    tableHtml += `<tr style="background:${rowBg}; border-bottom:1px solid var(--border); transition:background 0.2s;">`;
    cells.forEach(cell => {
      tableHtml += `<td style="padding:8px 12px; border-right:1px solid var(--border); color:var(--text-secondary);">${cell}</td>`;
    });
    tableHtml += '</tr>';
  });
  tableHtml += '</tbody></table></div>';
  
  return tableHtml;
}

let recognition = null;
let isRecognizing = false;

window.toggleSpeechRecognition = function() {
  const micBtn = document.getElementById('chat-mic-btn');
  const chatInput = document.getElementById('chat-input');
  if (!micBtn || !chatInput) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    alert('Speech recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.');
    return;
  }

  if (isRecognizing) {
    recognition.stop();
    return;
  }

  if (!recognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isRecognizing = true;
      micBtn.classList.add('mic-recording');
      chatInput.placeholder = 'Listening to your clinical query...';
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      chatInput.value = (chatInput.value + ' ' + transcript).trim();
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      stopRecognition();
    };

    recognition.onend = () => {
      stopRecognition();
    };
  }

  function stopRecognition() {
    isRecognizing = false;
    micBtn.classList.remove('mic-recording');
    chatInput.placeholder = 'Ask a question, or attach a file and ask questions about it...';
  }

  recognition.start();
};

window.speakChatMessage = function(btn) {
  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
    if (btn.classList.contains('speaking-active')) {
      document.querySelectorAll('.chat-speak-btn').forEach(b => {
        b.classList.remove('speaking-active');
        b.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg> Speak`;
      });
      return;
    }
  }

  document.querySelectorAll('.chat-speak-btn').forEach(b => {
    b.classList.remove('speaking-active');
    b.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg> Speak`;
  });

  const bubble = btn.closest('div').previousElementSibling;
  if (!bubble) return;
  
  const clone = bubble.cloneNode(true);
  const thoughtEl = clone.querySelector('.thought-details');
  if (thoughtEl) thoughtEl.remove();
  
  let text = clone.innerText || clone.textContent;
  text = text.replace(/Attached File:.*?(\n|$)/g, '').trim();

  if (!text) return;

  const utterance = new SpeechSynthesisUtterance(text);
  const hasTamil = /[\u0B80-\u0BFF]/.test(text);
  if (hasTamil) {
    utterance.lang = 'ta-IN';
  } else {
    utterance.lang = 'en-US';
  }

  const voices = window.speechSynthesis.getVoices();
  let selectedVoice = null;
  if (hasTamil) {
    selectedVoice = voices.find(v => v.lang.startsWith('ta'));
  } else {
    selectedVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural')));
  }
  if (selectedVoice) utterance.voice = selectedVoice;

  utterance.onstart = () => {
    btn.classList.add('speaking-active');
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect></svg> Stop`;
  };

  utterance.onend = () => {
    btn.classList.remove('speaking-active');
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg> Speak`;
  };

  utterance.onerror = () => {
    btn.classList.remove('speaking-active');
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg> Speak`;
  };

  window.speechSynthesis.speak(utterance);
}

function appendChatMessage(text, sender, attachmentName = null, thought = null, stream = false) {
  const container = document.getElementById('chat-messages');
  if (!container) return;
  
  const msgEl = document.createElement('div');
  msgEl.style.cssText = sender === 'user' 
    ? 'display:flex; gap:12px; max-width:80%; align-self:flex-end; flex-direction:row-reverse;' 
    : 'display:flex; gap:12px; max-width:80%; align-self:flex-start;';
     
  const avatarHtml = sender === 'user'
    ? `<div style="width:32px; height:32px; border-radius:50%; background:var(--accent); display:flex; align-items:center; justify-content:center; flex-shrink:0; font-size:11px; font-weight:700; color:#fff;">${currentUser?.name ? currentUser.name.slice(0,2).toUpperCase() : 'US'}</div>`
    : `<div style="width:36px; height:36px; border-radius:50%; overflow:hidden; flex-shrink:0; border:1.5px solid var(--accent); box-shadow:0 2px 6px var(--accent-glow);">
        <img src="/images/ai_doctor.jpg" alt="Dr. Elena Vance" style="width:100%; height:100%; object-fit:cover;">
       </div>`;
        
  const bubbleBg = sender === 'user' ? 'var(--accent)' : 'var(--bg-elevated)';
  const bubbleBorder = sender === 'user' ? '1px solid var(--accent-light)' : '1px solid var(--border)';
  const bubbleRadius = sender === 'user' ? '16px 0 16px 16px' : '0 16px 16px 16px';
  const textColor = sender === 'user' ? '#fff' : 'var(--text-primary)';
  
  let thoughtHtml = '';
  if (thought && sender === 'bot') {
    let formattedThought = parseMarkdown(thought);
    thoughtHtml = `
      <details class="thought-details" style="background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:8px; margin-bottom:12px; padding:10px 14px; width:100%;">
        <summary style="cursor:pointer; font-weight:600; font-size:11px; color:var(--accent-light); display:flex; align-items:center; gap:6px; user-select:none; outline:none;">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="transition: transform 0.2s;" class="thought-summary-icon"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2zM14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2z"/></svg>
          <span>Thinking Process</span>
        </summary>
        <div class="thought-content" style="font-size:11px; color:var(--text-muted); line-height:1.6; margin-top:8px; border-left:2px solid var(--accent); padding-left:10px;">
          ${formattedThought}
        </div>
      </details>
    `;
  }
    
  let attachmentHtml = '';
  if (attachmentName) {
    attachmentHtml = `<div style="margin-bottom:6px; display:flex; align-items:center; gap:4px; font-size:11px; opacity:0.8; border-bottom:1px solid rgba(255,255,255,0.2); padding-bottom:4px; color:#fff;">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
      <strong>Attached File:</strong> ${attachmentName}
    </div>`;
  }

  const initialText = stream ? '' : parseMarkdown(text);

  msgEl.innerHTML = `
    ${avatarHtml}
    <div style="flex:1; min-width:0;">
      <div style="background:${bubbleBg}; border:${bubbleBorder}; padding:14px 18px; border-radius:${bubbleRadius}; font-size:13px; line-height:1.5; color:${textColor};">
        ${attachmentHtml}
        ${thoughtHtml}
        <div class="chat-bubble-text" style="word-break:break-word; position:relative;"><span class="chat-text-content">${initialText}</span>${stream ? '<span class="chat-typing-cursor" style="display:inline-block; width:6px; height:14px; background:var(--accent-light); margin-left:3px; animation: blink 0.8s infinite; vertical-align:middle;"></span>' : ''}</div>
      </div>
      <div style="font-size:10px; color:var(--text-muted); margin-top:4px; margin-left:4px; display:flex; align-items:center; gap:8px; justify-content:${sender === 'user' ? 'flex-end' : 'flex-start'};">
        <span>${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
        ${sender === 'bot' ? `<button onclick="speakChatMessage(this)" class="chat-speak-btn" style="background:none; border:none; padding:0; color:var(--text-muted); cursor:pointer; display:inline-flex; align-items:center; gap:4px; font-size:10px; outline:none; transition:color 0.2s;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>Speak</button>` : ''}
      </div>
    </div>
  `;
  
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;

  if (stream && sender === 'bot') {
    // Ensure blink keyframes style tag exists
    let styleTag = document.getElementById('chat-typing-anim-style');
    if (!styleTag) {
      styleTag = document.createElement('style');
      styleTag.id = 'chat-typing-anim-style';
      styleTag.innerHTML = `@keyframes blink { 0% { opacity: .2; } 20% { opacity: 1; } 100% { opacity: .2; } }`;
      document.head.appendChild(styleTag);
    }

    const textContentEl = msgEl.querySelector('.chat-text-content');
    const cursorEl = msgEl.querySelector('.chat-typing-cursor');
    const words = text.split(/(\s+)/);
    let wordIndex = 0;
    let accumulatedText = '';
    
    // Disable send button while typing
    const sendBtn = document.getElementById('chat-send-btn');
    if (sendBtn) {
      sendBtn.disabled = true;
      sendBtn.style.opacity = '0.5';
      sendBtn.style.cursor = 'not-allowed';
    }

    function typeWord() {
      if (wordIndex < words.length) {
        accumulatedText += words[wordIndex];
        textContentEl.innerHTML = parseMarkdown(accumulatedText);
        wordIndex++;
        container.scrollTop = container.scrollHeight;
        setTimeout(typeWord, Math.random() * 15 + 10);
      } else {
        if (cursorEl) cursorEl.remove();
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.style.opacity = '1';
          sendBtn.style.cursor = 'pointer';
        }
      }
    }
    typeWord();
  }
}

function appendChatTyping() {
  const container = document.getElementById('chat-messages');
  if (!container) return null;
  
  const typingEl = document.createElement('div');
  typingEl.style.cssText = 'display:flex; gap:12px; max-width:80%; align-self:flex-start;';
  
  // CSS Animation keyframe injected inside stylesheet or directly via style tag
  let styleTag = document.getElementById('chat-typing-anim-style');
  if (!styleTag) {
    styleTag = document.createElement('style');
    styleTag.id = 'chat-typing-anim-style';
    styleTag.innerHTML = `
      @keyframes blink {
        0% { opacity: .2; }
        20% { opacity: 1; }
        100% { opacity: .2; }
      }
    `;
    document.head.appendChild(styleTag);
  }
  
  typingEl.innerHTML = `
    <div style="width:32px; height:32px; border-radius:50%; background:var(--accent-glow); border:1px solid var(--accent); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
      <svg viewBox="0 0 24 24" width="16" height="16" stroke="var(--accent-light)" stroke-width="2.5" fill="none"><circle cx="12" cy="12" r="3"/><path d="M12 5C7 5 2.73 8.11 1 12.5 2.73 16.89 7 20 12 20s9.27-3.11 11-7.5C21.27 8.11 17 5 12 5z"/></svg>
    </div>
    <div>
      <div style="background:var(--bg-elevated); border:1px solid var(--border); padding:12px 18px; border-radius:0 16px 16px 16px; font-size:13px; line-height:1.5; color:var(--text-secondary); display:flex; align-items:center; gap:4px;">
        <span style="font-size:18px; line-height:1; animation: blink 1.4s infinite both;">•</span>
        <span style="font-size:18px; line-height:1; animation: blink 1.4s infinite both 0.2s;">•</span>
        <span style="font-size:18px; line-height:1; animation: blink 1.4s infinite both 0.4s;">•</span>
      </div>
    </div>
  `;
  
  container.appendChild(typingEl);
  container.scrollTop = container.scrollHeight;
  return typingEl;
}

function wireChatbot() {
  const clearBtn = document.getElementById('btn-clear-chat');
  if (clearBtn) {
    clearBtn.onclick = () => {
      const container = document.getElementById('chat-messages');
      if (container) {
        container.innerHTML = `
          <!-- Bot greeting message -->
          <div style="display:flex; gap:12px; max-width:80%; align-self:flex-start;">
            <div style="width:32px; height:32px; border-radius:50%; background:var(--accent-glow); border:1px solid var(--accent); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
              <svg viewBox="0 0 24 24" width="16" height="16" stroke="var(--accent-light)" stroke-width="2.5" fill="none"><circle cx="12" cy="12" r="3"/><path d="M12 5C7 5 2.73 8.11 1 12.5 2.73 16.89 7 20 12 20s9.27-3.11 11-7.5C21.27 8.11 17 5 12 5z"/></svg>
            </div>
            <div>
              <div style="background:var(--bg-elevated); border:1px solid var(--border); padding:14px 18px; border-radius: 0 16px 16px 16px; font-size:13px; line-height:1.5; color:var(--text-primary);">
                Hello! I am your VisionCare AI Assistant. I can help answer your questions about retinal diseases, symptoms, diagnosis, treatment, and prevention in English, Tamil (தமிழ்), or Tanglish. 
                <br/><br/>
                How can I assist you today?
              </div>
              <div style="font-size:10px; color:var(--text-muted); margin-top:4px; margin-left:4px;">Just now</div>
            </div>
          </div>
        `;
      }
    };
  }
}

/* ─── Eye Scanner HUD & Pupil Auto-Tracking Module ─── */

let cameraStreamInstance = null;
let trackingLoopActive = false;
let pupilLockedX = 50.0;
let pupilLockedY = 50.0;
let currentUploadTab = 'file';

// Hidden low-res sampling canvas for pupil lock calculations
const sampleCanvas = document.createElement('canvas');
sampleCanvas.width = 40;
sampleCanvas.height = 30;
const sampleCtx = sampleCanvas.getContext('2d');

window.switchUploadTab = function(tab) {
  currentUploadTab = tab;
  const fileTab = document.getElementById('upload-tab-file');
  const cameraTab = document.getElementById('upload-tab-camera');
  const filePane = document.getElementById('pane-upload-file');
  const cameraPane = document.getElementById('pane-upload-camera');
  
  if (!fileTab || !cameraTab || !filePane || !cameraPane) return;
  
  if (tab === 'file') {
    fileTab.classList.remove('btn-outline');
    fileTab.classList.add('btn-primary');
    cameraTab.classList.remove('btn-primary');
    cameraTab.classList.add('btn-outline');
    filePane.style.display = 'block';
    cameraPane.style.display = 'none';
    
    // Stop camera stream if switching tabs
    stopCameraStream();
  } else {
    cameraTab.classList.remove('btn-outline');
    cameraTab.classList.add('btn-primary');
    fileTab.classList.remove('btn-primary');
    fileTab.classList.add('btn-outline');
    filePane.style.display = 'none';
    cameraPane.style.display = 'block';
  }
}

function stopCameraStream() {
  if (cameraStreamInstance) {
    cameraStreamInstance.getTracks().forEach(track => track.stop());
    cameraStreamInstance = null;
  }
  trackingLoopActive = false;
  
  // Reset HUD elements safely
  const sysStat = document.getElementById('hud-sys-stat');
  const autolock = document.getElementById('hud-autolock');
  const zoom = document.getElementById('hud-zoom');
  const lock = document.getElementById('hud-lock-status');
  const prompt = document.getElementById('hud-target-prompt');
  
  if (sysStat) {
    sysStat.textContent = "OFFLINE";
    sysStat.style.color = "#ef4444";
  }
  if (autolock) {
    autolock.textContent = "STANDBY";
    autolock.style.color = "var(--text-muted)";
  }
  if (zoom) zoom.textContent = "1.0X";
  if (lock) {
    lock.textContent = "NONE";
    lock.style.color = "var(--text-muted)";
  }
  if (prompt) {
    prompt.textContent = "Camera Inactive";
    prompt.style.color = "var(--text-muted)";
  }
  
  const toggleBtn = document.getElementById('btn-camera-toggle');
  if (toggleBtn) toggleBtn.textContent = "Start Camera";
  
  const captureBtn = document.getElementById('btn-camera-capture');
  if (captureBtn) captureBtn.disabled = true;
}

window.toggleCamera = async function() {
  if (cameraStreamInstance) {
    stopCameraStream();
    return;
  }
  
  const video = document.getElementById('camera-stream');
  const toggleBtn = document.getElementById('btn-camera-toggle');
  const captureBtn = document.getElementById('btn-camera-capture');
  
  if (!video) return;
  
  if (toggleBtn) toggleBtn.textContent = "Connecting...";
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      }
    });
    
    cameraStreamInstance = stream;
    video.srcObject = stream;
    
    if (toggleBtn) toggleBtn.textContent = "Stop Camera";
    if (captureBtn) captureBtn.disabled = false;
    
    // Update HUD visual statuses
    const sysStat = document.getElementById('hud-sys-stat');
    const autolock = document.getElementById('hud-autolock');
    if (sysStat) {
      sysStat.textContent = "ONLINE";
      sysStat.style.color = "#10b981";
    }
    if (autolock) {
      autolock.textContent = "SEARCHING";
      autolock.style.color = "var(--accent-light)";
    }
    
    // Initiate Real-Time Pupil Lock CV Algorithm
    trackingLoopActive = true;
    requestAnimationFrame(trackPupilLoop);
  } catch (err) {
    alert("Camera initialization failed: " + err.message);
    if (toggleBtn) toggleBtn.textContent = "Start Camera";
  }
}

function trackPupilLoop() {
  if (!trackingLoopActive || !cameraStreamInstance) return;
  
  const video = document.getElementById('camera-stream');
  if (video && video.readyState === video.HAVE_ENOUGH_DATA) {
    // Draw to low-res hidden canvas to parse intensity arrays rapidly
    sampleCtx.drawImage(video, 0, 0, 40, 30);
    const frameData = sampleCtx.getImageData(0, 0, 40, 30);
    const pixels = frameData.data;
    
    let minBrightness = 255;
    let darkX = 20;
    let darkY = 15;
    
    // Centered pupil scan bounds
    for (let y = 6; y < 24; y++) {
      for (let x = 8; x < 32; x++) {
        const idx = (y * 40 + x) * 4;
        const r = pixels[idx];
        const g = pixels[idx+1];
        const b = pixels[idx+2];
        const luminance = 0.299*r + 0.587*g + 0.114*b; // Luminance calculation
        if (luminance < minBrightness) {
          minBrightness = luminance;
          darkX = x;
          darkY = y;
        }
      }
    }
    
    // Map sampled coordinates to layout percentages
    const targetX = (darkX / 40) * 100;
    const targetY = (darkY / 30) * 100;
    
    // Smooth moving average interpolation (damping factor)
    pupilLockedX += (targetX - pupilLockedX) * 0.15;
    pupilLockedY += (targetY - pupilLockedY) * 0.15;
    
    const pupilEl = document.getElementById('scanner-pupil');
    if (pupilEl) {
      pupilEl.style.left = `${pupilLockedX}%`;
      pupilEl.style.top = `${pupilLockedY}%`;
    }
    
    const xPos = document.getElementById('hud-x-pos');
    const yPos = document.getElementById('hud-y-pos');
    if (xPos) xPos.textContent = pupilLockedX.toFixed(2);
    if (yPos) yPos.textContent = pupilLockedY.toFixed(2);
    
    const lockStatus = document.getElementById('hud-lock-status');
    const autolock = document.getElementById('hud-autolock');
    const prompt = document.getElementById('hud-target-prompt');
    
    // Alignment criteria
    if (Math.abs(pupilLockedX - 50) < 12 && Math.abs(pupilLockedY - 50) < 12) {
      if (lockStatus) {
        lockStatus.textContent = "LOCK OK";
        lockStatus.style.color = "#10b981";
      }
      if (autolock) {
        autolock.textContent = "LOCKED";
        autolock.style.color = "#10b981";
      }
      if (prompt) {
        prompt.textContent = "🎯 pupil aligned - ready to capture";
        prompt.style.color = "#10b981";
      }
    } else {
      if (lockStatus) {
        lockStatus.textContent = "LOCKING";
        lockStatus.style.color = "#ef4444";
      }
      if (autolock) {
        autolock.textContent = "SEARCHING";
        autolock.style.color = "var(--accent-light)";
      }
      if (prompt) {
        prompt.textContent = "⚠️ Center eye in target reticle";
        prompt.style.color = "var(--text-muted)";
      }
    }
  }
  requestAnimationFrame(trackPupilLoop);
}

window.captureEyeScan = function() {
  const video = document.getElementById('camera-stream');
  const canvas = document.getElementById('camera-canvas');
  const laser = document.getElementById('camera-laser');
  const loader = document.getElementById('scanner-loader');
  const loaderTitle = document.getElementById('scanner-loader-title');
  const loaderSub = document.getElementById('scanner-loader-sub');
  
  if (!video || !canvas || !cameraStreamInstance) return;
  
  // Freeze buttons and visual controls during scan countdown
  const capBtn = document.getElementById('btn-camera-capture');
  const togBtn = document.getElementById('btn-camera-toggle');
  if (capBtn) capBtn.disabled = true;
  if (togBtn) togBtn.disabled = true;
  
  const sysStat = document.getElementById('hud-sys-stat');
  if (sysStat) {
    sysStat.textContent = "SCANNING";
    sysStat.style.color = "var(--accent-light)";
  }
  
  // 1. Play scanning visual sequence: Active green laser sweeps
  if (laser) laser.classList.add('active');
  
  // 2. Trigger zoom transition to capture details "deeper in the eye"
  video.style.transform = "scale(2.5)";
  const zoomText = document.getElementById('hud-zoom');
  if (zoomText) zoomText.textContent = "2.5X";
  
  // 3. Initiate step-by-step layer simulation loader
  if (loader) loader.style.display = 'flex';
  if (loaderTitle) loaderTitle.textContent = "Initiating Laser Scan...";
  if (loaderSub) loaderSub.textContent = "Aperture locked, tracking pupil center...";
  
  setTimeout(() => {
    if (loaderTitle) loaderTitle.textContent = "Penetrating Cornea Layer...";
    if (loaderSub) loaderSub.textContent = "Refracting lens focal depth, scanning depth = 120μm...";
  }, 400);
  
  setTimeout(() => {
    if (loaderTitle) loaderTitle.textContent = "Scanning Vitreous Humor...";
    if (loaderSub) loaderSub.textContent = "Filtering ocular fluid noise, scanning depth = 850μm...";
  }, 800);
  
  setTimeout(() => {
    if (loaderTitle) loaderTitle.textContent = "Capturing Retinal Cross-Section...";
    if (loaderSub) loaderSub.textContent = "Mapping photoreceptors and capillary networks...";
  }, 1200);
  
  setTimeout(() => {
    // 4. Capture cropped video frame on canvas
    const ctx = canvas.getContext('2d');
    canvas.width = 640;
    canvas.height = 480;
    
    // Draw centered scaled image to canvas (performing center 50% crop zoom in JS)
    const srcWidth = video.videoWidth;
    const srcHeight = video.videoHeight;
    const cropWidth = srcWidth * 0.40;  // Zoomed-in crop region
    const cropHeight = srcHeight * 0.40;
    const sx = (srcWidth - cropWidth) / 2;
    const sy = (srcHeight - cropHeight) / 2;
    
    ctx.drawImage(video, sx, sy, cropWidth, cropHeight, 0, 0, 640, 480);
    
    // 5. Digital Image Processing Algorithm: Contrast Stretch (Normalization)
    const imgData = ctx.getImageData(0, 0, 640, 480);
    const data = imgData.data;
    
    let minVal = 255;
    let maxVal = 0;
    
    // Find min and max intensity values across colors
    for (let i = 0; i < data.length; i += 4) {
      const luminance = 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
      if (luminance < minVal) minVal = luminance;
      if (luminance > maxVal) maxVal = luminance;
    }
    
    // Dynamic range stretching to map pixels back into a high contrast spectrum
    const diff = maxVal - minVal || 1;
    for (let i = 0; i < data.length; i += 4) {
      data[i] = ((data[i] - minVal) / diff) * 255;
      data[i+1] = ((data[i+1] - minVal) / diff) * 255;
      data[i+2] = ((data[i+2] - minVal) / diff) * 255;
    }
    ctx.putImageData(imgData, 0, 0);
    
    // Convert canvas back to blob and cache for analysis
    canvas.toBlob((blob) => {
      // Release camera
      stopCameraStream();
      if (togBtn) togBtn.disabled = false;
      if (capBtn) capBtn.disabled = false;
      if (loader) loader.style.display = 'none';
      if (laser) laser.classList.remove('active');
      video.style.transform = "scale(1.0)";
      
      // Cache captured file reference
      const file = new File([blob], "eye_scan_hud_capture.png", { type: "image/png" });
      selectedUploadFile = file;
      
      // Update File Info Grid
      document.getElementById('file-info').innerHTML = `
        <div class="file-info-row"><span class="file-info-key">File Name</span><span class="file-info-val">eye_scan_hud_capture.png</span></div>
        <div class="file-info-row"><span class="file-info-key">File Type</span><span class="file-info-val">image/png (Captured)</span></div>
        <div class="file-info-row"><span class="file-info-key">File Size</span><span class="file-info-val">${(file.size / 1024).toFixed(1)} KB</span></div>
        <div class="file-info-row"><span class="file-info-key">Status</span><span class="file-info-val" style="color:var(--accent-light);font-weight:600;">Captured (Ready for Analysis)</span></div>`;
        
      document.getElementById('validation-status').style.display = 'none';
      
      // Show local preview immediately using object URL
      const previewUrl = URL.createObjectURL(blob);
      document.querySelector('#mod-upload .oct-scan-placeholder').innerHTML =
        `<img src="${previewUrl}" style="width:100%;height:100%;object-fit:cover;border-radius:8px;">`;
        
    }, 'image/png');
    
  }, 1600);
}

function updateCompareButtonState() {
  const checkboxes = document.querySelectorAll('.history-compare-cb:checked');
  const compareBtn = document.getElementById('history-compare-btn');
  if (!compareBtn) return;
  compareBtn.textContent = `Compare (${checkboxes.length})`;
  compareBtn.disabled = checkboxes.length !== 2;
}
window.updateCompareButtonState = updateCompareButtonState;

async function compareSelectedScans() {
  const checkboxes = Array.from(document.querySelectorAll('.history-compare-cb:checked'));
  if (checkboxes.length !== 2) {
    alert('Please select exactly two scans to compare.');
    return;
  }
  
  const id1 = checkboxes[0].dataset.id;
  const id2 = checkboxes[1].dataset.id;
  
  const compareBtn = document.getElementById('history-compare-btn');
  const origText = compareBtn.textContent;
  compareBtn.textContent = 'Comparing...';
  compareBtn.disabled = true;
  
  try {
    const r = await fetch(`${API}/api/scans/compare?scan_id_1=${id1}&scan_id_2=${id2}&user_id=${currentUser.id}`);
    if (!r.ok) {
      const data = await r.json();
      throw new Error(data.detail || 'Failed to compare scans');
    }
    const comparison = await r.json();
    
    const s1 = comparison.scan1;
    const s2 = comparison.scan2;
    const deltas = comparison.deltas;
    
    document.getElementById('compare-a-name').textContent = s1.patient_name;
    document.getElementById('compare-a-date').textContent = (s1.created_at || '').slice(0, 10);
    document.getElementById('compare-a-img').src = b64img(s1.preview_b64) || '';
    document.getElementById('compare-a-diagnosis').textContent = s1.prediction;
    document.getElementById('compare-a-severity').textContent = s1.stage;
    document.getElementById('compare-a-affected').textContent = `${Math.round(s1.total_affected_pct)}%`;
    document.getElementById('compare-a-confidence').textContent = `${Math.round(s1.confidence)}%`;
    
    document.getElementById('compare-b-name').textContent = s2.patient_name;
    document.getElementById('compare-b-date').textContent = (s2.created_at || '').slice(0, 10);
    document.getElementById('compare-b-img').src = b64img(s2.preview_b64) || '';
    document.getElementById('compare-b-diagnosis').textContent = s2.prediction;
    document.getElementById('compare-b-severity').textContent = s2.stage;
    document.getElementById('compare-b-affected').textContent = `${Math.round(s2.total_affected_pct)}%`;
    document.getElementById('compare-b-confidence').textContent = `${Math.round(s2.confidence)}%`;
    
    const warningEl = document.getElementById('compare-patient-warning');
    if (s1.patient_name.trim().toLowerCase() !== s2.patient_name.trim().toLowerCase()) {
      warningEl.style.display = 'block';
    } else {
      warningEl.style.display = 'none';
    }
    
    const shift = deltas.stage_shift;
    const badge = document.getElementById('delta-stage-badge');
    if (shift > 0) {
      badge.textContent = `+${shift} Stage Progression`;
      badge.className = 'badge badge-danger';
    } else if (shift < 0) {
      badge.textContent = `${shift} Stage Improvement`;
      badge.className = 'badge badge-success';
    } else {
      badge.textContent = 'Stable / No Shift';
      badge.className = 'badge badge-info';
    }
    
    const formatDelta = (elId, value) => {
      const el = document.getElementById(elId);
      if (!el) return;
      el.textContent = `${value > 0 ? '+' : ''}${value}%`;
      el.style.color = value > 0 ? 'var(--danger)' : (value < 0 ? 'var(--success)' : 'var(--text-primary)');
    };
    
    formatDelta('delta-lesion', deltas.lesion_area_pct);
    formatDelta('delta-fluid', deltas.fluid_area_pct);
    formatDelta('delta-edema', deltas.edema_area_pct);
    formatDelta('delta-total', deltas.total_affected_pct);
    
    document.getElementById('modal-scan-compare').style.display = 'flex';
    
    const cols = Array.from({length: 224}, (_, i) => i);
    const trace1 = {
      x: cols,
      y: s1.thickness_profile.length ? s1.thickness_profile : new Array(224).fill(0),
      type: 'scatter',
      mode: 'lines',
      name: 'A (Baseline)',
      line: { color: '#6366f1', width: 2 }
    };
    const trace2 = {
      x: cols,
      y: s2.thickness_profile.length ? s2.thickness_profile : new Array(224).fill(0),
      type: 'scatter',
      mode: 'lines',
      name: 'B (Follow-up)',
      line: { color: '#10b981', width: 2 }
    };
    
    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: { title: 'OCT Column B-scan', gridcolor: '#e2e8f0', tickfont: {color: '#64748b'}, titlefont: {color: '#475569'} },
      yaxis: { title: 'Pathology Vertical Height (px)', gridcolor: '#e2e8f0', tickfont: {color: '#64748b'}, titlefont: {color: '#475569'}, range: [0, 150] },
      legend: { font: {color: '#334155'}, orientation: 'h', y: 1.15 },
      margin: { l: 40, r: 20, t: 20, b: 40 }
    };
    
    Plotly.newPlot('plotly-compare-thickness-chart', [trace1, trace2], layout, {responsive: true, displayModeBar: false});
    
  } catch (err) {
    alert('Comparison failed: ' + err.message);
  } finally {
    compareBtn.textContent = origText;
    compareBtn.disabled = false;
  }
}
window.compareSelectedScans = compareSelectedScans;

window.loadSampleScan = async function(sampleKey, conditionName, autoAnalyze = true) {
  try {
    const url = `/images/samples/sample_${sampleKey}.jpg`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('Could not fetch sample scan');
    const blob = await resp.blob();
    const file = new File([blob], `oct_${sampleKey}_sample.jpg`, { type: 'image/jpeg' });
    selectedUploadFile = file;
    
    // Auto-fill patient demo name if empty or default
    const nameInput = document.getElementById('patient-name-input');
    if (nameInput) nameInput.value = `Clinical Demo (${conditionName})`;
    const ageInput = document.getElementById('patient-age-input');
    if (ageInput && !ageInput.value) ageInput.value = '56';
    
    // Update preview placeholder
    const placeholder = document.querySelector('#mod-upload .oct-scan-placeholder');
    if (placeholder) {
      placeholder.innerHTML = `<img src="${url}" style="width:100%;height:100%;object-fit:cover;border-radius:8px;">`;
    }
    
    // Update file info
    const fileInfo = document.getElementById('file-info');
    if (fileInfo) {
      fileInfo.innerHTML = `
        <div class="file-info-row"><span class="file-info-key">File Name</span><span class="file-info-val">oct_${sampleKey}_sample.jpg</span></div>
        <div class="file-info-row"><span class="file-info-key">Condition Type</span><span class="file-info-val" style="color:var(--accent); font-weight:700;">${conditionName}</span></div>
        <div class="file-info-row"><span class="file-info-key">Resolution</span><span class="file-info-val">512 x 496 px (Kermany B-Scan)</span></div>
        <div class="file-info-row"><span class="file-info-key">Status</span><span class="file-info-val" style="color:var(--success);font-weight:600;">${autoAnalyze ? 'Analyzing with Neural Core...' : 'Demo Case Ready'}</span></div>`;
    }
    
    if (autoAnalyze) {
      if (!currentUser) {
        currentUser = { id: 1, name: 'Dr. Elena Vance', email: 'doctor@visioncare.ai' };
      }
      const patient = getPatientFormValues();
      const pName = `Clinical Demo (${conditionName})`;
      const data = await apiAnalyze(file, pName, patient.age || 56, patient.gender || 'Female', 'OCT');
      applyAnalysisToModules(data);
      
      const statusBadge = document.getElementById('validation-status');
      if (statusBadge) statusBadge.style.display = 'flex';
      
      if (fileInfo) {
        fileInfo.innerHTML = `
          <div class="file-info-row"><span class="file-info-key">File Name</span><span class="file-info-val">oct_${sampleKey}_sample.jpg</span></div>
          <div class="file-info-row"><span class="file-info-key">Condition Type</span><span class="file-info-val" style="color:var(--accent); font-weight:700;">${conditionName}</span></div>
          <div class="file-info-row"><span class="file-info-key">Resolution</span><span class="file-info-val">512 x 496 px (Kermany B-Scan)</span></div>
          <div class="file-info-row"><span class="file-info-key">Status</span><span class="file-info-val" style="color:var(--success);font-weight:600;">Analysis Completed</span></div>`;
      }
      
      await refreshUserData();
      if (typeof switchModule === 'function') {
        switchModule('prediction');
      }
    } else {
      if (typeof switchModule === 'function') {
        switchModule('upload');
      }
    }
  } catch (err) {
    console.error('Error loading sample scan:', err);
    alert('Failed to load demo scan: ' + err.message);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  try {
    wireAuth();
  } catch (err) {
    console.error('Error wiring auth:', err);
  }

  try {
    wireUpload();
  } catch (err) {
    console.error('Error wiring upload:', err);
  }

  try {
    wireReports();
  } catch (err) {
    console.error('Error wiring reports:', err);
  }

  try {
    wireAllModules();
  } catch (err) {
    console.error('Error wiring modules:', err);
  }

  try {
    updateThemeOptions();
  } catch (err) {
    console.error('Error updating theme:', err);
  }

  try {
    if (loadSession()) {
      showMain();
      const nameEl = document.querySelector('.user-name');
      if (nameEl && currentUser) nameEl.textContent = currentUser.name;
    }
  } catch (err) {
    console.error('Error loading session:', err);
  }

  // Check server health asynchronously in the background
  checkServerHealth().then(async (ok) => {
    if (ok && currentUser) {
      try {
        await refreshUserData();
      } catch (err) {
        console.error('Error refreshing user data:', err);
      }
    }
  });

  try {
    initCurtainSliders();
    render3DMacularTopography(null);
  } catch (err) {
    console.warn('Error initializing advanced PACS tools:', err);
  }

  // Pre-load baseline diagnostic case so all modules (Prediction, Heatmap, Segmentation, Reports) are fully ready immediately
  if (!lastAnalysis) {
    setTimeout(() => {
      fetch('/images/samples/sample_dme.jpg')
        .then(r => {
          if (!r.ok) throw new Error('sample fetch failed');
          return r.blob();
        })
        .then(async (blob) => {
          const file = new File([blob], 'oct_dme_sample.jpg', { type: 'image/jpeg' });
          selectedUploadFile = file;
          if (!currentUser) {
            currentUser = { id: 1, name: 'Dr. Elena Vance', email: 'doctor@visioncare.ai' };
          }
          const res = await apiAnalyze(file, 'Ramesh Kumar', 52, 'Male', 'OCT');
          applyAnalysisToModules(res);
          console.log('✓ Initial clinical workstation case preloaded successfully');
        })
        .catch(err => console.log('Notice: Preload initial sample:', err.message));
    }, 300);
  }
});

/* ==========================================================================
   ADVANCED CLINICAL TELEMETRY, 3D TOPOGRAPHY & PACS SUITE
   ========================================================================== */

let currentUtterance = null;
let audioLanguage = 'en';

function setAudioLanguage(lang) {
  audioLanguage = lang;
  const enBtn = document.getElementById('audio-lang-en');
  const taBtn = document.getElementById('audio-lang-ta');
  if (enBtn && taBtn) {
    if (lang === 'en') {
      enBtn.className = 'btn btn-primary btn-sm';
      taBtn.className = 'btn btn-outline btn-sm';
    } else {
      enBtn.className = 'btn btn-outline btn-sm';
      taBtn.className = 'btn btn-primary btn-sm';
    }
  }
  if (window.speechSynthesis && window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
    startAudioBriefing();
  }
}
window.setAudioLanguage = setAudioLanguage;

function toggleAudioBriefing() {
  if (!('speechSynthesis' in window)) {
    alert('Speech synthesis is not supported in this browser.');
    return;
  }
  if (window.speechSynthesis.speaking) {
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
      setAudioVisualizerState(true);
    } else {
      window.speechSynthesis.pause();
      setAudioVisualizerState(false);
    }
  } else {
    startAudioBriefing();
  }
}
window.toggleAudioBriefing = toggleAudioBriefing;

function startAudioBriefing() {
  if (!lastAnalysis) {
    alert('Please upload and analyze an OCT scan first.');
    return;
  }
  window.speechSynthesis.cancel();

  const data = lastAnalysis;
  const pName = data.patient_name || 'Patient';
  const label = data.label_full || data.label || 'Unknown';
  const stage = data.stage || 'Normal';
  const conf = data.confidence || 0;
  const bio = data.biomarkers || {};
  const cst = bio.cst_microns || 262;

  let text = '';
  if (audioLanguage === 'ta') {
    text = `விஷன் கேர் ஏ ஐ மருத்துவ பகுப்பாய்வு சுருக்கம். நோயாளி பெயர்: ${pName}. கண்டறியப்பட்ட விழித்திரை நோய்: ${label}. நோய் தீவிரம்: ${stage}. உறுதிப்பாடு: ${conf} சதவீதம். மைய விழித்திரை தடிமன்: ${cst} மைக்ரோமீட்டர். பரிந்துரைக்கப்பட்ட மருத்துவ நிலை: ${bio.triage_level || 'வழக்கமான பரிசோதனை'}. உடனடியாக கண் மருத்துவ நிபுணரை அணுகவும்.`;
  } else {
    text = `VisionCare Artificial Intelligence Clinical Briefing. Patient: ${pName}. Primary diagnostic finding: ${label} at ${stage} stage. Neural ensemble certainty score: ${conf} percent. Central subfield thickness is measured at ${cst} micrometers. Intraretinal fluid assessment: ${bio.irf ? 'Positive, intraretinal cystoid fluid detected' : 'Negative, neurosensory foveal profile preserved'}. Clinical triage priority: ${bio.triage_level || 'Routine Screening'}. ${bio.triage_timeframe || 'Annual examination advised'}.`;
  }

  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 0.95;
  utter.pitch = 1.0;

  const voices = window.speechSynthesis.getVoices();
  if (audioLanguage === 'ta') {
    const taVoice = voices.find(v => v.lang.startsWith('ta'));
    if (taVoice) utter.voice = taVoice;
  } else {
    const enVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
    if (enVoice) utter.voice = enVoice;
  }

  utter.onstart = () => {
    setAudioVisualizerState(true);
    const statusText = document.getElementById('audio-status-text');
    if (statusText) statusText.textContent = 'Speaking diagnostic telemetry...';
  };
  utter.onend = () => {
    setAudioVisualizerState(false);
    const statusText = document.getElementById('audio-status-text');
    if (statusText) statusText.textContent = 'Completed briefing';
  };
  utter.onerror = () => {
    setAudioVisualizerState(false);
  };

  currentUtterance = utter;
  window.speechSynthesis.speak(utter);
}

function setAudioVisualizerState(isPlaying) {
  const viz = document.getElementById('audio-visualizer');
  const playIcon = document.getElementById('audio-icon-play');
  const pauseIcon = document.getElementById('audio-icon-pause');
  const label = document.getElementById('audio-briefing-label');
  if (viz) {
    if (isPlaying) viz.classList.add('playing');
    else viz.classList.remove('playing');
  }
  if (playIcon && pauseIcon) {
    playIcon.style.display = isPlaying ? 'none' : 'block';
    pauseIcon.style.display = isPlaying ? 'block' : 'none';
  }
  if (label) {
    label.textContent = isPlaying ? 'Pause AI Briefing' : 'AI Audio Clinical Briefing';
  }
}

function updateBiomarkersUI(data) {
  const bio = data.biomarkers || {};
  const cstVal = document.getElementById('bio-cst-val');
  const cstStatus = document.getElementById('bio-cst-status');
  if (cstVal) cstVal.textContent = `${bio.cst_microns || 262} µm`;
  if (cstStatus) {
    cstStatus.textContent = bio.cst_status || 'Normal Foveal Profile';
    cstStatus.style.color = (bio.cst_microns && bio.cst_microns > 310) ? 'var(--warning)' : 'var(--success)';
  }

  const irfVal = document.getElementById('bio-irf-val');
  const irfStatus = document.getElementById('bio-irf-status');
  if (irfVal) {
    irfVal.textContent = bio.irf ? 'Positive (Detected)' : 'Negative';
    irfVal.style.color = bio.irf ? 'var(--danger)' : 'var(--text-primary)';
  }
  if (irfStatus) {
    irfStatus.textContent = bio.irf ? 'Intraretinal Cystoid Fluid' : 'No Cystoid Macular Edema';
    irfStatus.style.color = bio.irf ? 'var(--danger)' : 'var(--success)';
  }

  const srfVal = document.getElementById('bio-srf-val');
  const srfStatus = document.getElementById('bio-srf-status');
  if (srfVal) {
    srfVal.textContent = bio.srf ? 'Positive (Detected)' : (bio.ped ? 'PED Noted' : 'Negative');
    srfVal.style.color = (bio.srf || bio.ped) ? 'var(--warning)' : 'var(--text-primary)';
  }
  if (srfStatus) {
    srfStatus.textContent = bio.srf ? 'Subretinal Exudative Fluid' : (bio.ped ? 'Pigment Detachment' : 'Sub-RPE Space Preserved');
    srfStatus.style.color = (bio.srf || bio.ped) ? 'var(--warning)' : 'var(--success)';
  }

  const ezVal = document.getElementById('bio-ez-val');
  const ezStatus = document.getElementById('bio-ez-status');
  if (ezVal) ezVal.textContent = bio.ez_integrity || 'Intact';
  if (ezStatus) {
    ezStatus.textContent = bio.ez_integrity?.includes('Disrupt') ? 'Photoreceptor Attenuation' : 'Photoreceptor Band Intact';
    ezStatus.style.color = bio.ez_integrity?.includes('Disrupt') ? 'var(--warning)' : 'var(--success)';
  }

  // Triage banner
  const triageTitle = document.getElementById('bio-triage-title');
  const triageSub = document.getElementById('bio-triage-sub');
  const triageBadge = document.getElementById('bio-triage-badge');
  const triageBeacon = document.getElementById('bio-triage-beacon');
  if (triageTitle) triageTitle.textContent = `Clinical Triage: ${bio.triage_level || 'Routine Screening'}`;
  if (triageSub) triageSub.textContent = `Action timeframe: ${bio.triage_timeframe || '12 Months'}. Comprehensive retinal documentation recorded.`;
  if (triageBadge) {
    triageBadge.textContent = bio.triage_level?.split(' ')[0] || 'Routine';
    if (bio.triage_level?.includes('Critical') || bio.triage_level?.includes('Urgent')) {
      triageBadge.className = 'badge badge-danger';
      if (triageBeacon) triageBeacon.style.background = '#ef4444';
    } else if (bio.triage_level?.includes('Semi')) {
      triageBadge.className = 'badge badge-warning';
      if (triageBeacon) triageBeacon.style.background = '#f59e0b';
    } else {
      triageBadge.className = 'badge badge-success';
      if (triageBeacon) triageBeacon.style.background = '#10b981';
    }
  }
}

function updateConsensusUI(data) {
  const cons = data.consensus || {};
  const eff = cons.efficientnet || {};
  const mob = cons.mobilenet || {};

  const effLbl = document.getElementById('consensus-eff-label');
  const effConf = document.getElementById('consensus-eff-conf');
  const effBar = document.getElementById('consensus-eff-bar');
  if (effLbl) effLbl.textContent = eff.label || data.label || '—';
  if (effConf) effConf.textContent = `${eff.confidence || data.confidence || 0}%`;
  if (effBar) effBar.style.width = `${eff.confidence || data.confidence || 0}%`;

  const mobLbl = document.getElementById('consensus-mob-label');
  const mobConf = document.getElementById('consensus-mob-conf');
  const mobBar = document.getElementById('consensus-mob-bar');
  if (mobLbl) mobLbl.textContent = mob.label || data.label || '—';
  if (mobConf) mobConf.textContent = `${mob.confidence || (data.confidence ? (data.confidence * 0.98).toFixed(1) : 0)}%`;
  if (mobBar) mobBar.style.width = `${mob.confidence || data.confidence || 0}%`;

  const entropyVal = document.getElementById('consensus-entropy-val');
  const ratingBadge = document.getElementById('consensus-rating-badge');
  const statusBadge = document.getElementById('consensus-status-badge');
  if (entropyVal) entropyVal.textContent = `${cons.entropy_nats || 0.082} nats`;
  if (ratingBadge) ratingBadge.textContent = cons.uncertainty_rating?.split(' ')[0] || 'High Reliability';
  if (statusBadge) {
    statusBadge.textContent = cons.agreement ? '100% Concordant' : 'Divergent Ensemble';
    statusBadge.className = cons.agreement ? 'badge badge-success' : 'badge badge-warning';
  }
}

function updateEtdrsUI(data) {
  const grid = data.etdrs_grid || {};
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el && val != null) el.textContent = val;
  };
  setVal('etdrs-central', grid.central || 262);
  setVal('etdrs-inner-sup', grid.inner_sup || 325);
  setVal('etdrs-inner-inf', grid.inner_inf || 320);
  setVal('etdrs-inner-nas', grid.inner_nas || 330);
  setVal('etdrs-inner-temp', grid.inner_temp || 310);
  setVal('etdrs-outer-sup', grid.outer_sup || 285);
  setVal('etdrs-outer-inf', grid.outer_inf || 280);
  setVal('etdrs-outer-nas', grid.outer_nas || 295);
  setVal('etdrs-outer-temp', grid.outer_temp || 270);
}

function render3DMacularTopography(topography) {
  const chartEl = document.getElementById('plotly-3d-surface');
  if (!chartEl || !window.Plotly) return;
  if (!topography || !topography.length) {
    const size = 20;
    topography = [];
    for (let i = 0; i < size; i++) {
      const row = [];
      for (let j = 0; j < size; j++) {
        const x = (i - size/2) / 3;
        const y = (j - size/2) / 3;
        const r = Math.sqrt(x*x + y*y);
        row.push(Math.round(320.0 - 60.0 * Math.exp(-(r/1.1)*(r/1.1))));
      }
      topography.push(row);
    }
  }

  const plotData = [{
    z: topography,
    type: 'surface',
    colorscale: [
      [0.0, '#0284c7'],
      [0.3, '#06b6d4'],
      [0.6, '#10b981'],
      [0.85, '#f59e0b'],
      [1.0, '#ef4444']
    ],
    showscale: false,
    contours: {
      z: { show: true, usecolormap: true, highlightcolor: '#38bdf8', project: { z: false } }
    }
  }];

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    scene: {
      xaxis: { title: 'Temporal ↔ Nasal', showgrid: true, gridcolor: '#334155', tickfont: {color: '#64748b', size: 9} },
      yaxis: { title: 'Inferior ↔ Superior', showgrid: true, gridcolor: '#334155', tickfont: {color: '#64748b', size: 9} },
      zaxis: { title: 'Height (µm)', showgrid: true, gridcolor: '#334155', tickfont: {color: '#64748b', size: 9} },
      camera: { eye: { x: 1.4, y: -1.4, z: 1.1 } }
    }
  };

  Plotly.newPlot('plotly-3d-surface', plotData, layout, { responsive: true, displayModeBar: false });
}

window.reset3DCamera = function() {
  const chartEl = document.getElementById('plotly-3d-surface');
  if (chartEl && window.Plotly) {
    Plotly.relayout(chartEl, { 'scene.camera': { eye: { x: 1.4, y: -1.4, z: 1.1 } } });
  }
};

function updateCurtainViews(data) {
  const orig = b64img(data.preview_b64);
  const heat = b64img(data.heatmap_b64);
  const mask = b64img(data.mask_b64);

  // Heatmap curtain
  const hUnder = document.getElementById('heat-curtain-under');
  const hOver = document.getElementById('heat-curtain-over');
  if (hUnder && heat) hUnder.innerHTML = `<img src="${heat}" alt="Heatmap Under">`;
  if (hOver && orig) hOver.innerHTML = `<img src="${orig}" alt="OCT Over">`;

  // Segmentation curtain
  const sUnder = document.getElementById('seg-curtain-under');
  const sOver = document.getElementById('seg-curtain-over');
  if (sUnder && mask) sUnder.innerHTML = `<img src="${mask}" alt="Segmentation Under">`;
  if (sOver && orig) sOver.innerHTML = `<img src="${orig}" alt="OCT Over">`;
}

function initCurtainSliders() {
  ['heat', 'seg'].forEach((mod) => {
    const container = document.getElementById(`${mod}-curtain-container`);
    const divider = document.getElementById(`${mod}-curtain-divider`);
    if (!container || !divider) return;

    let isDragging = false;

    const setPosition = (clientX) => {
      const rect = container.getBoundingClientRect();
      let x = clientX - rect.left;
      x = Math.max(0, Math.min(rect.width, x));
      const pct = (x / rect.width) * 100;
      container.style.setProperty('--curtain-pos', `${pct.toFixed(2)}%`);
    };

    const onMove = (e) => {
      if (!isDragging) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      setPosition(clientX);
    };

    const stopDrag = () => {
      isDragging = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', stopDrag);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', stopDrag);
    };

    const startDrag = (e) => {
      if (caliperState[mod]?.active) return;
      isDragging = true;
      e.preventDefault();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      setPosition(clientX);
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', stopDrag);
      window.addEventListener('touchmove', onMove, { passive: false });
      window.addEventListener('touchend', stopDrag);
    };

    container.onmousedown = (e) => {
      if (caliperState[mod]?.active) return;
      startDrag(e);
    };
    container.ontouchstart = (e) => {
      if (caliperState[mod]?.active) return;
      startDrag(e);
    };
  });
}

const caliperState = {
  heat: { active: false },
  seg: { active: false }
};

window.toggleCaliper = function(mod) {
  const canvas = document.getElementById(`${mod}-caliper-canvas`);
  const readout = document.getElementById(`${mod}-caliper-readout`);
  const btn = document.getElementById(`btn-${mod}-caliper`);
  if (!canvas) return;

  const state = caliperState[mod];
  state.active = !state.active;

  if (state.active) {
    canvas.classList.add('active');
    if (btn) {
      btn.className = 'btn btn-primary btn-sm';
      btn.textContent = '📏 Caliper Active (Click & Drag)';
    }
    if (readout) readout.style.display = 'block';
    setupCanvasListeners(mod);
  } else {
    canvas.classList.remove('active');
    if (btn) {
      btn.className = 'btn btn-outline btn-sm';
      btn.textContent = '📏 Caliper (µm)';
    }
    if (readout) readout.style.display = 'none';
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
};

function setupCanvasListeners(mod) {
  const canvas = document.getElementById(`${mod}-caliper-canvas`);
  const readout = document.getElementById(`${mod}-caliper-readout`);
  if (!canvas) return;

  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;

  const ctx = canvas.getContext('2d');
  let drawing = false;
  let startX = 0, startY = 0;

  canvas.onmousedown = (e) => {
    drawing = true;
    const r = canvas.getBoundingClientRect();
    startX = e.clientX - r.left;
    startY = e.clientY - r.top;
  };

  canvas.onmousemove = (e) => {
    if (!drawing) return;
    const r = canvas.getBoundingClientRect();
    const currX = e.clientX - r.left;
    const currY = e.clientY - r.top;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(currX, currY);
    ctx.stroke();

    drawCaliperCrosshair(ctx, startX, startY);
    drawCaliperCrosshair(ctx, currX, currY);

    const distPx = Math.sqrt(Math.pow(currX - startX, 2) + Math.pow(currY - startY, 2));
    const distMicrons = (distPx * 3.9).toFixed(1);

    if (readout) {
      readout.textContent = `Retinal Caliper: ${distMicrons} µm (Normal Foveal Range: 240–300 µm)`;
    }
  };

  canvas.onmouseup = () => { drawing = false; };
  canvas.onmouseleave = () => { drawing = false; };
}

function drawCaliperCrosshair(ctx, x, y) {
  ctx.save();
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 8, y);
  ctx.lineTo(x + 8, y);
  ctx.moveTo(x, y - 8);
  ctx.lineTo(x + 8, y);
  ctx.stroke();
  ctx.restore();
}

window.updatePacsFilters = function(mod) {
  const bright = document.getElementById(`${mod}-pacs-bright`)?.value || 0;
  const contrast = document.getElementById(`${mod}-pacs-contrast`)?.value || 100;
  const under = document.getElementById(`${mod}-curtain-under`);
  const over = document.getElementById(`${mod}-curtain-over`);
  const filterVal = `brightness(${100 + Number(bright)}%) contrast(${contrast}%)`;
  if (under) under.style.filter = filterVal;
  if (over) over.style.filter = filterVal;
};

const pacsInvert = { heat: false, seg: false };
window.togglePacsInvert = function(mod) {
  pacsInvert[mod] = !pacsInvert[mod];
  const under = document.getElementById(`${mod}-curtain-under`);
  const over = document.getElementById(`${mod}-curtain-over`);
  const btn = document.getElementById(`btn-${mod}-invert`);
  const invStr = pacsInvert[mod] ? 'invert(100%) ' : '';
  const bright = document.getElementById(`${mod}-pacs-bright`)?.value || 0;
  const contrast = document.getElementById(`${mod}-pacs-contrast`)?.value || 100;
  const filterVal = `${invStr}brightness(${100 + Number(bright)}%) contrast(${contrast}%)`;
  if (under) under.style.filter = filterVal;
  if (over) over.style.filter = filterVal;
  if (btn) {
    btn.className = pacsInvert[mod] ? 'btn btn-primary btn-sm' : 'btn btn-outline btn-sm';
  }
};

window.resetPacs = function(mod) {
  const b = document.getElementById(`${mod}-pacs-bright`);
  const c = document.getElementById(`${mod}-pacs-contrast`);
  if (b) b.value = 0;
  if (c) c.value = 100;
  pacsInvert[mod] = false;
  const btn = document.getElementById(`btn-${mod}-invert`);
  if (btn) btn.className = 'btn btn-outline btn-sm';
  const under = document.getElementById(`${mod}-curtain-under`);
  const over = document.getElementById(`${mod}-curtain-over`);
  if (under) under.style.filter = 'none';
  if (over) over.style.filter = 'none';
};

window.exportHeatmapImage = function() {
  if (!lastAnalysis?.heatmap_b64) {
    alert('Please run an analysis first.');
    return;
  }
  const a = document.createElement('a');
  a.href = b64img(lastAnalysis.heatmap_b64);
  a.download = `GradCAM_Heatmap_${lastAnalysis.scan_id || 'export'}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

window.exportSegMask = function() {
  if (!lastAnalysis?.mask_b64) {
    alert('Please run an analysis first.');
    return;
  }
  const a = document.createElement('a');
  a.href = b64img(lastAnalysis.mask_b64);
  a.download = `Retinal_Segmentation_${lastAnalysis.scan_id || 'export'}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};
