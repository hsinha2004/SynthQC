/**
 * SynthQC Vision Station - Unslopped Frontend Controller
 * Minimal, quiet, precise client logic.
 */

// Application State
const state = {
  activeTab: 'inspect',
  systemStatus: null,
  activeInspection: null,
  inspectionViewMode: 'clean', // 'clean' or 'heatmap'
  catalogPresets: {},
  studioParams: {
    flawCategory: 'broken_large',
    preset: 'rim',
    scale: 1.0,
    rotation: 0
  },
  lastStudioResult: null,
  trainingPollingInterval: null,
  scarcityChart: null
};

// ==============================================================================
// Initialization & Navigation
// ==============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  await refreshSystemStatus();
  await loadCatalogPresets();
  setupAnalyticsChart();
  startBackgroundJobPolling();

  if (state.catalogPresets['preset_large']) {
    loadSamplePreset('preset_large');
  } else {
    generateStudioSample();
  }
});

function switchTab(tabId) {
  state.activeTab = tabId;

  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.remove('active');
  });
  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  if (activeBtn) activeBtn.classList.add('active');

  document.querySelectorAll('.station-view').forEach(view => {
    view.classList.add('hidden');
    view.classList.remove('block');
  });
  const targetView = document.getElementById(`station-${tabId}`);
  if (targetView) {
    targetView.classList.remove('hidden');
    targetView.classList.add('block');
  }

  if (window.lucide) lucide.createIcons();

  if (tabId === 'studio' && !state.lastStudioResult) {
    generateStudioSample();
  } else if (tabId === 'analytics') {
    refreshAuditLog();
    if (state.scarcityChart) state.scarcityChart.update();
  }
}

// ==============================================================================
// System Status & Deployed Models
// ==============================================================================

async function refreshSystemStatus() {
  try {
    const res = await fetch('/api/system/status');
    const data = await res.json();
    state.systemStatus = data;

    document.getElementById('header-specimen').textContent = data.specimen_profile || 'Glass bottle';
    document.getElementById('header-active-model').textContent = data.active_model?.name || 'Active model';

    renderDeployedModels(data.available_models || []);

    if (data.background_job?.is_running) {
      showBackgroundBanner(data.background_job);
    }
  } catch (err) {
    console.error('Status fetch error:', err);
  }
}

function renderDeployedModels(models) {
  const container = document.getElementById('models-list-container');
  if (!container) return;

  container.innerHTML = models.map(m => `
    <div class="flex items-center justify-between p-3 rounded-lg ${m.is_active ? 'bg-[#19191b] border border-[rgba(59,130,246,0.3)]' : 'bg-[#141415] border border-[rgba(237,237,237,0.08)]'}">
      <div>
        <div class="text-xs font-medium text-[#ededed] flex items-center gap-2">
          <span>${m.name}</span>
          ${m.is_active ? '<span class="text-[11px] font-mono text-[#3b82f6]">Active</span>' : ''}
        </div>
        <div class="text-xs font-mono text-[#85858b] mt-0.5">Accuracy: ${m.accuracy}% — Latency: 18 ms</div>
      </div>
      ${!m.is_active ? `
        <button onclick="deployModel('${m.key}')" class="btn-neutral text-xs">
          Deploy
        </button>
      ` : '<span class="text-xs font-mono text-[#4cc38a]">Online</span>'}
    </div>
  `).join('');
}

async function deployModel(modelKey) {
  try {
    const res = await fetch('/api/models/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_key: modelKey })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Activated ${data.name}`, 'info');
      await refreshSystemStatus();
    }
  } catch (err) {
    showToast('Failed to switch model', 'warn');
  }
}

// ==============================================================================
// Station 1: Live Inspection
// ==============================================================================

async function loadCatalogPresets() {
  try {
    const res = await fetch('/api/catalog/presets');
    const data = await res.json();
    data.presets.forEach(p => {
      state.catalogPresets[p.id] = p;
    });
  } catch (err) {
    console.error('Presets error:', err);
  }
}

async function loadSamplePreset(presetId) {
  const preset = state.catalogPresets[presetId];
  if (!preset) return;

  const loading = document.getElementById('inspect-loading');
  loading.classList.remove('hidden');

  try {
    const res = await fetch('/api/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ img_path: preset.img_path })
    });
    const result = await res.json();
    renderInspectionResult(result);
  } catch (err) {
    showToast('Inspection request failed', 'warn');
  } finally {
    loading.classList.add('hidden');
  }
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const loading = document.getElementById('inspect-loading');
  loading.classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/inspect/upload', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    renderInspectionResult(result);
    showToast(`Inspected ${file.name}`, 'info');
  } catch (err) {
    showToast('Inspection upload failed', 'warn');
  } finally {
    loading.classList.add('hidden');
    event.target.value = '';
  }
}

function renderInspectionResult(data) {
  state.activeInspection = data;

  document.getElementById('empty-view-placeholder').classList.add('hidden');
  const imgEl = document.getElementById('main-view-img');
  imgEl.classList.remove('hidden');

  setInspectionViewMode(state.inspectionViewMode);

  const banner = document.getElementById('verdict-banner');
  const title = document.getElementById('verdict-title');
  const sub = document.getElementById('verdict-subtitle');
  const icon = document.getElementById('verdict-icon');

  banner.className = 'p-3.5 rounded-lg flex items-center gap-3 transition-all';
  if (data.is_defective) {
    banner.classList.add('badge-fail');
    icon.innerHTML = '<i data-lucide="alert-triangle" class="w-4 h-4 stroke-[1.5]"></i>';
    title.textContent = 'Defect detected';
    sub.textContent = `${data.defect_type} (${data.confidence_percentage}% confidence)`;
  } else {
    banner.classList.add('badge-pass');
    icon.innerHTML = '<i data-lucide="check" class="w-4 h-4 stroke-[1.5]"></i>';
    title.textContent = 'Passed';
    sub.textContent = `Nominal surface (${data.confidence_percentage}% confidence)`;
  }

  document.getElementById('inspect-latency').textContent = `${data.latency_ms} ms`;
  document.getElementById('metric-defect-type').textContent = data.defect_type;
  document.getElementById('metric-confidence').textContent = `${data.confidence_percentage}%`;

  const matchEl = document.getElementById('metric-match-score');
  if (data.focus_match_iou) {
    matchEl.textContent = `${data.focus_match_iou}% IoU match with ground-truth defect`;
  } else {
    matchEl.textContent = 'Nominal part (no defect expected)';
  }

  if (window.lucide) lucide.createIcons();
}

function setInspectionViewMode(mode) {
  state.inspectionViewMode = mode;
  document.getElementById('view-mode-clean').classList.toggle('active', mode === 'clean');
  document.getElementById('view-mode-heatmap').classList.toggle('active', mode === 'heatmap');

  const imgEl = document.getElementById('main-view-img');
  const captionEl = document.getElementById('lens-caption');

  if (!state.activeInspection) return;

  if (mode === 'clean') {
    imgEl.src = `data:image/jpeg;base64,${state.activeInspection.clean_image_b64}`;
    captionEl.textContent = 'Raw optical feed';
  } else {
    imgEl.src = `data:image/jpeg;base64,${state.activeInspection.heatmap_image_b64}`;
    captionEl.textContent = 'Grad-CAM visual localization heatmap';
  }
}

async function logOperatorDecision(decision) {
  if (!state.activeInspection) {
    showToast('Inspect a specimen before logging disposition', 'warn');
    return;
  }

  const payload = {
    image_name: state.activeInspection.image_name || 'specimen.png',
    defect_type: state.activeInspection.defect_type,
    confidence: state.activeInspection.confidence_percentage,
    decision: decision,
    notes: state.activeInspection.recommendation
  };

  try {
    const res = await fetch('/api/audit/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Logged disposition: ${decision}`, 'info');
      refreshAuditLog();
    }
  } catch (err) {
    showToast('Failed to record disposition', 'warn');
  }
}

async function runBatchInspection() {
  showToast('Screening 12 specimens in parallel...', 'info');
  try {
    const res = await fetch('/api/inspect/batch?limit=12');
    const data = await res.json();

    const container = document.getElementById('batch-results-container');
    const grid = document.getElementById('batch-cards-grid');
    const summary = document.getElementById('batch-summary-text');

    summary.textContent = `${data.total_screened} evaluated — ${data.rejected_count} rejected, ${data.passed_count} passed`;

    grid.innerHTML = data.items.map(item => `
      <div class="bg-[#09090a] p-2 rounded border ${item.is_flagged ? 'border-[rgba(229,72,77,0.3)]' : 'border-[rgba(76,195,138,0.25)]'} text-center space-y-1">
        <img src="data:image/jpeg;base64,${item.thumbnail_b64}" class="w-full h-24 object-contain rounded bg-[#0f0f10]" alt="${item.stem}">
        <div class="flex items-center justify-between text-[11px] font-mono px-0.5">
          <span class="${item.is_flagged ? 'text-[#e5484d]' : 'text-[#4cc38a]'} font-medium">${item.verdict}</span>
          <span class="text-[#85858b]">${item.confidence}%</span>
        </div>
        <div class="text-[11px] text-[#85858b] truncate px-0.5">${item.defect_type}</div>
      </div>
    `).join('');

    container.classList.remove('hidden');
    if (window.lucide) lucide.createIcons();
  } catch (err) {
    showToast('Batch screening failed', 'warn');
  }
}

// ==============================================================================
// Station 2: Defect Studio
// ==============================================================================

function setStudioFlaw(category, btn) {
  state.studioParams.flawCategory = category;
  document.querySelectorAll('.studio-flaw-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  generateStudioSample();
}

function setStudioPreset(preset, btn) {
  state.studioParams.preset = preset;
  document.querySelectorAll('.studio-pos-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  generateStudioSample();
}

function updateStudioSliderVals() {
  const scale = parseFloat(document.getElementById('studio-scale').value);
  const rot = parseInt(document.getElementById('studio-rot').value);
  document.getElementById('slider-scale-val').textContent = `${scale.toFixed(2)}x`;
  document.getElementById('slider-rot-val').textContent = `${rot}°`;
  state.studioParams.scale = scale;
  state.studioParams.rotation = rot;
}

async function generateStudioSample() {
  const payload = {
    flaw_category: state.studioParams.flawCategory,
    placement_preset: state.studioParams.preset,
    scale: state.studioParams.scale,
    rotation: state.studioParams.rotation,
    blend_style: 'poisson_normal'
  };

  try {
    const res = await fetch('/api/studio/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      state.lastStudioResult = data;
      document.getElementById('studio-preview-clean').src = `data:image/jpeg;base64,${data.clean_b64}`;
      document.getElementById('studio-preview-patch').src = `data:image/jpeg;base64,${data.patch_b64}`;
      document.getElementById('studio-preview-result').src = `data:image/jpeg;base64,${data.result_b64}`;
      document.getElementById('studio-preview-mask').src = `data:image/jpeg;base64,${data.mask_b64}`;
    }
  } catch (err) {
    console.error('Studio generation error:', err);
  }
}

async function commitStudioSample() {
  if (!state.lastStudioResult) {
    showToast('Generate a preview first', 'warn');
    return;
  }

  try {
    const res = await fetch('/api/studio/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        img_path: state.lastStudioResult.temp_img_path,
        mask_path: state.lastStudioResult.temp_mask_path
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast('Sample committed to training pool', 'info');
    }
  } catch (err) {
    showToast('Failed to commit sample', 'warn');
  }
}

// ==============================================================================
// Station 3: Model Hub & Background Training
// ==============================================================================

function updateScarcityLabel() {
  const val = parseInt(document.getElementById('train-scarcity').value);
  const parts = Math.max(3, Math.round(39 * (val / 100)));
  document.getElementById('train-scarcity-label').textContent = `${val}% (${parts} samples)`;
}

async function startBackgroundTraining() {
  const mode = document.querySelector('input[name="train-mode"]:checked').value;
  const scarcity = parseInt(document.getElementById('train-scarcity').value);

  const payload = {
    training_mode: mode,
    scarcity_percent: scarcity,
    epochs: 5
  };

  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.success) {
      showToast('Background training initiated', 'info');
      startBackgroundJobPolling();
    } else {
      showToast(data.detail || 'Training failed to start', 'warn');
    }
  } catch (err) {
    showToast('Error requesting training', 'warn');
  }
}

function startBackgroundJobPolling() {
  if (state.trainingPollingInterval) clearInterval(state.trainingPollingInterval);

  state.trainingPollingInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/train/status');
      const job = await res.json();

      if (job.is_running) {
        showBackgroundBanner(job);
        updateTrainingStationMonitor(job);
      } else {
        hideBackgroundBanner();
        updateTrainingStationMonitor(job);
        if (job.result && job.progress === 1.0) {
          await refreshSystemStatus();
        }
      }
    } catch (err) {
      console.warn('Poll error:', err);
    }
  }, 1500);
}

function showBackgroundBanner(job) {
  const banner = document.getElementById('bg-job-banner');
  banner.classList.remove('hidden');
  banner.classList.add('flex');

  document.getElementById('bg-banner-msg').textContent = job.message;
  const pct = Math.round(job.progress * 100);
  document.getElementById('bg-banner-progress').style.width = `${pct}%`;
  document.getElementById('bg-banner-pct').textContent = `${pct}%`;
}

function hideBackgroundBanner() {
  const banner = document.getElementById('bg-job-banner');
  banner.classList.add('hidden');
  banner.classList.remove('flex');
}

function updateTrainingStationMonitor(job) {
  const badge = document.getElementById('train-worker-status-badge');
  const bar = document.getElementById('train-worker-bar');
  const pctText = document.getElementById('train-worker-pct');
  const msgText = document.getElementById('train-worker-msg');
  const logLine = document.getElementById('train-log-line-1');

  if (job.is_running) {
    badge.className = 'text-xs font-mono text-[#3b82f6]';
    badge.textContent = `Running — Ep ${job.current_epoch}/${job.total_epochs}`;
    const pct = Math.round(job.progress * 100);
    bar.style.width = `${pct}%`;
    pctText.textContent = `${pct}%`;
    msgText.textContent = job.message;
    logLine.textContent = `Epoch ${job.current_epoch}/${job.total_epochs} — Loss: ${job.loss}`;
  } else if (job.result) {
    badge.className = 'text-xs font-mono text-[#4cc38a]';
    badge.textContent = 'Completed';
    bar.style.width = '100%';
    pctText.textContent = '100%';
    msgText.textContent = job.message;
    logLine.textContent = `Accuracy ${job.result.accuracy}% — Catch rate ${job.result.recall}%`;
  }
}

// ==============================================================================
// Station 4: Analytics & Audit
// ==============================================================================

function setupAnalyticsChart() {
  const ctx = document.getElementById('scarcityChart');
  if (!ctx) return;

  state.scarcityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['10% (3 samples)', '25% (10 samples)', '50% (19 samples)', '100% (39 samples)'],
      datasets: [
        {
          label: 'SynthQC augmented',
          data: [83.1, 88.9, 93.4, 96.5],
          borderColor: '#3b82f6',
          backgroundColor: 'transparent',
          tension: 0.2,
          borderWidth: 2,
          pointBackgroundColor: '#3b82f6',
          pointRadius: 3
        },
        {
          label: 'Standard baseline',
          data: [72.4, 84.6, 91.2, 96.9],
          borderColor: '#85858b',
          borderDash: [4, 4],
          tension: 0.2,
          borderWidth: 1.5,
          pointBackgroundColor: '#85858b',
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#a1a1a6', font: { size: 12, family: 'sans-serif' }, boxWidth: 12 }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}% F1`
          }
        }
      },
      scales: {
        y: {
          min: 65,
          max: 100,
          grid: { color: 'rgba(237, 237, 237, 0.04)' },
          ticks: { color: '#85858b', font: { size: 11, family: 'monospace' } }
        },
        x: {
          grid: { color: 'rgba(237, 237, 237, 0.04)' },
          ticks: { color: '#85858b', font: { size: 11, family: 'sans-serif' } }
        }
      }
    }
  });
}

function updateSensitivitySetting() {
  const val = parseFloat(document.getElementById('sensitivity-slider').value);
  let label = `${val.toFixed(2)}`;
  if (val <= 0.30) label += ' (Strict: Zero escapes)';
  else if (val >= 0.70) label += ' (Lenient: Zero false scrap)';
  else label += ' (Balanced: Factory standard)';

  document.getElementById('sensitivity-val-label').textContent = label;
}

async function refreshAuditLog() {
  try {
    const res = await fetch('/api/audit/history');
    const data = await res.json();
    const tbody = document.getElementById('audit-table-body');
    const countEl = document.getElementById('audit-log-count');

    if (!data.audit_log || data.audit_log.length === 0) {
      countEl.textContent = '0 logged';
      tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-[#85858b] font-sans">No operator dispositions recorded yet.</td></tr>';
      return;
    }

    countEl.textContent = `${data.audit_log.length} logged`;
    tbody.innerHTML = data.audit_log.map(item => `
      <tr class="hover:bg-[#19191b] transition-colors">
        <td class="py-2 px-3 text-[#85858b] font-mono">${item.timestamp}</td>
        <td class="py-2 px-3 font-sans text-[#ededed]">${item.image_name}</td>
        <td class="py-2 px-3 font-sans text-[#a1a1a6]">${item.defect_type}</td>
        <td class="py-2 px-3 text-[#ededed] font-mono">${item.confidence}</td>
        <td class="py-2 px-3 font-mono text-xs ${
          item.operator_decision === 'Approved' ? 'text-[#4cc38a]' :
          item.operator_decision === 'Rejected' ? 'text-[#e5484d]' :
          'text-[#f59e0b]'
        }">
          ${item.operator_decision}
        </td>
        <td class="py-2 px-3 font-sans text-[#85858b]">${item.notes}</td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('Audit fetch error:', err);
  }
}

function exportAuditCsv() {
  window.location.href = '/api/audit/export.csv';
  showToast('Downloading audit log CSV', 'info');
}

// ==============================================================================
// Toast Notifications
// ==============================================================================

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast';

  let iconName = 'info';
  if (type === 'warn') iconName = 'alert-triangle';

  toast.innerHTML = `
    <i data-lucide="${iconName}" class="w-4 h-4 stroke-[1.5] text-[#85858b] shrink-0"></i>
    <span class="text-xs text-[#ededed]">${message}</span>
  `;

  container.appendChild(toast);
  if (window.lucide) lucide.createIcons();

  setTimeout(() => toast.classList.add('show'), 10);
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 250);
  }, 3000);
}
