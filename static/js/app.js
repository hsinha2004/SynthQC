/**
 * SynthQC Vision Station - Enterprise Frontend Controller
 * Handles Station switching, optical inference, flaw synthesis,
 * asynchronous background job polling, and analytics rendering.
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
  console.log('[SynthQC] Initializing Vision Station...');
  await refreshSystemStatus();
  await loadCatalogPresets();
  setupAnalyticsChart();
  startBackgroundJobPolling();

  // Load first default preset automatically so the view isn't empty!
  if (state.catalogPresets['preset_large']) {
    loadSamplePreset('preset_large');
  } else {
    // Generate default flaw preview in studio
    generateStudioSample();
  }
});

function switchTab(tabId) {
  state.activeTab = tabId;

  // Update nav button states
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.remove('active');
  });
  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  if (activeBtn) activeBtn.classList.add('active');

  // Toggle view visibility
  document.querySelectorAll('.station-view').forEach(view => {
    view.classList.add('hidden');
    view.classList.remove('block');
  });
  const targetView = document.getElementById(`station-${tabId}`);
  if (targetView) {
    targetView.classList.remove('hidden');
    targetView.classList.add('block');
  }

  // Refresh icons
  if (window.lucide) lucide.createIcons();

  // Specific tab triggers
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

    // Header labels
    document.getElementById('header-specimen').textContent = data.specimen_profile || 'Glass Bottle';
    document.getElementById('header-active-model').textContent = data.active_model?.name || 'Active Model';

    // Update deployed models list in Station 3
    renderDeployedModels(data.available_models || []);

    // Check if background job was already running
    if (data.background_job?.is_running) {
      showBackgroundBanner(data.background_job);
    }
  } catch (err) {
    console.error('Failed to fetch system status:', err);
  }
}

function renderDeployedModels(models) {
  const container = document.getElementById('models-list-container');
  if (!container) return;

  container.innerHTML = models.map(m => `
    <div class="flex items-center justify-between p-3 rounded-lg ${m.is_active ? 'bg-blue-950/30 border border-blue-500/40' : 'bg-[#161F33] border border-[#1E2B45]'} transition">
      <div>
        <div class="text-xs font-semibold text-slate-200 flex items-center gap-2">
          <span>${m.name}</span>
          ${m.is_active ? '<span class="text-[10px] px-2 py-0.2 rounded bg-blue-500/20 text-blue-400 font-normal">ACTIVE</span>' : ''}
        </div>
        <div class="text-[11px] text-slate-400 mt-0.5">Accuracy: ${m.accuracy}% | Latency: 18ms</div>
      </div>
      ${!m.is_active ? `
        <button onclick="deployModel('${m.key}')" class="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-[#1E2B45] transition">
          Deploy
        </button>
      ` : '<span class="text-xs text-blue-400 font-medium px-2">Online</span>'}
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
      showToast(`Switched active vision model to ${data.name}`, 'success');
      await refreshSystemStatus();
    }
  } catch (err) {
    showToast('Failed to switch model', 'error');
  }
}

// ==============================================================================
// Station 1: Inspection & Visual Explainability
// ==============================================================================

async function loadCatalogPresets() {
  try {
    const res = await fetch('/api/catalog/presets');
    const data = await res.json();
    data.presets.forEach(p => {
      state.catalogPresets[p.id] = p;
    });
  } catch (err) {
    console.error('Failed to load presets:', err);
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
    showToast('Inspection request failed', 'error');
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
    showToast(`Inspected uploaded image: ${file.name}`, 'info');
  } catch (err) {
    showToast('Upload inspection failed', 'error');
  } finally {
    loading.classList.add('hidden');
    event.target.value = ''; // Reset input
  }
}

function renderInspectionResult(data) {
  state.activeInspection = data;

  // Hide placeholder, show image
  document.getElementById('empty-view-placeholder').classList.add('hidden');
  const imgEl = document.getElementById('main-view-img');
  imgEl.classList.remove('hidden');

  // Set default view based on active mode
  setInspectionViewMode(state.inspectionViewMode);

  // Big Verdict Banner
  const banner = document.getElementById('verdict-banner');
  const title = document.getElementById('verdict-title');
  const sub = document.getElementById('verdict-subtitle');
  const icon = document.getElementById('verdict-icon');

  banner.className = 'p-4 rounded-xl flex items-center gap-3 transition-all';
  if (data.is_defective) {
    banner.classList.add('badge-fail');
    icon.className = 'w-8 h-8 rounded-lg flex items-center justify-center bg-rose-500/20 text-rose-400';
    icon.innerHTML = '<i data-lucide="alert-triangle" class="w-5 h-5"></i>';
    title.textContent = 'DEFECT DETECTED';
    sub.textContent = `${data.defect_type} (${data.confidence_percentage}% confidence)`;
  } else {
    banner.classList.add('badge-pass');
    icon.className = 'w-8 h-8 rounded-lg flex items-center justify-center bg-emerald-500/20 text-emerald-400';
    icon.innerHTML = '<i data-lucide="check-circle" class="w-5 h-5"></i>';
    title.textContent = 'PART PASSED';
    sub.textContent = `Clean / Flawless (${data.confidence_percentage}% confidence)`;
  }

  // Telemetry details
  document.getElementById('inspect-latency').textContent = `Latency: ${data.latency_ms} ms`;
  document.getElementById('metric-defect-type').textContent = data.defect_type;
  document.getElementById('metric-confidence').textContent = `${data.confidence_percentage}%`;

  const matchEl = document.getElementById('metric-match-score');
  if (data.focus_match_iou) {
    matchEl.textContent = `${data.focus_match_iou}% match with true flaw`;
  } else {
    matchEl.textContent = 'N/A (Pristine or Custom Part)';
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
    captionEl.textContent = 'Viewing raw optical camera feed';
  } else {
    imgEl.src = `data:image/jpeg;base64,${state.activeInspection.heatmap_image_b64}`;
    captionEl.textContent = 'Heatmap shows where the AI detected flaw textures';
  }
}

async function logOperatorDecision(decision) {
  if (!state.activeInspection) {
    showToast('Please inspect a part before recording a decision', 'warn');
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
      showToast(`Decision recorded: ${decision}`, 'success');
      refreshAuditLog();
    }
  } catch (err) {
    showToast('Failed to log decision', 'error');
  }
}

async function runBatchInspection() {
  showToast('Screening 12 conveyor samples in parallel...', 'info');
  try {
    const res = await fetch('/api/inspect/batch?limit=12');
    const data = await res.json();

    const container = document.getElementById('batch-results-container');
    const grid = document.getElementById('batch-cards-grid');
    const summary = document.getElementById('batch-summary-text');

    summary.textContent = `${data.total_screened} Screened | ${data.rejected_count} Rejected | ${data.passed_count} Passed`;

    grid.innerHTML = data.items.map(item => `
      <div class="bg-[#060911] p-2 rounded-lg border ${item.is_flagged ? 'border-rose-500/40 bg-rose-950/10' : 'border-emerald-500/40 bg-emerald-950/10'} text-center space-y-1.5">
        <img src="data:image/jpeg;base64,${item.thumbnail_b64}" class="w-full h-28 object-contain rounded bg-[#0A0E17]" alt="${item.stem}">
        <div class="flex items-center justify-between px-1 text-[10px]">
          <span class="font-bold ${item.is_flagged ? 'text-rose-400' : 'text-emerald-400'}">${item.verdict}</span>
          <span class="text-slate-400">${item.confidence}%</span>
        </div>
        <div class="text-[10px] text-slate-400 truncate px-1">${item.defect_type}</div>
      </div>
    `).join('');

    container.classList.remove('hidden');
    if (window.lucide) lucide.createIcons();
    showToast(`Batch complete: ${data.rejected_count} flaws stopped`, 'success');
  } catch (err) {
    showToast('Batch screening failed', 'error');
  }
}

// ==============================================================================
// Station 2: Defect Studio (Data Augmentation)
// ==============================================================================

function setStudioFlaw(category, btn) {
  state.studioParams.flawCategory = category;
  document.querySelectorAll('.studio-flaw-btn').forEach(b => {
    b.classList.remove('active', 'bg-blue-600/20', 'border-blue-500/40', 'text-blue-400');
    b.classList.add('bg-[#161F33]', 'border-[#1E2B45]', 'text-slate-300');
  });
  btn.classList.add('active', 'bg-blue-600/20', 'border-blue-500/40', 'text-blue-400');
  generateStudioSample();
}

function setStudioPreset(preset, btn) {
  state.studioParams.preset = preset;
  document.querySelectorAll('.studio-pos-btn').forEach(b => {
    b.classList.remove('active', 'bg-blue-600/20', 'border-blue-500/40', 'text-blue-400');
    b.classList.add('bg-[#161F33]', 'border-[#1E2B45]', 'text-slate-300');
  });
  btn.classList.add('active', 'bg-blue-600/20', 'border-blue-500/40', 'text-blue-400');
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
    console.error('Studio generation failed:', err);
  }
}

async function commitStudioSample() {
  if (!state.lastStudioResult) {
    showToast('Generate a defect preview first', 'warn');
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
      showToast(data.message, 'success');
    }
  } catch (err) {
    showToast('Failed to save sample to pool', 'error');
  }
}

// ==============================================================================
// Station 3: Model Hub & Background Training
// ==============================================================================

function updateScarcityLabel() {
  const val = parseInt(document.getElementById('train-scarcity').value);
  const parts = Math.max(3, Math.round(39 * (val / 100)));
  document.getElementById('train-scarcity-label').textContent = `${val}% (${parts} physical broken parts)`;
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
      showToast('Training initiated in background! You can keep inspecting parts.', 'info');
      startBackgroundJobPolling();
    } else {
      showToast(data.detail || 'Training failed to start', 'error');
    }
  } catch (err) {
    showToast('Error requesting training start', 'error');
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
          // Completed
          await refreshSystemStatus();
        }
      }
    } catch (err) {
      console.warn('Status poll error:', err);
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

  document.getElementById('train-tab-badge').classList.remove('hidden');
}

function hideBackgroundBanner() {
  const banner = document.getElementById('bg-job-banner');
  banner.classList.add('hidden');
  banner.classList.remove('flex');
  document.getElementById('train-tab-badge').classList.add('hidden');
}

function updateTrainingStationMonitor(job) {
  const badge = document.getElementById('train-worker-status-badge');
  const bar = document.getElementById('train-worker-bar');
  const pctText = document.getElementById('train-worker-pct');
  const msgText = document.getElementById('train-worker-msg');
  const logLine = document.getElementById('train-log-line-1');

  if (job.is_running) {
    badge.className = 'text-xs px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 font-mono animate-pulse';
    badge.textContent = `Running (Ep ${job.current_epoch}/${job.total_epochs})`;
    const pct = Math.round(job.progress * 100);
    bar.style.width = `${pct}%`;
    pctText.textContent = `${pct}%`;
    msgText.textContent = job.message;
    logLine.textContent = `Loss: ${job.loss} | Epoch ${job.current_epoch}/${job.total_epochs}`;
  } else if (job.result) {
    badge.className = 'text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 font-mono';
    badge.textContent = 'Completed';
    bar.style.width = '100%';
    pctText.textContent = '100%';
    msgText.textContent = job.message;
    logLine.textContent = `Deployed: Accuracy ${job.result.accuracy}% | Catch Rate ${job.result.recall}%`;
  }
}

// ==============================================================================
// Station 4: Analytics & Audit Log
// ==============================================================================

function setupAnalyticsChart() {
  const ctx = document.getElementById('scarcityChart');
  if (!ctx) return;

  state.scarcityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['10% (3 parts)', '25% (10 parts)', '50% (19 parts)', '100% (39 parts)'],
      datasets: [
        {
          label: 'SynthQC Augmented (Arm B)',
          data: [83.1, 88.9, 93.4, 96.5],
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.3,
          borderWidth: 2.5,
          pointBackgroundColor: '#3B82F6',
          pointRadius: 4,
          fill: true
        },
        {
          label: 'Standard Baseline (Arm A)',
          data: [72.4, 84.6, 91.2, 96.9],
          borderColor: '#94A3B8',
          borderDash: [5, 5],
          tension: 0.3,
          borderWidth: 2,
          pointBackgroundColor: '#94A3B8',
          pointRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#CBD5E1', font: { size: 11 } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}% F1 Score`
          }
        }
      },
      scales: {
        y: {
          min: 65,
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94A3B8', font: { size: 10 } }
        },
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94A3B8', font: { size: 10 } }
        }
      }
    }
  });
}

function updateSensitivitySetting() {
  const val = parseFloat(document.getElementById('sensitivity-slider').value);
  let label = `${val.toFixed(2)}`;
  if (val <= 0.30) label += ' (Strict: Zero Defect Escape)';
  else if (val >= 0.70) label += ' (Lenient: Zero Waste)';
  else label += ' (Balanced: Factory Recommended)';

  document.getElementById('sensitivity-val-label').textContent = label;
}

async function refreshAuditLog() {
  try {
    const res = await fetch('/api/audit/history');
    const data = await res.json();
    const tbody = document.getElementById('audit-table-body');
    const countEl = document.getElementById('audit-log-count');

    if (!data.audit_log || data.audit_log.length === 0) {
      countEl.textContent = '0 records logged';
      tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-slate-500 font-sans">No operator dispositions recorded yet. Inspect parts in Station 1 to log decisions.</td></tr>';
      return;
    }

    countEl.textContent = `${data.audit_log.length} records logged`;
    tbody.innerHTML = data.audit_log.map(item => `
      <tr class="hover:bg-[#161F33]/50 transition">
        <td class="py-2 px-3 text-slate-400">${item.timestamp}</td>
        <td class="py-2 px-3 font-sans font-medium text-slate-200">${item.image_name}</td>
        <td class="py-2 px-3 font-sans text-slate-300">${item.defect_type}</td>
        <td class="py-2 px-3 text-blue-400">${item.confidence}</td>
        <td class="py-2 px-3">
          <span class="px-2 py-0.5 rounded text-[10px] font-sans font-semibold ${
            item.operator_decision === 'Approved' ? 'bg-emerald-500/20 text-emerald-400' :
            item.operator_decision === 'Rejected' ? 'bg-rose-500/20 text-rose-400' :
            'bg-amber-500/20 text-amber-400'
          }">
            ${item.operator_decision}
          </span>
        </td>
        <td class="py-2 px-3 font-sans text-slate-400">${item.notes}</td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('Failed to fetch audit log:', err);
  }
}

function exportAuditCsv() {
  window.location.href = '/api/audit/export.csv';
  showToast('Downloading operator audit report...', 'info');
}

// ==============================================================================
// Toast Notifications
// ==============================================================================

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast';

  let iconName = 'info';
  let iconColor = 'text-blue-400';
  if (type === 'success') {
    iconName = 'check-circle';
    iconColor = 'text-emerald-400';
  } else if (type === 'warn') {
    iconName = 'alert-triangle';
    iconColor = 'text-amber-400';
  } else if (type === 'error') {
    iconName = 'alert-circle';
    iconColor = 'text-rose-400';
  }

  toast.innerHTML = `
    <i data-lucide="${iconName}" class="w-4 h-4 ${iconColor} shrink-0"></i>
    <span class="flex-1">${message}</span>
  `;

  container.appendChild(toast);
  if (window.lucide) lucide.createIcons();

  // Animate in
  setTimeout(() => toast.classList.add('show'), 10);

  // Auto remove after 3.5s
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
