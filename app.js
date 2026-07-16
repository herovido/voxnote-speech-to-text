const fileInput = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const selectedFile = document.querySelector('#selected-file');
const fileName = document.querySelector('#file-name');
const fileSize = document.querySelector('#file-size');
const removeFile = document.querySelector('#remove-file');
const processButton = document.querySelector('#process-button');
const progressWrap = document.querySelector('#progress-wrap');
const progressValue = document.querySelector('#progress-value');
const progressBar = document.querySelector('#progress-bar');
const progressLabel = progressWrap.querySelector('.progress-copy span');
const workspace = document.querySelector('#workspace');
const toast = document.querySelector('#toast');
const maxFileSize = 2 * 1024 * 1024 * 1024;
const allowedExtensions = ['mp3', 'wav', 'm4a', 'mp4', 'mov', 'webm', 'ogg'];
let currentFile = null;
let toastTimer;
const apiBase = window.location.port === '4173' ? 'http://127.0.0.1:8000' : '';

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toastTimer = window.setTimeout(() => { toast.hidden = true; }, 3500);
}

function validateFile(file) {
  const extension = file.name.split('.').pop().toLowerCase();
  if (!allowedExtensions.includes(extension)) return 'Định dạng file chưa được hỗ trợ.';
  if (file.size > maxFileSize) return 'File vượt quá giới hạn 2 GB.';
  return '';
}

function setFile(file) {
  if (!file) return;
  const error = validateFile(file);
  if (error) {
    showToast(error);
    return;
  }
  currentFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = `${formatSize(file.size)} · Sẵn sàng xử lý`;
  selectedFile.hidden = false;
  dropZone.hidden = true;
  processButton.disabled = false;
  processButton.querySelector('span').textContent = 'Chuyển thành văn bản';
}

fileInput.addEventListener('change', (event) => setFile(event.target.files[0]));

['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add('dragging');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragging');
  });
});

dropZone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));

removeFile.addEventListener('click', () => {
  currentFile = null;
  fileInput.value = '';
  selectedFile.hidden = true;
  dropZone.hidden = false;
  processButton.disabled = true;
  progressWrap.hidden = true;
  progressBar.style.width = '0%';
});

function updateProgress(job) {
  const progress = Math.max(0, Math.min(Number(job.progress) || 0, 100));
  progressLabel.textContent = job.stage || 'Đang xử lý…';
  progressValue.textContent = `${progress}%`;
  progressBar.style.width = `${progress}%`;
}

function apiErrorMessage(error) {
  return error instanceof Error ? error.message : 'Không thể kết nối backend.';
}

async function readApiError(response) {
  try {
    const payload = await response.json();
    return payload.detail || `Yêu cầu thất bại (${response.status}).`;
  } catch (error) {
    return `Yêu cầu thất bại (${response.status}).`;
  }
}

async function pollJob(jobId) {
  while (true) {
    const response = await fetch(`${apiBase}/api/jobs/${jobId}`);
    if (!response.ok) throw new Error(await readApiError(response));
    const job = await response.json();
    updateProgress(job);

    if (job.status === 'completed') return job;
    if (job.status === 'failed') throw new Error(job.error || 'Backend không thể xử lý file.');
    await new Promise((resolve) => window.setTimeout(resolve, 450));
  }
}

processButton.addEventListener('click', async () => {
  if (!currentFile) return;
  processButton.disabled = true;
  processButton.querySelector('span').textContent = 'Đang tải lên…';
  progressWrap.hidden = false;
  updateProgress({ progress: 2, stage: 'Đang gửi file tới backend…' });

  try {
    const formData = new FormData();
    formData.append('file', currentFile, currentFile.name);
    const response = await fetch(`${apiBase}/api/jobs`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error(await readApiError(response));
    const createdJob = await response.json();
    updateProgress(createdJob);
    const completedJob = await pollJob(createdJob.id);
    renderJobResult(completedJob.result);
    processButton.querySelector('span').textContent = 'Đã hoàn tất';
    workspace.hidden = false;
    showToast(`Đã xử lý file ở chế độ ${completedJob.mode}.`);
    window.setTimeout(() => workspace.scrollIntoView({ behavior: 'smooth', block: 'start' }), 250);
  } catch (error) {
    processButton.disabled = false;
    processButton.querySelector('span').textContent = 'Thử lại';
    progressLabel.textContent = 'Xử lý chưa thành công';
    showToast(apiErrorMessage(error));
  }
});

const tabs = document.querySelectorAll('.tab');
tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    tabs.forEach((item) => {
      const panel = document.querySelector(`#${item.dataset.panel}`);
      const active = item === tab;
      item.classList.toggle('active', active);
      item.setAttribute('aria-selected', String(active));
      panel.hidden = !active;
    });
  });
});

const searchButton = document.querySelector('#search-button');
const searchPanel = document.querySelector('#search-panel');
const searchInput = document.querySelector('#transcript-search');
const clearSearch = document.querySelector('#clear-search');
const searchResult = document.querySelector('#search-result');
let utterances = [...document.querySelectorAll('.utterance')];

function prepareUtterances() {
  utterances = [...document.querySelectorAll('.utterance')];
  utterances.forEach((utterance) => {
    const paragraph = utterance.querySelector('p');
    paragraph.dataset.original = paragraph.textContent;
  });
}

prepareUtterances();

searchButton.addEventListener('click', () => {
  const opening = searchPanel.hidden;
  searchPanel.hidden = !opening;
  searchButton.setAttribute('aria-expanded', String(opening));
  if (opening) searchInput.focus();
});

function updateSearch() {
  const query = searchInput.value.trim();
  let matches = 0;
  utterances.forEach((utterance) => {
    const paragraph = utterance.querySelector('p');
    const speaker = utterance.querySelector('[data-speaker-name]').textContent;
    const source = paragraph.dataset.original;
    const isMatch = !query || `${speaker} ${source}`.toLocaleLowerCase('vi').includes(query.toLocaleLowerCase('vi'));
    utterance.classList.toggle('search-hidden', !isMatch);
    paragraph.textContent = source;
    if (isMatch && query) {
      const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      paragraph.innerHTML = source.replace(new RegExp(escaped, 'giu'), '<mark>$&</mark>');
      matches += 1;
    }
  });
  searchResult.textContent = query ? `Tìm thấy ${matches} đoạn phù hợp.` : 'Nhập từ khóa để bắt đầu tìm kiếm.';
}

searchInput.addEventListener('input', updateSearch);
clearSearch.addEventListener('click', () => {
  searchInput.value = '';
  updateSearch();
  searchInput.focus();
});

const renameDialog = document.querySelector('#rename-dialog');
const renameForm = document.querySelector('#rename-form');
const speakerNameInput = document.querySelector('#speaker-name-input');
const cancelRename = document.querySelector('#cancel-rename');
let activeSpeakerId = '';

function initials(name) {
  return name.trim().split(/\s+/).slice(-2).map((part) => part[0]).join('').toLocaleUpperCase('vi');
}

function bindSpeakerButtons() {
  document.querySelectorAll('[data-speaker]').forEach((button) => {
    button.addEventListener('click', () => {
      activeSpeakerId = button.dataset.speaker;
      speakerNameInput.value = document.querySelector(`[data-speaker-name="${activeSpeakerId}"]`).textContent;
      renameDialog.showModal();
      speakerNameInput.select();
    });
  });
}

function applySavedSpeakerNames() {
  document.querySelectorAll('[data-speaker-name]').forEach((nameNode) => {
    const id = nameNode.dataset.speakerName;
    try {
      const savedName = localStorage.getItem(`voxnote-${id}`);
      if (savedName) {
        nameNode.textContent = savedName;
        const button = document.querySelector(`[data-speaker="${id}"]`);
        button.textContent = initials(savedName);
        button.setAttribute('aria-label', `Đổi tên ${savedName}`);
      }
    } catch (error) { /* Continue without persistence. */ }
  });
}

function formatTimestamp(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function renderJobResult(result) {
  if (!result) return;
  document.querySelector('#meeting-title').textContent = result.title || currentFile.name;
  document.querySelector('#summary-text').textContent = result.summary || 'Chưa có tóm tắt.';
  document.querySelector('#speaker-count').textContent = `${result.speaker_count || 0} người nói`;

  const speakerMap = new Map((result.speakers || []).map((speaker) => [speaker.id, speaker]));
  const transcriptPanel = document.querySelector('#transcript-panel');
  transcriptPanel.replaceChildren();

  (result.segments || []).forEach((segment, index) => {
    const speaker = speakerMap.get(segment.speaker_id) || {
      id: segment.speaker_id,
      name: 'Người nói',
      initials: 'N',
      color: 'speaker-one',
    };
    const utterance = document.createElement('div');
    utterance.className = `utterance${index === 0 ? ' active-utterance' : ''}`;

    const speakerButton = document.createElement('button');
    speakerButton.className = `speaker ${speaker.color || 'speaker-one'}`;
    speakerButton.type = 'button';
    speakerButton.dataset.speaker = speaker.id;
    speakerButton.title = 'Đổi tên người nói';
    speakerButton.setAttribute('aria-label', `Đổi tên ${speaker.name}`);
    speakerButton.textContent = speaker.initials || initials(speaker.name);

    const content = document.createElement('div');
    const meta = document.createElement('div');
    meta.className = 'utterance-meta';
    const name = document.createElement('strong');
    name.dataset.speakerName = speaker.id;
    name.textContent = speaker.name;
    const time = document.createElement('time');
    time.textContent = formatTimestamp(segment.start_seconds || 0);
    const paragraph = document.createElement('p');
    paragraph.textContent = segment.text;
    meta.append(name, time);
    content.append(meta, paragraph);
    utterance.append(speakerButton, content);
    transcriptPanel.append(utterance);
  });

  prepareUtterances();
  bindSpeakerButtons();
  applySavedSpeakerNames();
  searchInput.value = '';
  updateSearch();
}

bindSpeakerButtons();
applySavedSpeakerNames();

cancelRename.addEventListener('click', () => renameDialog.close());
renameForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const newName = speakerNameInput.value.trim();
  if (!newName) return;
  document.querySelectorAll(`[data-speaker-name="${activeSpeakerId}"]`).forEach((node) => { node.textContent = newName; });
  const speakerButton = document.querySelector(`[data-speaker="${activeSpeakerId}"]`);
  speakerButton.textContent = initials(newName);
  speakerButton.setAttribute('aria-label', `Đổi tên ${newName}`);
  try { localStorage.setItem(`voxnote-${activeSpeakerId}`, newName); } catch (error) { /* Private mode can block storage. */ }
  renameDialog.close();
  showToast(`Đã đổi tên người nói thành ${newName}.`);
  updateSearch();
});

const playButton = document.querySelector('#play-button');
playButton.addEventListener('click', () => {
  const playing = playButton.classList.toggle('playing');
  playButton.setAttribute('aria-label', playing ? 'Tạm dừng bản ghi' : 'Phát bản ghi');
  playButton.innerHTML = playing
    ? '<svg viewBox="0 0 24 24"><path d="M8 6v12M16 6v12"/></svg>'
    : '<svg viewBox="0 0 24 24"><path d="m8 5 11 7-11 7V5Z"/></svg>';
  showToast(playing ? 'Đang phát bản ghi mẫu.' : 'Đã tạm dừng.');
});

document.querySelectorAll('.task-item input').forEach((checkbox) => {
  try { checkbox.checked = localStorage.getItem(`voxnote-${checkbox.id}`) === 'done'; } catch (error) { /* Continue without persistence. */ }
  checkbox.addEventListener('change', () => {
    try { localStorage.setItem(`voxnote-${checkbox.id}`, checkbox.checked ? 'done' : 'open'); } catch (error) { /* Continue without persistence. */ }
    showToast(checkbox.checked ? 'Đã đánh dấu công việc hoàn thành.' : 'Đã mở lại công việc.');
  });
});

document.querySelector('#export-button').addEventListener('click', () => {
  const lines = utterances.map((utterance) => {
    const speaker = utterance.querySelector('[data-speaker-name]').textContent;
    const time = utterance.querySelector('time').textContent;
    const text = utterance.querySelector('p').dataset.original;
    return `[${time}] ${speaker}: ${text}`;
  });
  const header = 'VoxNote — Cuộc họp tuần, Nhóm sản phẩm\n\n';
  const blob = new Blob([header + lines.join('\n\n')], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'voxnote-ban-ghi-cuoc-hop.txt';
  link.click();
  URL.revokeObjectURL(url);
  showToast('Đã xuất bản ghi dạng TXT.');
});

document.querySelector('.button-ghost').addEventListener('click', () => showToast('Đăng nhập sẽ được kết nối ở giai đoạn backend.'));
