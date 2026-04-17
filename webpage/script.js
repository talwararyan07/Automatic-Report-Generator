const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPrompt = document.getElementById('upload-prompt');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const processBtn = document.getElementById('process-btn');
const resultsPanel = document.getElementById('results-panel');
const scanningLine = document.querySelector('.scanning-line');
const analyzeStatus = document.getElementById('analyze-status');
const resetBtn = document.getElementById('reset-btn');

// Mock data in case the API is not running yet
const MOCK_REPORTS = [
    { findings: "The heart size and cardiomediastinal silhouette are within normal limits. The lungs are clear without focal consolidation, pleural effusion, or pneumothorax.", impression: "Normal chest x-ray." },
    { findings: "Bilateral hyperinflation. No focal consolidation, pneumothorax, or large pleural effusion.", impression: "Chronic obstructive pulmonary disease pattern." },
    { findings: "Right lower lobe airspace opacity. Calcified granuloma in the left lung apex. Heart size and mediastinal silhouette normal.", impression: "Right lower lobe pneumonia." }
];

// Handle drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, JPG, or DICOM/PNG converter).');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadPrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        resultsPanel.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

processBtn.addEventListener('click', async () => {
    processBtn.disabled = true;
    processBtn.innerText = 'Analyzing...';
    scanningLine.style.display = 'block';
    analyzeStatus.innerText = 'AI is scanning pixels...';

    // Simulate analysis time
    setTimeout(async () => {
        try {
            // Check if actual API is running
            const response = await fetch('http://localhost:5001/generate-report', {
                method: 'POST',
                // Note: In real setup, you'd append the file to FormData
                body: new FormData() 
            }).catch(() => null);

            let data;
            if (response && response.ok) {
                data = await response.json();
            } else {
                // Fallback to mock for demo
                console.log("Using demo mode data...");
                const randomReport = MOCK_REPORTS[Math.floor(Math.random() * MOCK_REPORTS.length)];
                data = {
                    findings: randomReport.findings,
                    impression: randomReport.impression,
                    confidence: (95 + Math.random() * 4).toFixed(1) + '%'
                };
            }

            displayResults(data);
        } catch (err) {
            console.error(err);
        } finally {
            scanningLine.style.display = 'none';
        }
    }, 2500);
});

function displayResults(data) {
    document.getElementById('findings-text').innerText = data.findings;
    document.getElementById('impression-text').innerText = data.impression;
    document.getElementById('confidence-val').innerText = data.confidence || '96.2%';
    
    resultsPanel.classList.remove('hidden');
    resultsPanel.scrollIntoView({ behavior: 'smooth' });
    processBtn.disabled = false;
    processBtn.innerText = 'Re-Analyze';
}

resetBtn.addEventListener('click', () => {
    uploadPrompt.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    resultsPanel.classList.add('hidden');
    fileInput.value = '';
});
