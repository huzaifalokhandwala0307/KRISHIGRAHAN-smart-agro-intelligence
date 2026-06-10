// ── Smooth Scroll ──
document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
        e.preventDefault();
        document.querySelector(a.getAttribute('href'))
            ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
});

// ── Form Validation ──
document.querySelector('form').addEventListener('submit', function(e) {
    const fields = {
        N:        { val: parseFloat(this.N.value),        min: 0,   max: 140, label: 'Nitrogen' },
        P:        { val: parseFloat(this.P.value),        min: 0,   max: 145, label: 'Phosphorus' },
        K:        { val: parseFloat(this.K.value),        min: 0,   max: 205, label: 'Potassium' },
        temp:     { val: parseFloat(this.temp.value),     min: 0,   max: 50,  label: 'Temperature' },
        humidity: { val: parseFloat(this.humidity.value), min: 0,   max: 100, label: 'Humidity' },
        ph:       { val: parseFloat(this.ph.value),       min: 0,   max: 14,  label: 'pH Level' },
        rainfall: { val: parseFloat(this.rainfall.value), min: 0,   max: 500, label: 'Rainfall' },
        moisture: { val: parseFloat(this.moisture.value), min: 0,   max: 100, label: 'Moisture' },
    };

    // Clear old errors
    document.querySelectorAll('.field-error').forEach(el => el.remove());
    document.querySelectorAll('.field input, .field select').forEach(el => {
        el.style.borderColor = '#d8d0c4';
    });

    let valid = true;

    // Check soil type
    if (!this.soil_type.value) {
        showError(this.soil_type, 'Please select a soil type');
        valid = false;
    }

    // Check numeric fields
    for (const [name, f] of Object.entries(fields)) {
        const input = this[name];
        if (input.value === '' || isNaN(f.val)) {
            showError(input, `${f.label} is required`);
            valid = false;
        } else if (f.val < f.min || f.val > f.max) {
            showError(input, `${f.label} must be ${f.min}–${f.max}`);
            valid = false;
        }
    }

    if (!valid) {
        e.preventDefault();
        // Scroll to first error
        document.querySelector('.field-error')
            ?.closest('.field')
            ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});

function showError(input, msg) {
    input.style.borderColor = '#d04030';
    const err = document.createElement('span');
    err.className = 'field-error';
    err.textContent = msg;
    err.style.cssText = 'font-size:10px;color:#d04030;letter-spacing:0.5px;margin-top:2px;';
    input.closest('.field').appendChild(err);
}

// ── Plant Disease Detection UI and Logic ──

const tabUploadBtn = document.getElementById('tab-upload-btn');
const tabCameraBtn = document.getElementById('tab-camera-btn');
const uploadContent = document.getElementById('upload-content');
const cameraContent = document.getElementById('camera-content');

const dragArea = document.getElementById('drag-area');
const browseBtn = document.getElementById('browse-btn');
const fileInput = document.getElementById('disease-file-input');

const uploadPreviewContainer = document.getElementById('upload-preview-container');
const uploadPreview = document.getElementById('upload-preview');
const removeUploadBtn = document.getElementById('remove-upload-btn');

const webcam = document.getElementById('webcam');
const photoCanvas = document.getElementById('photo-canvas');
const startCameraBtn = document.getElementById('start-camera-btn');
const captureBtn = document.getElementById('capture-btn');
const cameraPreviewContainer = document.getElementById('camera-preview-container');
const cameraPreview = document.getElementById('camera-preview');
const removeCameraBtn = document.getElementById('remove-camera-btn');

const detectBtn = document.getElementById('detect-btn');
const spinner = document.getElementById('disease-spinner');
const diseaseResult = document.getElementById('disease-result');

let activeTab = 'upload'; // 'upload' or 'camera'
let webcamStream = null;
let selectedBlob = null; // Store current file/capture blob to submit

if (tabUploadBtn && tabCameraBtn) {
    // Tab Switcher
    tabUploadBtn.addEventListener('click', () => {
        switchTab('upload');
    });

    tabCameraBtn.addEventListener('click', () => {
        switchTab('camera');
    });
}

function switchTab(tab) {
    if (activeTab === tab) return;
    activeTab = tab;
    
    if (tab === 'upload') {
        tabUploadBtn.classList.add('active');
        tabCameraBtn.classList.remove('active');
        uploadContent.style.display = 'block';
        cameraContent.style.display = 'none';
        stopWebcam();
        resetCameraState();
        updateDetectBtnState();
    } else {
        tabCameraBtn.classList.add('active');
        tabUploadBtn.classList.remove('active');
        cameraContent.style.display = 'block';
        uploadContent.style.display = 'none';
        resetUploadState();
        updateDetectBtnState();
    }
}

if (browseBtn && fileInput) {
    // Upload Tab logic
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });
}

if (dragArea) {
    // Drag and Drop
    dragArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dragArea.classList.add('active');
    });

    dragArea.addEventListener('dragleave', () => {
        dragArea.classList.remove('active');
    });

    dragArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dragArea.classList.remove('active');
        if (e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }
    selectedBlob = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadPreview.src = e.target.result;
        dragArea.style.display = 'none';
        uploadPreviewContainer.style.display = 'flex';
        updateDetectBtnState();
    };
    reader.readAsDataURL(file);
}

if (removeUploadBtn) {
    removeUploadBtn.addEventListener('click', () => {
        resetUploadState();
        updateDetectBtnState();
    });
}

function resetUploadState() {
    if (fileInput) fileInput.value = '';
    selectedBlob = null;
    if (uploadPreview) uploadPreview.src = '';
    if (dragArea) dragArea.style.display = 'flex';
    if (uploadPreviewContainer) uploadPreviewContainer.style.display = 'none';
}

if (startCameraBtn) {
    // Camera Tab logic
    startCameraBtn.addEventListener('click', async () => {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: { ideal: "environment" } },
                audio: false
            });
            webcam.srcObject = webcamStream;
            startCameraBtn.style.display = 'none';
            captureBtn.style.display = 'inline-block';
            webcam.style.display = 'block';
        } catch (err) {
            console.error("Webcam access failed: ", err);
            alert("Could not access the camera. Please make sure camera permissions are granted.");
        }
    });
}

if (captureBtn) {
    captureBtn.addEventListener('click', () => {
        if (!webcamStream) return;
        
        const context = photoCanvas.getContext('2d');
        photoCanvas.width = webcam.videoWidth || 640;
        photoCanvas.height = webcam.videoHeight || 480;
        
        // Draw current frame
        context.drawImage(webcam, 0, 0, photoCanvas.width, photoCanvas.height);
        
        // Convert canvas to blob
        photoCanvas.toBlob((blob) => {
            selectedBlob = blob;
            const url = URL.createObjectURL(blob);
            cameraPreview.src = url;
            
            // Hide webcam stream, show preview
            webcam.style.display = 'none';
            captureBtn.style.display = 'none';
            cameraPreviewContainer.style.display = 'flex';
            
            stopWebcam();
            updateDetectBtnState();
        }, 'image/jpeg');
    });
}

if (removeCameraBtn) {
    removeCameraBtn.addEventListener('click', () => {
        resetCameraState();
        // Restart camera automatically
        if (startCameraBtn) startCameraBtn.click();
    });
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (webcam) webcam.srcObject = null;
}

function resetCameraState() {
    stopWebcam();
    selectedBlob = null;
    if (cameraPreview) cameraPreview.src = '';
    if (webcam) webcam.style.display = 'block';
    if (startCameraBtn) startCameraBtn.style.display = 'inline-block';
    if (captureBtn) captureBtn.style.display = 'none';
    if (cameraPreviewContainer) cameraPreviewContainer.style.display = 'none';
}

function updateDetectBtnState() {
    if (detectBtn) {
        detectBtn.disabled = !selectedBlob;
    }
}

// Format class name
function formatDiseaseName(className) {
    if (!className) return { plant: '', disease: '' };
    // E.g., "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    let plantName = '';
    let diseaseName = className;
    
    if (className.includes('___')) {
        const parts = className.split('___');
        plantName = parts[0].replace(/_/g, ' ');
        diseaseName = parts[1].replace(/_/g, ' ');
    } else {
        diseaseName = className.replace(/_/g, ' ');
    }
    
    return {
        plant: plantName,
        disease: diseaseName
    };
}

if (detectBtn) {
    // Fetch prediction
    detectBtn.addEventListener('click', async () => {
        if (!selectedBlob) return;
        
        // Disable interface
        detectBtn.style.display = 'none';
        if (spinner) spinner.style.display = 'flex';
        if (diseaseResult) diseaseResult.style.display = 'none';
        
        const formData = new FormData();
        formData.append('leaf_image', selectedBlob, 'leaf_image.jpg');
        
        try {
            const response = await fetch('/predict_disease', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                renderDiseaseResult(data);
            } else {
                showErrorCard(data.error || 'Failed to detect disease.');
            }
        } catch (err) {
            console.error("Error predicting disease: ", err);
            showErrorCard("Network error. Unable to connect to server.");
        } finally {
            detectBtn.style.display = 'inline-block';
            if (spinner) spinner.style.display = 'none';
        }
    });
}

function renderDiseaseResult(data) {
    if (!diseaseResult) return;
    const { disease, confidence, is_healthy } = data;
    const info = formatDiseaseName(disease);
    
    let htmlContent = '';
    
    if (is_healthy) {
        htmlContent = `
            <div class="disease-result-card healthy">
                <h2>✅ Healthy Plant</h2>
                <p>The analyzed leaf from the <strong>${info.plant || 'plant'}</strong> appears to be healthy.</p>
                <div class="confidence-badge">Confidence: ${confidence}%</div>
            </div>
        `;
    } else {
        htmlContent = `
            <div class="disease-result-card diseased">
                <h2>⚠️ Diseased Plant Detected</h2>
                <p><strong>Affected Plant:</strong> ${info.plant || 'Unknown'}</p>
                <p><strong>Diagnosed Condition:</strong> ${info.disease}</p>
                <div class="confidence-badge">Confidence: ${confidence}%</div>
                <div class="advisory-note">
                    <strong>💡 What does this mean?</strong><br>
                    An infection or pest issue has been identified on your plant's leaf. We advise isolating the affected plants (if in pots) and consulting with a local agronomist or farming extension service to get an exact treatment plan.
                </div>
            </div>
        `;
    }
    
    diseaseResult.innerHTML = htmlContent;
    diseaseResult.style.display = 'block';
    
    // Scroll smoothly to results
    diseaseResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showErrorCard(msg) {
    if (!diseaseResult) return;
    diseaseResult.innerHTML = `
        <div class="disease-result-card diseased" style="background: #fff0ed; border-color: #d04030;">
            <h2 style="color: #b03020;">❌ Error</h2>
            <p style="color: #c04030;">${msg}</p>
        </div>
    `;
    diseaseResult.style.display = 'block';
    diseaseResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Mobile Navigation Toggle ──
document.addEventListener('DOMContentLoaded', () => {
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
    }
});
