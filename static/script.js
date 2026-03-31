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
