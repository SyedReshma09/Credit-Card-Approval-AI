// Custom JavaScript for Credit Card Approval AI

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Auto-calculate family members based on children
    const childrenInput = document.getElementById('children');
    const familyMembersInput = document.getElementById('family_members');
    
    if (childrenInput && familyMembersInput) {
        childrenInput.addEventListener('change', function() {
            const children = parseInt(this.value) || 0;
            const currentFamily = parseInt(familyMembersInput.value) || 1;
            if (currentFamily < children + 1) {
                familyMembersInput.value = children + 1;
            }
        });
    }
    
    // Income formatting
    const incomeInput = document.getElementById('income');
    if (incomeInput) {
        incomeInput.addEventListener('blur', function() {
            const value = parseInt(this.value);
            if (value > 0) {
                this.value = value.toLocaleString();
            }
        });
        
        incomeInput.addEventListener('focus', function() {
            const value = this.value.replace(/,/g, '');
            if (value) {
                this.value = value;
            }
        });
    }
    
    // Prediction form submit with loading state
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
                
                // Show loading overlay
                showLoading();
            }
        });
    }
    
    // Age validation (18-100)
    const ageInput = document.getElementById('age');
    if (ageInput) {
        ageInput.addEventListener('input', function() {
            const value = parseInt(this.value);
            if (value < 18) {
                this.setCustomValidity('Age must be at least 18');
            } else if (value > 100) {
                this.setCustomValidity('Age must be less than 100');
            } else {
                this.setCustomValidity('');
            }
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Loading overlay functions
function showLoading() {
    let overlay = document.querySelector('.spinner-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'spinner-overlay';
        overlay.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" style="width: 3rem; height: 3rem;" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">Processing your application...</p>
            </div>
        `;
        document.body.appendChild(overlay);
    }
    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.querySelector('.spinner-overlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

// API call function
async function makePrediction(data) {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Chart interaction handlers
function handleChartClick(chartId, data) {
    console.log(`Chart ${chartId} clicked:`, data);
    // Add custom chart interaction logic here
}

// Export functions for use in other scripts
window.creditApp = {
    showLoading,
    hideLoading,
    makePrediction,
    handleChartClick
};