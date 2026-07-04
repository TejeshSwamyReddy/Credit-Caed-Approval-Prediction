/**
 * Apex Bank Portal Interactivity Script
 */

document.addEventListener('DOMContentLoaded', function() {
    // 1. Navbar Scroll Effect
    const navbar = document.getElementById('mainNavbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 20) {
                navbar.classList.add('shadow-lg');
                navbar.style.padding = '10px 0';
            } else {
                navbar.classList.remove('shadow-lg');
                navbar.style.padding = '16px 0';
            }
        });
    }

    // 2. Auto-Dismiss Flash Alerts
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // 3. Client-Side Form Validation (Bootstrap 5)
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // 4. Dynamic Income Calculation & Field Disabling
    const annualIncomeInput = document.getElementById('annual_income');
    const monthlyIncomeInput = document.getElementById('monthly_income');
    const empStatusSelect = document.getElementById('employment_status');
    const empDurationInput = document.getElementById('employment_duration');

    if (annualIncomeInput && monthlyIncomeInput) {
        const calculateMonthly = function() {
            const annual = parseFloat(annualIncomeInput.value);
            if (!isNaN(annual) && annual >= 0) {
                const monthly = annual / 12.0;
                monthlyIncomeInput.value = monthly.toFixed(2);
            } else {
                monthlyIncomeInput.value = '';
            }
        };

        annualIncomeInput.addEventListener('input', calculateMonthly);
        annualIncomeInput.addEventListener('change', calculateMonthly);
        
        // Run initial calculation (in case validation failed and values were re-injected)
        calculateMonthly();
    }

    if (empStatusSelect && empDurationInput) {
        const configureEmploymentFields = function() {
            const status = empStatusSelect.value;
            if (status === 'Unemployed') {
                empDurationInput.value = 0;
                empDurationInput.setAttribute('readonly', 'true');
            } else {
                empDurationInput.removeAttribute('readonly');
            }
        };

        empStatusSelect.addEventListener('change', configureEmploymentFields);
        
        // Run initial configuration
        configureEmploymentFields();
    }
});
