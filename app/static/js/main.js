document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupThemeToggle();
    setupActiveNav();
});

function initTheme() {
    const savedTheme = localStorage.getItem('kg_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function setupThemeToggle() {
    const themeBtn = document.getElementById('themeToggleBtn');
    if (!themeBtn) return;

    themeBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('kg_theme', newTheme);
        updateThemeIcon(newTheme);

        if (window.renderDashboardCharts) {
            window.renderDashboardCharts();
        }
    });
}

function updateThemeIcon(theme) {
    const icon = document.querySelector('#themeToggleBtn i');
    if (!icon) return;
    if (theme === 'dark') {
        icon.className = 'fas fa-sun text-warning';
    } else {
        icon.className = 'fas fa-moon text-primary';
    }
}

function setupActiveNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link-custom');
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}
