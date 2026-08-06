document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('correctVsWrongChart')) {
        renderDashboardCharts();
    }
    if (document.getElementById('knowledgeGapGauge')) {
        animateKnowledgeGapGauge();
    }
    setupRecommendationFilters();
});

window.renderDashboardCharts = function() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#cbd5e1' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)';

    const chartData = window.DASHBOARD_DATA || {
        correct: 0,
        wrong: 0,
        unanswered: 0,
        skills: [],
        skillScores: []
    };

    // 1. Donut Chart - Correct vs Wrong
    const ctxPie = document.getElementById('correctVsWrongChart')?.getContext('2d');
    if (ctxPie) {
        if (window.donutChartInstance) window.donutChartInstance.destroy();

        window.donutChartInstance = new Chart(ctxPie, {
            type: 'doughnut',
            data: {
                labels: ['Correct Answers', 'Wrong Answers', 'Unanswered'],
                datasets: [{
                    data: [chartData.correct, chartData.wrong, chartData.unanswered],
                    backgroundColor: ['#10b981', '#ef4444', '#f59e0b'],
                    borderWidth: 0,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: textColor, font: { family: 'Inter', size: 12, weight: 600 } }
                    }
                },
                cutout: '72%'
            }
        });
    }

    // 2. Bar Chart - Skill-wise Performance
    const ctxBar = document.getElementById('skillPerformanceChart')?.getContext('2d');
    if (ctxBar) {
        if (window.barChartInstance) window.barChartInstance.destroy();

        window.barChartInstance = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: chartData.skills,
                datasets: [{
                    label: 'Skill Mastery (%)',
                    data: chartData.skillScores,
                    backgroundColor: chartData.skillScores.map(score => 
                        score < 50 ? '#ef4444' : score < 75 ? '#f59e0b' : '#6366f1'
                    ),
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: textColor, callback: value => value + '%' },
                        grid: { color: gridColor }
                    },
                    x: {
                        ticks: { color: textColor },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
};

function animateKnowledgeGapGauge() {
    const gaugeCircle = document.getElementById('knowledgeGapGaugeCircle');
    const gapScoreText = document.getElementById('knowledgeGapScoreText');
    if (!gaugeCircle || !gapScoreText) return;

    const gapPct = parseFloat(gapScoreText.getAttribute('data-gap-score') || '30');

    if (gapPct > 50) {
        gaugeCircle.style.stroke = '#ef4444';
    } else if (gapPct > 25) {
        gaugeCircle.style.stroke = '#f59e0b';
    } else {
        gaugeCircle.style.stroke = '#10b981';
    }

    const strokeDasharray = 440;
    const offset = strokeDasharray - (strokeDasharray * (gapPct / 100));

    setTimeout(() => {
        gaugeCircle.style.strokeDashoffset = offset;
    }, 300);
}

function setupRecommendationFilters() {
    const filterBtns = document.querySelectorAll('.rec-filter-btn');
    const recCards = document.querySelectorAll('.rec-card-item');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => {
                b.classList.remove('active', 'btn-brand');
                b.classList.add('btn-brand-outline');
            });

            btn.classList.remove('btn-brand-outline');
            btn.classList.add('btn-brand', 'active');

            const filterValue = btn.getAttribute('data-filter');

            recCards.forEach(card => {
                const category = card.getAttribute('data-category');
                if (filterValue === 'all') {
                    card.style.display = 'block';
                } else if (filterValue === 'critical') {
                    card.style.display = (category === 'critical') ? 'block' : 'none';
                } else if (filterValue === 'practice') {
                    card.style.display = (category === 'practice') ? 'block' : 'none';
                } else {
                    card.style.display = (category === filterValue) ? 'block' : 'none';
                }
            });
        });
    });
}
