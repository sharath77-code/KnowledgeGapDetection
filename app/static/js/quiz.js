let currentQuestionIndex = 0;
let userAnswers = {};
let quizTimer = null;
let totalTimeSeconds = 15 * 60; // 15 Minutes

document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('quizFormContainer')) {
        initQuiz();
    }
});

function initQuiz() {
    startTimer();
    renderQuestionPalette();
    loadQuestion(0);

    document.getElementById('prevBtn')?.addEventListener('click', navigatePrev);
    document.getElementById('nextBtn')?.addEventListener('click', navigateNext);
    document.getElementById('submitQuizBtn')?.addEventListener('click', submitQuiz);
}

function startTimer() {
    const timerDisplay = document.getElementById('timerText');
    const timerBadge = document.getElementById('timerBadge');

    quizTimer = setInterval(() => {
        totalTimeSeconds--;

        const minutes = Math.floor(totalTimeSeconds / 60);
        const seconds = totalTimeSeconds % 60;
        const formatted = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        if (timerDisplay) {
            timerDisplay.textContent = formatted;
        }

        if (totalTimeSeconds <= 180 && timerBadge) {
            timerBadge.classList.add('warning-time');
        }

        if (totalTimeSeconds <= 0) {
            clearInterval(quizTimer);
            submitQuizPayload();
        }
    }, 1000);
}

function renderQuestionPalette() {
    const palette = document.getElementById('questionPalette');
    if (!palette || !window.QUIZ_QUESTIONS) return;

    palette.innerHTML = '';
    window.QUIZ_QUESTIONS.forEach((q, index) => {
        const dot = document.createElement('div');
        dot.className = `q-dot ${index === 0 ? 'active' : ''}`;
        dot.textContent = index + 1;
        dot.id = `qDot_${index}`;
        dot.addEventListener('click', () => loadQuestion(index));
        palette.appendChild(dot);
    });
}

function loadQuestion(index) {
    const questions = window.QUIZ_QUESTIONS;
    if (!questions || index < 0 || index >= questions.length) return;

    currentQuestionIndex = index;
    const q = questions[index];

    document.getElementById('questionCounterText').textContent = `Question ${index + 1} of ${questions.length}`;
    document.getElementById('skillTag').textContent = q.skill_name || q.skill || 'General Skill';
    document.getElementById('questionText').textContent = q.question;

    const progressPct = ((index + 1) / questions.length) * 100;
    document.getElementById('quizProgressBar').style.width = `${progressPct}%`;

    const optionsGrid = document.getElementById('optionsGrid');
    optionsGrid.innerHTML = '';

    const qId = q.id !== undefined ? q.id : (index + 1);

    q.options.forEach((optText, optIdx) => {
        const isSelected = userAnswers[qId] === optIdx;
        const optionCard = document.createElement('div');
        optionCard.className = `option-card ${isSelected ? 'selected' : ''}`;
        optionCard.innerHTML = `
            <div class="option-radio-btn"></div>
            <div class="option-label">${optText}</div>
        `;

        optionCard.addEventListener('click', () => selectOption(qId, optIdx, optionCard));
        optionsGrid.appendChild(optionCard);
    });

    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitQuizBtn');

    if (prevBtn) prevBtn.style.visibility = index === 0 ? 'hidden' : 'visible';
    if (nextBtn) nextBtn.style.display = index === questions.length - 1 ? 'none' : 'inline-flex';
    if (submitBtn) submitBtn.style.display = index === questions.length - 1 ? 'inline-flex' : 'none';

    updatePaletteHighlights();
}

function selectOption(questionId, optionIndex, element) {
    userAnswers[questionId] = optionIndex;

    const siblings = document.querySelectorAll('.option-card');
    siblings.forEach(card => card.classList.remove('selected'));
    element.classList.add('selected');

    updatePaletteHighlights();
}

function updatePaletteHighlights() {
    window.QUIZ_QUESTIONS.forEach((q, index) => {
        const dot = document.getElementById(`qDot_${index}`);
        if (!dot) return;
        const qId = q.id !== undefined ? q.id : (index + 1);

        dot.classList.remove('active', 'answered');
        if (index === currentQuestionIndex) {
            dot.classList.add('active');
        } else if (userAnswers[qId] !== undefined) {
            dot.classList.add('answered');
        }
    });
}

function navigatePrev() {
    if (currentQuestionIndex > 0) {
        loadQuestion(currentQuestionIndex - 1);
    }
}

function navigateNext() {
    if (currentQuestionIndex < window.QUIZ_QUESTIONS.length - 1) {
        loadQuestion(currentQuestionIndex + 1);
    }
}

function submitQuiz() {
    const totalQ = window.QUIZ_QUESTIONS.length;
    const answeredQ = Object.keys(userAnswers).length;

    if (answeredQ < totalQ) {
        if (!confirm(`You have answered ${answeredQ} out of ${totalQ} questions. Submit quiz?`)) {
            return;
        }
    }

    submitQuizPayload();
}

function submitQuizPayload() {
    clearInterval(quizTimer);
    
    fetch('/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
            answers: userAnswers,
            time_spent_seconds: (15 * 60) - totalTimeSeconds
        })
    })
    .then(res => res.json())
    .then(data => {
        window.location.href = '/loading';
    })
    .catch(err => {
        console.error('Submission error:', err);
        window.location.href = '/loading';
    });
}
