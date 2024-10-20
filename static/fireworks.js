document.getElementById('fireworks-btn').addEventListener('click', function(e) {
    e.preventDefault(); // Prevent default link behavior
    createFireworks();
    showWelcomeText();
});

function createFireworks() {
    const container = document.getElementById('fireworks-container');

    // Create a fireworks element
    const firework = document.createElement('div');
    firework.className = 'firework';
    container.appendChild(firework);

    // Remove the firework after animation
    firework.addEventListener('animationend', () => {
        container.removeChild(firework);
    });
}

function showWelcomeText() {
    const textElement = document.getElementById('welcome-text');
    
    // Ensure any existing animation is cleared
    textElement.style.animation = 'none';
    textElement.offsetHeight; // Trigger a reflow to reset the animation
    textElement.style.animation = '';

    textElement.textContent = 'WELCOME';
    
    // Remove the text after animation
    textElement.addEventListener('animationend', () => {
        textElement.textContent = '';
    });
}
