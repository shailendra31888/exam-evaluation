function saveInput() {
    var inputField = document.getElementById('omrText');
    localStorage.setItem('savedInput', inputField.value);
}

function loadFormData() {
    var savedInput = localStorage.getItem('savedInput');
    if (savedInput) {
        document.getElementById('omrText').value = savedInput;
    }
}

function updateCount() {
    var inputField = document.getElementById('omrText');
    var charCountDisplay = document.getElementById('charCount');
    charCountDisplay.textContent = 'Character count: ' + inputField.value.length;
}
function loading_functions(){
    loadFormData();
    updateCount();
}

window.onload = loading_functions;