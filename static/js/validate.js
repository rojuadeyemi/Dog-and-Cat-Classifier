function validateFile() {
    const fileInput = document.querySelector('input[type="file"]');
    const errorMessage = document.getElementById('error-message');
    const file = fileInput.files[0];
    
    // Clear any previous message
    errorMessage.textContent = '';

    // Check if a file is selected
    if (!file) {
        errorMessage.textContent = "Please upload a file.";
        return false;
    }

    // Check if the file is an image
    const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        errorMessage.textContent = "Only image files (JPEG, PNG, GIF) are allowed.";
        return false;
    }

    return true;
}

