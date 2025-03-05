document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file-upload");
    const previewImage = document.getElementById("previewImage");
    const loadingDiv = document.getElementById("loading");
    const uploadForm = document.querySelector("form");

    if (fileInput) {
        fileInput.addEventListener("change", function (event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    if (previewImage) {
                        previewImage.src = e.target.result;
                        previewImage.style.display = "block";
                    }
                };
                reader.readAsDataURL(file);
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener("submit", function () {
            if (loadingDiv) {
                loadingDiv.style.display = "block"; // Show loading message on form submit
            }
        });
    }
});

function goBack() {
    window.history.back();
}
