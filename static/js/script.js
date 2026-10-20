const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const filePrompt = document.getElementById("filePrompt");
const previewImage = document.getElementById("previewImage");
const form = document.querySelector("form");

const allowedExtensions = /\.(png|jpe?g|webp)$/i;
const maxFileSize = 15 * 1024 * 1024; // 15 MB
let previewUrl = null;

function showError(message) {
  let errorBox = document.getElementById("clientError");

  if (!errorBox) {
    errorBox = document.createElement("p");
    errorBox.id = "clientError";
    errorBox.className = "error-message";
    errorBox.setAttribute("role", "alert");
    form.appendChild(errorBox);
  }

  errorBox.textContent = message;
}

function clearError() {
  const errorBox = document.getElementById("clientError");
  if (errorBox) {
    errorBox.remove();
  }
}

function showFile(file) {
  clearError();

  if (!file) {
    return;
  }

  if (!allowedExtensions.test(file.name)) {
    showError("Choose a PNG, JPG, JPEG, or WebP image.");
    imageInput.value = "";
    return;
  }

  if (file.size > maxFileSize) {
    showError("The image is too large. The maximum size is 15 MB.");
    imageInput.value = "";
    return;
  }

  fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  filePrompt.textContent = "Image ready to analyze";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.hidden = false;
}

imageInput.addEventListener("change", () => {
  showFile(imageInput.files[0]);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");

  const file = event.dataTransfer.files[0];
  if (!file) {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;

  showFile(file);
});