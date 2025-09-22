document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const chatBox = document.getElementById("chatBox");
    const imageUpload = document.getElementById("image-upload");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const previewText = document.getElementById("preview-text");
    const removeImage = document.getElementById("remove-image");

    let selectedFile = null;

    function addUserMessage(text) {
        const msgHtml = `
            <div class="message user">
                <div class="message-content">${text}</div>
            </div>`;
        chatBox.insertAdjacentHTML('beforeend', msgHtml);
        scrollToBottom();
    }

    function addUserImage(src) {
        const msgHtml = `
            <div class="message user">
                <div class="message-content">
                    <img src="${src}" style="max-width: 250px; border-radius: 8px;" />
                </div>
            </div>`;
        chatBox.insertAdjacentHTML('beforeend', msgHtml);
        scrollToBottom();
    }

    function addBotMessage(text) {
        const renderedHtml = window.marked.parse(text);
        const msgHtml = `
            <div class="message bot">
                <div class="avatar">🤖</div>
                <div class="message-content">
                    ${renderedHtml}
                </div>
            </div>`;
        chatBox.insertAdjacentHTML('beforeend', msgHtml);
        scrollToBottom();
    }
    
    function addLoadingIndicator() {
        const loadingHtml = `
            <div class="message bot" id="loading-indicator">
                <div class="avatar">🤖</div>
                <div class="message-content loading">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>`;
        chatBox.insertAdjacentHTML('beforeend', loadingHtml);
        scrollToBottom();
    }

    function removeLoadingIndicator() {
        const indicator = document.getElementById("loading-indicator");
        if (indicator) indicator.remove();
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Fetch and render chat history for the current session
    async function fetchAndRenderHistory() {
        let sessionId = localStorage.getItem("session_id");
        if (!sessionId) {
            sessionId = crypto.randomUUID();
            localStorage.setItem("session_id", sessionId);
        }
        // Show loading indicator while fetching history
        addLoadingIndicator();
        try {
            const response = await fetch("http://127.0.0.1:8000/history", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId })
            });
            if (!response.ok) throw new Error("Failed to fetch history");
            const data = await response.json();
            chatBox.innerHTML = "";
            if (Array.isArray(data.history)) {
                data.history.forEach(msg => {
                    if (msg.role === "user") {
                        addUserMessage(msg.content);
                    } else if (msg.role === "assistant") {
                        addBotMessage(msg.content);
                    }
                });
            }
        } catch (err) {
            console.error("Error fetching chat history:", err);
        } finally {
            removeLoadingIndicator();
        }
    }

    // Call on page load
    fetchAndRenderHistory();

    // Auto-resize textarea as user types
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = (messageInput.scrollHeight) + 'px';
    });

    // Trigger form submit on Enter (without Shift)
    messageInput.addEventListener("keydown", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault(); // Prevent newline
            chatForm.requestSubmit(); // Trigger the form's submit event
        }
    });

    // Handle image selection, rejecting other file types
    imageUpload.addEventListener("change", function(e) {
        const file = this.files[0];
        if (!file) return;

        // Check if the selected file is an image
        if (file.type.startsWith("image/")) {
            selectedFile = file;
            previewText.textContent = ''; // Clear any previous text
            
            const reader = new FileReader();
            reader.onload = (event) => {
                previewImage.src = event.target.result;
                previewImage.style.display = 'block';
            };
            reader.readAsDataURL(file);
            previewContainer.style.display = "flex";
        } else {
            // If not an image, alert the user and clear the selection
            alert("Invalid file type. Please select an image.");
            selectedFile = null;
            imageUpload.value = ""; // Reset the file input
            previewContainer.style.display = "none";
        }
    });

    // Handle removal of a selected image preview
    removeImage.addEventListener("click", () => {
        imageUpload.value = "";
        previewContainer.style.display = "none";
        selectedFile = null;
    });

    // Handle form submission
    chatForm.addEventListener("submit", async function(e) {
        e.preventDefault(); // Stop the default page refresh

        const message = messageInput.value.trim();
        if (!message && !selectedFile) return; // Do nothing if input is empty

        const userMessage = message;
        const userFile = selectedFile;
        const fileSrc = previewImage.src;

        // Get or create session_id
        let sessionId = localStorage.getItem("session_id");
        if (!sessionId) {
            sessionId = crypto.randomUUID();
            localStorage.setItem("session_id", sessionId);
        }

        // Reset the form immediately for a responsive feel
        messageInput.value = "";
        messageInput.style.height = 'auto';
        imageUpload.value = "";
        previewContainer.style.display = "none";
        selectedFile = null;

        // Display user's message and/or image in the chat
        if (userMessage) addUserMessage(userMessage);
        if (userFile) {
            addUserImage(fileSrc);
        }
        addLoadingIndicator(); // Show loader while waiting for response

        // Prepare data for the backend
        const formData = new FormData();
        let endpoint = "";
        if (userFile) {
            // Image upload: use /upload
            formData.append("file", userFile);
            formData.append("session_id", sessionId);
            formData.append("message", userMessage || "");
            endpoint = "http://127.0.0.1:8000/upload";
        } else {
            // Text only: use /chat
            formData.append("message", userMessage || "");
            formData.append("session_id", sessionId);
            endpoint = "http://127.0.0.1:8000/chat";
        }

        try {
            // Make the API call to your backend
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            const data = await response.json();
            removeLoadingIndicator();
            addBotMessage(data.response || data.reply || "Sorry, I couldn't process that.");
        } catch (err) {
            console.error("Error submitting form:", err);
            removeLoadingIndicator();
            addBotMessage("⚠️ Oops! Something went wrong. Please try again.");
        }
    });
});
