const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const detectionIndicator = document.getElementById("detectionIndicator");
const currentEvent = document.getElementById("currentEvent");
const detectionFields = {
    normal: document.getElementById("normalProbability"),
    fire: document.getElementById("fireProbability"),
    violence: document.getElementById("violenceProbability"),
    accident: document.getElementById("accidentProbability")
};
const emergencyMessage = document.createElement("p");
emergencyMessage.className = "emergency-response";
emergencyMessage.textContent = "No emergency detected";
currentEvent.parentElement.appendChild(emergencyMessage);
let detectionTimer = null;

function updateEmergencyMessage(event) {
    const messages = {
        fire: "Calling Fire Department...",
        violence: "Calling Police...",
        accident: "Calling Emergency Medical Services...",
        normal: "No emergency detected"
    };
    emergencyMessage.textContent = messages[event] || messages.normal;
}

function resetDetection() {
    currentEvent.textContent = "NORMAL";
    detectionIndicator.className = "indicator normal";
    Object.values(detectionFields).forEach((field) => field.textContent = "0%");
    updateEmergencyMessage("normal");
}

async function updateDetection() {
    try {
        const response = await fetch("/detection");
        const detection = await response.json();
        currentEvent.textContent = detection.event.toUpperCase();
        detectionIndicator.className = `indicator ${detection.event === "normal" ? "normal" : "alert"}`;
        updateEmergencyMessage(detection.event);
        Object.entries(detectionFields).forEach(([name, field]) => {
            field.textContent = `${(detection[name] * 100).toFixed(0)}%`;
        });
    } catch (error) {
        console.error("Detection request failed:", error);
    }
}


startBtn.addEventListener("click", async () => {

    await fetch("/start_camera");

    const videoStream = document.getElementById("videoStream");
    const videoPlaceholder = document.getElementById("videoPlaceholder");

    videoStream.src = "/video_feed";

    videoPlaceholder.classList.add("hidden");
    videoStream.classList.remove("hidden");

    startBtn.disabled = true;
    stopBtn.disabled = false;
    updateDetection();
    detectionTimer = setInterval(updateDetection, 1000);

});


stopBtn.addEventListener("click", async () => {

    await fetch("/stop_camera");

    const videoStream = document.getElementById("videoStream");
    const videoPlaceholder = document.getElementById("videoPlaceholder");

    videoStream.src = "";

    videoStream.classList.add("hidden");
    videoPlaceholder.classList.remove("hidden");

    startBtn.disabled = false;
    stopBtn.disabled = true;
    clearInterval(detectionTimer);
    detectionTimer = null;
    resetDetection();

});