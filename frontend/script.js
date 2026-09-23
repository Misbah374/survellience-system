const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const detectionIndicator = document.getElementById("detectionIndicator");
const multiclassFields = {
    normal: document.getElementById("multiclassNormalProbability"),
    fire: document.getElementById("multiclassFireProbability"),
    accident: document.getElementById("multiclassAccidentProbability"),
    violence: document.getElementById("multiclassViolenceProbability")
};
const binaryFields = {
    fire: document.getElementById("binaryFireProbability"),
    violence: document.getElementById("binaryViolenceProbability"),
    accident: document.getElementById("binaryAccidentProbability")
};
const binaryEvent = document.getElementById("binaryEvent");
const multiclassEvent = document.getElementById("multiclassEvent");
const comparisonMulticlassEvent = document.getElementById("comparisonMulticlassEvent");
const comparisonBinaryEvent = document.getElementById("comparisonBinaryEvent");
const comparisonAgreement = document.getElementById("comparisonAgreement");
let detectionTimer = null;

function formatPercentage(probability) {
    return `${(probability * 100).toFixed(2)}%`;
}

function resetDetection() {
    detectionIndicator.className = "indicator normal";
    Object.values(multiclassFields).forEach((field) => field.textContent = "0.00%");
    Object.values(binaryFields).forEach((field) => field.textContent = "0.00%");
    binaryEvent.textContent = "NORMAL";
    multiclassEvent.textContent = "NORMAL";
    comparisonMulticlassEvent.textContent = "NORMAL";
    comparisonBinaryEvent.textContent = "NORMAL";
    comparisonAgreement.textContent = "YES";
}

async function updateDetection() {
    try {
        const response = await fetch("/detection");
        const detection = await response.json();
        Object.entries(multiclassFields).forEach(([name, field]) => {
            field.textContent = formatPercentage(detection.multiclass[name]);
        });
        Object.entries(binaryFields).forEach(([name, field]) => {
            field.textContent = formatPercentage(detection.binary[name]);
        });
        binaryEvent.textContent = detection.binary.event.toUpperCase();
        multiclassEvent.textContent = detection.multiclass.event.toUpperCase();
        comparisonMulticlassEvent.textContent = detection.comparison.multiclass_event.toUpperCase();
        comparisonBinaryEvent.textContent = detection.comparison.binary_event.toUpperCase();
        comparisonAgreement.textContent = detection.comparison.agreement ? "YES" : "NO";
        const hasAlert = detection.multiclass.event !== "normal" || detection.binary.event !== "normal";
        detectionIndicator.className = `indicator ${hasAlert ? "alert" : "normal"}`;
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