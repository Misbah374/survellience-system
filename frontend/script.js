const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");


startBtn.addEventListener("click", async () => {

    await fetch("/start_camera");

    const videoStream = document.getElementById("videoStream");
    const videoPlaceholder = document.getElementById("videoPlaceholder");

    videoStream.src = "/video_feed";

    videoPlaceholder.classList.add("hidden");
    videoStream.classList.remove("hidden");

    startBtn.disabled = true;
    stopBtn.disabled = false;

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

});