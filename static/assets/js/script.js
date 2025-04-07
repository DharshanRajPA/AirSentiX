// DOM elements
const analyzeBtn = document.getElementById("analyzeBtn");
const tweetInput = document.getElementById("tweetInput");
const loader = document.getElementById("loader");
const results = document.getElementById("results");

const sentimentResult = document.getElementById("sentimentResult");
const confidenceResult = document.getElementById("confidenceResult");
const emotionResult = document.getElementById("emotionResult");
const urgencyResult = document.getElementById("urgencyResult");

const darkModeToggle = document.getElementById("darkModeToggle");

// Event: Analyze
analyzeBtn.addEventListener("click", async () => {
  const text = tweetInput.value.trim();
  if (!text) {
    alert("Please enter a tweet!");
    return;
  }

  results.classList.add("hidden");
  loader.classList.remove("hidden");

  try {
    const response = await fetch("http://localhost:8000/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) throw new Error("Prediction failed.");

    const data = await response.json();

    sentimentResult.textContent = data.sentiment;
    confidenceResult.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    emotionResult.textContent = data.emotional_tone;
    urgencyResult.textContent = data.urgency_score.toFixed(3);

    renderChart(data.all_scores);

    loader.classList.add("hidden");
    results.classList.remove("hidden");
  } catch (error) {
    loader.classList.add("hidden");
    alert("Error: " + error.message);
  }
});

// Event: Dark mode toggle
darkModeToggle.addEventListener("click", () => {
  document.documentElement.classList.toggle("dark");
});
