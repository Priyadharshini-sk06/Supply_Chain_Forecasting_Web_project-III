
"use strict";

// Same-origin API: FastAPI serves both the frontend and API.
const API_BASE = "";

const skuSelect = document.getElementById("skuSelect");
const horizonInput = document.getElementById("horizonInput");
const horizonValue = document.getElementById("horizonValue");
const loadBtn = document.getElementById("loadBtn");
const statsRow = document.getElementById("statsRow");
const wmapeValue = document.getElementById("wmapeValue");
const categoryValue = document.getElementById("categoryValue");
const importanceBars = document.getElementById("importanceBars");
const errorCard = document.getElementById("errorCard");
const forecastCanvas = document.getElementById("forecastChart");

let chart = null;
let freq = "daily";
let horizonUnit = "days";
let isLoading = false;

// ------------------------------
// INITIALISE APPLICATION
// ------------------------------

async function init() {
  setLoading(true);

  try {
    const [healthResponse, skusResponse, importanceResponse] =
      await Promise.all([
        fetch(`${API_BASE}/health`),
        fetch(`${API_BASE}/skus`),
        fetch(`${API_BASE}/feature-importance`)
      ]);

    if (!healthResponse.ok) {
      throw new Error("Could not connect to the forecasting service.");
    }

    if (!skusResponse.ok) {
      throw new Error("Could not load the SKU list.");
    }

    if (!importanceResponse.ok) {
      throw new Error("Could not load feature importance.");
    }

    const health = await healthResponse.json();
    const skus = await skusResponse.json();
    const importance = await importanceResponse.json();

    if (health.status !== "ok") {
      throw new Error(
        "The forecasting model is not loaded. Check the backend terminal."
      );
    }

    if (!Array.isArray(skus) || skus.length === 0) {
      throw new Error("No SKU data is available.");
    }

    freq = health.freq || "daily";
    horizonUnit = freq === "monthly" ? "months" : "days";

    horizonInput.max = freq === "monthly" ? 24 : 60;
    horizonInput.value = freq === "monthly" ? 6 : 14;

    updateHorizonLabel();

    // Populate SKU selector safely.
    skuSelect.replaceChildren();

    skus.forEach((item) => {
      const option = document.createElement("option");

      option.value = item.sku_id;
      option.textContent = `${item.sku_id} (${item.category})`;

      skuSelect.appendChild(option);
    });

    renderImportance(importance.feature_importance || []);

    await loadForecast();
  } catch (error) {
    showError(error.message || "Unable to initialise the application.");
  } finally {
    setLoading(false);
  }
}

// ------------------------------
// FORECAST HORIZON
// ------------------------------

function updateHorizonLabel() {
  const value = Number(horizonInput.value);

  horizonValue.textContent =
    `${value} ${horizonUnit}`;
}

horizonInput.addEventListener("input", updateHorizonLabel);

// ------------------------------
// FORECAST BUTTON
// ------------------------------

loadBtn.addEventListener("click", loadForecast);

// Refresh automatically when the selected SKU changes.
skuSelect.addEventListener("change", loadForecast);

// ------------------------------
// LOAD FORECAST
// ------------------------------

async function loadForecast() {
  if (isLoading) return;

  const sku = skuSelect.value;
  const horizon = Number(horizonInput.value);

  if (!sku) {
    showError("Please select a SKU.");
    return;
  }

  if (!Number.isFinite(horizon) || horizon < 1) {
    showError("Please select a valid forecast horizon.");
    return;
  }

  clearError();
  setLoading(true);

  try {
    const url =
      `${API_BASE}/forecast` +
      `?sku_id=${encodeURIComponent(sku)}` +
      `&horizon=${encodeURIComponent(horizon)}`;

    const response = await fetch(url);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));

      throw new Error(
        errorData.detail || `Forecast request failed (${response.status}).`
      );
    }

    const data = await response.json();

    validateForecastData(data);
    renderForecast(data);
  } catch (error) {
    showError(error.message || "Unable to generate the forecast.");
  } finally {
    setLoading(false);
  }
}

// ------------------------------
// VALIDATE RESPONSE
// ------------------------------

function validateForecastData(data) {
  if (!Array.isArray(data.recent_actuals)) {
    throw new Error("The API returned invalid historical sales data.");
  }

  if (!Array.isArray(data.forecast)) {
    throw new Error("The API returned invalid forecast data.");
  }

  if (data.recent_actuals.length === 0) {
    throw new Error("No historical sales data is available for this SKU.");
  }

  if (data.forecast.length === 0) {
    throw new Error("The API returned an empty forecast.");
  }
}

// ------------------------------
// RENDER FORECAST
// ------------------------------

function renderForecast(data) {
  statsRow.hidden = false;

  wmapeValue.textContent =
    data.model_wmape_pct != null &&
    Number.isFinite(Number(data.model_wmape_pct))
      ? `${Number(data.model_wmape_pct).toFixed(2)}%`
      : "--";

  categoryValue.textContent = data.category || "Uncategorised";

  const actualLabels = data.recent_actuals.map((row) => row.date);
  const actualValues = data.recent_actuals.map((row) => Number(row.sales));

  const forecastLabels = data.forecast.map((row) => row.date);
  const forecastValues = data.forecast.map((row) => Number(row.forecast));

  const labels = [...actualLabels, ...forecastLabels];

  // Actual series ends where forecasting begins.
  const actualSeries = [
    ...actualValues,
    ...Array(forecastLabels.length).fill(null)
  ];

  // Start forecast line at the last actual value for visual continuity.
  const forecastSeries = [
    ...Array(Math.max(actualLabels.length - 1, 0)).fill(null),
    actualValues[actualValues.length - 1],
    ...forecastValues
  ];

  const context = forecastCanvas.getContext("2d");

  if (!context) {
    throw new Error("Unable to initialise the forecasting chart.");
  }

  if (chart) {
    chart.destroy();
  }

  chart = new Chart(context, {
    type: "line",

    data: {
      labels,

      datasets: [
        {
          label: "Actual sales",
          data: actualSeries,

          borderColor: "#3478e5",
          backgroundColor: "rgba(52, 120, 229, 0.10)",

          borderWidth: 2.5,
          tension: 0.28,
          pointRadius: 0,
          pointHoverRadius: 5,

          fill: false,
          spanGaps: false
        },

        {
          label: "Forecast demand",
          data: forecastSeries,

          borderColor: "#14a88a",
          backgroundColor: "rgba(20, 168, 138, 0.08)",

          borderWidth: 2.5,
          borderDash: [7, 5],

          tension: 0.28,
          pointRadius: 0,
          pointHoverRadius: 5,

          fill: false,
          spanGaps: false
        }
      ]
    },

    options: {
      responsive: true,
      maintainAspectRatio: true,

      interaction: {
        mode: "index",
        intersect: false
      },

      animation: {
        duration: 450
      },

      scales: {
        x: {
          ticks: {
            color: "#74839b",
            maxTicksLimit: 10,
            maxRotation: 0
          },

          grid: {
            color: "#edf1f7",
            drawTicks: false
          },

          border: {
            color: "#e5eaf2"
          }
        },

        y: {
          beginAtZero: true,

          ticks: {
            color: "#74839b"
          },

          grid: {
            color: "#edf1f7",
            drawTicks: false
          },

          border: {
            display: false
          }
        }
      },

      plugins: {
        legend: {
          position: "top",
          align: "start",

          labels: {
            color: "#172b4d",
            usePointStyle: true,
            pointStyle: "circle",
            padding: 20,
            boxWidth: 8
          }
        },

        tooltip: {
          backgroundColor: "#172b4d",
          titleColor: "#ffffff",
          bodyColor: "#eaf0fa",
          padding: 12,
          cornerRadius: 8,
          displayColors: true
        }
      }
    }
  });
}

// ------------------------------
// SHAP FEATURE IMPORTANCE
// ------------------------------

function renderImportance(items) {
  importanceBars.replaceChildren();

  if (!Array.isArray(items) || items.length === 0) {
    const message = document.createElement("p");

    message.textContent = "Feature importance is not available.";
    message.style.color = "#74839b";

    importanceBars.appendChild(message);
    return;
  }

  const validItems = items.filter(
    (item) =>
      Array.isArray(item) &&
      item.length >= 2 &&
      Number.isFinite(Number(item[1]))
  );

  const maxValue = Math.max(
    ...validItems.map((item) => Number(item[1])),
    0
  );

  validItems.forEach(([name, value]) => {
    const numericValue = Number(value);

    const row = document.createElement("div");
    row.className = "imp-row";

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = String(name);

    const track = document.createElement("span");
    track.className = "track";

    const fill = document.createElement("span");
    fill.className = "fill";

    const width =
      maxValue > 0
        ? Math.max(0, Math.min(100, (numericValue / maxValue) * 100))
        : 0;

    fill.style.width = `${width}%`;

    track.appendChild(fill);

    const amount = document.createElement("span");
    amount.textContent = numericValue.toFixed(2);
    amount.style.fontWeight = "700";
    amount.style.color = "#172b4d";

    row.append(label, track, amount);
    importanceBars.appendChild(row);
  });
}

// ------------------------------
// LOADING STATE
// ------------------------------

function setLoading(loading) {
  isLoading = loading;

  loadBtn.disabled = loading;
  skuSelect.disabled = loading;

  loadBtn.textContent = loading
    ? "Loading..."
    : "Forecast";

  loadBtn.setAttribute("aria-busy", String(loading));
}

// ------------------------------
// ERROR HANDLING
// ------------------------------

function showError(message) {
  errorCard.textContent = `Unable to load forecast: ${message}`;
  errorCard.hidden = false;
}

function clearError() {
  errorCard.textContent = "";
  errorCard.hidden = true;
}

init();
