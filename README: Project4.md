# Pseudo-AI Student Report Consolidation System: Architectural Map

This document outlines the comprehensive architectural design, module hierarchies, data flow pathways, and dependency relationships for the Pseudo-AI Student Report Consolidation System. This structure is designed to be highly modular, scalable, and maintainable, facilitating seamless future transitions to real client-side or server-side AI models.

---

## 1. High-Level Architectural Diagram

The system is organized into five distinct layers, separating concerns from configuration and input processing to core inference, output rendering, and shared utilities.

```
+-----------------------------------------------------------------------------+
|                             CONFIGURATION LAYER                             |
|  - Settings & Constants      - API/Webhook Config      - Feature Flags      |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                           INPUT PROCESSING MODULE                           |
|  - Webhook Ingestion         - API Poller              - Data Validator     |
+-----------------------------------------------------------------------------+
                                       |
                                       v (Cleaned & Validated JSON)
+-----------------------------------------------------------------------------+
|                            CORE INFERENCE ENGINE                            |
|  - Metric Calculator         - Trend Analyzer          - Sentiment Extractor|
|  - Keyword Classifier        - Predictive Modeler      - Insight Generator  |
+-----------------------------------------------------------------------------+
                                       |
                                       v (Consolidated Metrics & Insights)
+-----------------------------------------------------------------------------+
|                          OUTPUT GENERATION MODULE                           |
|  - DOM Renderer              - Chart.js Visualizer     - Report Exporter    |
+-----------------------------------------------------------------------------+
                                       ^
                                       | (Shared Helpers)
+-----------------------------------------------------------------------------+
|                               UTILITY LIBRARY                               |
|  - Math & Stats Helpers      - DOM Utilities           - Notification Engine|
+-----------------------------------------------------------------------------+
```

---

## 2. Detailed Sub-Component Breakdown & File Organization

To support modularity and easy integration into existing websites, the codebase is organized into a structured directory hierarchy.

```
project-root/
│
├── index.html                      # Demo page integrating the modular system
├── integration.html                # Integration guide and embed example
│
├── src/                            # Source code directory
│   │
│   ├── config/                     # Configuration Layer
│   │   └── settings.js             # System settings, thresholds, and API endpoints
│   │
│   ├── input/                      # Input Processing Module
│   │   ├── validator.js            # Schema validation for incoming student reports
│   │   └── receiver.js             # Webhook receiver and API polling coordinator
│   │
│   ├── core/                       # Core Inference Engine (The Pseudo-AI)
│   │   ├── calculator.js           # Statistical calculations (averages, consistency)
│   │   ├── analyzer.js             # Trend analysis and predictive modeling
│   │   ├── nlp.js                  # Pseudo-NLP (sentiment analysis, keyword extraction)
│   │   └── engine.js               # Orchestrator combining metrics and generating insights
│   │
│   ├── output/                     # Output Generation Module
│   │   ├── renderer.js             # Dynamic DOM rendering of reports and lists
│   │   ├── visualizer.js           # Chart.js wrapper for radar and trend charts
│   │   └── exporter.js             # Text/PDF export generator
│   │
│   └── utils/                      # Utility Library
│       ├── helpers.js              # Math, date, and string formatting helpers
│       └── notifier.js             # Toast notification system
│
├── css/                            # Stylesheets
│   └── ai-report-consolidation.css # Component-specific styles
│
└── dist/                           # Production Build (Optional)
    └── ai-report-consolidation.min.js # Bundled and minified single-file import
```

---

## 3. Data Flow Sequences

The diagram below illustrates the sequential flow of data from report ingestion to final visualization and export.

```
[ External Source / UI Form ]
             |
             | (Raw Report JSON)
             v
   +-------------------+
   |  input/receiver   | <--- Polling or Webhook Trigger
   +-------------------+
             |
             v
   +-------------------+
   |  input/validator  | ---> [ Invalid ] ---> Trigger Notification
   +-------------------+
             |
             | (Validated Report Object)
             v
   +-------------------+
   |    core/engine    | <--- Orchestrates the inference pipeline
   +-------------------+
     |       |       |
     |       |       +-----------------------------------+
     |       +-------------------------+                 |
     v                                 v                 v
+-------------------+         +-------------------+ +-------------------+
|  core/calculator  |         |   core/analyzer   | |     core/nlp      |
|  - Subject Avgs   |         |  - YoY Growth     | |  - Sentiment      |
|  - Consistency    |         |  - Trend Slopes   | |  - Keywords       |
+-------------------+         +-------------------+ +-------------------+
     |                                 |                 |
     +----------------+----------------+-----------------+
                      |
                      | (Consolidated Metrics & Insights JSON)
                      v
            +-------------------+
            |    core/engine    |
            +-------------------+
                      |
                      v
            +-------------------+
            |  output/renderer  |
            +-------------------+
             /                 \
            /                   \
           v                     v
+-------------------+   +-------------------+
| output/visualizer |   |  output/exporter  |
| - Radar Chart     |   | - Text Export     |
| - Line Chart      |   | - PDF Export      |
+-------------------+   +-------------------+
```

### Step-by-Step Sequence:
1. **Ingestion:** Reports are received either via manual form submission, periodic API polling (`input/receiver.js`), or an incoming webhook POST request.
2. **Validation:** The `input/validator.js` checks the report against a strict schema (ensuring student name, year, and numerical subject scores are present and valid).
3. **Orchestration:** The validated report is passed to `core/engine.js`, which coordinates the analysis.
4. **Statistical Calculation:** `core/calculator.js` computes overall averages, subject-specific averages, and a performance consistency score (using standard deviation).
5. **Trend & Predictive Analysis:** `core/analyzer.js` calculates year-over-year growth rates and applies linear regression to predict future performance trajectories.
6. **Pseudo-NLP Extraction:** `core/nlp.js` parses qualitative comments (if provided) to extract sentiment scores and classify key strengths/weaknesses.
7. **Insight Synthesis:** `core/engine.js` combines all metrics, trends, and NLP outputs to select and format highly personalized, actionable insights.
8. **Rendering & Visualization:** The synthesized report is sent to `output/renderer.js` to update the DOM, while `output/visualizer.js` updates the Chart.js radar and trend charts.
9. **Export:** The user can trigger `output/exporter.js` to download a clean text or PDF summary of the consolidated report.

---

## 4. Dependency Mapping

To maintain a clean architecture, dependencies are strictly unidirectional (top-down) and external libraries are isolated.

### Internal Dependencies:
* **Orchestrator (`core/engine.js`)** depends on `core/calculator.js`, `core/analyzer.js`, and `core/nlp.js`.
* **Output Module (`output/renderer.js`)** depends on `output/visualizer.js` and `output/exporter.js`.
* **All Modules** can optionally depend on `config/settings.js` for thresholds/endpoints and `utils/helpers.js` / `utils/notifier.js` for shared utilities.

### External Dependencies:
1. **Chart.js (v4.x):** Used exclusively by `output/visualizer.js` for rendering canvas-based charts. Isolated so that if the charting library changes, only `visualizer.js` needs modification.
2. **FontAwesome (v6.x):** Used for UI icons in `output/renderer.js` and `utils/notifier.js`.
3. **Tailwind CSS / Custom CSS:** Used for styling. The component styles are isolated in `css/ai-report-consolidation.css` to prevent style bleeding into the host website.

---

## 5. Functional Component Descriptions

### Configuration Layer
* **`config/settings.js`**: Centralizes all configurable parameters, such as grade thresholds (e.g., Excellent >= 90), trend slope sensitivity, API endpoints, polling intervals, and feature flags (e.g., enabling/disabling predictive analysis).

### Input Processing Module
* **`input/validator.js`**: Ensures data integrity. It validates that incoming JSON payloads contain all required fields with correct data types, preventing runtime errors in the inference engine.
* **`input/receiver.js`**: Manages data ingestion. It handles the polling interval for auto-entry and simulates/handles webhook endpoints, feeding clean data into the main application state.

### Core Inference Engine (The Pseudo-AI)
* **`core/calculator.js`**: Performs mathematical aggregations. It calculates averages across multiple reports and computes a "Consistency Score" based on the standard deviation of scores over time.
* **`core/analyzer.js`**: Analyzes historical trends. It calculates year-over-year growth rates and uses a linear regression slope to predict future performance trends.
* **`core/nlp.js`**: Implements rule-based natural language processing. It scans teacher comments for positive/negative sentiment keywords and extracts key terms to identify specific strengths and weaknesses.
* **`core/engine.js`**: The brain of the system. It orchestrates the calculation, trend, and NLP modules, and applies a rule-based expert system to generate highly tailored, natural-sounding AI insights.

### Output Generation Module
* **`output/renderer.js`**: Manages the user interface. It dynamically builds and updates the HTML structure for the report list, status displays, and the final consolidated report card.
* **`output/visualizer.js`**: Handles data visualization. It initializes, updates, and destroys Chart.js instances (radar and line charts) to reflect the active student's data.
* **`output/exporter.js`**: Handles file generation. It formats the consolidated report into a clean, readable text file or PDF for download.

### Utility Library
* **`utils/helpers.js`**: Contains reusable, non-business-logic functions such as rounding numbers, formatting dates, capitalizing strings, and safe DOM element creation.
* **`utils/notifier.js`**: A lightweight, self-contained toast notification system to alert users of auto-entry events, successful operations, or validation errors.
