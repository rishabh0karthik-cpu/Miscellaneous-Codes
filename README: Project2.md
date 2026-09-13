# Ocean Current Prediction Pipeline: Del Niño

This project provides a Python-based pipeline for predicting and visualizing ocean current dynamics. It leverages both a quadratic model for direct current pattern prediction and a linear dynamics model for forecasting sequences of changes in current patterns.

## Features

*   **OceanCurrentPredictor:** Utilizes a quadratic model to predict current patterns based on feature data.
*   **LinearCurrentDynamics:** Learns and applies a linear transition matrix to model the evolution of current patterns over time.
*   **Sequence Forecasting:** Predicts a sequence of future current patterns.
*   **Path Visualization:** Generates and plots predicted ocean-current paths from specified initial positions, utilizing velocity forecasts. Supports both Matplotlib and Cartopy for geographical plotting.
*   **Synthetic Data Generation:** Includes a `main` function to demonstrate the pipeline with randomly generated data.

## Getting Started

### Prerequisites

*   Python 3.x
*   NumPy
*   Matplotlib
*   Cartopy (optional, for enhanced plotting)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib
    # For enhanced plotting with Cartopy:
    pip install cartopy
    ```

### Usage

The primary functionality is demonstrated in the `Code-2:Del_Niño.py` script.

**Running the example:**

```bash
python Code-2:Del_Niño.py
```

This will:
1. Train the `OceanCurrentPredictor` and `LinearCurrentDynamics` models with synthetic data.
2. Generate a forecast of current patterns.
3. Plot the predicted ocean-current paths and save the plot as `current_forecast.png`.
4. Print the predicted current pattern, forecast, and residual covariance to the console.

**Using the classes programmatically:**

You can import and use the `OceanCurrentPredictor` and `LinearCurrentDynamics` classes in your own Python projects:

```python
from Code_2_Del_Niño import OceanCurrentPredictor, LinearCurrentDynamics

# Initialize and fit the predictor
predictor = OceanCurrentPredictor()
# ... fit predictor with your feature_data and current_patterns ...

# Initialize and fit the dynamics model
dynamics = LinearCurrentDynamics()
# ... fit dynamics with your current_patterns ...

# Predict a future pattern
predicted_pattern = predictor.predict(new_feature_data)

# Forecast a sequence of patterns
forecast_sequence = dynamics.predict_sequence(predicted_pattern, steps=5)

# Plot paths (if you have initial positions and velocity forecast)
# from Code_2_Del_Niño import plot_current_paths
# initial_positions = ...
# velocity_forecast = ... # needs to be reshaped to (steps, locations, 2)
# plot_current_paths(initial_positions, velocity_forecast, save_path="my_forecast.png")
```

## Project Structure

*   `Code-2:Del_Niño.py`: Main module containing the predictor, dynamics, and plotting functionalities.
*   `testing/`: Directory containing associated testing scripts and data (e.g., `quadratic_current_model.py`, dummy data files, evaluation scripts).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you encounter any problems or have suggestions for improvement.

