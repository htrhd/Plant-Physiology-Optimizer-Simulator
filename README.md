# Plant Physiology Optimizer & Simulator

This project is a high-fidelity biological simulation that models plant photosynthesis, transpiration, and stress responses. It uses **Differential Evolution** (a type of genetic algorithm) to "evolve" a set of plant parameters—like leaf thickness, Rubisco content, and root hydraulic conductance—to maximize photosynthetic gain under specific environmental conditions.



## 🔬 Core Features
- **Photosynthesis Modeling:** Implements Rubisco-limited, light-limited, and TPU-limited CO2 assimilation models.
- **Environmental Sensitivity:** Accounts for temperature, CO2 concentration, UV-B intensity, ozone levels, and soil nutrients.
- **Hydraulic Stress:** Models vapor pressure deficit (VPD) and xylem embolism repair rates.
- **AI Optimization:** Uses `scipy.optimize.differential_evolution` to find the "perfect" plant for a given climate.
- **Sensitivity Analysis:** Automated plotting of which plant traits (e.g., stomatal density vs. leaf lifespan) most impact survival.

## 📊 Visualizations
The script generates:
1. **Parameter Sensitivities:** A bar chart showing which traits have the biggest impact on growth.
2. **Response Curves:** Visualization of the top 3 most critical parameters.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/htrhd/Plant-Physiology-Optimizer-Simulator.git](https://github.com/htrhd/Plant-Physiology-Optimizer-Simulator.git)
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulation:**
   ```bash
   python main.py
   ```

## 🛠️ Technical Details
- **Optimization Algorithm:** Differential Evolution
- **Sampling:** Latin Hypercube Sampling (via `pyDOE`)
- **Physics:** Penman-Monteith transpiration and Beer-Lambert canopy light extinction.

---
*Developed for computational biology and botanical research simulation.*
