# 📡 SKYWAVE: HF Radio Reception Probability Prediction

> *MSc Advanced Computer Science (Data Analytics) Project | University of Leeds | 2025/2026*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[![Status](https://img.shields.io/badge/status-completed-success)](#)

---

##  Project Overview

**SKYWAVE** is a machine learning framework that predicts the probability of successful reception for High Frequency (HF) radio transmissions using historical operational logs.

HF radio (3–30 MHz) propagates by refracting off ionised layers in the upper atmosphere—a process governed by solar radiation, geomagnetic activity, time of day, and seasonal cycles. Because the ionosphere is inherently dynamic, reception success cannot be determined deterministically from transmitter settings alone.

This project addresses the operational challenge by:
-  Analysing **pre-transmission parameters** (frequency, SWR, power, temporal conditions)
-  Engineering **physics-informed features** (frequency-daylight interactions, cyclical encodings)
-  Training a **calibrated Random Forest classifier** with leakage prevention
-  Validating with **temporal splitting** for real-world readiness

---

##  Key Results

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.9371 | Excellent discrimination between success/failure |
| **Brier Score** | 0.0973 | Well-calibrated probabilities (lower is better) |
| **Accuracy** | 0.8560 | Overall classification accuracy |
| **Precision** | 0.9114 | 91% of predicted successes actually occur |
| **Recall** | 0.8495 | 85% of successful receptions correctly identified |

**Top Predictive Feature:** `freq_daylight_match` (51.6%) — Frequency × daylight interaction.

---

##  Repository Structure

```text
skywave-hf-prediction/
├── src/                          # Core pipeline code
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── predict.py
│   ├── config.py
│   ├── main.py
│   └── country_lookup.py
├── scripts/                      # Report generation utilities
│   ├── generate_report_figures.py
│   ├── generate_report_tables.py
│   └── generate_fig_5_3.py
├── releases/                     # Final deliverables
│   └── Final_report.md
├── README.md
├── requirements.txt
├── .gitignore
└── [Ignored by Git]              # Download from SharePoint
    ├── data/                     # Raw/processed datasets (~902MB)
    ├── figures/                  # Generated images (~34MB)
    ├── tables/                   # Table images
    ├── models/                   # Trained model files
    ├── model_outputs/            # Metrics & logs
    └── eda_outputs/              # EDA results
```

---

##  Installation

```bash
# 1. Create virtual environment
python3 -m venv skywave_env

# 2. Activate environment
source skywave_env/bin/activate  # On Mac/Linux
# OR
# skywave_env\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
cd src
python main.py
```

---

##  Data Access

Raw transmission logs are stored on **University of Leeds SharePoint** and updated regularly:

🔗 **Data Access Link**:  
https://leeds365-my.sharepoint.com/:f:/g/personal/gcsv9491_leeds_ac_uk/Eu3Rbx4xRFZOjo1O9XXeqVUBF-SANbB7hlWgi28fWoInFg

>  **Note**: The `data/` folder is excluded from Git due to size. Always download the latest version before running the pipeline.

---

##  Model Configuration

Key hyperparameters (from `src/config.py`):

```python
{
  "n_estimators": 100,           # Number of trees in Random Forest
  "max_depth": 15,               # Maximum tree depth
  "class_weight": "balanced",    # Handle class imbalance
  "calibration_method": "isotonic", # Probability calibration method
  "random_state": 42             # Reproducibility seed
}
```

---

##  Updating with New Data

When new transmission logs are added to SharePoint:

1. **Download latest data** to `data/raw/`
2. **Re-run preprocessing**: `python src/data_loader.py`
3. **Re-train model** (optional): `cd src && python main.py`
4. **Regenerate report assets**:
   ```bash
   cd ..
   python scripts/generate_report_figures.py
   python scripts/generate_report_tables.py
   ```
5. **Update report** in Google Docs → export to `releases/Final_report.md`
6. **Commit code changes** (not data):
   ```bash
   git add src/ scripts/ releases/
   git commit -m "Update: new data run [YYYY-MM-DD]"
   git push
   ```

---

## 🛠️ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| `ModuleNotFoundError` | Ensure virtual environment is activated: `source skywave_env/bin/activate` |
| `FileNotFoundError: data/raw/...` | Download latest data from SharePoint link above |
| `MemoryError` | Reduce `n_estimators` in `config.py` or use a machine with more RAM |
| Figures/tables not generating | Check `matplotlib` backend: `export MPLBACKEND=Agg` before running |

---


##  License

You are free to:
-  Use, modify, and distribute the code
-  Use for academic or commercial purposes
-  Provide attribution to the original author

---

## 👤 Author

**Kuppa Ganesh**  
MSc Advanced Computer Science (Artificial Intelligence)  
School of Computing, University of Leeds  

 [GitHub](https://github.com/GaneshKuppa)

📧 qdwh02121@leeds.ac.uk  

📧 ganeshkuppa04@gmail.com(personal mail)


**Supervisor**: Leandro Soares Indrusiak,Distributed Systems and Services Group, School of Computer Science, University of Leeds  - L.SoaresIndrusiak@leeds.ac.uk

**Assessor**: Antonio Marcos Alberti, Distributed Systems and Services Group, School of Computer Science, University of Leeds - A.M.Alberti@leeds.ac.uk  

**Academic Year**: 2025/2026