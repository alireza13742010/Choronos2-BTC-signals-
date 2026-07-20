# Choronos2-BTC-signals-
# ₿ Chronos-2 BTC/USD Forecast Signal App


<br/>

A real-time Bitcoin trading signal application powered by Amazon Chronos-2, a state-of-the-art zero-shot time series forecasting foundation model fine-tuned on historical Bitstamp BTC/USD minute-level data using AutoGluon TimeSeries.

The app generates actionable BUY / SELL / HOLD signals through a clean, interactive Streamlit dashboard backed by a rigorous backtest-calibrated confidence and threshold framework.

<br/>





## 📌 Overview

This project builds an end-to-end pipeline that:

- Ingests over a **decade of Bitstamp BTC/USD 1-minute OHLCV data**
- Computes technical indicators — **RSI, MACD, ATR, ADX**
- Feeds a rolling context window into a fine-tuned **Chronos-2 probabilistic forecaster**
- Derives trading signals calibrated against **historical backtest performance**

Confidence scores are computed by combining backtest directional precision, quantile consensus across prediction intervals, and predicted return magnitude — with thresholds dynamically adjusted per signal to control for forecast uncertainty and transaction costs.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔮 **Chronos-2 Forecasting** | Probabilistic forecasting via AutoGluon TimeSeries with full quantile prediction interval support |
| 📈 **Technical Indicators** | RSI, MACD, ATR, ADX computed on rolling context windows |
| 🎯 **Backtest-Calibrated Thresholds** | BUY/SELL thresholds independently calibrated using precision-support grid search |
| 🧠 **Confidence-Weighted Signals** | Blends backtest precision, quantile consensus, and move magnitude |
| ⚡ **Dynamic Threshold Adjustment** | Effective thresholds scale with live confidence vs. minimum required confidence |
| 🔄 **Auto-Refresh Mode** | Continuous signal monitoring at configurable intervals |
| 📝 **Signal CSV Logging** | Persistent trade journal with deduplication |
| 📊 **Interactive Streamlit UI** | Real-time charts, forecast tables, calibration diagnostics, sidebar controls |
| 🗂️ **Flexible Data Ingestion** | Handles historical `.csv.gz` + recent `.csv` with automatic schema normalization |

---

## 🧱 Tech Stack

| Component | Technology |
|---|---|
| Forecasting Model | Amazon Chronos-2 (fine-tuned) via AutoGluon |
| Training Framework | AutoGluon TimeSeries |
| UI | Streamlit |
| Data Source | Bitstamp BTC/USD 1-minute OHLCV |
| Indicators | Custom NumPy / Pandas (RSI, MACD, ATR, ADX) |
| Visualization | Matplotlib |

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/chronos2-btc-signal-app
cd chronos2-btc-signal-app
pip install -r requirements.txt
```

Or install dependencies directly:

```bash
pip install streamlit autogluon.timeseries pandas numpy matplotlib
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Configure all paths and hyperparameters from the **sidebar**:

- 📁 Point to your trained model directory
- 📂 Set your historical and recent Bitstamp CSV paths
- 📄 Set your backtest predictions CSV path
- ⚙️ Adjust thresholds and confidence parameters

---

## 📁 Project Structure

```
chronos2-btc-signal-app/
│
├── app.py                                      # Main Streamlit application
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
│
├── models/
│   └── btc_chronos2_bitstamp/                  # Trained model weights (see below ↓)
│
├── bitstamp-btcusd-minute-data/
│   ├── data/historical/
│   │   └── btcusd_bitstamp_1min_2012-2025.csv.gz
│   └── data/updates/
│       └── btcusd_bitstamp_1min_latest.csv
│
└── chronos2_bitstamp_training_outputs/
    └── chronos2_bitstamp_backtest_predictions.csv
```

---

## 🔐 Trained Model Weights

> The fine-tuned **Chronos-2 model weights** trained on Bitstamp BTC/USD minute-level data are **not included** in this repository due to file size constraints.

### How to Request Access

To request access to the pre-trained model weights, please reach out directly:

<div align="center">

**Alireza Tavakoli**  
University of Tehran  
📧 **[alireza.tavakol@ut.ac.ir](mailto:alireza.tavakol@ut.ac.ir)**

</div>

Please include the following in your email:

- ✅ Your **name and affiliation**
- ✅ **Intended use** — research / paper trading / academic
- ✅ A brief **description of your project**

> ❌ Requests for **commercial use** will not be accommodated.  
> ❌ **Redistribution** of the model weights without explicit written permission is prohibited.

---

## 📊 Signal Generation Logic

```
Raw OHLCV Data
      ↓
Technical Indicators (RSI, MACD, ATR, ADX)
      ↓
Rolling Context Window → Chronos-2 Forecast
      ↓
Quantile Prediction Intervals
      ↓
Backtest Calibration (MAE, RMSE, Direction Accuracy)
      ↓
Confidence Score = 0.60 × Backtest Precision
                 + 0.25 × Quantile Consensus
                 + 0.15 × Magnitude Score
      ↓
Dynamic Threshold Adjustment
      ↓
BUY / SELL / HOLD Signal
```

---

## 📸 App Preview

<img width="1842" height="1002" alt="Choronos2_signal_generation_platform2" src="https://github.com/user-attachments/assets/6833d826-4ed7-4057-aaba-ffcdb7dc25f0" />
<img width="1842" height="1002" alt="Choronos2_signal_generation_platform" src="https://github.com/user-attachments/assets/8f18129f-629c-4eea-8753-30e5c57f11ba" />

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{tavakoli2025chronos2btc,
  author    = {Alireza Tavakoli},
  title     = {Chronos-2 BTC/USD Forecast Signal App},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/your-username/chronos2-btc-signal-app}
}
```

---

## 📄 License

This repository is released for **non-commercial research use only**.  
Redistribution of the trained model weights without explicit permission from the author is strictly prohibited.

---

<div align="center">

## 📬 Contact

**Alireza Tavakoli**  
University of Tehran  
📧 [alireza.tavakol@ut.ac.ir](mailto:alireza.tavakol@ut.ac.ir)

<br/>

⭐ If you find this project useful, please consider starring the repository!

</div>
