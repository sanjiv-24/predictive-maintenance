# 🚀 Predictive Maintenance API

AI-powered Predictive Maintenance Backend using **Temporal Fusion Transformer (TFT)**, **FastAPI**, and **Supabase** for Remaining Useful Life (RUL) prediction on the NASA CMAPSS FD001 dataset.

---

# 📌 Project Overview

This project predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines using deep learning and serves predictions through a lightweight cloud-deployed REST API.

The system analyzes engine sensor data, estimates engine degradation, and classifies engine health into:

- ✅ NORMAL
- ⚠️ WARNING
- 🚨 CRITICAL

---

# 🧠 AI Model

The prediction system is based on:

## Temporal Fusion Transformer (TFT)

A state-of-the-art deep learning architecture for:

- Time-series forecasting
- Sequential pattern learning
- Multivariate sensor analysis
- Predictive maintenance

---

# 📂 Dataset

Dataset Used:

## NASA CMAPSS FD001 Dataset

Contains:
- Turbofan engine sensor readings
- Engine degradation simulation
- Operational cycles
- Remaining Useful Life labels

Dataset Source:
https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

---

# ⚙️ System Architecture

```text
NASA CMAPSS Dataset
        ↓
Data Preprocessing
        ↓
TFT Model Training
        ↓
Prediction Generation
        ↓
Supabase Cloud Database
        ↓
FastAPI Backend
        ↓
Frontend Dashboard