# 🏠 Hoorn Insulation Priority Dashboard

A data-driven tool to identify homes in the Municipality of Hoorn most in need of insulation support under the National Insulation Programme (NIP).

## 🔗 Live Demo
https://hoorn-insulation-dashboard-i8v85zibcjlglvidonsoos.streamlit.app/

## Built by
Group 4 — Inholland University of Applied Sciences  
Client: Municipality of Hoorn

## Data sources
- EP Online (energy performance database)
- Liander open gas consumption data (CC-BY 4.0)
- Municipality of Hoorn private field survey

## Model
LightGBM Regression on Warmtebehoefte (heat demand in kWh/m²)  
R² = 0.953 | MAE = 4.9 kWh/m²

## Run locally
pip install -r requirements.txt  
streamlit run app.py
