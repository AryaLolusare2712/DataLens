# 🔭 DataLens – AI Analytics Dashboard (Python)

A full-featured AI-powered data analytics dashboard built with **Streamlit** and **Gemini 1.5 Flash**.

## 🚀 Live Demo
Link : https://datalens-d8jw93nquhuwgnanrrrfgu.streamlit.app/

## Features
- 📤 **Upload** — drag & drop CSV/Excel files
- 📊 **Overview** — KPI cards, dataset profile, column types
- 📈 **Charts** — bar, line, area, pie charts with column selector
- 🟣 **Heatmap** — Pearson correlation heatmap
- ✦ **AI Insights** — auto-generated stat cards per column
- 🔍 **Explorer** — searchable data table
- 🤖 **AI Chat** — chat with Gemini 1.5 about your data

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run datalens_app.py
```

### 3. Add your Gemini API key
- Get a free key from: https://aistudio.google.com/app/apikey
- Enter it in the **sidebar** of the app (GEMINI API KEY field)
- Or set it as an environment variable:
  ```bash
  export GOOGLE_API_KEY="AIza..."
  streamlit run datalens_app.py
  ```

## Model Used
**gemini-1.5-flash-preview-04-17** — Gemini 1.5 Flash (latest)
