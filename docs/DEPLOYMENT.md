# Deployment & Operations Guide

## 1. System Requirements

### Hardware Requirements
- **Memory (RAM)**:
  - Minimum: 4 GB RAM.
  - Recommended: 8 GB+ RAM (especially for Olist raw dataset feature engineering).
- **CPU**: Dual-core 2.0 GHz or higher (x86_64 or Apple Silicon).
- **Disk Space**: ~2 GB free disk space (accounting for raw Excel/CSV datasets, virtual environments, and generated artifacts).

### Software Requirements
- **Operating System**: Windows 10/11, macOS 12+, or Ubuntu 20.04+ Linux.
- **Python Version**: Python 3.10, 3.11, 3.12, or 3.13.

---

## 2. Local Deployment (Windows)

### Option A: Double-Click Startup (Recommended)
1. Navigate to the project directory:
   `c:\Users\pgarv\OneDrive\Documents\HCLproject\project`
2. Double-click **`run_app.bat`**.
3. The launcher will automatically verify Python, check Streamlit installation, validate configuration, and launch the dashboard at **`http://localhost:8501`**.

### Option B: PowerShell / Command Line
```powershell
# 1. Open PowerShell and navigate to the project directory
cd c:\Users\pgarv\OneDrive\Documents\HCLproject\project

# 2. (Optional) Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Start the Streamlit application
python -m streamlit run app.py
```

---

## 3. Configuration & Environment Variables

Create a `.env` file in the project root based on `.env.example`:

```ini
# Google Gemini API credentials (Optional)
# If omitted, the dashboard operates fully in deterministic Offline Fallback Mode.
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
LLM_TEMPERATURE=0.2

# Optional OpenAI-compatible endpoint override
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

> **Security Note**: Never commit `.env` to Git. The repository's `.gitignore` explicitly excludes `.env` and `.env.*`.

---

## 4. Pipeline Execution (Generating New Models)

To regenerate feature sets, retrain clustering models, and refresh outputs:

```powershell
# Train both Online Retail II and Olist datasets
python run_all.py --dataset all

# Or train a single dataset
python run_all.py --dataset online_retail_ii
python run_all.py --dataset olist

# Generate publication-grade paper assets and benchmark reports
python make_paper_assets.py
```

---

## 5. Troubleshooting & Operational Diagnostics

### Port Already In Use
If port 8501 is occupied:
```powershell
python -m streamlit run app.py --server.port 8502
```

### Quota Exceeded (429) Notice in Chat
- If your Google Gemini key exhausts its free rate limit, the dashboard automatically transitions to **Offline Mode**.
- No crash or restart occurs. The advisor continues answering segment metric and comparison questions using verified dataset statistics.
- To reactivate live AI, update `GEMINI_API_KEY` in `.env` and press `R` in the browser to rerun.

### Calamine / Excel Engine Missing
If reading `.xlsx` throws an error:
```powershell
pip install openpyxl python-calamine
```

---

## 6. Optional Containerization (Docker)

To run the application inside an isolated Docker container:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### Building & Running Container:
```bash
docker build -t customer-loyalty-analytics .
docker run -p 8501:8501 --env-file .env customer-loyalty-analytics
```
