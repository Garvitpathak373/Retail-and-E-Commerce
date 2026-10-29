# Demo Startup & Presentation Guide

This guide describes how to run and present the **E-Commerce Customer Loyalty Analytics** dashboard reliably after restarting your laptop.

---

## 1. How to Start the Application After Laptop Restart

You have two simple options to launch the dashboard from Windows:

### Option A: Double-Click Launcher (Recommended)
1. Open File Explorer to your project folder:
   `c:\Users\pgarv\OneDrive\Documents\HCLproject\project`
2. Double-click **`run_app.bat`**.
3. A command window will verify Python, confirm your `.env` configuration, and launch Streamlit.

### Option B: PowerShell / Command Prompt
Open PowerShell or Terminal in the project directory and run:
```powershell
python -m streamlit run app.py
```

---

## 2. How to Confirm That the Dashboard Has Started

1. When the server starts, your default browser will automatically open, or you can navigate to:
   **`http://localhost:8501`**
2. In the terminal, you will see:
   ```text
   Local URL: http://localhost:8501
   Network URL: http://...:8501
   ```
3. On the dashboard:
   - Select **Online Retail II** or **Brazilian E-Commerce (Olist)** from the sidebar.
   - Confirm that the tabs (**Customer Segments & Profiles**, **Clustering Comparison**, **Data Quality & Pipeline Rules**) render clean data and metrics cards.

---

## 3. How to Recognize Live vs. Fallback Chatbot Mode

The dashboard chatbot and AI recommendation sections clearly label the response mode:

- **`🟢 LIVE GEMINI RESPONSE`**:
  Indicates that Google's Gemini API answered live using `gemini-flash-latest`.
- **`ℹ️ Offline Mode (Dataset Metrics)`**:
  Indicates that deterministic local fallback generated the response using verified project data from `outputs/` and computed cluster profiles.
  *(Displays: "AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics.")*
- **Conversational Greetings**:
  Greetings like `"hi"`, `"hello"`, `"how are you"`, and `"thanks"` return immediate assistant responses with zero API calls.

Both modes strictly respect privacy guardrails: customer IDs, order IDs, and email addresses are never sent to external providers or accepted in chat.

---

## 4. What to Do If Gemini Quota Is Exhausted

If your Google Gemini API key hits its free-tier rate or daily quota limit (`429 RESOURCE_EXHAUSTED`):

1. **No Code Debugging Required**:
   - The application automatically catches the 429 quota exhaustion in ~4 seconds and operates seamlessly in **Offline Mode**.
   - The local fallback provides 100% accurate, verified aggregate segment metrics (e.g., At-Risk Inactive, Recent Engaged, High Value Repeat), cross-segment comparisons, and clustering evaluations (K-Means, GMM, Silhouette, Davies-Bouldin, ARI).
2. **If You Have a Fresh Gemini API Key**:
   - Open `.env` in the project root:
     ```env
     GEMINI_API_KEY=your_new_gemini_api_key_here
     GEMINI_MODEL=gemini-flash-latest
     ```
   - Save `.env` and click **Rerun** (or press `R`) in the Streamlit browser window. Live requests will resume immediately.
3. **If You Do Not Have a New Key**:
   - You can confidently demo the dashboard in **Offline Mode**. The demo is fully functional, all charts render, and the advisor answers segment metric questions, comparisons, and retention strategies factually based on verified data.

---

## 5. Checklist Before Presenting Your Demo

Run these quick checks before your audience arrives:

- [ ] **Quick Test Suite**:
  Run PowerShell in the project directory:
  ```powershell
  pytest tests/test_advisor_reliability.py
  ```
  *(Confirms all 12 reliability scenarios pass in ~2.5 seconds).*
- [ ] **Start Dashboard**:
  Run `run_app.bat` or `python -m streamlit run app.py`.
- [ ] **Verify Page Loading**:
  - Open `http://localhost:8501`.
  - Check the **Clustering Comparison & Model Selection** tab to confirm algorithm metrics (K-Means, K=3, Silhouette ~0.2524, Davies-Bouldin ~1.4625).
- [ ] **Test the Loyalty Assistant Chat**:
  - Click the floating chat icon (💬) at the bottom right.
  - Ask: *"List all three customer segments with their average recency, purchase frequency, and monetary value in a table."*
  - Verify that the table displays cleanly with exact verified figures and zero unsolicited marketing actions.
- [ ] **Ready to Demo**:
  You are ready to present!
