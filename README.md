# ChurnIQ – Customer Churn Prediction App

ChurnIQ is a full-stack churn prediction application that combines a Python/FastAPI backend with a React/TypeScript frontend. A logistic regression model (via scikit-learn) is used to predict churn probability based on customer behavior, exposed via a RESTful API. The React UI allows users to input behavior data and view live model predictions.

## 🔍 Features
- Logistic regression churn model trained on real data
- FastAPI backend exposing prediction endpoint
- React + TypeScript UI for input and results display
- CORS-enabled API for local dev
- Swagger UI for exploring the API
- Docker-ready backend

## 📁 Project Structure

```
churniq/
├── app/                    # FastAPI backend
│   ├── api.py              # Main app with routes + CORS
│   ├── churn_model.py      # Model training script
│   └── churn_data.csv      # Sample training data
├── ui/                     # React frontend (TypeScript)
│   ├── src/
│   │   └── App.tsx         # UI to call API and show result
│   └── package.json
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## ⚙️ Backend Setup (FastAPI)

1. Create a virtual environment and activate it
```bash
python -m venv .venv
source .venv/Scripts/activate     # On Windows Git Bash
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Train the model
```bash
python app/churn_model.py
```

4. Start the API server
```bash
uvicorn app.api:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🖥 Frontend Setup (React + TypeScript)

```bash
cd ui
npm install
npm start
```

App runs at [http://localhost:3000](http://localhost:3000)

Ensure backend is running at `http://localhost:8000`

---

## 🔮 API Example

**POST** `/predict_churn`
```json
{
  "age": 30,
  "subscription_months": 12,
  "login_freq": 5
}
```

**Response**
```json
{
  "churn_probability": 0.435
}
```

---

## 🧠 Tech Stack

**Backend:** Python · FastAPI · scikit-learn · pandas · Swagger · CORS  
**Frontend:** React · TypeScript · Fetch API  
**Infrastructure:** Docker  
**ML:** Logistic Regression · AI/ML

---

## 🚀 Next Steps
- Add database persistence (PostgreSQL)
