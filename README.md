# Customer Churn Prediction App

Full-stack churn prediction application that combines a Python/FastAPI backend with a React/TypeScript frontend. A logistic regression model (via scikit-learn) is used to predict churn probability based on customer behavior, exposed via a RESTful API. The React UI allows users to input behavior data and view live model predictions.

## 🔍 Features
- Logistic regression churn model trained on real data
- FastAPI backend exposing prediction endpoint
- React + TypeScript UI for input and results display
- CORS-enabled API for local dev
- Swagger UI for exploring the API
- Docker-ready backend


## ⚙️ Backend Setup (FastAPI)

1. Create a virtual environment and activate it
```bash
python -m venv .venv
source .venv/bin/activate     
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

Use this command if your ui part is not setup yet : `python3 app/app.py`

Once UI part is set up you can use the below one:

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
{"user_id":"0f467e7f-31a2-4b20-8088-a3155fc7a4a6","recent_events":[{"event_id":"65uz256inmg","collector_tstamp":"2025-06-11T10:47:52.416Z","event_name":"page_view","domain_userid":"0f467e7f-31a2-4b20-8088-a3155fc7a4a6","session_id":"session_frustrated_1","page_urlpath":"/q/coverage_selection/gather_intent/","element_id":null,"error_message":null},{"event_id":"xgf2x4cx41j","collector_tstamp":"2025-06-11T10:48:04.078Z","event_name":"page_view","domain_userid":"0f467e7f-31a2-4b20-8088-a3155fc7a4a6","session_id":"session_frustrated_1","page_urlpath":"/q/coverage_selection/welcome/","element_id":null,"error_message":null},{"event_id":"4z19pummbu7","collector_tstamp":"2025-06-11T10:48:04.757Z","event_name":"pageset_section_changed","domain_userid":"0f467e7f-31a2-4b20-8088-a3155fc7a4a6","session_id":"session_frustrated_1","page_urlpath":"/q/coverage_selection/welcome/","element_id":null,"error_message":null}]}
```

**Response**
```json
{
    "churn_pattern_explanation": "Based on their activity, the user's churn pattern indicates: User is **recently active**.\n\nKey observations include:\n- Active with 1.0 sessions, averaging 0.2 min and 3 events per session.",
    "churn_probability": 0.79,
    "contributing_features": {
        "avg_events_per_session": {
            "importance": 0.1639344262295082,
            "value": 3
        },
        "avg_session_duration": {
            "importance": 0.09836065573770493,
            "value": 0.20568333333333333
        },
        "days_since_last_activity": {
            "importance": 0.0819672131147541,
            "value": 0
        },
        "excessive_help_text_usage": {
            "importance": 0.1475409836065574,
            "value": 0
        },
        "form_abandonment_rate": {
            "importance": 0,
            "value": 0
        },
        "funnel_completion_attempts": {
            "importance": 0.09836065573770493,
            "value": 0
        },
        "highest_funnel_step_reached": {
            "importance": 0.09836065573770493,
            "value": 0
        }
    },
    "funnel_step_metrics": [
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/business/",
            "step_number": 1,
            "visited": false
        },
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/business_split/",
            "step_number": 2,
            "visited": false
        },
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/your_employees/",
            "step_number": 3,
            "visited": false
        },
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/business_activities/",
            "step_number": 4,
            "visited": false
        },
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/business_activities_split/",
            "step_number": 5,
            "visited": false
        },
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/policy_details/",
            "step_number": 6,
            "visited": false
        },
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/policy_details_split/",
            "step_number": 7,
            "visited": false
        },
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/waiting_for_quotes/",
            "step_number": 8,
            "visited": false
        },
        {
            "completed_step": false,
            "duration_minutes": 0,
            "error_details": [],
            "errors_on_step": 0,
            "help_event_details": [],
            "help_events_on_step": 0,
            "step_name": "/q/usa/usa/quote_comparison/",
            "step_number": 9,
            "visited": false
        }
    ],
    "status": "High Risk",
    "user_id": "0f467e7f-31a2-4b20-8088-a3155fc7a4a6"
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
