import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

# --- Configuration (MUST BE IDENTICAL TO churn_model.py for consistency) ---
LOOKBACK_DAYS = 60
CHURN_INACTIVITY_DAYS = 60 # Used for 'days_since_last_activity'
HIGH_RISK_THRESHOLD = 0.55

# --- UPDATED FUNNEL_STEPS (MUST BE IDENTICAL TO churn_model.py) ---
FUNNEL_STEPS = {
    '/q/usa/usa/business/': 1,
    '/q/usa/usa/business_split/': 2,
    '/q/usa/usa/your_employees/': 3,
    '/q/usa/usa/business_activities/': 4,
    '/q/usa/usa/business_activities_split/': 5,
    '/q/usa/usa/policy_details/': 6,
    '/q/usa/usa/policy_details_split/': 7,
    '/q/usa/usa/waiting_for_quotes/': 8,
    '/q/usa/usa/quote_comparison/': 9
}

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load Model and Data Globally (once when the app starts) ---
try:
    loaded_model = joblib.load('../models/churn_prediction_rf_model.joblib')
    loaded_feature_columns = joblib.load('../models/trained_feature_columns.joblib')
    print("ML Model and Feature Columns loaded successfully.")
except Exception as e:
    print(f"Error loading model or feature columns: {e}")
    loaded_model = None
    loaded_feature_columns = []

try:
    # This global DataFrame is used by featurize_user_data_for_prediction to get
    # a user's *overall* last activity for 'days_since_last_activity' feature.
    raw_events_df_global = pd.read_csv('../data/raw_events_df.csv')
    raw_events_df_global['collector_tstamp'] = pd.to_datetime(raw_events_df_global['collector_tstamp'], utc=True)
    print("Historical raw_events_df loaded successfully for feature engineering.")
except FileNotFoundError:
    print("Error: 'raw_events_df.csv' not found. Ensure it's in the same directory as app.py.")
    raw_events_df_global = pd.DataFrame()
except Exception as e:
    print(f"Error loading raw_events_df.csv: {e}")
    raw_events_df_global = pd.DataFrame()

# --- Feature Engineering Function (MUST BE IDENTICAL TO churn_model.py) ---
def featurize_user_data_for_prediction(user_id, recent_events_data, all_historical_events_df):
    """
    Engineers features for a single user based on their recent events
    and their full historical context (for 'days_since_last_activity').

    Args:
        user_id (str): The ID of the user.
        recent_events_data (list of dict): List of recent event dictionaries for this user.
        all_historical_events_df (pd.DataFrame): Full historical event data (like raw_events_df used in training)
                                                to get last overall activity for 'days_since_last_activity'.

    Returns:
        pd.DataFrame: A single row DataFrame with engineered features, in the correct order.
    """

    current_time_for_inference = datetime.now(timezone.utc)
    inference_lookback_start = current_time_for_inference - timedelta(days=LOOKBACK_DAYS)

    recent_events_df = pd.DataFrame(recent_events_data)

    if not recent_events_df.empty:
        unix_ts_mask = recent_events_df['collector_tstamp'].astype(str).str.match(r'^\d{13}$')
        if unix_ts_mask.any():
            recent_events_df.loc[unix_ts_mask, 'collector_tstamp'] = pd.to_datetime(
                recent_events_df.loc[unix_ts_mask, 'collector_tstamp'], unit='ms', errors='coerce', utc=True
            )
        recent_events_df.loc[~unix_ts_mask.fillna(False), 'collector_tstamp'] = pd.to_datetime(
                recent_events_df.loc[~unix_ts_mask.fillna(False), 'collector_tstamp'], errors='coerce', infer_datetime_format=True, utc=True
            )
        recent_events_df.dropna(subset=['collector_tstamp'], inplace=True)
        recent_events_df = recent_events_df.sort_values(by='collector_tstamp').reset_index(drop=True)

    recent_events_in_window = recent_events_df[
        (recent_events_df['collector_tstamp'] >= inference_lookback_start) &
        (recent_events_df['collector_tstamp'] < current_time_for_inference)
    ].copy()

    features_dict = {f: 0 for f in loaded_feature_columns}

    last_overall_activity_tstamp = None
    if all_historical_events_df is not None and not all_historical_events_df.empty:
        user_overall_activity_in_history = all_historical_events_df[all_historical_events_df['domain_userid'] == user_id]['collector_tstamp']
        if not user_overall_activity_in_history.empty:
            last_overall_activity_tstamp = user_overall_activity_in_history.max()

    if pd.isna(last_overall_activity_tstamp) and not recent_events_df.empty:
         last_overall_activity_tstamp = recent_events_df['collector_tstamp'].max()

    if pd.notna(last_overall_activity_tstamp):
        features_dict['days_since_last_activity'] = (current_time_for_inference - last_overall_activity_tstamp).days
    else:
        features_dict['days_since_last_activity'] = LOOKBACK_DAYS + CHURN_INACTIVITY_DAYS 

    funnel_step_metrics = [] # To store metrics for each step

    if not recent_events_in_window.empty:
        features_dict['total_page_views'] = int((recent_events_in_window['event_name'] == 'page_view').sum())
        features_dict['total_clicks'] = int((recent_events_in_window['event_name'] == 'click').sum())
        features_dict['unique_pages_visited'] = int(recent_events_in_window[recent_events_in_window['event_name'] == 'page_view']['page_urlpath'].nunique())

        sessions = recent_events_in_window.groupby('session_id').agg(
            session_start=('collector_tstamp', 'min'),
            session_end=('collector_tstamp', 'max'),
            num_events=('event_name', 'count')
        ).reset_index()
        sessions['session_duration_minutes'] = (sessions['session_end'] - sessions['session_start']).dt.total_seconds() / 60

        features_dict['num_sessions'] = int(len(sessions))
        features_dict['avg_session_duration'] = float(sessions['session_duration_minutes'].mean()) if not sessions.empty else 0.0
        features_dict['avg_events_per_session'] = float(sessions['num_events'].mean()) if not sessions.empty else 0.0

        features_dict['num_error_events'] = int((recent_events_in_window['event_name'] == 'error_event').sum())

        rage_clicks = 0
        user_clicks = recent_events_in_window[recent_events_in_window['event_name'] == 'click'].sort_values('collector_tstamp')
        if len(user_clicks) > 1:
            user_clicks['time_diff'] = user_clicks['collector_tstamp'].diff().dt.total_seconds()
            user_clicks['same_element'] = user_clicks['element_id'] == user_clicks['element_id'].shift(1)
            rage_clicks = int(((user_clicks['time_diff'] < 1) & (user_clicks['same_element'])).sum())
        features_dict['rage_clicks'] = rage_clicks

        funnel_error_events = recent_events_in_window[
            recent_events_in_window['event_name'].isin(['form_error', 'invalid_field_submitted']) &
            recent_events_in_window['page_urlpath'].str.contains(
                '|'.join(re.escape(path) for path in FUNNEL_STEPS.keys()),
                na=False, regex=True
            )
        ]
        features_dict['repeated_validation_errors'] = int(len(funnel_error_events))
        features_dict['form_abandonment_rate'] = float(1.0) if features_dict['repeated_validation_errors'] > 0 else 0.0


        help_content_paths_regex = '/help|/contact_us|/error|' + '|'.join(re.escape(path) for path in FUNNEL_STEPS.keys())
        help_pages_df = recent_events_in_window[
            recent_events_in_window['page_urlpath'].str.contains(help_content_paths_regex, na=False, regex=True)
        ].copy()

        if not help_pages_df.empty:
            session_durations_on_help = help_pages_df.groupby('session_id')['collector_tstamp'].apply(
                lambda x: (x.max() - x.min()).total_seconds() / 60 if len(x) > 1 else 0.0 # Ensure float
            )
            features_dict['time_on_help_content'] = float(session_durations_on_help.sum())
        else:
            features_dict['time_on_help_content'] = 0.0

        explicit_help_events_count = int((recent_events_in_window['event_name'].isin(['help_text_opened', 'tooltip_viewed'])).sum())
        
        matching_funnel_paths_for_help_usage = int(recent_events_in_window[
            recent_events_in_window['page_urlpath'].str.contains(
                r'/q/usa/usa/business_split/|/q/usa/usa/your_employees/|/q/usa/usa/business_activities/|/q/usa/usa/business_activities_split/|/q/usa/usa/policy_details/|/q/usa/usa/policy_details_split/',
                na=False, regex=True
            )
        ]['page_urlpath'].nunique())
        
        features_dict['excessive_help_text_usage'] = explicit_help_events_count + matching_funnel_paths_for_help_usage


        # --- Funnel Progression & Step-by-Step Metrics ---
        funnel_paths_regex = '|'.join(re.escape(path) for path in FUNNEL_STEPS.keys())
        funnel_events_all_cols = recent_events_in_window[
            recent_events_in_window['page_urlpath'].str.contains(funnel_paths_regex, na=False, regex=True)
        ].sort_values('collector_tstamp')

        # Dictionary to store step-specific events and times and NEW detailed event lists
        step_data = {path: {'events': [], 'start_time': None, 'end_time': None} for path in FUNNEL_STEPS.keys()}
        
        if not funnel_events_all_cols.empty:
            current_session_id = None
            
            for idx, event in funnel_events_all_cols.iterrows():
                if event['session_id'] != current_session_id:
                    current_session_id = event['session_id']

                base_path = None
                for funnel_prefix in FUNNEL_STEPS.keys():
                    if event['page_urlpath'].startswith(funnel_prefix):
                        base_path = funnel_prefix
                        break
                
                if base_path:
                    step_data[base_path]['events'].append(event)
                    if step_data[base_path]['start_time'] is None:
                        step_data[base_path]['start_time'] = event['collector_tstamp']
                    step_data[base_path]['end_time'] = event['collector_tstamp']


            features_dict['highest_funnel_step_reached'] = 0
            features_dict['num_funnel_backtracks'] = 0
            features_dict['funnel_completion_attempts'] = 0

            current_step_num = 0
            for idx, event in funnel_events_all_cols.iterrows():
                base_path = None
                for funnel_prefix in FUNNEL_STEPS.keys():
                    if event['page_urlpath'].startswith(funnel_prefix):
                        base_path = funnel_prefix
                        break
                step_num = FUNNEL_STEPS.get(base_path, 0)
                
                if step_num > 0:
                    if step_num > features_dict['highest_funnel_step_reached']:
                        features_dict['highest_funnel_step_reached'] = step_num
                    elif step_num < current_step_num and current_step_num != 0:
                        features_dict['num_funnel_backtracks'] += 1
                    current_step_num = step_num

            if features_dict['highest_funnel_step_reached'] == max(FUNNEL_STEPS.values()):
                last_step_key = [k for k,v in FUNNEL_STEPS.items() if v == max(FUNNEL_STEPS.values())][0]
                last_step_events = funnel_events_all_cols[funnel_events_all_cols['page_urlpath'].str.startswith(last_step_key)]
                if not last_step_events.empty and (last_step_events['event_name'] == 'page_view').any():
                    features_dict['funnel_completion_attempts'] = 1


        # --- CALCULATE STEP-SPECIFIC METRICS ---
        for path_prefix, step_num in FUNNEL_STEPS.items():
            # Filter events for this specific step prefix
            step_events_in_df = pd.DataFrame([e for e_list in step_data.values() for e in e_list['events'] if e['page_urlpath'].startswith(path_prefix)])
            
            # Initialize detailed lists for this step
            errors_details = []
            help_event_details = []

            if step_events_in_df.empty:
                funnel_step_metrics.append({
                    'step_name': path_prefix,
                    'step_number': int(step_num),
                    'visited': False,
                    'duration_minutes': 0.0,
                    'errors_on_step': 0,
                    'help_events_on_step': 0,
                    'completed_step': False,
                    'error_details': [], # NEW
                    'help_event_details': [] # NEW
                })
                continue

            step_events_in_df['collector_tstamp'] = pd.to_datetime(step_events_in_df['collector_tstamp'], utc=True, errors='coerce')
            step_events_in_df.dropna(subset=['collector_tstamp'], inplace=True)
            step_events_in_df = step_events_in_df.sort_values('collector_tstamp')

            first_event_time = step_events_in_df['collector_tstamp'].min()
            last_event_time = step_events_in_df['collector_tstamp'].max()
            duration = (last_event_time - first_event_time).total_seconds() / 60 if len(step_events_in_df) > 1 else 0.0

            # Populate detailed error and help event lists
            for idx, event in step_events_in_df.iterrows():
                if event['event_name'] in ['form_error', 'invalid_field_submitted']:
                    errors_details.append({
                        'event_name': event['event_name'],
                        'timestamp': event['collector_tstamp'].isoformat(), # Convert to string
                        'message': event['error_message'] if pd.notna(event['error_message']) else 'N/A',
                        'element_id': event['element_id'] if pd.notna(event['element_id']) else 'N/A'
                    })
                elif event['event_name'] in ['help_text_opened', 'tooltip_viewed']:
                    help_event_details.append({
                        'event_name': event['event_name'],
                        'timestamp': event['collector_tstamp'].isoformat(), # Convert to string
                        'element_id': event['element_id'] if pd.notna(event['element_id']) else 'N/A'
                    })

            errors_on_step = len(errors_details) # Use len of list for count
            help_events_on_step = len(help_event_details) # Use len of list for count

            completed_step = features_dict['highest_funnel_step_reached'] >= step_num + 1

            if step_num == max(FUNNEL_STEPS.values()) and features_dict['funnel_completion_attempts'] > 0:
                completed_step = True


            funnel_step_metrics.append({
                'step_name': path_prefix,
                'step_number': int(step_num),
                'visited': True,
                'duration_minutes': float(duration),
                'errors_on_step': int(errors_on_step), # Ensure int
                'help_events_on_step': int(help_events_on_step), # Ensure int
                'completed_step': completed_step,
                'error_details': errors_details, # NEW
                'help_event_details': help_event_details # NEW
            })

    feature_df = pd.DataFrame([features_dict])[loaded_feature_columns]
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(0, inplace=True)
    
    return feature_df, funnel_step_metrics


# --- NEW HELPER FUNCTION: Generate Churn Pattern Explanation (no changes from last version) ---
def generate_churn_pattern_explanation(user_features):
    summary_phrases = []
    detailed_points = []
    
    days_inactive = user_features.get('days_since_last_activity', 0)
    num_sessions = user_features.get('num_sessions', 0)

    if days_inactive >= CHURN_INACTIVITY_DAYS:
        summary_phrases.append("This user is **highly disengaged** due to prolonged inactivity.")
        detailed_points.append(f"Inactive for {days_inactive} days, signaling potential churn.")
    elif days_inactive > 30:
        summary_phrases.append("Engagement is **declining**, with recent low activity.")
        detailed_points.append(f"Last interacted {days_inactive} days ago, indicating fading interest.")
    elif num_sessions == 0:
        summary_phrases.append("No recent active sessions observed within the lookback window.")
    else:
        summary_phrases.append("User is **recently active**.")
        avg_session_duration = user_features.get('avg_session_duration', 0)
        avg_events_per_session = user_features.get('avg_events_per_session', 0)
        detailed_points.append(f"Active with {num_sessions} sessions, averaging {avg_session_duration:.1f} min and {avg_events_per_session:.0f} events per session.")

    repeated_errors = user_features.get('repeated_validation_errors', 0)
    form_abandonment = user_features.get('form_abandonment_rate', 0)
    
    if repeated_errors > 5 and form_abandonment > 0.5:
        summary_phrases.append("They encountered **critical form errors leading to abandonment**.")
        detailed_points.append(f"Suffered {repeated_errors} validation errors, leading to a major drop-off in a key form process.")
    elif repeated_errors > 0:
        summary_phrases.append("Experienced **form validation issues**.")
        detailed_points.append(f"Encountered {repeated_errors} validation errors, causing user frustration.")
    
    excessive_help = user_features.get('excessive_help_text_usage', 0)
    time_on_help = user_features.get('time_on_help_content', 0)

    if excessive_help > 5 and time_on_help > 5:
        summary_phrases.append("Showed **significant confusion and high support needs**.")
        detailed_points.append(f"Accessed help content/elements {excessive_help} times and spent {time_on_help:.1f} minutes on help, indicating severe difficulty.")
    elif excessive_help > 0:
        summary_phrases.append("Exhibited **frequent help-seeking behavior**.")
        detailed_points.append(f"Accessed help content/elements {excessive_help} times, suggesting points of confusion.")
    elif time_on_help > 0:
        summary_phrases.append("Spent time on **dedicated help resources**.")
        detailed_points.append(f"Spent {time_on_help:.1f} minutes on help-related content, seeking clarification.")

    highest_funnel_step = user_features.get('highest_funnel_step_reached', 0)
    if highest_funnel_step > 0:
        total_funnel_steps = max(FUNNEL_STEPS.values()) if FUNNEL_STEPS else 0
        detailed_points.append(f"Progressed to step {highest_funnel_step} out of {total_funnel_steps} in the main funnel.")
        if user_features.get('num_funnel_backtracks', 0) > 0:
            detailed_points.append(f"Note: User had {user_features.get('num_funnel_backtracks', 0)} backtracks in the funnel.")

    if not summary_phrases and not detailed_points:
        return "This user shows no distinct churn patterns based on available data, indicating a stable or new user."
    
    overall_summary = "Based on their activity, the user's churn pattern indicates: " + " ".join(summary_phrases)
    if detailed_points:
        return overall_summary + "\n\nKey observations include:\n- " + "\n- ".join(detailed_points)
    else:
        return overall_summary


# --- API Endpoint for Prediction ---
@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    if loaded_model is None or loaded_feature_columns is None or raw_events_df_global.empty:
        return jsonify({'error': 'Service not fully initialized. Model or data not loaded.'}), 500

    data = request.json
    if not data or 'user_id' not in data or 'recent_events' not in data:
        return jsonify({'error': 'Missing user_id or recent_events in request payload.'}), 400

    user_id = data['user_id']
    recent_events = data['recent_events']

    print(f"Received prediction request for user: {user_id}")

    try:
        user_features_df, funnel_step_metrics = featurize_user_data_for_prediction(
            user_id,
            recent_events_data=recent_events,
            all_historical_events_df=raw_events_df_global
        )
        user_calculated_features = user_features_df.iloc[0].to_dict()
        
        print(f"Engineered features for {user_id}: {user_calculated_features}")
    except Exception as e:
        print(f"Error during feature engineering for user {user_id}: {e}")
        return jsonify({'error': f'Feature engineering failed: {str(e)}'}), 500

    try:
        churn_proba = loaded_model.predict_proba(user_features_df)[:, 1][0]
        status = "High Risk" if churn_proba >= HIGH_RISK_THRESHOLD else "Low Risk"

        feature_importances_dict = {}
        if hasattr(loaded_model, 'feature_importances_'):
            importances = pd.Series(loaded_model.feature_importances_, index=loaded_feature_columns)
            sorted_importances = importances.sort_values(ascending=False)

            top_n_features = 5
            user_specific_features = user_features_df.iloc[0].to_dict() 

            for feature_name, importance_score in sorted_importances.head(top_n_features).items():
                feature_value = user_specific_features.get(feature_name, 0.0)
                feature_importances_dict[feature_name] = {
                    'value': float(feature_value),
                    'importance': float(importance_score)
                }

        requested_features_for_display = [
            'form_abandonment_rate',
            'time_on_help_content',
            'excessive_help_text_usage',
            'repeated_validation_errors',
            'days_since_last_activity'
        ]

        for feature_name in requested_features_for_display:
            if feature_name in user_specific_features:
                val = user_specific_features[feature_name]
                
                importance = 0.0
                if hasattr(loaded_model, 'feature_importances_') and feature_name in loaded_feature_columns:
                    importance = importances.get(feature_name, 0.0)
                
                if val > 0 or feature_name in ['form_abandonment_rate', 'days_since_last_activity']:
                    feature_importances_dict[feature_name] = {
                        'value': float(val),
                        'importance': float(importance)
                    }
        
        churn_pattern_explanation = generate_churn_pattern_explanation(user_calculated_features)


        response = {
            'user_id': user_id,
            'churn_probability': float(churn_proba),
            'status': status,
            'contributing_features': feature_importances_dict,
            'churn_pattern_explanation': churn_pattern_explanation,
            'funnel_step_metrics': funnel_step_metrics # NEW FIELD: Funnel Health Metrics
        }
        print(f"Prediction for {user_id}: Proba={churn_proba:.2f}, Status={status}")
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction for user {user_id}: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5124)
