import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import re

# --- Configuration (MUST BE IDENTICAL TO app.py) ---
LOOKBACK_DAYS = 60
CHURN_INACTIVITY_DAYS = 60 # Check this value. If your raw_events_df.csv doesn't have old data, try reducing it TEMPORARILY to 10 or 30
HIGH_RISK_THRESHOLD = 0.50

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
# --- Feature Engineering Function (MUST BE IDENTICAL TO app.py) ---
def featurize_user_data(user_events_df, current_time_for_features):
    """
    Engineers features for churn prediction from raw user event data.

    Args:
        user_events_df (pd.DataFrame): DataFrame of raw events for all users.
        current_time_for_features (datetime): The reference time for calculating recency features.

    Returns:
        pd.DataFrame: A DataFrame with engineered features per user.
    """
    # Ensure collector_tstamp is datetime and UTC
    user_events_df['collector_tstamp'] = pd.to_datetime(user_events_df['collector_tstamp'], utc=True, errors='coerce')
    user_events_df.dropna(subset=['collector_tstamp'], inplace=True)
    user_events_df = user_events_df.sort_values(by=['domain_userid', 'collector_tstamp'])

    # Define the lookback window for recent activity
    lookback_start = current_time_for_features - timedelta(days=LOOKBACK_DAYS)

    all_users_features = []

    # Iterate through users ONLY if user_events_df is not empty
    if not user_events_df.empty: # <--- Added this check
        for user_id, user_df in user_events_df.groupby('domain_userid'):
            recent_user_events = user_df[
                (user_df['collector_tstamp'] >= lookback_start) &
                (user_df['collector_tstamp'] < current_time_for_features)
            ].copy()

            # Initialize features with defaults
            # This dictionary must contain all expected feature names
            # (including 'days_since_last_activity' which was discussed)
            features = {
                'total_page_views': 0,
                'total_clicks': 0,
                'unique_pages_visited': 0,
                'num_sessions': 0,
                'avg_session_duration': 0,
                'avg_events_per_session': 0,
                'num_error_events': 0,
                'rage_clicks': 0,
                'form_abandonment_rate': 0,
                'time_on_help_content': 0,
                'excessive_help_text_usage': 0,
                'highest_funnel_step_reached': 0,
                'num_funnel_backtracks': 0,
                'funnel_completion_attempts': 0,
                'repeated_validation_errors': 0,
                'days_since_last_activity': (current_time_for_features - user_df['collector_tstamp'].max()).days
            }

            if not recent_user_events.empty:
                features['total_page_views'] = (recent_user_events['event_name'] == 'page_view').sum()
                features['total_clicks'] = (recent_user_events['event_name'] == 'click').sum()
                features['unique_pages_visited'] = recent_user_events[recent_user_events['event_name'] == 'page_view']['page_urlpath'].nunique()

                sessions = recent_user_events.groupby('session_id').agg(
                    session_start=('collector_tstamp', 'min'),
                    session_end=('collector_tstamp', 'max'),
                    num_events=('event_name', 'count')
                ).reset_index()
                sessions['session_duration_minutes'] = (sessions['session_end'] - sessions['session_start']).dt.total_seconds() / 60

                features['num_sessions'] = len(sessions)
                features['avg_session_duration'] = sessions['session_duration_minutes'].mean() if not sessions.empty else 0
                features['avg_events_per_session'] = sessions['num_events'].mean() if not sessions.empty else 0

                features['num_error_events'] = (recent_user_events['event_name'] == 'error_event').sum()

                user_clicks = recent_user_events[recent_user_events['event_name'] == 'click'].sort_values('collector_tstamp')
                if len(user_clicks) > 1:
                    user_clicks['time_diff'] = user_clicks['collector_tstamp'].diff().dt.total_seconds()
                    user_clicks['same_element'] = user_clicks['element_id'] == user_clicks['element_id'].shift(1)
                    features['rage_clicks'] = ((user_clicks['time_diff'] < 1) & (user_clicks['same_element'])).sum()
                
                funnel_error_events = recent_user_events[
                    recent_user_events['event_name'].isin(['form_error', 'invalid_field_submitted']) &
                    recent_user_events['page_urlpath'].str.contains(
                        '|'.join(re.escape(path) for path in FUNNEL_STEPS.keys()),
                        na=False, regex=True
                    )
                ]
                features['repeated_validation_errors'] = len(funnel_error_events)
                features['form_abandonment_rate'] = 1.0 if features['repeated_validation_errors'] > 0 else 0.0


                help_content_paths_regex = '/help|/contact_us|/error|' + '|'.join(re.escape(path) for path in FUNNEL_STEPS.keys())
                help_pages_df = recent_user_events[
                    recent_user_events['page_urlpath'].str.contains(help_content_paths_regex, na=False, regex=True)
                ].copy()

                if not help_pages_df.empty:
                    session_durations_on_help = help_pages_df.groupby('session_id')['collector_tstamp'].apply(
                        lambda x: (x.max() - x.min()).total_seconds() / 60 if len(x) > 1 else 0
                    )
                    features['time_on_help_content'] = session_durations_on_help.sum()
                else:
                    features['time_on_help_content'] = 0

                explicit_help_events_count = (recent_user_events['event_name'].isin(['help_text_opened', 'tooltip_viewed'])).sum()
                
                matching_funnel_paths_for_help_usage = recent_user_events[
                    recent_user_events['page_urlpath'].str.contains(
                        r'/q/usa/usa/business_split/|/q/usa/usa/your_employees|/q/usa/usa/business_activities|/q/usa/usa/business_activities_split|/q/usa/usa/policy_details|/q/usa/usa/policy_details_split',
                        na=False, regex=True
                    )
                ]['page_urlpath'].nunique()
                
                features['excessive_help_text_usage'] = explicit_help_events_count + matching_funnel_paths_for_help_usage

                funnel_paths_regex = '|'.join(re.escape(path) for path in FUNNEL_STEPS.keys())
                funnel_events = recent_user_events[
                    recent_user_events['page_urlpath'].str.contains(funnel_paths_regex, na=False, regex=True)
                ].sort_values('collector_tstamp')

                if not funnel_events.empty:
                    current_step = 0
                    for idx, event in funnel_events.iterrows():
                        base_path = None
                        for funnel_prefix in FUNNEL_STEPS.keys():
                            if event['page_urlpath'].startswith(funnel_prefix):
                                base_path = funnel_prefix
                                break
                        step = FUNNEL_STEPS.get(base_path, 0)

                        if step > 0:
                            if step > features['highest_funnel_step_reached']:
                                features['highest_funnel_step_reached'] = step
                            elif step < current_step and current_step != 0:
                                features['num_funnel_backtracks'] += 1
                            current_step = step
                    
                    if features['highest_funnel_step_reached'] == max(FUNNEL_STEPS.values()):
                        last_step_events = funnel_events[funnel_events['page_urlpath'].str.startswith(
                            [k for k,v in FUNNEL_STEPS.items() if v == max(FUNNEL_STEPS.values())][0]
                        )]
                        if not last_step_events.empty:
                            features['funnel_completion_attempts'] = 1

            all_users_features.append({'domain_userid': user_id, **features})

    # Always return a DataFrame, even if empty
    features_df = pd.DataFrame(all_users_features)
    # Ensure all columns that will be expected by the model (loaded_feature_columns) exist, filling missing with 0
    # This step is more critical in app.py's featurize_user_data_for_prediction when aligning with loaded_feature_columns.
    # For training, we construct based on calculated features, then align X to trained_feature_columns.
    
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)

    return features_df


# --- Main Training Script Logic ---
if __name__ == '__main__':
    # Load your historical raw_events_df.csv
    try:
        raw_events_df = pd.read_csv('../data/raw_events_df.csv')
        raw_events_df['collector_tstamp'] = pd.to_datetime(raw_events_df['collector_tstamp'], utc=True, errors='coerce')
        raw_events_df.dropna(subset=['collector_tstamp'], inplace=True)
        print("Historical raw_events_df.csv loaded for training.")
    except FileNotFoundError:
        print("Error: 'raw_events_df.csv' not found. Please ensure it's in the same directory.")
        exit()

    training_current_time = raw_events_df['collector_tstamp'].max()
    print(f"Training current time set to: {training_current_time}")

    print("Starting feature engineering...")
    engineered_features_df = featurize_user_data(raw_events_df, training_current_time)
    print("Feature engineering complete.")
    print(f"Engineered features for {len(engineered_features_df)} users.")

    # --- Churn Labeling ---
    last_activity_per_user = raw_events_df.groupby('domain_userid')['collector_tstamp'].max().reset_index()
    last_activity_per_user.rename(columns={'collector_tstamp': 'last_activity_tstamp'}, inplace=True)
    labeled_data = engineered_features_df.merge(last_activity_per_user, on='domain_userid', how='left')
    labeled_data['days_inactive_for_churn_label'] = (training_current_time - labeled_data['last_activity_tstamp']).dt.days
    labeled_data['churn'] = (labeled_data['days_inactive_for_churn_label'] >= CHURN_INACTIVITY_DAYS).astype(int)

    # --- CRITICAL DEBUG OUTPUTS ---
    print("\n--- Detailed Churn Labeling Check ---")
    print(f"Labeled Data Head:\n{labeled_data[['domain_userid', 'last_activity_tstamp', 'days_inactive_for_churn_label', 'churn']].head()}")
    print(f"Full Churn Value Counts:\n{labeled_data['churn'].value_counts()}")
    print(f"Users labeled as churn=1:\n{labeled_data[labeled_data['churn'] == 1]['domain_userid'].tolist()}")
    print(f"Total labeled users: {len(labeled_data)}")
    # --- END CRITICAL DEBUG OUTPUTS ---


    features_to_drop = ['domain_userid', 'last_activity_tstamp', 'days_inactive_for_churn_label']
    X = labeled_data.drop(columns=features_to_drop + ['churn'])
    y = labeled_data['churn']

    trained_feature_columns = X.columns.tolist()
    print(f"\nFeatures used for training (final list):\n{trained_feature_columns}")
    joblib.dump(trained_feature_columns, 'trained_feature_columns.joblib')

    # --- CRITICAL DEBUG OUTPUTS FOR SPLIT ---
    print("\n--- Train/Test Split Check ---")
    print(f"y value counts before split:\n{y.value_counts()}")
    # --- END CRITICAL DEBUG OUTPUTS FOR SPLIT ---

    # --- SOLUTION FOR VALUEERROR: The least populated class ...
    # This checks if the minority class has at least 2 members.
    # If not, it means stratification is impossible.
    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        print(f"WARNING: Least populated class has only {min_class_count} member(s). Stratification will be skipped.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # No stratify
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"y_train distribution after split:\n{y_train.value_counts()}")
    print(f"y_test distribution after split:\n{y_test.value_counts()}")

    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    print("Model training complete.")

    # --- CRITICAL DEBUG OUTPUTS FOR PREDICT_PROBA ---
    print("\n--- predict_proba Debug ---")
    print(f"Shape of X_test for prediction: {X_test.shape}")
    print(f"Model classes learned during fit: {rf_model.classes_}")

    # Check if the model actually learned two classes
    if len(rf_model.classes_) < 2:
        print(f"ERROR: Model only learned {len(rf_model.classes_)} class(es). Cannot get probability for index 1.")
        # Handle this gracefully, e.g., exit or return a default
        exit("Model trained on single class. Cannot proceed with predict_proba[:, 1].")
    # --- END CRITICAL DEBUG OUTPUTS FOR PREDICT_PROBA ---

    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1] # This is the problematic line

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")

    joblib.dump(rf_model, 'churn_prediction_rf_model.joblib')
    print("Model and trained_feature_columns saved successfully.")

    print("\n--- Feature Importances (from trained model) ---")
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_importances.head(10))