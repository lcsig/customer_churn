"""
Feature Set 2
Enhanced feature generator with temporal, trend, and demographic features
Extends FeatureSet1 with additional churn prediction features
"""

import hashlib

import pandas as pd

from ml_training.features.feature_set1 import FeatureSet1


class FeatureSet2(FeatureSet1):
    """
    Enhanced feature set extending FeatureSet1 with critical churn prediction features.

    Adds 52 new features to the 38 base features from FeatureSet1 (90 total):
    - Temporal/recency features (12 features)
    - Engagement trend features (15 features) - declining activity detection
    - Subscription/demographic features (5 features) - includes location hash
    - Additional event features (9 features)
    - Time pattern features (11 features)

    Inherits all base features from FeatureSet1.
    """

    def generate_features(
        self, cutoff_date=None, observation_window_days=30, active_users_only=True
    ):
        """
        Override to store cutoff_date for temporal features.
        """
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        # Set cutoff_date to max date if not provided
        if cutoff_date is None:
            cutoff_date = df["ts_dt"].max()

        # Store cutoff_date for temporal features
        self.cutoff_date = cutoff_date

        # Call parent's generate_features logic
        return super().generate_features(cutoff_date, observation_window_days, active_users_only)

    def _generate_user_features(self, user_df, user_id):
        """
        Generate all features for a single user.
        Extends parent's 38 base features with 52 additional churn prediction features.

        Total: 90 features per user
        """
        # Get all base features from FeatureSet1 (38 features)
        features = super()._generate_user_features(user_df, user_id)

        # Add new features (52 features)
        features.update(self._temporal_recency_features(user_df))  # 12 features
        features.update(self._engagement_trend_features(user_df))  # 15 features
        features.update(
            self._subscription_demographics_features(user_df)
        )  # 5 features (includes location hash)
        features.update(self._additional_event_features(user_df))  # 9 features
        features.update(self._time_pattern_features(user_df))  # 11 features

        return features

    def _temporal_recency_features(self, user_df):
        """
        Critical temporal and recency features.

        Generates 12 features:
        - Days/hours since last activity
        - User tenure (days/weeks since registration)
        - Active days metrics (count, ratio)
        - Gap analysis between sessions (max, avg, min, std)
        """
        features = {}

        # Days/hours since last activity (VERY IMPORTANT)
        last_activity = user_df["ts_dt"].max()
        features["days_since_last_activity"] = (self.cutoff_date - last_activity).days
        features["hours_since_last_activity"] = (
            self.cutoff_date - last_activity
        ).total_seconds() / 3600

        # User tenure (days since registration)
        if len(user_df) > 0:
            # Get first non-null registration_dt (some users have NaT in first rows)
            # Look in observation window first, but if all NaT, check full dataset
            non_null_registration = user_df["registration_dt"].dropna()

            if len(non_null_registration) == 0:
                # All registration_dt are NaT in observation window
                # Look up registration from full dataset for this user
                user_id = user_df["userId"].iloc[0]
                full_user_data = self.df[self.df["userId"] == user_id]
                non_null_registration = full_user_data["registration_dt"].dropna()

            if len(non_null_registration) > 0:
                registration_date = non_null_registration.iloc[0]
                features["days_since_registration"] = (self.cutoff_date - registration_date).days
                # Account age in weeks
                features["account_age_weeks"] = features["days_since_registration"] / 7.0
            else:
                # Missing registration data - use 0 or median
                features["days_since_registration"] = 0
                features["account_age_weeks"] = 0
        else:
            features["days_since_registration"] = 0
            features["account_age_weeks"] = 0

        # Active days metrics
        active_days = user_df["ts_dt"].dt.date.nunique()
        observation_days = (user_df["ts_dt"].max() - user_df["ts_dt"].min()).days + 1

        features["active_days_count"] = active_days
        features["observation_window_days"] = observation_days
        features["active_days_ratio"] = (
            active_days / observation_days if observation_days > 0 else 0
        )

        # Gap analysis between sessions
        session_dates = user_df.groupby("sessionId")["ts_dt"].min().sort_values()
        if len(session_dates) > 1:
            gaps = session_dates.diff().dt.total_seconds() / 3600  # in hours
            # Remove NaN from diff (first value is always NaN)
            gaps = gaps.dropna()
            if len(gaps) > 0:
                features["max_gap_between_sessions_hours"] = gaps.max()
                features["avg_gap_between_sessions_hours"] = gaps.mean()
                features["min_gap_between_sessions_hours"] = gaps.min()
                features["std_gap_between_sessions_hours"] = gaps.std() if len(gaps) > 1 else 0
            else:
                features["max_gap_between_sessions_hours"] = 0
                features["avg_gap_between_sessions_hours"] = 0
                features["min_gap_between_sessions_hours"] = 0
                features["std_gap_between_sessions_hours"] = 0
        else:
            features["max_gap_between_sessions_hours"] = 0
            features["avg_gap_between_sessions_hours"] = 0
            features["min_gap_between_sessions_hours"] = 0
            features["std_gap_between_sessions_hours"] = 0

        # Days since first activity in observation window
        first_activity = user_df["ts_dt"].min()
        features["days_since_first_activity_in_window"] = (self.cutoff_date - first_activity).days

        return features

    def _engagement_trend_features(self, user_df):
        """
        Detect declining engagement - CRITICAL for churn prediction.

        Generates 15 features:
        - Songs played: early vs recent period (count, ratio, diff)
        - Session frequency: early vs recent period (count, ratio, diff)
        - Active days: early vs recent period (count, ratio)
        - Overall events: early vs recent period (count, ratio)
        - Thumbs up trend ratio
        """
        features = {}

        # Split observation window in half
        min_date = user_df["ts_dt"].min()
        max_date = user_df["ts_dt"].max()
        mid_point = min_date + (max_date - min_date) / 2

        early_period = user_df[user_df["ts_dt"] < mid_point]
        recent_period = user_df[user_df["ts_dt"] >= mid_point]

        # Songs played trend
        early_songs = len(early_period[early_period["page"] == "NextSong"])
        recent_songs = len(recent_period[recent_period["page"] == "NextSong"])

        features["songs_early_period"] = early_songs
        features["songs_recent_period"] = recent_songs
        features["songs_trend_ratio"] = recent_songs / (
            early_songs + 1
        )  # >1 increasing, <1 declining
        features["songs_trend_diff"] = recent_songs - early_songs

        # Session frequency trend
        early_sessions = early_period["sessionId"].nunique()
        recent_sessions = recent_period["sessionId"].nunique()

        features["sessions_early_period"] = early_sessions
        features["sessions_recent_period"] = recent_sessions
        features["sessions_trend_ratio"] = recent_sessions / (early_sessions + 1)
        features["sessions_trend_diff"] = recent_sessions - early_sessions

        # Active days trend
        early_days = early_period["ts_dt"].dt.date.nunique()
        recent_days = recent_period["ts_dt"].dt.date.nunique()

        features["active_days_early_period"] = early_days
        features["active_days_recent_period"] = recent_days
        features["active_days_trend_ratio"] = recent_days / (early_days + 1)

        # Overall activity trend (total events)
        early_events = len(early_period)
        recent_events = len(recent_period)

        features["events_early_period"] = early_events
        features["events_recent_period"] = recent_events
        features["events_trend_ratio"] = recent_events / (early_events + 1)

        # Thumbs interaction trend
        early_thumbs_up = len(early_period[early_period["page"] == "Thumbs Up"])
        recent_thumbs_up = len(recent_period[recent_period["page"] == "Thumbs Up"])
        features["thumbs_up_trend_ratio"] = recent_thumbs_up / (early_thumbs_up + 1)

        return features

    def _subscription_demographics_features(self, user_df):
        """
        Subscription level and demographic features.

        Generates 5 features:
        - Current subscription level (paid vs free)
        - Level change indicator
        - Gender (male/female flags)
        - Location hash (numeric representation of location string)
        """
        features = {}

        # Current subscription level (free vs paid) - use last known level
        # Note: data is sorted by ts_dt at load time
        if "level" in user_df.columns and len(user_df) > 0:
            latest_level = user_df["level"].iloc[-1]
            features["is_paid_user"] = 1 if latest_level == "paid" else 0

            # Check if level changed during observation window
            level_changes = user_df["level"].nunique()
            features["level_changed"] = 1 if level_changes > 1 else 0
        else:
            features["is_paid_user"] = 0
            features["level_changed"] = 0

        # Gender
        if "gender" in user_df.columns and len(user_df) > 0:
            gender = user_df["gender"].iloc[0]
            features["is_male"] = 1 if gender == "M" else 0
            features["is_female"] = 1 if gender == "F" else 0
        else:
            features["is_male"] = 0
            features["is_female"] = 0

        # Location hash - convert location string to numeric value
        if "location" in user_df.columns and len(user_df) > 0:
            location = user_df["location"].iloc[0]
            if pd.notna(location) and location != "":
                # Use hashlib for consistent hashing across sessions
                location_str = str(location)
                # Create hash and convert to integer, use modulo to keep in reasonable range
                hash_obj = hashlib.md5(location_str.encode())
                # Convert to int and normalize to reasonable range (0 to 999999)
                features["location_hash"] = int(hash_obj.hexdigest(), 16) % 1000000
            else:
                features["location_hash"] = 0
        else:
            features["location_hash"] = 0

        return features

    def _additional_event_features(self, user_df):
        """
        Additional event-based features.

        Generates 9 features:
        - Page visit counts (Home, Login, Logout, Error, Save Settings, Submit Registration)
        - Event ratios (logout, home, error)
        """
        features = {}

        # Home page visits
        features["num_home_visits"] = len(user_df[user_df["page"] == "Home"])

        # Logout events
        features["num_logout"] = len(user_df[user_df["page"] == "Logout"])

        # Login events
        features["num_login"] = len(user_df[user_df["page"] == "Login"])

        # Error page visits
        features["num_error_page"] = len(user_df[user_df["page"] == "Error"])

        # Save Settings
        features["num_save_settings"] = len(user_df[user_df["page"] == "Save Settings"])

        # Submit Registration (probably rare in observation window)
        features["num_submit_registration"] = len(user_df[user_df["page"] == "Submit Registration"])

        # Calculate ratios for key events
        total_events = len(user_df)
        if total_events > 0:
            features["logout_ratio"] = features["num_logout"] / total_events
            features["home_ratio"] = features["num_home_visits"] / total_events
            features["error_ratio"] = features["num_error_page"] / total_events
        else:
            features["logout_ratio"] = 0
            features["home_ratio"] = 0
            features["error_ratio"] = 0

        return features

    def _time_pattern_features(self, user_df):
        """
        Time-of-day and day-of-week patterns.

        Generates 11 features:
        - Hour diversity (unique hours active, hour diversity ratio)
        - Day of week diversity
        - Weekend vs weekday activity (counts, ratio)
        - Peak hour analysis (peak hour, activity count, ratio)
        - Daily activity variance (std, mean)
        """
        features = {}

        if len(user_df) > 0:
            # Extract hour and day of week
            user_df_copy = user_df.copy()
            user_df_copy["hour"] = user_df_copy["ts_dt"].dt.hour
            user_df_copy["day_of_week"] = user_df_copy["ts_dt"].dt.dayofweek

            # Hour diversity (how many different hours user is active)
            features["unique_hours_active"] = user_df_copy["hour"].nunique()
            features["hour_diversity"] = features["unique_hours_active"] / 24.0

            # Day of week diversity
            features["unique_days_of_week_active"] = user_df_copy["day_of_week"].nunique()

            # Weekend vs weekday activity
            weekend_events = len(user_df_copy[user_df_copy["day_of_week"].isin([5, 6])])
            weekday_events = len(user_df_copy[~user_df_copy["day_of_week"].isin([5, 6])])

            features["weekend_activity_count"] = weekend_events
            features["weekday_activity_count"] = weekday_events
            features["weekend_to_weekday_ratio"] = weekend_events / (weekday_events + 1)

            # Peak hour activity (most active hour)
            if len(user_df_copy) > 0:
                hour_counts = user_df_copy["hour"].value_counts()
                features["peak_hour"] = hour_counts.index[0] if len(hour_counts) > 0 else 0
                features["peak_hour_activity_count"] = (
                    hour_counts.iloc[0] if len(hour_counts) > 0 else 0
                )
                features["peak_hour_ratio"] = features["peak_hour_activity_count"] / len(
                    user_df_copy
                )
            else:
                features["peak_hour"] = 0
                features["peak_hour_activity_count"] = 0
                features["peak_hour_ratio"] = 0

            # Variance in daily activity (consistency)
            daily_activity = user_df_copy.groupby(user_df_copy["ts_dt"].dt.date).size()
            features["daily_activity_std"] = daily_activity.std() if len(daily_activity) > 1 else 0
            features["daily_activity_mean"] = (
                daily_activity.mean() if len(daily_activity) > 0 else 0
            )
        else:
            features["unique_hours_active"] = 0
            features["hour_diversity"] = 0
            features["unique_days_of_week_active"] = 0
            features["weekend_activity_count"] = 0
            features["weekday_activity_count"] = 0
            features["weekend_to_weekday_ratio"] = 0
            features["peak_hour"] = 0
            features["peak_hour_activity_count"] = 0
            features["peak_hour_ratio"] = 0
            features["daily_activity_std"] = 0
            features["daily_activity_mean"] = 0

        return features
