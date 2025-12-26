"""
Feature Set 1
Concrete implementation of feature generator with comprehensive churn features
"""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
from features.base_features import BaseFeatureGenerator


class FeatureSet1(BaseFeatureGenerator):
    """
    First feature set implementation with comprehensive user behavior features.

    Generates 38 features per user covering:
    - Song interaction features (7 features)
    - Session features (5 features)
    - Device features (3 features)
    - HTTP status features (5 features)
    - Engagement features (3 features)
    - User action features (9 features)
    - Song string features (6 features)
    """

    def load_data(self):
        """Load the JSONL data into a pandas DataFrame"""
        data = []
        with open(self.data_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        self.df = pd.DataFrame(data)

        # Convert timestamps to datetime
        self.df["ts_dt"] = pd.to_datetime(self.df["ts"], unit="ms")
        self.df["registration_dt"] = pd.to_datetime(self.df["registration"], unit="ms")

        # Sort by timestamp once at load time for all timestamp-dependent features
        self.df = self.df.sort_values("ts_dt").reset_index(drop=True)

        return self

    def generate_features(
        self, cutoff_date=None, observation_window_days=30, active_users_only=True
    ):
        """
        Generate features for all users using a sliding observation window.

        Args:
            cutoff_date: datetime object. The end of the observation window. If None, use max date in data.
            observation_window_days: Number of days to look back from cutoff_date for feature generation.
                                     Features will be computed using events in the window
                                     [cutoff_date - observation_window_days, cutoff_date)
            active_users_only: If True, only include users who haven't cancelled yet by cutoff_date

        Returns:
            DataFrame with features per userId
        """
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        # Set cutoff_date to max date if not provided
        if cutoff_date is None:
            cutoff_date = df["ts_dt"].max()

        # Calculate observation window start date
        observation_start = cutoff_date - timedelta(days=observation_window_days)

        # Filter to observation window: [observation_start, cutoff_date)
        df = df[(df["ts_dt"] >= observation_start) & (df["ts_dt"] < cutoff_date)]

        # Identify cancelled users (those who have Cancellation Confirmation within the window)
        cancelled_users = set(df[df["page"] == "Cancellation Confirmation"]["userId"].unique())

        # Filter to active users only if requested
        if active_users_only:
            df = df[~df["userId"].isin(cancelled_users)]

        # Generate features per user
        features = []
        for user_id in df["userId"].unique():
            user_df = df[df["userId"] == user_id]
            features.append(self._generate_user_features(user_df, user_id))

        feature_df = pd.DataFrame(features)
        return feature_df

    def generate_labels(self, cutoff_date, label_window_days=30):
        """
        Generate churn labels based on events after cutoff_date.
        A user is labeled as churned if they cancel within label_window_days after cutoff_date.

        Args:
            cutoff_date: datetime object. The date to split features from labels.
            label_window_days: Number of days after cutoff to look for churn events.

        Returns:
            DataFrame with userId and churn label (1 = churned, 0 = not churned)
        """
        if self.df is None:
            self.load_data()

        label_end_date = cutoff_date + timedelta(days=label_window_days)

        # Find users who cancelled between cutoff_date and label_end_date
        future_events = self.df[
            (self.df["ts_dt"] >= cutoff_date) & (self.df["ts_dt"] < label_end_date)
        ]

        churned_users = set(
            future_events[future_events["page"] == "Cancellation Confirmation"]["userId"].unique()
        )

        # Get all users who were active before cutoff
        active_users = self.df[self.df["ts_dt"] < cutoff_date]["userId"].unique()

        labels = []
        for user_id in active_users:
            labels.append({"userId": user_id, "churn": 1 if user_id in churned_users else 0})

        return pd.DataFrame(labels)

    def _generate_user_features(self, user_df, user_id):
        """Generate all features for a single user"""
        features = {"userId": user_id}

        # Song interaction features
        features.update(self._song_interaction_features(user_df))

        # Session features
        features.update(self._session_features(user_df))

        # Device features
        features.update(self._device_features(user_df))

        # HTTP status features
        features.update(self._http_status_features(user_df))

        # Engagement features
        features.update(self._engagement_features(user_df))

        # User action features
        features.update(self._user_action_features(user_df))

        # Song string features
        features.update(self._song_string_features(user_df))

        return features

    def _song_interaction_features(self, user_df):
        """Features related to song interactions"""
        features = {}

        # Filter to NextSong events
        songs = user_df[user_df["page"] == "NextSong"]
        total_songs = len(songs)

        features["total_songs_played"] = total_songs

        if total_songs > 0:
            # Thumbs up/down
            thumbs_up = len(user_df[user_df["page"] == "Thumbs Up"])
            thumbs_down = len(user_df[user_df["page"] == "Thumbs Down"])

            features["thumbs_up_count"] = thumbs_up
            features["thumbs_down_count"] = thumbs_down
            features["thumbs_up_ratio"] = thumbs_up / total_songs
            features["thumbs_down_ratio"] = thumbs_down / total_songs
            features["thumbs_up_down_ratio"] = thumbs_up / (
                thumbs_down + 1
            )  # Avoid division by zero

            # Song length analysis
            avg_song_length = songs["length"].mean()
            features["avg_song_length"] = avg_song_length if pd.notna(avg_song_length) else 0
        else:
            features["thumbs_up_count"] = 0
            features["thumbs_down_count"] = 0
            features["thumbs_up_ratio"] = 0
            features["thumbs_down_ratio"] = 0
            features["thumbs_up_down_ratio"] = 0
            features["avg_song_length"] = 0

        return features

    def _session_features(self, user_df):
        """Features related to user sessions"""
        features = {}

        sessions = user_df.groupby("sessionId")
        num_sessions = len(sessions)

        features["num_sessions"] = num_sessions

        if num_sessions > 0:
            # Songs per session
            songs_per_session = []
            session_lengths = []

            for session_id, session_df in sessions:
                # Count songs in this session
                songs_in_session = len(session_df[session_df["page"] == "NextSong"])
                songs_per_session.append(songs_in_session)

                # Calculate session length (login to logout or last event)
                session_start = session_df["ts_dt"].min()
                session_end = session_df["ts_dt"].max()
                session_length = (session_end - session_start).total_seconds() / 60  # in minutes
                session_lengths.append(session_length)

            features["avg_songs_per_session"] = (
                np.mean(songs_per_session) if songs_per_session else 0
            )
            features["avg_session_length_minutes"] = (
                np.mean(session_lengths) if session_lengths else 0
            )
            features["max_session_length_minutes"] = (
                np.max(session_lengths) if session_lengths else 0
            )
            features["min_session_length_minutes"] = (
                np.min(session_lengths) if session_lengths else 0
            )
        else:
            features["avg_songs_per_session"] = 0
            features["avg_session_length_minutes"] = 0
            features["max_session_length_minutes"] = 0
            features["min_session_length_minutes"] = 0

        return features

    def _device_features(self, user_df):
        """Features related to device types"""
        features = {}

        user_agents = user_df["userAgent"].dropna().unique()

        is_mobile = 0
        is_desktop = 0

        mobile_indicators = ["mobile", "android", "iphone", "ipad"]
        desktop_indicators = ["windows", "macintosh", "x11"]

        for ua in user_agents:
            ua_lower = str(ua).lower()

            # Check mobile first
            if any(mobile in ua_lower for mobile in mobile_indicators):
                is_mobile = 1
            # Only check desktop if not mobile (avoids android/linux overlap)
            elif any(desktop in ua_lower for desktop in desktop_indicators):
                is_desktop = 1

        features["is_mobile"] = is_mobile
        features["is_desktop"] = is_desktop
        features["is_both_mobile_desktop"] = 1 if (is_mobile and is_desktop) else 0

        return features

    def _http_status_features(self, user_df):
        """Features related to HTTP status codes"""
        features = {}

        features["count_404"] = len(user_df[user_df["status"] == 404])
        features["count_307"] = len(user_df[user_df["status"] == 307])
        features["count_200"] = len(user_df[user_df["status"] == 200])

        total_requests = len(user_df)
        if total_requests > 0:
            features["ratio_404"] = features["count_404"] / total_requests
            features["ratio_307"] = features["count_307"] / total_requests
        else:
            features["ratio_404"] = 0
            features["ratio_307"] = 0

        return features

    def _engagement_features(self, user_df):
        """Features related to user engagement"""
        features = {}

        # Ads seen (Roll Advert page)
        ads = user_df[user_df["page"] == "Roll Advert"]
        features["total_ads_seen"] = len(ads)

        # Ads per session
        num_sessions = user_df["sessionId"].nunique()
        features["ads_per_session"] = (
            features["total_ads_seen"] / num_sessions if num_sessions > 0 else 0
        )

        # Unique artists per session
        songs = user_df[user_df["page"] == "NextSong"]
        if num_sessions > 0 and len(songs) > 0:
            artists_per_session = []
            for session_id in user_df["sessionId"].unique():
                session_songs = songs[songs["sessionId"] == session_id]
                unique_artists = session_songs["artist"].nunique()
                artists_per_session.append(unique_artists)
            features["avg_unique_artists_per_session"] = (
                np.mean(artists_per_session) if artists_per_session else 0
            )
        else:
            features["avg_unique_artists_per_session"] = 0

        return features

    def _user_action_features(self, user_df):
        """Features related to user actions"""
        features = {}

        features["num_add_friend"] = len(user_df[user_df["page"] == "Add Friend"])
        features["num_add_to_playlist"] = len(user_df[user_df["page"] == "Add to Playlist"])
        features["num_about_page"] = len(user_df[user_df["page"] == "About"])
        features["num_settings_page"] = len(user_df[user_df["page"] == "Settings"])
        features["num_help_page"] = len(user_df[user_df["page"] == "Help"])

        # Downgrade/Upgrade
        features["is_downgrade_clicked"] = (
            1 if len(user_df[user_df["page"] == "Downgrade"]) > 0 else 0
        )
        features["is_upgrade_clicked"] = 1 if len(user_df[user_df["page"] == "Upgrade"]) > 0 else 0

        # Submit Downgrade/Upgrade (actual actions)
        features["is_downgraded"] = (
            1 if len(user_df[user_df["page"] == "Submit Downgrade"]) > 0 else 0
        )
        features["is_upgraded"] = 1 if len(user_df[user_df["page"] == "Submit Upgrade"]) > 0 else 0

        return features

    def _song_string_features(self, user_df):
        """String-based features from songs listened"""
        features = {}

        songs = user_df[user_df["page"] == "NextSong"]

        # Total unique artists
        features["num_unique_artists"] = songs["artist"].nunique()

        # Total unique songs
        features["num_unique_songs"] = songs["song"].nunique()

        # Song diversity (unique songs / total songs)
        total_songs = len(songs)
        if total_songs > 0:
            features["song_diversity"] = features["num_unique_songs"] / total_songs
            features["artist_diversity"] = features["num_unique_artists"] / total_songs
        else:
            features["song_diversity"] = 0
            features["artist_diversity"] = 0

        # Most played artist frequency
        if len(songs) > 0 and songs["artist"].notna().any():
            artist_counts = songs["artist"].value_counts()
            features["top_artist_play_count"] = (
                artist_counts.iloc[0] if len(artist_counts) > 0 else 0
            )
            features["top_artist_ratio"] = (
                features["top_artist_play_count"] / total_songs if total_songs > 0 else 0
            )
        else:
            features["top_artist_play_count"] = 0
            features["top_artist_ratio"] = 0

        return features
