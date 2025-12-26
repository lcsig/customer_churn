#!/usr/bin/env python3
"""
Script to check if cancelled users appeared again in the dataset.
"""

import json
from collections import defaultdict


def check_cancelled_users(file_path):
    """Check if users were cancelled but appeared again later."""
    user_activity = defaultdict(list)

    # Read file
    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                user_id = data.get("userId", "")

                if not user_id:
                    continue

                user_activity[user_id].append(
                    {
                        "ts": data.get("ts"),
                        "sessionId": data.get("sessionId"),
                        "page": data.get("page", ""),
                        "auth": data.get("auth", ""),
                    }
                )
            except json.JSONDecodeError:
                continue

    # Find cancelled users who logged in again
    cancelled_and_returned = []

    for user_id, activities in user_activity.items():
        activities_sorted = sorted(activities, key=lambda x: x["ts"])

        # Find last cancellation
        last_cancel_idx = -1
        for idx, activity in enumerate(activities_sorted):
            if activity["auth"] == "Cancelled":
                last_cancel_idx = idx

        # Check if user logged in after cancellation
        if last_cancel_idx != -1:
            logged_in_after = [
                a for a in activities_sorted[last_cancel_idx + 1 :] if a["auth"] == "Logged In"
            ]

            if logged_in_after:
                cancelled_and_returned.append(
                    {
                        "userId": user_id,
                        "cancellation": activities_sorted[last_cancel_idx],
                        "logged_in_after": logged_in_after,
                    }
                )

    # Print results
    print(f"\nTotal users: {len(user_activity)}")
    print(f"Users who cancelled and logged in again: {len(cancelled_and_returned)}\n")

    if cancelled_and_returned:
        for user_data in cancelled_and_returned:
            print(f"User ID: {user_data['userId']}")
            print(
                f"  Cancellation: Session {user_data['cancellation']['sessionId']}, "
                f"Page: {user_data['cancellation']['page']}"
            )
            print(
                f"  Logged in again: {len(user_data['logged_in_after'])} times after cancellation"
            )
            print()


if __name__ == "__main__":
    check_cancelled_users("data_clean_process/customer_churn.json")

"""


Total users: 448
Users who cancelled and logged in again: 0


"""
