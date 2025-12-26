#!/usr/bin/env python3
"""
Script to validate required fields and analyze session-to-userID relationships.

A. Checks if required fields ('ts', 'sessionId', 'page', 'auth', 'level') are missing
B. Checks if sessions can be reassigned to user IDs (analyzes sessionId-userId mapping)
"""

import json
from collections import defaultdict


def validate_required_fields(file_path):
    """
    Validate required fields and analyze session-to-userID relationships.

    Args:
        file_path: Path to the JSONL file
    """
    # Required fields to check
    required_fields = ["ts", "sessionId", "page", "auth", "level", "itemInSession"]

    # Track missing fields statistics
    missing_fields_count = defaultdict(int)
    records_with_missing_fields = []

    # Track session-to-userID mapping
    session_to_users = defaultdict(set)  # sessionId -> set of userIds

    line_count = 0
    valid_records = 0

    print(f"Analyzing: {file_path}\n")

    # Read file line by line
    with open(file_path, "r") as f:
        for line in f:
            line_count += 1

            # Show progress every 100,000 lines
            if line_count % 100000 == 0:
                print(f"  Processing... {line_count:,} lines")

            # Parse JSON line
            try:
                data = json.loads(line.strip())

                # Check for missing required fields
                missing_fields = []
                for field in required_fields:
                    if field not in data or data[field] is None or data[field] == "":
                        missing_fields.append(field)
                        missing_fields_count[field] += 1

                # Record if any required fields are missing
                if missing_fields:
                    records_with_missing_fields.append(
                        {
                            "line": line_count,
                            "missing_fields": missing_fields,
                            "data": data,
                        }
                    )
                else:
                    valid_records += 1

                # Track session-to-userID mapping (even if fields are missing)
                session_id = data.get("sessionId")
                user_id = data.get("userId")

                if session_id is not None and user_id is not None:
                    session_to_users[session_id].add(user_id)

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_count}: {e}")
                continue

    print(f"\n{'='*60}")
    print("A. REQUIRED FIELDS VALIDATION")
    print(f"{'='*60}")
    print(f"Total records:        {line_count:>10,}")
    print(f"Valid records:        {valid_records:>10,}  ({valid_records/line_count*100:.1f}%)")
    print(
        f"Invalid records:      {len(records_with_missing_fields):>10,}  ({len(records_with_missing_fields)/line_count*100:.1f}%)"
    )

    if missing_fields_count:
        print("\nMissing fields breakdown:")
        for field in required_fields:
            count = missing_fields_count[field]
            if count > 0:
                print(f"  {field:12s}  {count:>8,}  ({count/line_count*100:.2f}%)")
    else:
        print("\n[+] All records have all required fields!")

    # Show examples if needed
    if records_with_missing_fields and len(records_with_missing_fields) <= 5:
        print("\nExamples (line, missing fields):")
        for record in records_with_missing_fields[:5]:
            print(f"  Line {record['line']}: {', '.join(record['missing_fields'])}")

    print(f"\n{'='*60}")
    print("B. SESSION-TO-USERID MAPPING")
    print(f"{'='*60}")

    # Categorize sessions by number of users
    one_to_one_sessions = 0
    one_to_many_sessions = 0
    problematic_sessions = []

    for session_id, user_ids in session_to_users.items():
        num_users = len(user_ids)
        if num_users == 1:
            one_to_one_sessions += 1
        else:
            one_to_many_sessions += 1
            problematic_sessions.append(
                {
                    "sessionId": session_id,
                    "userIds": sorted(user_ids),
                    "num_users": num_users,
                }
            )

    print(f"Total sessions:       {len(session_to_users):>10,}")
    print(
        f"1 session -> 1 user:  {one_to_one_sessions:>10,}  ({one_to_one_sessions/len(session_to_users)*100:.1f}%)"
    )
    print(
        f"1 session -> N users: {one_to_many_sessions:>10,}  ({one_to_many_sessions/len(session_to_users)*100:.1f}%)"
    )

    # Conclusion
    print(f"\n{'='*60}")
    if one_to_many_sessions == 0:
        print("[+] Sessions can be reassigned to user IDs (1:1 mapping)")
    else:
        print(f"[-] Sessions CANNOT be reassigned ({one_to_many_sessions:,} conflicts)")

        # Show top examples
        if problematic_sessions:
            problematic_sessions.sort(key=lambda x: x["num_users"], reverse=True)
            print(f"\nTop {min(5, len(problematic_sessions))} conflicts:")
            for session_info in problematic_sessions[:5]:
                print(
                    f"  Session {session_info['sessionId']} -> {session_info['num_users']} users: {session_info['userIds']}"
                )
    print(f"{'='*60}")

    # Return summary statistics
    return {
        "total_records": line_count,
        "valid_records": valid_records,
        "invalid_records": len(records_with_missing_fields),
        "missing_fields_count": dict(missing_fields_count),
        "total_sessions": len(session_to_users),
        "one_to_one_sessions": one_to_one_sessions,
        "one_to_many_sessions": one_to_many_sessions,
        "can_reassign": one_to_many_sessions == 0,
    }


if __name__ == "__main__":
    jsonl_file = "data_clean_process/customer_churn.json"
    validate_required_fields(jsonl_file)


"""

Analyzing: customer_churn.json

  Processing... 100,000 lines
  Processing... 200,000 lines
  Processing... 300,000 lines
  Processing... 400,000 lines
  Processing... 500,000 lines

============================================================
A. REQUIRED FIELDS VALIDATION
============================================================
Total records:           543,705
Valid records:           543,705  (100.0%)
Invalid records:               0  (0.0%)

[+] All records have all required fields!

============================================================
B. SESSION-TO-USERID MAPPING
============================================================
Total sessions:            4,590
1 session -> 1 user:       1,443  (31.4%)
1 session -> N users:      3,147  (68.6%)

============================================================
[-] Sessions CANNOT be reassigned (3,147 conflicts)

Top 5 conflicts:
  Session 292 -> 5 users: ['', '100043', '200008', '293', '300036']
  Session 97 -> 5 users: ['', '100016', '200045', '300027', '98']
  Session 178 -> 5 users: ['', '100041', '179', '200038', '300021']
  Session 245 -> 5 users: ['', '100003', '200050', '246', '300035']
  Session 38 -> 5 users: ['', '100038', '200038', '300038', '39']
============================================================



"""
