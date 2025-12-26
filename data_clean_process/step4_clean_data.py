#!/usr/bin/env python3
"""
Data Cleaning Script for Customer Churn JSONL
- Removes all Guest auth records
- Fills missing userId using sessionId + itemInSession reset detection
- Removes records where userId cannot be determined
- Uses itemInSession decreases to detect session boundaries
- Uses timestamp ranges to fill missing userIds (avoids overlapping itemInSession ranges)
- Removes any records that occur after a user's last cancellation
- Preserves data order
"""

import json
from collections import defaultdict


def clean_customer_data(input_file, output_file):
    """
    Clean customer churn data by:
    1. Removing Guest auth records
    2. Filling missing userId based on sessionId and itemInSession
    3. Removing records where userId cannot be determined
    4. Removing records that occur after a user's last cancellation

    Strategy:
    - Pass 1: Build session segments mapping (sessionId, timestamp range) -> userId
      * Use itemInSession to DETECT when a new user takes over (when it decreases)
      * Store TIMESTAMP ranges for each segment (not itemInSession ranges to avoid overlap)
    - Pass 2: Find last cancellation timestamp for each user
    - Pass 3: Clean data and fill missing userIds using timestamp-based segment matching
      * Records with unfillable userId are removed from output (e.g. not logged in yet, or never logged in)
      * Records after user cancellation are removed from output

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output cleaned JSONL file
    """

    # Statistics
    stats = {
        "total_lines": 0,
        "guest_removed": 0,
        "empty_userId_found": 0,
        "empty_userId_filled": 0,
        "empty_userId_unfillable": 0,
        "unfillable_removed": 0,
        "records_written": 0,
        "segments_created": 0,
        "resets_detected": 0,
        "cancelled_users": 0,
        "records_after_cancellation": 0,
    }

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    # FIRST PASS: Build session-user mapping using itemInSession to detect boundaries
    print("\n[PASS 1/2] Building session-user mapping based on itemInSession...")

    # Structure: {sessionId: [(start_ts, end_ts, userId), ...]}
    # Use itemInSession to detect when sessions change, but store timestamp ranges
    session_segments = defaultdict(list)

    # Temporary tracking: {sessionId: {'userId': str, 'start_ts': int, 'last_item': int}}
    active_sessions = {}

    with open(input_file, "r") as infile:
        for line_num, line in enumerate(infile, 1):
            if line_num % 100000 == 0:
                print(f"  Pass 1: Processed {line_num:,} lines...")

            try:
                record = json.loads(line.strip())

                # Skip Guest records
                if record.get("auth") == "Guest":
                    continue

                sessionId = str(record.get("sessionId", ""))
                userId = record.get("userId", "").strip()
                itemInSession = record.get("itemInSession")
                ts = record.get("ts", 0)

                # Skip if no sessionId or itemInSession
                if not sessionId or itemInSession is None:
                    continue

                try:
                    itemInSession = int(itemInSession)
                except (ValueError, TypeError):
                    continue

                # Only process records with userId to build the mapping
                if not userId:
                    continue

                if sessionId in active_sessions:
                    last_item = active_sessions[sessionId]["last_item"]
                    current_userId = active_sessions[sessionId]["userId"]

                    # Detect reset: itemInSession decreased - NEW USER/SESSION
                    if itemInSession < last_item:
                        stats["resets_detected"] += 1
                        # Close previous segment with timestamp range
                        start_ts = active_sessions[sessionId]["start_ts"]
                        session_segments[sessionId].append((start_ts, ts - 1, current_userId))
                        stats["segments_created"] += 1
                        # Start new segment
                        active_sessions[sessionId] = {
                            "userId": userId,
                            "start_ts": ts,
                            "last_item": itemInSession,
                        }
                    # Different user with same or increasing itemInSession - shouldn't happen but handle it
                    elif userId != current_userId:
                        stats["resets_detected"] += 1
                        # Close previous segment
                        start_ts = active_sessions[sessionId]["start_ts"]
                        session_segments[sessionId].append((start_ts, ts - 1, current_userId))
                        stats["segments_created"] += 1
                        # Start new segment
                        active_sessions[sessionId] = {
                            "userId": userId,
                            "start_ts": ts,
                            "last_item": itemInSession,
                        }
                    else:
                        # Same user, update last_item
                        active_sessions[sessionId]["last_item"] = itemInSession
                else:
                    # Start new session tracking
                    active_sessions[sessionId] = {
                        "userId": userId,
                        "start_ts": ts,
                        "last_item": itemInSession,
                    }

            except json.JSONDecodeError:
                continue

    # Close any remaining open segments (end of file)
    for sessionId, session_info in active_sessions.items():
        session_segments[sessionId].append(
            (
                session_info["start_ts"],
                float("inf"),  # Open-ended
                session_info["userId"],
            )
        )
        stats["segments_created"] += 1

    print(f"  Found {len(session_segments):,} sessions with {stats['segments_created']:,} segments")
    print(f"  Detected {stats['resets_detected']:,} itemInSession resets")

    # SECOND PASS: Find last cancellation timestamp for each user
    print("\n[PASS 2/3] Finding cancellation timestamps...")

    # Dictionary: userId -> last cancellation timestamp
    user_cancellation_ts = {}

    with open(input_file, "r") as infile:
        for line_num, line in enumerate(infile, 1):
            if line_num % 100000 == 0:
                print(f"  Pass 2: Processed {line_num:,} lines...")

            try:
                record = json.loads(line.strip())

                if record.get("auth") == "Cancelled":
                    userId = record.get("userId", "").strip()
                    ts = record.get("ts", 0)

                    if userId:
                        user_cancellation_ts[userId] = ts
                    else:
                        print(f"Warning: No userId found for cancellation at line {line_num}")

            except json.JSONDecodeError:
                continue

    stats["cancelled_users"] = len(user_cancellation_ts)
    print(f"  Found {stats['cancelled_users']:,} users with cancellation")

    # THIRD PASS: Clean and write data
    print("\n[PASS 3/3] Cleaning and writing data...")

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            stats["total_lines"] += 1

            if stats["total_lines"] % 100000 == 0:
                print(f"  Pass 3: Processed {stats['total_lines']:,} lines...")

            try:
                record = json.loads(line.strip())

                # Remove Guest records
                if record.get("auth") == "Guest":
                    stats["guest_removed"] += 1
                    continue

                # Handle missing userId
                userId = record.get("userId", "").strip()
                sessionId = str(record.get("sessionId", ""))
                ts = record.get("ts", 0)

                # Skip records after user's last cancellation
                if userId and userId in user_cancellation_ts:
                    if ts > user_cancellation_ts[userId]:
                        stats["records_after_cancellation"] += 1
                        continue

                if not userId:
                    stats["empty_userId_found"] += 1

                    # Try to find userId from session segments using timestamp ranges
                    filled = False
                    if sessionId in session_segments:
                        for start_ts, end_ts, segment_userId in session_segments[sessionId]:
                            if start_ts <= ts <= end_ts:
                                record["userId"] = segment_userId
                                stats["empty_userId_filled"] += 1
                                filled = True
                                break

                    if not filled:
                        # Could not fill userId - remove this record
                        stats["empty_userId_unfillable"] += 1
                        stats["unfillable_removed"] += 1
                        continue

                # Write cleaned record (only if it has a userId)
                outfile.write(json.dumps(record) + "\n")
                stats["records_written"] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {stats['total_lines']}: {e}")
                continue

    # Print statistics
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"Total lines read:              {stats['total_lines']:,}")
    print("\nSession Segmentation:")
    print(f"  Segments created:            {stats['segments_created']:,}")
    print(f"  itemInSession resets:        {stats['resets_detected']:,}")
    print("\nUserId Filling:")
    print(f"  Empty userId found:          {stats['empty_userId_found']:,}")
    print(f"  - Successfully filled:       {stats['empty_userId_filled']:,}")
    print(f"  - Could not fill (removed):  {stats['empty_userId_unfillable']:,}")
    print("\nCancellation Handling:")
    print(f"  Cancelled users:             {stats['cancelled_users']:,}")
    print(f"  Records after cancellation:  {stats['records_after_cancellation']:,}")
    print("\nRecords Removed:")
    print(f"  Guest records:               {stats['guest_removed']:,}")
    print(f"  Unfillable userId:           {stats['unfillable_removed']:,}")
    print(f"  After cancellation:          {stats['records_after_cancellation']:,}")
    print(
        f"  Total removed:               {stats['guest_removed'] + stats['unfillable_removed'] + stats['records_after_cancellation']:,}"
    )
    print(f"\nRecords written to output:     {stats['records_written']:,}")
    print(
        f"Data retention rate:           {(stats['records_written'] / stats['total_lines'] * 100):.2f}%"
    )
    print("=" * 80)

    return stats


if __name__ == "__main__":
    input_file = "data_clean_process/customer_churn.json"
    output_file = "data_clean_process/customer_churn_cleaned.json"

    # Clean the data
    stats = clean_customer_data(input_file, output_file)


"""


Reading from: customer_churn.json
Writing to: customer_churn_cleaned.json

[PASS 1/2] Building session-user mapping based on itemInSession...
  Pass 1: Processed 100,000 lines...
  Pass 1: Processed 200,000 lines...
  Pass 1: Processed 300,000 lines...
  Pass 1: Processed 400,000 lines...
  Pass 1: Processed 500,000 lines...
  Found 4,470 sessions with 6,080 segments
  Detected 1,610 itemInSession resets

[PASS 2/3] Finding cancellation timestamps...
  Pass 2: Processed 100,000 lines...
  Pass 2: Processed 200,000 lines...
  Pass 2: Processed 300,000 lines...
  Pass 2: Processed 400,000 lines...
  Pass 2: Processed 500,000 lines...
  Found 99 users with cancellation

[PASS 3/3] Cleaning and writing data...
  Pass 3: Processed 100,000 lines...
  Pass 3: Processed 200,000 lines...
  Pass 3: Processed 300,000 lines...
  Pass 3: Processed 400,000 lines...
  Pass 3: Processed 500,000 lines...

================================================================================
CLEANING SUMMARY
================================================================================
Total lines read:              543,705

Session Segmentation:
  Segments created:            6,080
  itemInSession resets:        1,610

UserId Filling:
  Empty userId found:          15,606
  - Successfully filled:       14,140
  - Could not fill (removed):  1,466

Cancellation Handling:
  Cancelled users:             99
  Records after cancellation:  0

Records Removed:
  Guest records:               94
  Unfillable userId:           1,466
  After cancellation:          0
  Total removed:               1,560

Records written to output:     542,145
Data retention rate:           99.71%
================================================================================


"""
