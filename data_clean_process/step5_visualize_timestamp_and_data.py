#!/usr/bin/env python3
"""
Step 1: Visualize Timestamp and Data Distribution
- Plot the number of events per day
- Show cancellations per day as bars
"""

import json
from collections import defaultdict
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def load_and_aggregate_events(input_file):
    """
    Load cleaned data and aggregate events by day.

    Returns:
        events_per_day: dict mapping date -> count of events
        cancellations_per_day: dict mapping date -> count of cancellations
    """
    events_per_day = defaultdict(int)
    cancellations_per_day = defaultdict(int)

    print(f"Reading from: {input_file}")
    print("Processing events...")

    line_count = 0
    with open(input_file, "r") as infile:
        for line_num, line in enumerate(infile, 1):
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} lines...")

            try:
                record = json.loads(line.strip())

                # Extract timestamp
                ts = record.get("ts", 0)
                if ts == 0:
                    continue

                # Convert timestamp (milliseconds) to date
                dt = datetime.fromtimestamp(ts / 1000.0)
                date = dt.date()

                # Count total events
                events_per_day[date] += 1
                line_count += 1

                # Count cancellations
                auth = record.get("auth", "")
                if auth == "Cancelled":
                    cancellations_per_day[date] += 1

            except (json.JSONDecodeError, ValueError):
                continue

    print(f"\nTotal records processed: {line_count:,}")
    print(f"Total days with events: {len(events_per_day)}")
    print(f"Days with cancellations: {len(cancellations_per_day)}")

    return events_per_day, cancellations_per_day


def plot_events_and_cancellations(
    events_per_day,
    cancellations_per_day,
    output_file="data_clean_process/event_timeline.png",
):
    """
    Create a visualization showing events per day and cancellations per day.

    Args:
        events_per_day: dict mapping date -> event count
        cancellations_per_day: dict mapping date -> cancellation count
        output_file: path to save the plot
    """
    # Sort dates
    dates = sorted(events_per_day.keys())

    if not dates:
        print("No data to plot!")
        return

    # Prepare data for plotting
    event_counts = [events_per_day[d] for d in dates]
    cancellation_counts = [cancellations_per_day.get(d, 0) for d in dates]

    # Convert dates to datetime for matplotlib
    dates_dt = [datetime.combine(d, datetime.min.time()) for d in dates]

    # Create figure with two subplots (stacked)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Event Timeline Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Total events per day (line plot)
    ax1.plot(dates_dt, event_counts, color="steelblue", linewidth=2, label="Events per Day")
    ax1.fill_between(dates_dt, event_counts, alpha=0.3, color="steelblue")
    ax1.set_ylabel("Number of Events", fontsize=12, fontweight="bold")
    ax1.set_title("Total Events per Day", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)

    # Add statistics to plot 1
    avg_events = np.mean(event_counts)
    max_events = max(event_counts)
    ax1.axhline(
        y=avg_events,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Average: {avg_events:.0f}",
    )
    ax1.text(
        0.02,
        0.98,
        f"Total Events: {sum(event_counts):,}\n"
        f"Avg per Day: {avg_events:.0f}\n"
        f"Max per Day: {max_events:,}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    # Plot 2: Cancellations per day (bar plot)
    ax2.bar(
        dates_dt,
        cancellation_counts,
        color="crimson",
        alpha=0.7,
        label="Cancellations per Day",
    )
    ax2.set_ylabel("Number of Cancellations", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax2.set_title("Cancellations per Day", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper right", fontsize=10)

    # Add statistics to plot 2
    total_cancellations = sum(cancellation_counts)
    days_with_cancellations = sum(1 for c in cancellation_counts if c > 0)
    if days_with_cancellations > 0:
        avg_cancellations = total_cancellations / days_with_cancellations
    else:
        avg_cancellations = 0

    ax2.text(
        0.02,
        0.98,
        f"Total Cancellations: {total_cancellations:,}\n"
        f"Days with Cancellations: {days_with_cancellations}\n"
        f"Avg per Active Day: {avg_cancellations:.1f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        fontsize=10,
    )

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    # Also create a combined overlay plot
    create_overlay_plot(
        dates_dt,
        event_counts,
        cancellation_counts,
        "data_clean_process/event_timeline_overlay.png",
    )


def create_overlay_plot(dates, event_counts, cancellation_counts, output_file):
    """
    Create a single plot with events as line and cancellations as bars overlaid.
    """
    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.suptitle("Events and Cancellations Timeline (Overlay)", fontsize=16, fontweight="bold")

    # Plot events on primary y-axis (line)
    color = "steelblue"
    ax1.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Events", color=color, fontsize=12, fontweight="bold")
    ax1.plot(dates, event_counts, color=color, linewidth=2, label="Events per Day")
    ax1.fill_between(dates, event_counts, alpha=0.2, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot cancellations on secondary y-axis (bars)
    ax2 = ax1.twinx()
    color = "crimson"
    ax2.set_ylabel("Number of Cancellations", color=color, fontsize=12, fontweight="bold")
    ax2.bar(
        dates,
        cancellation_counts,
        color=color,
        alpha=0.6,
        width=0.8,
        label="Cancellations per Day",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Overlay plot saved to: {output_file}")


def print_summary_statistics(events_per_day, cancellations_per_day):
    """
    Print detailed statistics about the data.
    """
    print("\n" + "=" * 80)
    print("DATA SUMMARY STATISTICS")
    print("=" * 80)

    dates = sorted(events_per_day.keys())
    event_counts = [events_per_day[d] for d in dates]
    cancellation_counts = [cancellations_per_day.get(d, 0) for d in dates]

    print("\nDate Range:")
    print(f"  First date: {dates[0]}")
    print(f"  Last date:  {dates[-1]}")
    print(f"  Span:       {(dates[-1] - dates[0]).days + 1} days")

    print("\nEvents:")
    print(f"  Total events:      {sum(event_counts):,}")
    print(f"  Average per day:   {np.mean(event_counts):.1f}")
    print(f"  Median per day:    {np.median(event_counts):.1f}")
    print(f"  Min per day:       {min(event_counts):,}")
    print(f"  Max per day:       {max(event_counts):,}")
    print(f"  Std dev:           {np.std(event_counts):.1f}")

    print("\nCancellations:")
    print(f"  Total cancellations:       {sum(cancellation_counts):,}")
    print(f"  Days with cancellations:   {sum(1 for c in cancellation_counts if c > 0)}")
    print(f"  Days without cancellations: {sum(1 for c in cancellation_counts if c == 0)}")

    if sum(1 for c in cancellation_counts if c > 0) > 0:
        active_cancellations = [c for c in cancellation_counts if c > 0]
        print(f"  Average per active day:    {np.mean(active_cancellations):.2f}")
        print(f"  Max cancellations per day: {max(cancellation_counts)}")

    print("=" * 80)


if __name__ == "__main__":
    input_file = "data_clean_process/customer_churn_cleaned.json"

    # Load and aggregate data
    events_per_day, cancellations_per_day = load_and_aggregate_events(input_file)

    # Print statistics
    print_summary_statistics(events_per_day, cancellations_per_day)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_events_and_cancellations(events_per_day, cancellations_per_day)

    print("\n[+] Complete! Visualizations created successfully.")

"""

Reading from: ../data_clean_process/customer_churn_cleaned.json
Processing events...
  Processed 100,000 lines...
  Processed 200,000 lines...
  Processed 300,000 lines...
  Processed 400,000 lines...
  Processed 500,000 lines...

Total records processed: 542,145
Total days with events: 62
Days with cancellations: 53

================================================================================
DATA SUMMARY STATISTICS
================================================================================

Date Range:
  First date: 2018-10-01
  Last date:  2018-12-01
  Span:       62 days

Events:
  Total events:      542,145
  Average per day:   8744.3
  Median per day:    9082.5
  Min per day:       1,100
  Max per day:       13,138
  Std dev:           2359.9

Cancellations:
  Total cancellations:       99
  Days with cancellations:   53
  Days without cancellations: 9
  Average per active day:    1.87
  Max cancellations per day: 6
================================================================================

Creating visualizations...

Plot saved to: event_timeline.png
Overlay plot saved to: event_timeline_overlay.png

[+] Complete! Visualizations created successfully.

"""
