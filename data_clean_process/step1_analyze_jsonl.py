#!/usr/bin/env python3
"""
Script to analyze JSONL file and extract unique values for each field.
Reads customer_churn.json line by line and prints unique values for all fields except 'ts'.
"""

import json
from collections import defaultdict


def analyze_jsonl(file_path):
    """
    Read JSONL file and extract unique values for each field.

    Args:
        file_path: Path to the JSONL file
    """
    # Dictionary to store unique values for each field
    unique_values = defaultdict(set)

    # Track all field names
    all_fields = set()

    line_count = 0

    print(f"Reading file: {file_path}")
    print("Processing lines...\n")

    # Read file line by line
    with open(file_path, "r") as f:
        for line in f:
            line_count += 1

            # Show progress every 50,000 lines
            if line_count % 50000 == 0:
                print(f"Processed {line_count:,} lines...")

            # Parse JSON line
            try:
                data = json.loads(line.strip())

                # Extract fields and values
                for field, value in data.items():
                    all_fields.add(field)

                    # Skip 'ts' field as requested
                    if field == "ts":
                        continue

                    # Convert value to string for consistent storage
                    # Handle None/null values
                    if value is None:
                        unique_values[field].add("NULL")
                    else:
                        unique_values[field].add(str(value))

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_count}: {e}")
                continue

    print(f"\nTotal lines processed: {line_count:,}\n")
    print("=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nAll fields found: {sorted(all_fields)}")
    print(f"\nTotal number of fields: {len(all_fields)}\n")
    print("=" * 80)

    # Print unique values for each field (sorted alphabetically by field name)
    for field in sorted(unique_values.keys()):
        values = unique_values[field]
        print(f"\nField: '{field}'")
        print(f"Number of unique values: {len(values)} ----> Contains Null {'NULL' in values}")
        print("-" * 40)

        # Sort values for better readability
        sorted_values = sorted(values)

        # If there are too many unique values, show first 25 and summary
        if len(sorted_values) > 25:
            print("First 25 unique values:")
            for value in sorted_values[:25]:
                print(f"  - {value}")
            print(f"  ... and {len(sorted_values) - 25} more unique values")
        else:
            print("All unique values:")
            for value in sorted_values:
                print(f"  - {value}")

        print("-" * 40)


if __name__ == "__main__":
    jsonl_file = "data_clean_process/customer_churn.json"
    analyze_jsonl(jsonl_file)


"""

Reading file: customer_churn.json
Processing lines...

Processed 50,000 lines...
Processed 100,000 lines...
Processed 150,000 lines...
Processed 200,000 lines...
Processed 250,000 lines...
Processed 300,000 lines...
Processed 350,000 lines...
Processed 400,000 lines...
Processed 450,000 lines...
Processed 500,000 lines...

Total lines processed: 543,705

================================================================================
ANALYSIS RESULTS
================================================================================

All fields found: ['artist', 'auth', 'firstName', 'gender', 'itemInSession', 'lastName', 'length', 'level', 'location', 'method', 'page', 'registration', 'sessionId', 'song', 'status', 'ts', 'userAgent', 'userId']

Total number of fields: 18

================================================================================

Field: 'artist'
Number of unique values: 21247 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - !!!
  - & And Oceans
  - '68 Comeback
  - 'N Sync/Phil Collins
  - 'Til Tuesday
  - 't Hof Van Commerce & Brahim
  - (Love) Tattoo
  - (hed) p.e.
  - *NSYNC featuring Nelly
  - + / - {Plus/Minus}
  - +44
  - -123 minut
  - -123min.
  - -M-
  - ...And Oceans
  - ...And You Will Know Us By The Trail Of Dead
  - 1 40 4 20
  - 1 Giant Leap feat. Grant Lee Phillips & Horace Andy
  - 1 Giant Leap feat. Michael Stipe & Asha Bhosle
  - 1 Giant Leap feat. Robbie Wlliams & Maxi Jazz
  - 1 Giant Leap feat. Speech & Neneh Cherry
  - 1 Mile North
  - 10 Years
  - 10000 Maniacs
  - 1000names
  ... and 21222 more unique values
----------------------------------------

Field: 'auth'
Number of unique values: 4 ----> Contains Null False
----------------------------------------
All unique values:
  - Cancelled
  - Guest
  - Logged In
  - Logged Out
----------------------------------------

Field: 'firstName'
Number of unique values: 345 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - Aaliyah
  - Abel
  - Abraham
  - Adal
  - Adam
  - Addison
  - Adelaida
  - Adelaide
  - Adrian
  - Adriana
  - Adriel
  - Adrienne
  - Aiden
  - Ainsley
  - Alejandra
  - Alex
  - Alexander
  - Alexandria
  - Alexi
  - Allan
  - Allison
  - Allisson
  - Alyssa
  - Alyssia
  - Amberlynn
  ... and 320 more unique values
----------------------------------------

Field: 'gender'
Number of unique values: 2 ----> Contains Null False
----------------------------------------
All unique values:
  - F
  - M
----------------------------------------

Field: 'itemInSession'
Number of unique values: 1006 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - 0
  - 1
  - 10
  - 100
  - 1000
  - 1001
  - 1002
  - 1003
  - 1004
  - 1005
  - 101
  - 102
  - 103
  - 104
  - 105
  - 106
  - 107
  - 108
  - 109
  - 11
  - 110
  - 111
  - 112
  - 113
  - 114
  ... and 981 more unique values
----------------------------------------

Field: 'lastName'
Number of unique values: 275 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - Abbott
  - Adams
  - Aguilar
  - Alexander
  - Allen
  - Anderson
  - Atkinson
  - Bailey
  - Baker
  - Ball
  - Barber
  - Barnes
  - Barnett
  - Beck
  - Benitez
  - Bennett
  - Berry
  - Bird
  - Black
  - Blackburn
  - Blackwell
  - Boone
  - Boyd
  - Bradley
  - Bridges
  ... and 250 more unique values
----------------------------------------

Field: 'length'
Number of unique values: 16679 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - 0.78322
  - 1.12281
  - 10.03057
  - 10.52689
  - 100.07465
  - 100.10077
  - 100.12689
  - 100.15302
  - 100.17914
  - 100.23138
  - 100.25751
  - 100.28363
  - 100.33587
  - 100.41424
  - 100.46649
  - 100.49261
  - 100.51873
  - 100.54485
  - 100.57098
  - 100.5971
  - 100.67546
  - 100.70159
  - 100.72771
  - 100.80608
  - 100.8322
  ... and 16654 more unique values
----------------------------------------

Field: 'level'
Number of unique values: 2 ----> Contains Null False
----------------------------------------
All unique values:
  - free
  - paid
----------------------------------------

Field: 'location'
Number of unique values: 192 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - Akron, OH
  - Albany, OR
  - Albany-Schenectady-Troy, NY
  - Albemarle, NC
  - Alexandria, LA
  - Alexandria, MN
  - Allentown-Bethlehem-Easton, PA-NJ
  - Anchorage, AK
  - Appleton, WI
  - Athens, TX
  - Atlanta-Sandy Springs-Roswell, GA
  - Atlantic City-Hammonton, NJ
  - Auburn, IN
  - Augusta-Waterville, ME
  - Austin-Round Rock, TX
  - Bakersfield, CA
  - Baltimore-Columbia-Towson, MD
  - Beaumont-Port Arthur, TX
  - Big Spring, TX
  - Billings, MT
  - Birmingham-Hoover, AL
  - Boise City, ID
  - Boston-Cambridge-Newton, MA-NH
  - Boulder, CO
  - Bowling Green, KY
  ... and 167 more unique values
----------------------------------------

Field: 'method'
Number of unique values: 2 ----> Contains Null False
----------------------------------------
All unique values:
  - GET
  - PUT
----------------------------------------

Field: 'page'
Number of unique values: 22 ----> Contains Null False
----------------------------------------
All unique values:
  - About
  - Add Friend
  - Add to Playlist
  - Cancel
  - Cancellation Confirmation
  - Downgrade
  - Error
  - Help
  - Home
  - Login
  - Logout
  - NextSong
  - Register
  - Roll Advert
  - Save Settings
  - Settings
  - Submit Downgrade
  - Submit Registration
  - Submit Upgrade
  - Thumbs Down
  - Thumbs Up
  - Upgrade
----------------------------------------

Field: 'registration'
Number of unique values: 448 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - 1509854193000
  - 1519397713000
  - 1521380675000
  - 1522076012000
  - 1522793334000
  - 1523464964000
  - 1523721066000
  - 1523777521000
  - 1526739206000
  - 1526838391000
  - 1527013865000
  - 1527341164000
  - 1527476190000
  - 1528403713000
  - 1528560242000
  - 1528772084000
  - 1528780738000
  - 1528833373000
  - 1528996419000
  - 1529024969000
  - 1529026199000
  - 1529027541000
  - 1529252604000
  - 1529643103000
  - 1529934689000
  ... and 423 more unique values
----------------------------------------

Field: 'sessionId'
Number of unique values: 4590 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - 1
  - 10
  - 100
  - 1000
  - 1001
  - 1002
  - 1003
  - 1004
  - 1005
  - 1006
  - 1007
  - 1008
  - 1009
  - 101
  - 1010
  - 1011
  - 1012
  - 1013
  - 1014
  - 1015
  - 1016
  - 1017
  - 1018
  - 1019
  - 102
  ... and 4565 more unique values
----------------------------------------

Field: 'song'
Number of unique values: 80292 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - ÃÂ	g ÃÂtti GrÃÂ¡a ÃÂsku
  -  - kumo -
  -  Energy To Burn
  -  I Will Not Reap Destruction
  - !@*$%#
  - # 1
  - #!*@ Me (Amended Version)
  - #!*@ You Tonight [Featuring R. Kelly] (Explicit Album Version)
  - #1
  - #1 Crew In The Area (Explicit)
  - #1 Fan [Feat. Keyshia Cole & J. Holiday] (Album Version)
  - #1 Stunna
  - #13 (Album Version)
  - #16
  - #2 For Prepared Wire-Strung Harp With Tremolo Pedal
  - #24
  - #3 For Bray Harp
  - #40
  - $
  - $1000 Wedding
  - $29.00
  - $in$
  - $timulus Plan
  - & Down
  - & Down (Teenage Bad Girl Remix)
  ... and 80267 more unique values
----------------------------------------

Field: 'status'
Number of unique values: 3 ----> Contains Null False
----------------------------------------
All unique values:
  - 200
  - 307
  - 404
----------------------------------------

Field: 'userAgent'
Number of unique values: 71 ----> Contains Null False
----------------------------------------
First 25 unique values:
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.3 (KHTML, like Gecko) Version/8.0 Safari/600.1.3"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.8 (KHTML, like Gecko) Version/8.0 Safari/600.1.8"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/534.59.10 (KHTML, like Gecko) Version/5.1.9 Safari/534.59.10"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/6.1.5 Safari/537.77.4"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/6.1.5 Safari/537.77.4"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.74.9 (KHTML, like Gecko) Version/7.0.2 Safari/537.74.9"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.76.4 (KHTML, like Gecko) Version/7.0.4 Safari/537.76.4"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"
  - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"
  - "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"
  - "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"
  ... and 46 more unique values
----------------------------------------

Field: 'userId'
Number of unique values: 449 ----> Contains Null False
----------------------------------------
First 25 unique values:
  -
  - 10
  - 100
  - 100001
  - 100002
  - 100003
  - 100004
  - 100005
  - 100006
  - 100007
  - 100008
  - 100009
  - 100010
  - 100011
  - 100012
  - 100013
  - 100014
  - 100015
  - 100016
  - 100017
  - 100018
  - 100019
  - 100020
  - 100021
  - 100022
  ... and 424 more unique values
----------------------------------------


"""
