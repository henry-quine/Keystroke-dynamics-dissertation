"""
1. Read the clean export CSV.
2. Keep only participants who have the required sessions.
3. Turn keyboard events into typing features.
4. Rank the features with Fisher Score.
5. Train one One-Class SVM model per participant.
6. Test the final authentication model.

The experiment session setup is controlled by run_experiments.py.
"""
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import pandas as pd
from sklearn.svm import OneClassSVM
from scipy.stats import skew, median_abs_deviation


# =============================================================================
# BASIC SETTINGS
# =============================================================================

# Define the path to the clean input CSV file.
THIS_FOLDER = Path(__file__).resolve().parent 
CSV_FILE = THIS_FOLDER / "input" / "clean_model_events_all (19).csv"


# Try these model settings during tuning and choose the best validation result.
MODEL_SETTINGS_TO_TRY = {
    "feature_count": [
        10, 15, 20, 25, 30, 35, 40, 45,
        50, 55, 60, 67,
    ],
    "nu": [0.03, 0.05, 0.08, 0.1],
    "gamma": [0.00005, 0.0001, 0.0005, 0.001],
}

## Participants must have these sessions to be included.
REQUIRED_SESSIONS_FOR_PARTICIPANT = [1, 2, 3, 4, 5]

## Free-text windows
FREE_TEXT_WINDOW_SIZE = 50

## Data types that can be used in experiments.
## The left side is the simple name we use in run_experiments.py.
## The right side is how that data appears in the clean CSV.
DATA_TYPES = {
    "fixed normal": ("fixed", "normal"),
    "fixed rushed": ("fixed", "rushed"),
    "fixed distracted": ("fixed", "distracted"),
    "semi-fixed normal": ("semi_fixed", "normal"),
    "free normal": ("free", "normal"),
}


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================


## Basic Key Timing Features
CLASSIC_FEATURES = ["hold", "flight", "dd", "uu"]
CLASSIC_TIMING_STATS = ["mean", "median", "p10", "p25", "p75", "p90", "std", "iqr", "mad", "cv", "skew"]

CLASSIC_TIMING_FEATURES = []
for feature_name in CLASSIC_FEATURES:
    for stat_name in CLASSIC_TIMING_STATS:
        CLASSIC_TIMING_FEATURES.append(feature_name + "_" + stat_name)



## Individual Key Hold Features to Track
KEYS_TO_TRACK = ["space", "t", "h", "e", "a", "n", "d", "r", "i", "o", "s"]

KEY_HOLD_FEATURES = []
for key_name in KEYS_TO_TRACK:
    KEY_HOLD_FEATURES.append("keyhold_" + key_name + "_mean")



## Digraph Features - Entering, Typing, and Leaving the Word "the"
DIGRAPH_LIST = {
    "spacet": (" ", "t"),
    "th": ("t", "h"),
    "he": ("h", "e"),
    "espace": ("e", " "),
}

DIGRAPH_FEATURES = []
for name in DIGRAPH_LIST:
    DIGRAPH_FEATURES.append("digraph_" + name + "_dd_mean")



## Trigraph Features - Entering, Typing, and Leaving the Word "the"
TRIGRAPH_LIST = {
    "spaceth": (" ", "t", "h"),
    "the": ("t", "h", "e"),
    "hespace": ("h", "e", " "),
}

TRIGRAPH_FEATURES = []
for name in TRIGRAPH_LIST:
    TRIGRAPH_FEATURES.append("trigraph_" + name + "_span_mean")
    


## General Typing Behaviour Features
BEHAVIOUR_FEATURES = [
    "typing_speed",
    "negative_flight_ratio",
    "pause_ratio_300",
    "pause_ratio_500",
    "backspace_rate",
]
  
## Put All the Features Together
ALL_FEATURES = CLASSIC_TIMING_FEATURES + KEY_HOLD_FEATURES + DIGRAPH_FEATURES + TRIGRAPH_FEATURES + BEHAVIOUR_FEATURES


    
# Label each feature with its family for the feature ranking CSV
FEATURE_FAMILY = {}
for feature in CLASSIC_TIMING_FEATURES:
    FEATURE_FAMILY[feature] = "timing"
for feature in BEHAVIOUR_FEATURES:
    FEATURE_FAMILY[feature] = "behaviour"
for feature in KEY_HOLD_FEATURES:
    FEATURE_FAMILY[feature] = "key_hold"
for feature in DIGRAPH_FEATURES:
    FEATURE_FAMILY[feature] = "digraph"
for feature in TRIGRAPH_FEATURES:
    FEATURE_FAMILY[feature] = "trigraph"


# Each Sample's Metadata Columns For the Feature Table
METADATA_COLUMNS = [
    "participant_code",
    "session_number",
    "task_order",
    "task_type",
    "condition",
    "prompt_id",
    "task_id",
    "sample_id",
    "repetition_index",
    "sample_status",
    "event_rows",
    "paired_presses",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Make an output folder if it doesn't exist yet.
def make_output_folder(folder):
    folder.mkdir(parents=True, exist_ok=True)

# Return NaN to represent a missing value or a feature that could not be calculated.
def blank_number():
    return float("nan")

# Convert a decimal value to a percentage string with two decimal places.
def percent(value):
    if pd.isna(value):
        return "missing"
    return str(round(value * 100, 2)) + "%"

# Convert a session list like [1, 2, 3] into text like "1, 2, 3".
def session_list_text(sessions):
    return ", ".join(str(session) for session in sessions)

# Convert a simple data type name like "fixed normal" into task_type and condition.
def get_task_type_and_condition(data_type):
    if data_type not in DATA_TYPES:
        raise ValueError("Unknown data type: " + str(data_type))

    task_type, condition = DATA_TYPES[data_type]
    return task_type, condition

# Pick rows for one data type and a list of sessions.
def choose_rows(data, data_type, sessions):
    task_type, condition = get_task_type_and_condition(data_type)

    rows = data[
        (data["task_type"] == task_type)
        & (data["condition"] == condition)
        & (data["session_number"].isin(sessions))
    ].copy()

    return rows

# Choose which feature table to use for an experiment data type.
# Free normal always uses the 50-key free-text window table.
def get_feature_table_for_data_type(prepared_data, data_type):
    if data_type == "free normal":
        feature_table = prepared_data["free_windows"]
    else:
        feature_table = prepared_data["model_features"]

    return feature_table

# Clean a list of values before using it in statistical calculations.
# Values that are not valid numbers, are missing, or are infinite are removed.
def make_numbers(values):
    numbers = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    numbers = numbers.replace([float("inf"), float("-inf")], blank_number())
    numbers = numbers.dropna()
    return numbers

# Calculate the average of a list of numbers.
# If there are no valid numbers, return a missing value instead.
def my_mean(numbers):
    numbers = make_numbers(numbers)
    if len(numbers) == 0:
        return blank_number()
    return float(numbers.mean())

# Return the middle value of of an ordered list of numbers.
# If there are no valid numbers, return a missing value instead.
def my_median(numbers):
    numbers = make_numbers(numbers)
    if len(numbers) == 0:
        return blank_number()
    return float(numbers.median())

# Return the value at a given percentile of an ordered list of numbers.
# If there are no valid numbers, return a missing value instead.
def my_percentile(numbers, percent):
    numbers = make_numbers(numbers)
    if len(numbers) == 0:
        return blank_number()
    return float(numbers.quantile(percent / 100))

# Return the standard deviation of a list of numbers.
# If there are fewer than two valid numbers, return a missing value instead.
def my_std(numbers):
    numbers = make_numbers(numbers)
    if len(numbers) < 2:
        return blank_number()
    return float(numbers.std(ddof=1))

# Return the median absolute deviation (MAD) of a list of numbers.
# If there are no valid numbers, return a missing value instead.
def my_mad(numbers):
    numbers = make_numbers(numbers)
    if len(numbers) == 0:
        return blank_number()
    return float(median_abs_deviation(numbers, scale=1.0))

# Return the skewness of a list of numbers.
# If there are fewer than three valid numbers, return a missing value instead.
def my_skew(numbers):
    numbers = make_numbers(numbers)
    if len(numbers) < 3:
        return blank_number()
    return float(skew(numbers, bias=False))

# Return the interquartile range (IQR) of a list of numbers.
# If there are no valid numbers, return a missing value instead.
def my_iqr(numbers):
    numbers = make_numbers(numbers)
    if len(numbers) == 0:
        return blank_number()
    p25 = my_percentile(numbers, 25)
    p75 = my_percentile(numbers, 75)
    return p75 - p25 

# Return the coefficient of variation (CV) of a list of numbers.
# If there are fewer than two valid numbers, or if the mean is zero, return a missing value instead.
def my_cv(numbers):
    numbers = make_numbers(numbers)
    if len(numbers) < 2:
        return blank_number()
    average = my_mean(numbers)
    standard_deviation = my_std(numbers)
    if average == 0:
        return blank_number()
    return standard_deviation / average

# Calculate all the statistical features used in the classic timing features (hold, flight, dd, uu).
def timing_summary(values):
    return {
        "mean": my_mean(values),
        "median": my_median(values),
        "p10": my_percentile(values, 10),
        "p25": my_percentile(values, 25),
        "p75": my_percentile(values, 75),
        "p90": my_percentile(values, 90),
        "std": my_std(values),
        "iqr": my_iqr(values),
        "mad": my_mad(values),
        "cv": my_cv(values),
        "skew": my_skew(values),
    }


# =============================================================================
# STEP 1: READ THE CSV
# =============================================================================

# Read the clean export CSV and make sure the columns are the right type.
def read_the_csv():
    print("Reading export:")
    print(CSV_FILE)

    # If the CSV file is not found stop the script
    if not CSV_FILE.exists():
        raise FileNotFoundError("Could not find the clean export CSV.")

    data = pd.read_csv(CSV_FILE, low_memory=False)

    # These columns are numbers types 
    number_columns = [
        "session_number",
        "task_order",
        "repetition_index",
        "character_index_in_repetition",
        "rushed_limit_ms",
        "event_sequence",
        "timestamp_ms",
        "relative_time_ms",
    ]
    for column in number_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # These columns are text types
    text_columns = [
        "participant_code",
        "task_type",
        "condition",
        "prompt_id",
        "task_id",
        "sample_id",
        "sample_status",
        "press_id",
        "key",
        "code",
        "event_type",
    ]
    for column in text_columns:
        if column in data.columns:
            data[column] = data[column].fillna("").astype(str)

    # Sort the rows so that the events are in the right order for each sample.
    data = data.sort_values(["sample_id", "event_sequence", "relative_time_ms"])
    data = data.reset_index(drop=True)

    # Print some basic information about the data
    print("Rows:", len(data))
    print("Samples:", data["sample_id"].nunique())
    print("Participants:", data["participant_code"].nunique())

    return data


# =============================================================================
# STEP 2: KEEP ONLY PARTICIPANTS WITH THE REQUIRED SESSIONS
# =============================================================================

# Return a list of participants who have all 5 required sessions in the data.
def find_people_with_required_sessions(data):
    complete_participants = []
    
    # Group the data by participant and check whether each required session is present.
    for participant, participant_rows in data.groupby("participant_code"):
        
        sessions = participant_rows["session_number"].unique()

        has_all_required_sessions = True

        for required_session in REQUIRED_SESSIONS_FOR_PARTICIPANT:
            if required_session not in sessions:
                has_all_required_sessions = False

        if has_all_required_sessions:
            complete_participants.append(participant)

    return sorted(complete_participants)


# Save a CSV file that lists all participants and whether they were used in the analysis.
def save_participant_list(all_participants, complete_participants, output_folder):
    excluded_participants = sorted(set(all_participants) - set(complete_participants))

    rows = []

    # Add yes for participants who were used in the analysis
    for participant in complete_participants:
        rows.append({
            "participant_code": participant,
            "used_in_analysis": "yes"
        })

    # Add no for participants who were excluded from the analysis
    for participant in excluded_participants:
        rows.append({
            "participant_code": participant,
            "used_in_analysis": "no"
        })

    # Turn the list into a table and sort it by participant code.
    participant_table = pd.DataFrame(rows)
    participant_table = participant_table.sort_values("participant_code")

    # Save the table so we have a clear record of who was included and excluded.
    participant_table.to_csv(output_folder / "participants_selected.csv", index=False)

    return excluded_participants


# =============================================================================
# STEP 3: PAIR KEYDOWN AND KEYUP EVENTS AND CALCULATE TIMINGS
# =============================================================================

# Joins every keydown event with its matching keyup event.
# Returns a list of key pairs with their down time, up time, and hold time.
def make_key_pairs(sample):
    
    # Store all the keydown events that are waiting for a matching keyup event.
    waiting_keydowns = {}
    
    # Store the final list of key pairs with their down time, up time, and hold time.
    key_pairs = []

    # Loop through each event in the sample and pair keydown and keyup events.
    for event in sample.to_dict("records"):
        # Get the press_id and event_type for the current event.
        press_id = event["press_id"]
        event_type = event["event_type"]

        # If the event is a keydown, store it in the waiting_keydowns dictionary.
        if event_type == "keydown":
            waiting_keydowns[press_id] = event

        # If the event is a keyup, check if there is a matching keydown event.
        elif event_type == "keyup" and press_id in waiting_keydowns:
            # Get the matching keydown event from the waiting_keydowns dictionary.
            keydown_row = waiting_keydowns[press_id]
            # Remove the keydown from the waiting list now that it has been paired with a keyup event.
            del waiting_keydowns[press_id]

            # Calculate the hold time by subtracting the keydown time from the keyup time.
            keydown_time = keydown_row["relative_time_ms"]
            keyup_time = event["relative_time_ms"]
            hold_time = keyup_time - keydown_time

            # Save the key pair with its down time, up time, and hold time in the key_pairs list.
            key_pairs.append({
                "press_id": press_id,
                "key": keydown_row["key"],
                "code": keydown_row["code"],
                "down_ms": keydown_time,
                "up_ms": keyup_time,
                "hold_ms": hold_time,
            })

    # Sort the key pairs by their down time so that they are in the order they were typed.
    key_pairs = sorted(key_pairs, key=lambda pair: pair["down_ms"])
    return key_pairs

# Calculate the hold times, flight times, down-to-down times, and up-to-up times for a list of key pairs.
def get_timing_lists(key_pairs):
    hold_times = []
    flight_times = []
    down_to_down_times = []
    up_to_up_times = []

    # Create a list of hold times for each key pair.
    for key_pair in key_pairs:
        hold_times.append(key_pair["hold_ms"])

    # Calculate the flight times, down-to-down times, and up-to-up times for each pair and return them as lists.
    for i in range(len(key_pairs) - 1):
        current_key = key_pairs[i]
        next_key = key_pairs[i + 1]

        # Flight = current key up to next key down.
        flight_time = next_key["down_ms"] - current_key["up_ms"]
        
        # DD = current key down to next key down.
        down_to_down_time  = next_key["down_ms"] - current_key["down_ms"]
        
        # UU = current key up to next key up.
        up_to_up_time = next_key["up_ms"] - current_key["up_ms"]

        flight_times.append(flight_time)
        down_to_down_times.append(down_to_down_time)
        up_to_up_times.append(up_to_up_time)

    return hold_times, flight_times, down_to_down_times, up_to_up_times


# =============================================================================
# STEP 4: SELECT AND PREPARE KEYS FOR LATER FEATURE CALCULATION
# =============================================================================

# Check whether a key is one of the specific keys in KEYS_TO_TRACK.
# These are the keys we want to calculate individual key-hold features for.
def get_tracked_key_name(key, code):

    # If the key is a space, set the tracked key name to "space".
    if code == "Space":
        tracked_key_name = "space"

    # If the key is a letter, set the tracked key name to the lowercase letter.
    elif code.startswith("Key"):
        tracked_key_name = key.lower()

    # Ignore all other keys.
    else:
        return None

    # If the tracked key name is in KEYS_TO_TRACK, return it. Otherwise, return None.
    if tracked_key_name in KEYS_TO_TRACK:
        return tracked_key_name

    return None

# Build an ordered character stream for later digraph and trigraph matching 
# Return a list of tuples with the character and its down time for each key pair.
def make_character_stream(key_pairs):
    
    # Store the character stream as a list of tuples with the character and its down time.
    character_stream = []

    # Loop through each key pair and determine the character for each key.
    for key_pair in key_pairs:
        key = key_pair["key"]
        code = key_pair["code"]

        # If the key is a space, set the character to a space. 
        if code == "Space":
            character = " "

        # If the key is a letter, set the character to the lowercase letter.
        elif code.startswith("Key"):
            character = key.lower()

        # All other keys are ignored and set to None to act as a stream break for digraph and trigraph matching.
        else:
            character = None

        
        character_stream.append((character, key_pair["down_ms"]))

    return character_stream


# =============================================================================
# STEP 5: MAKE FEATURES FOR ONE SAMPLE
# =============================================================================

# Calculate all the features for one sample and return them as a dictionary.
def make_features_for_one_sample(sample):
    
    #1. Basic setup
    # Get the metadata for the sample
    # This includes participant code, session number, task type, etc for this sample.
    sample_metadata = sample.iloc[0]

    # Run the key pairing function to get a list of key pairs with their down time, up time, and hold time.
    key_pairs = make_key_pairs(sample)
    
    # If there are no completed key presses, we cannot make features
    if len(key_pairs) == 0:
        return None

    #2. Classic timing lists
    # Run the timing calculation function to get lists of hold times, flight times, down-to-down times, and up-to-up times.
    hold_times, flight_times, dd_times, uu_times = get_timing_lists(key_pairs)

    
    #3. Behaviour counts and rates
    # Get all the keydown events in the sample to calculate backspace rates
    keydown_rows = sample[sample["event_type"] == "keydown"]
    total_keydowns = len(keydown_rows)
    number_of_backspaces = int((keydown_rows["key"] == "Backspace").sum())
    backspace_rate = number_of_backspaces / total_keydowns

    # Calculate the duration of the sample in seconds by subtracting the start time from the end time.
    start_time = sample["relative_time_ms"].min()
    end_time = sample["relative_time_ms"].max()
    duration_seconds = (end_time - start_time) / 1000
    typing_speed = total_keydowns / duration_seconds 
    
    # Calculate the ratios of negative flight times and long pauses in the flight times.
    negative_flight_ratio = sum(v < 0 for v in flight_times) / len(flight_times)
    pause_ratio_300 = sum(v >= 300 for v in flight_times) / len(flight_times)
    pause_ratio_500 = sum(v >= 500 for v in flight_times) / len(flight_times)
    
    # 4. Classic timing statistics
    # Calculate the classic timing statistics for hold, flight, down-to-down, and up-to-up times.
    hold_features = timing_summary(hold_times)
    flight_features = timing_summary(flight_times)
    dd_features = timing_summary(dd_times)
    uu_features = timing_summary(uu_times)
    
    # 5. Individual key-hold features
    # Creates an empty list of hold times for each tracked key
    hold_time_for_each_target_key = {}
    for tracked_key in KEYS_TO_TRACK:
        hold_time_for_each_target_key[tracked_key] = []

    # Collects the hold key times for each tracked key
    for key_pair in key_pairs:
        key_name = get_tracked_key_name(key_pair["key"], key_pair["code"])

        if key_name is not None:
            hold_time_for_each_target_key[key_name].append(key_pair["hold_ms"])

    # Calculates the mean hold time for each tracked key
    keyhold_means = {}
    for tracked_key in KEYS_TO_TRACK:
        keyhold_means[tracked_key] = my_mean(hold_time_for_each_target_key[tracked_key])


    # 6. Digraph and Trigraph Features
    # Run the character stream function to get an ordered list of characters and their down times for later digraph and trigraph matching.
    character_stream = make_character_stream(key_pairs)
    
    # Digraph: Calculate the mean down-to-down time for each digraph in DIGRAPH_LIST.
    digraph_means = {}
    for digraph_name, target_digraph in DIGRAPH_LIST.items():
        digraph_dd_times = []
        
        # Loop through the character stream and find neighboring pairs of characters that match the target digraph.
        for i in range(len(character_stream) - 1):
            first_character = character_stream[i][0]
            second_character = character_stream[i + 1][0]
            
            if first_character is None or second_character is None:
                continue

            first_down_time = character_stream[i][1]
            second_down_time = character_stream[i + 1][1]

            actual_pair = (first_character, second_character)

            # If the actual pair of characters matches the target digraph, calculate the down-to-down time and add it to the timings list.
            if actual_pair == target_digraph:
                digraph_dd_time = second_down_time - first_down_time
                digraph_dd_times.append(digraph_dd_time)

        # Calculate the mean down-to-down time for the digraph and add it to the row dictionary.
        digraph_means[digraph_name] = my_mean(digraph_dd_times)
    

    # Trigraph: Calculate the mean span time for each trigraph in TRIGRAPH_LIST.
    trigraph_means = {}
    for trigraph_name, target_trigraph in TRIGRAPH_LIST.items():
        trigraph_span_times  = []

        # Loop through the character stream and find neighboring triplets of characters that match the target trigraph.
        for i in range(len(character_stream) - 2):
            first_character = character_stream[i][0]
            second_character = character_stream[i + 1][0]
            third_character = character_stream[i + 2][0]
            
            if first_character is None or second_character is None or third_character is None:
                continue
    
            first_down_time = character_stream[i][1]
            third_down_time = character_stream[i + 2][1]

            actual_three = (first_character, second_character, third_character)

            # If the actual triplet of characters matches the target trigraph, calculate the span time and add it to the timings list.
            if actual_three == target_trigraph:
                trigraph_span_time = third_down_time - first_down_time
                trigraph_span_times.append(trigraph_span_time)

        # Calculate the mean span time for the trigraph and add it to the row dictionary.
        trigraph_means[trigraph_name] = my_mean(trigraph_span_times)
    
    # 7. Build the final complete feature row
    row = {
        # Metadata
        "participant_code": sample_metadata["participant_code"],
        "session_number": int(sample_metadata["session_number"]),
        "task_order": int(sample_metadata["task_order"]),
        "task_type": sample_metadata["task_type"],
        "condition": sample_metadata["condition"],
        "prompt_id": sample_metadata["prompt_id"],
        "task_id": sample_metadata["task_id"],
        "sample_id": sample_metadata["sample_id"],
        "repetition_index": sample_metadata["repetition_index"],
        "sample_status": sample_metadata["sample_status"],
        "event_rows": len(sample),
        "paired_presses": len(key_pairs),

        # Behaviour features
        "typing_speed": typing_speed,
        "negative_flight_ratio": negative_flight_ratio,
        "pause_ratio_300": pause_ratio_300,
        "pause_ratio_500": pause_ratio_500,
        "backspace_rate": backspace_rate,
    }
    
    # Add classic timing features using the list 
    for statistic in CLASSIC_TIMING_STATS:
        row["hold_" + statistic] = hold_features[statistic]
        row["flight_" + statistic] = flight_features[statistic]
        row["dd_" + statistic] = dd_features[statistic]
        row["uu_" + statistic] = uu_features[statistic]

    # Add key-hold features
    for tracked_key in KEYS_TO_TRACK:
        row["keyhold_" + tracked_key + "_mean"] = keyhold_means[tracked_key]

    # Add digraph features
    for digraph_name in DIGRAPH_LIST:
        row["digraph_" + digraph_name + "_dd_mean"] = digraph_means[digraph_name]

    # Add trigraph features
    for trigraph_name in TRIGRAPH_LIST:
        row["trigraph_" + trigraph_name + "_span_mean"] = trigraph_means[trigraph_name]

    return row

# Calculate all the features for each sample in the data and return them as a DataFrame.
def make_sample_feature_table(data):
    print("Making features for each sample...")

    # Create an empty list to hold the feature rows for each sample.
    sample_feature_rows = []

    # Loops through each sample in the data and calculates its features using the make_features_for_one_sample function.
    for sample_id, sample in data.groupby("sample_id"):
        # Calculate the features for the sample and add them to the list of feature rows.
        sample_features = make_features_for_one_sample(sample)
        sample_feature_rows.append(sample_features)

    # Combine all the sample feature rows into a single DataFrame 
    feature_table = pd.DataFrame(sample_feature_rows)
    
    # Arrange the columns with the sample metadata first, followed by all the calculated features.
    feature_table = feature_table[METADATA_COLUMNS + ALL_FEATURES]

    print("Feature rows:", len(feature_table))
    print("Feature columns:", len(ALL_FEATURES))

    return feature_table


# Split the free-text samples into windows of a fixed number of key pairs and calculate features for each window.
def make_free_text_windows(data):
    print("Making free-text windows...")
    
    # Store one feature row for each completed free-text window.
    window_feature_rows = []
    
    # Filter the data to include only the free-text samples
    free_text_data = data[data["task_type"] == "free"]

    # Loop through each free-text sample in the data at a time
    for sample_id, sample in free_text_data.groupby("sample_id"):
        
        # Create completed key pairs for this sample.
        key_pairs = make_key_pairs(sample)

        # Calculate how many complete windows can be made.
        number_of_full_windows = len(key_pairs) // FREE_TEXT_WINDOW_SIZE

        # Loop through each full window and calculate its features using the make_features_for_one_sample function.
        for window_number in range(number_of_full_windows):
            
            # Calculate where this window starts and ends in the key-pair list.
            window_start = window_number * FREE_TEXT_WINDOW_SIZE
            window_end = window_start + FREE_TEXT_WINDOW_SIZE
            
            # Get the key pairs that belong to this window. 
            window_key_pairs = key_pairs[window_start:window_end]
            
            # Get the press IDs for the key pairs in this window.
            window_press_ids = []
            for key_pair in window_key_pairs:
                window_press_ids.append(key_pair["press_id"])
            
            # Keep only the rows in the sample that belong to this window and make a copy of them.
            window_sample = sample[sample["press_id"].isin(window_press_ids)].copy()
            
            # Give the window its own unique sample ID.
            window_sample["sample_id"] = sample_id + "_WIN" + str(window_number + 1).zfill(3)

            # Calculate the features for this window.
            window_features = make_features_for_one_sample(window_sample)
            
            # Store the window features in the list of window feature rows.
            window_feature_rows.append(window_features)

    # Combine all the window feature rows into a single DataFrame.
    window_feature_table = pd.DataFrame(window_feature_rows)
    
    # Arrange the metadata columns first, followed by all calculated features.
    window_feature_table = window_feature_table[METADATA_COLUMNS + ALL_FEATURES]

    print("Free-text window rows:", len(window_feature_table))
    return window_feature_table


# Make a simple summary showing how many feature rows exist for each data type.
def make_data_type_summary(model_features, free_windows):
    summary_rows = []

    # Loop through the data types we allow in experiments.
    for data_type in DATA_TYPES:
        task_type, condition = get_task_type_and_condition(data_type)

        # Free normal is stored in the separate 50-key window feature table.
        # All other data types are stored in the normal sample feature table.
        if data_type == "free normal":
            feature_table = free_windows
        else:
            feature_table = model_features

        # Keep only the rows for this data type across all 5 required sessions.
        data_type_rows = choose_rows(feature_table, data_type, REQUIRED_SESSIONS_FOR_PARTICIPANT)

        # Count how many rows this data type has in each session.
        session_1_rows = len(data_type_rows[data_type_rows["session_number"] == 1])
        session_2_rows = len(data_type_rows[data_type_rows["session_number"] == 2])
        session_3_rows = len(data_type_rows[data_type_rows["session_number"] == 3])
        session_4_rows = len(data_type_rows[data_type_rows["session_number"] == 4])
        session_5_rows = len(data_type_rows[data_type_rows["session_number"] == 5])

        participants = data_type_rows["participant_code"].nunique()
        average_rows_per_participant = len(data_type_rows) / participants

        summary_rows.append({
            "data_type": data_type,
            "task_type": task_type,
            "condition": condition,
            "feature_rows": len(data_type_rows),
            "participants": participants,
            "sessions_present": data_type_rows["session_number"].nunique(),
            "session_1_rows": session_1_rows,
            "session_2_rows": session_2_rows,
            "session_3_rows": session_3_rows,
            "session_4_rows": session_4_rows,
            "session_5_rows": session_5_rows,
            "average_rows_per_participant": average_rows_per_participant,
        })

    summary_table = pd.DataFrame(summary_rows)
    return summary_table


# =============================================================================
# STEP 6: FISHER SCORE FEATURE RANKING
# =============================================================================

# Give each Fisher Score a descriptive rating based on its value.  This is used in the feature ranking CSV.
def fisher_score_rating(fisher_score):
    if pd.isna(fisher_score):
        return "not enough data"
    if fisher_score >= 1:
        return "discriminative"
    else:
        return "not discriminative"


# Calculate a Fisher Score for every feature and rank the features from best to worst.
def rank_features(model_features, sessions_to_use, data_type="fixed normal"):
    print("Ranking features with Fisher Score...")
    print("Data type used:", data_type)
    print("Sessions used:", sessions_to_use)

    
    # Keep only the selected data type and sessions for the Fisher Score calculation.
    rows_to_rank = choose_rows(model_features, data_type, sessions_to_use)

    if len(rows_to_rank) == 0:
        raise ValueError("No rows found for Fisher Score ranking.")

    # Store one ranking row for each feature.
    ranking_rows = []

    # Loop through all the features and calculate their Fisher Scores.
    for feature in ALL_FEATURES:
        
        # Take one feature column and convert it to numeric values, replacing any invalid or infinite values with NaN.
        values = pd.to_numeric(rows_to_rank[feature], errors="coerce")
        values = values.replace([float("inf"), float("-inf")], blank_number())

        # Create a DataFrame with each participant's code and their corresponding feature value
        one_feature_data = pd.DataFrame({
            "participant_code": rows_to_rank["participant_code"],
            "value": values,
        })

        # Remove any blank or missing values from the feature values DataFrame.
        one_feature_data = one_feature_data.dropna()

        # Calculate how much of the total filtered data has valid values for this feature.
        valid_samples = len(one_feature_data)
        coverage = valid_samples / len(rows_to_rank)

        # Calculate the overall average of the feature values across all participants.
        overall_feature_average = one_feature_data["value"].mean()

        between_participants_score = 0
        within_participant_score = 0
        participants_used = 0

        # For each participant, calculate their average and variance for this feature, and use these to calculate the Fisher Score components.
        # Fisher Score rewards features where participant averages are far apart.
        # Fisher Score penalizes features where participants have high variance in their own values.
        for participant, participant_rows in one_feature_data.groupby("participant_code"):
            participants_used = participants_used + 1


            # Get the feature values for this participant and calculate their average and variance.
            participant_values = participant_rows["value"]
            number_of_samples = len(participant_values)
            participant_average = participant_values.mean()
            participant_variance = participant_values.var(ddof=0)

            # Measure how far this participant's average is from the overall feature average, and add it to the between-participant score.
            between_participants_score = between_participants_score + (
                number_of_samples * (participant_average - overall_feature_average) ** 2
            )

            # Measure how much this participant's values vary across their own samples, and add it to the within-participant score.
            within_participant_score = within_participant_score + (
                number_of_samples * participant_variance
            )

        # Check if there are enough participants and if the within-participant score is greater than zero to avoid division by zero.
        if participants_used >= 2 and within_participant_score > 0:
            # Calculate the Fisher Score as the ratio of between-participants score to within-participant score.
            fisher_score = between_participants_score / within_participant_score
        else:
            # If there are not enough participants or the within-participant score is zero, set the Fisher Score to NaN to indicate that it cannot be calculated.
            fisher_score = blank_number()

        # Store the results for that given feature in the ranking rows list
        # Call the fisher_score_rating function to give a descriptive rating for the Fisher Score.
        ranking_rows.append(
            {
                "feature": feature,
                "feature_family": FEATURE_FAMILY[feature],
                "coverage": coverage,
                "valid_samples": valid_samples,
                "participants_used": participants_used,
                "fisher_between_score": between_participants_score,
                "fisher_within_score": within_participant_score,
                "fisher_score": fisher_score,
                "fisher_score_rating": fisher_score_rating(fisher_score),
            }
        )

    # Combine all the feature ranking rows into a single DataFrame 
    ranking = pd.DataFrame(ranking_rows)

    # Rank features by Fisher Score, then coverage, then feature name.
    ranking = ranking.sort_values(
        ["fisher_score", "coverage", "feature"],
        ascending=[False, False, True],
        na_position="last",
    )

    # Drop the old index and add a new column for the rank of each feature, starting from 1 for the best feature.
    ranking = ranking.reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    return ranking


# =============================================================================
# STEP 7: PREPARE FEATURES FOR MODELING
# =============================================================================

# Check which features are available for a given participant based on their training rows and the list of features we want to use.
def get_available_features_for_participant(participant_training_rows, features_we_want_to_use):
    
    # Create an empty list to hold the features that are available for this participant.
    available_features = []

    # Check each selected feature one at a time.
    for feature in features_we_want_to_use:
        
        # Get the feature values for the participant's training rows and convert them to numeric values, replacing any invalid or infinite values with NaN.
        values = pd.to_numeric(participant_training_rows[feature], errors="coerce")
        values = values.replace([float("inf"), float("-inf")], blank_number())

        # Check which values are not missing.
        valid_values = values.notna()
        
        # Check whether at least one valid value exists.
        has_valid_values = valid_values.any()
        
        # Keep the feature if it has at least one valid value.
        if has_valid_values:
            available_features.append(feature)

    return available_features

# Fill in missing feature values in the training and testing data with the median value calculated from the training data. 
def fill_blanks_with_training_median(training_values, testing_values):
    
    # Find the typical value for this feature using only the training data.
    training_median = training_values.median()

    # Fill in any missing values with the training median. 
    filled_training_values = training_values.fillna(training_median)
    
    # Fill missing testing values using the same training median.
    # This ensures that the testing data is scaled in the same way as the training data.
    filled_testing_values = testing_values.fillna(training_median)

    return filled_training_values, filled_testing_values, training_median

# Scale values using the training median and training IQR.
def scale_with_training_iqr(filled_training_values, filled_testing_values, training_median):

    
    # Calculate the interquartile range (IQR) of the training values to use for scaling.
    training_iqr = my_iqr(filled_training_values)

    # Avoid dividing by zero if the feature values have no spread.
    if pd.isna(training_iqr) or training_iqr == 0:
        training_iqr = 1

    # Scale the training and testing values using the training median and IQR.
    scaled_training_values = (filled_training_values - training_median) / training_iqr

    # Scale the testing values using the same training median and IQR.
    scaled_testing_values = (filled_testing_values - training_median) / training_iqr

    return scaled_training_values, scaled_testing_values

# Prepare the training and testing data for modeling by filling in missing values and scaling the features based on the training data.
def prepare_features_for_model(training_data, testing_data, available_features):

    # Prepare the training and testing data for modeling by filling in missing values and scaling the features based on the training data.
    prepared_training_features  = {}
    prepared_testing_features  = {}

    # Loop through each available feature and prepare it for modeling.
    for feature in available_features:
        
        # Convert the feature values to numeric, replacing any invalid or infinite values with NaN.
        training_values  = pd.to_numeric(training_data[feature], errors="coerce")
        testing_values  = pd.to_numeric(testing_data[feature], errors="coerce")
        training_values = training_values.replace([float("inf"), float("-inf")], blank_number())
        testing_values = testing_values.replace([float("inf"), float("-inf")], blank_number())

        # Fill in missing values in the training and testing data with the median value calculated from the training data.
        filled_training_values, filled_testing_values, training_median = fill_blanks_with_training_median(training_values, testing_values)
        
        # Scale the training and testing values using the training median and training IQR.
        scaled_training_values, scaled_testing_values = scale_with_training_iqr(filled_training_values, filled_testing_values, training_median)
        
        # Store the prepared training and testing values for this feature 
        prepared_training_features[feature] = scaled_training_values
        prepared_testing_features[feature] = scaled_testing_values

    # Convert the prepared training and testing features into DataFrames with the same index as the original training and testing data.
    prepared_training_data = pd.DataFrame(prepared_training_features, index=training_data.index)
    prepared_testing_data = pd.DataFrame(prepared_testing_features, index=testing_data.index)

    return prepared_training_data, prepared_testing_data

# =============================================================================
# STEP 8: EVALUATE MODEL PERFORMANCE
# =============================================================================

# Calculate the authentication numbers for a given set genuine and impostor claims
def calculate_authentication_numbers(claims):
    
    # Split the claims into genuine and impostor claims based on the "is_genuine" column.
    # - genuine claims: the participant really is the participant they claim to be
    # - impostor claims: the participant is pretending to be that participant
    genuine_claims = claims[claims["is_genuine"] == True]
    impostor_claims = claims[claims["is_genuine"] == False]

    # Count how many genuine and impostor authentication attempts were made
    genuine_attempts = len(genuine_claims)
    impostor_attempts = len(impostor_claims)

    # Count how many genuine users were correctly accepted.
    genuine_accepts = int(genuine_claims["accepted"].sum())
    
    # Count how many impostors were wrongly accepted.
    false_accepts = int(impostor_claims["accepted"].sum())

    # Genuine attempts that were not accepted were wrongly rejected.
    false_rejections = genuine_attempts - genuine_accepts
    
    # Impostor attempts that were not accepted were correctly rejected.
    impostor_rejections = impostor_attempts - false_accepts

    # Calculate the Genuine Acceptance Rate (GAR) and False Rejection Rate (FRR)
    # How well the system accepts genuine users and rejects impostors.
    # GAR = how often genuine users were correctly accepted
    # FRR = how often genuine users were wrongly rejected
    if genuine_attempts > 0:
        genuine_acceptance_rate  = genuine_accepts / genuine_attempts
        false_rejection_rate  = false_rejections / genuine_attempts
    else:
        genuine_acceptance_rate = blank_number()
        false_rejection_rate = blank_number()

    # Calculate the False Acceptance Rate (FAR) and Impostor Rejection Rate (IRR)
    # How well the system handles impostor attempts and rejects them.
    # FAR = how often impostors were wrongly accepted
    # IRR = how often impostors were correctly rejected
    if impostor_attempts > 0:
        false_acceptance_rate = false_accepts / impostor_attempts
        impostor_rejection_rate  = impostor_rejections / impostor_attempts
    else:
        false_acceptance_rate  = blank_number()
        impostor_rejection_rate = blank_number()

    # Calculate the Balanced Accuracy as the average of GAR and IRR
    # Balanced Accuracy = Correctly accepted genuine users + Correctly rejected impostors / Total attempts
    if pd.notna(genuine_acceptance_rate) and pd.notna(impostor_rejection_rate):
        balanced_accuracy = (genuine_acceptance_rate + impostor_rejection_rate) / 2
    else:
        balanced_accuracy = blank_number()
        
    # Return all the calculated authentication numbers and rates as a dictionary
    return {
        "genuine_attempts": genuine_attempts,
        "impostor_attempts": impostor_attempts,
        "genuine_accepts": genuine_accepts,
        "false_rejections": false_rejections,
        "false_accepts": false_accepts,
        "impostor_rejections": impostor_rejections,
        "genuine_acceptance_rate": genuine_acceptance_rate,
        "false_rejection_rate": false_rejection_rate,
        "false_acceptance_rate": false_acceptance_rate,
        "impostor_rejection_rate": impostor_rejection_rate,
        "balanced_accuracy": balanced_accuracy,
    }

# Calculate the Equal Error Rate (EER) and the threshold where FAR equals FRR for each claim.
def calculate_eer(claims):
    
    # Split the claims into genuine and impostor claims based on the "is_genuine" column.
    # - genuine claims: the participant really is the participant they claim to be
    # - impostor claims: the participant is pretending to be that participant
    genuine_claims = claims[claims["is_genuine"] == True]
    impostor_claims = claims[claims["is_genuine"] == False]
    
    # Get the scores for the genuine and impostor claims
    genuine_scores = genuine_claims["model_score"].tolist()  
    impostor_scores = impostor_claims["model_score"].tolist()
    
    # Combine the genuine and imposter scores list 
    all_scores = genuine_scores + impostor_scores
    
    # Track the smallest difference between FAR and FRR
    smallest_difference = None
    eer_threshold = None
    eer_far = None
    eer_frr = None
    
    
    # Loop through each claim score and calculate the FAR and FRR at that threshold.
    for threshold in all_scores:
        
        # Count how many genuine scores are below the threshold (wrongly rejected) 
        genuine_rejections = 0
        for score in genuine_scores:
            if score < threshold:
                genuine_rejections = genuine_rejections + 1
        
        # Count how many impostor scores are above the threshold (wrongly accepted)
        impostor_accepts = 0
        for score in impostor_scores:
            if score >= threshold:
                impostor_accepts = impostor_accepts + 1
        
        # Calculate the False Rejection Rate (FRR) 
        frr = genuine_rejections / len(genuine_scores)
        
        # Calculate the False Acceptance Rate (FAR)
        far = impostor_accepts / len(impostor_scores)
        
        # Calculate the absolute difference between FAR and FRR to find the threshold where they are closest.
        difference = abs(far - frr)
        
        # Remember the threshold where FAR and FRR are closest to each other
        if smallest_difference is None or difference < smallest_difference:
            smallest_difference = difference
            eer_threshold = threshold
            eer_far = far
            eer_frr = frr
          
    # EER is the average of FAR and FRR at the threshold where they are closest.  
    eer = (eer_far + eer_frr) / 2
        
    return eer, eer_threshold 
    
    
# Calculate the authentication numbers for each participant based on their claims and return a DataFrame with the results.
def calculate_participant_metrics(all_claims):

    # Store the results for each participant
    participant_results = []

    # Loop through each participant and calculate their authentication numbers based on their claims.
    for target_participant, participant_claims in all_claims.groupby("target_participant"):
        
        authentication_result = calculate_authentication_numbers(participant_claims)
        eer, eer_threshold = calculate_eer(participant_claims)
        authentication_result["eer"] = eer
        authentication_result["eer_threshold"] = eer_threshold
        authentication_result["target_participant"] = target_participant
        participant_results.append(authentication_result)

    participant_metrics  = pd.DataFrame(participant_results)
    participant_metrics = participant_metrics.sort_values("target_participant")
    return participant_metrics


# Make the participant metrics more readable by converting rates to percentages and rounding them to two decimal places.
def make_readable_participant_metrics(participant_metrics):

    # Create a copy of the participant metrics DataFrame to avoid modifying the original data.
    metrics = participant_metrics.copy()

    metrics["GAR_percent"] = (metrics["genuine_acceptance_rate"] * 100).round(2)
    metrics["FRR_percent"] = (metrics["false_rejection_rate"] * 100).round(2)
    metrics["FAR_percent"] = (metrics["false_acceptance_rate"] * 100).round(2)
    metrics["IRR_percent"] = (metrics["impostor_rejection_rate"] * 100).round(2)
    metrics["balanced_accuracy_percent"] = (metrics["balanced_accuracy"] * 100).round(2)
    metrics["EER_percent"] = (metrics["eer"] * 100).round(2)
    metrics["EER_threshold"] = metrics["eer_threshold"].round(9)

    # Keep only the relevant columns for the final output and create a new DataFrame with them.
    metrics = metrics[
        [
            "target_participant",
            "genuine_attempts",
            "genuine_accepts",
            "false_rejections",
            "impostor_attempts",
            "false_accepts",
            "impostor_rejections",
            "GAR_percent",
            "FRR_percent",
            "FAR_percent",
            "IRR_percent",
            "balanced_accuracy_percent",
            "EER_percent",
            "EER_threshold",
        ]
    ].copy()

    # Sort the metrics by target participant and reset the index to create a clean DataFrame for output.
    metrics = metrics.sort_values("target_participant").reset_index(drop=True)
    return metrics


# =============================================================================
# STEP 9: TRAIN ONE MODEL CONFIGURATION
# =============================================================================

# If the user has specified "scale" or "auto" for the gamma parameter in the model settings, return the corresponding string. 
# Otherwise if the user has specified a numeric value, convert it to a float and return it.
def gamma_from_tuning_result(value):
    if value == "scale":
        return "scale"
    if value == "auto":
         return "auto"
    return float(value)

# Run a model experiment for a given set of training and testing data, participants, features, and model settings.
def run_model_experiment(
    training_feature_table,
    testing_feature_table,
    participants_to_use,
    selected_features,
    training_sessions,
    testing_session,
    model_nu,
    model_gamma,
    show_each_participant,
    training_data_type="fixed normal",
    testing_data_type="fixed normal",
):

    # Keep only the selected training data type and training sessions.
    training_rows = choose_rows(training_feature_table, training_data_type, training_sessions)

    # Keep only the selected testing data type and testing session.
    testing_rows = choose_rows(testing_feature_table, testing_data_type, [testing_session])

    # Check that there are enough training and testing rows for the experiment. If not, raise an error with a descriptive message.
    if len(training_rows) == 0:
        raise ValueError("No training rows found for: " + training_data_type)

    if len(testing_rows) == 0:
        raise ValueError("No testing rows found for: " + testing_data_type)

    # Store all the claims and model details for each participant in lists to combine later.
    all_claim_rows = []
    model_detail_rows = []
    
    
    # Train and test a model for each participant in the list of participants to use.
    for target_participant  in participants_to_use:
        if show_each_participant:
            print("  model for", target_participant)

        # Get the training rows for this participant and make a copy of them to avoid modifying the original data.
        this_participant_training = training_rows[training_rows["participant_code"] == target_participant].copy()
        this_participant_testing = testing_rows.copy()

        # Skip this participant if there are not enough training samples.
        if len(this_participant_training) < 2:
            continue

        
        genuine_test_rows = this_participant_testing[this_participant_testing["participant_code"] == target_participant]
        if len(genuine_test_rows) == 0:
            continue
        
        # Check which of the selected features are actually available for this participant based on their training data.
        usable_features = get_available_features_for_participant(this_participant_training, selected_features)

        # Fill missing values and scale the features using training data only.
        train_scaled, test_scaled = prepare_features_for_model(
            this_participant_training,
            this_participant_testing,
            usable_features,
        )

        # Train a One-Class SVM using only this participant's genuine training data.
        model = OneClassSVM(kernel="rbf", nu=model_nu, gamma=model_gamma)
        model.fit(train_scaled)

        # Give every testing sample a score based on how well it fits the model trained on this participant's genuine data.
        scores = model.decision_function(test_scaled)
        
        # Create one authentication claim row for each testing sample.
        claims = this_participant_testing[
            ["participant_code", "session_number", "task_type", "condition", "prompt_id", "sample_id"]
        ].copy()
        
        # Rename the "participant_code" column to "actual_participant" to indicate the true identity of the participant for each claim.
        claims = claims.rename(columns={"participant_code": "actual_participant"})
        
        # Record 
        claims.insert(0, "target_participant", target_participant)
        claims["is_genuine"] = claims["actual_participant"] == target_participant
        claims["model_score"] = scores
        claims["accepted"] = claims["model_score"] >= 0

        # Add the claims for this participant to the list of all claims for all participants.
        all_claim_rows.append(claims)
        
        # Store useful details about the model for this participant for reporting and analysis purposes.
        model_detail_rows.append(
            {
                "target_participant": target_participant,
                "training_data_type": training_data_type,
                "testing_data_type": testing_data_type,
                "genuine_training_samples": len(this_participant_training),
                "genuine_test_samples": int(claims["is_genuine"].sum()),
                "features_requested": len(selected_features),
                "features_used": len(usable_features),
                "model_nu": model_nu,
                "model_gamma": model_gamma,
                "feature_names_used": ";".join(usable_features),
            }
        )
    
    if len(all_claim_rows) == 0:
        raise ValueError("No model claims were made. Check the training and testing settings.")

    # Combine all the claims and model details for all participants into DataFrames for further analysis and reporting.
    all_claims = pd.concat(all_claim_rows, ignore_index=True)
    model_details = pd.DataFrame(model_detail_rows)
    
    # Calculate the overall authentication results 
    overall_metrics  = calculate_authentication_numbers(all_claims)
    
    overall_eer, overall_eer_threshold = calculate_eer(all_claims)
    overall_metrics["eer"] = overall_eer
    overall_metrics["eer_threshold"] = overall_eer_threshold
    
    # Calculate the authentication results for each participant individually and return them as a DataFrame.
    participant_numbers = calculate_participant_metrics(all_claims)

    return all_claims, model_details, overall_metrics, participant_numbers


# -----------------------------------------------------------------------------
# STEP 10: TUNE FEATURE COUNT AND MODEL SETTINGS
# -----------------------------------------------------------------------------

# Try different feature counts and model settings for nu and gamma, and return the best combination based on validation results.
def tune_model_settings(
    training_feature_table,
    participants_to_use,
    feature_ranking,
    tuning_training_sessions,
    tuning_validation_session,
    model_settings_to_try,
    training_data_type="fixed normal",
    experiment_name="",
):
    print("Tuning feature count and model settings...")
    if experiment_name != "":
        print("Experiment:", experiment_name)
    print("Training data type:", training_data_type)
    print("Validation data type:", training_data_type)
    print("Training sessions:", tuning_training_sessions)
    print("Validation session:", tuning_validation_session)


    tuning_results = []

    # Try each feature count in the list of feature counts 
    for feature_count in model_settings_to_try["feature_count"]:
        
        selected_features = feature_ranking["feature"].head(feature_count).tolist()
        
        # Try each combination of nu and gamma values in the list of model settings to try.
        for model_nu in model_settings_to_try["nu"]:
            for model_gamma in model_settings_to_try["gamma"]:
                claims, model_details, overall_metrics, participant_metrics = run_model_experiment(
                    training_feature_table,
                    training_feature_table,
                    participants_to_use=participants_to_use,
                    selected_features=selected_features,
                    training_sessions=tuning_training_sessions, 
                    testing_session=tuning_validation_session,
                    show_each_participant=False,
                    model_nu=model_nu,
                    model_gamma=model_gamma,
                    training_data_type=training_data_type,
                    testing_data_type=training_data_type,
                )

                tuning_result = {
                    "experiment_name": experiment_name,
                    "training_data_type": training_data_type,
                    "validation_data_type": training_data_type,
                    "feature_count": feature_count,
                    "nu": model_nu,
                    "gamma": model_gamma,
                    "average_features_actually_used": model_details["features_used"].mean(),
                    "genuine_acceptance_rate": overall_metrics["genuine_acceptance_rate"],
                    "false_rejection_rate": overall_metrics["false_rejection_rate"],
                    "false_acceptance_rate": overall_metrics["false_acceptance_rate"],
                    "impostor_rejection_rate": overall_metrics["impostor_rejection_rate"],
                    "balanced_accuracy": overall_metrics["balanced_accuracy"],
                }
                tuning_results.append(tuning_result)

                # Print the results for each combibnation of feature count, nu, and gamma to the console for tracking progress.
                if experiment_name != "":
                    print(f"  [{experiment_name}] top {feature_count}  nu {model_nu}  gamma {model_gamma} -> validation balanced accuracy {percent(overall_metrics['balanced_accuracy'])}")
                else:
                    print(f"  top {feature_count}  nu {model_nu}  gamma {model_gamma} -> validation balanced accuracy {percent(overall_metrics['balanced_accuracy'])}")


    # Put all results into a table
    tuning_table = pd.DataFrame(tuning_results)

    # Choose the best feature count
    # First, prefer the highest balanced accuracy
    # If tied, prefer lower false acceptance rate
    # If still tied, prefer fewer features 
    tuning_table = tuning_table.sort_values(
        ["balanced_accuracy", "false_acceptance_rate", "feature_count"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    # Best results is the first row ranked by the sorting criteria above.
    best_result = tuning_table.iloc[0]
    
    # Get the number of top features to use for the final model from the best result.
    selected_feature_count = int(best_result["feature_count"])
    selected_nu = float(best_result["nu"])
    selected_gamma = gamma_from_tuning_result(best_result["gamma"])

    print("Selected feature count:", selected_feature_count)
    print("Selected nu:", selected_nu)
    print("Selected gamma:", selected_gamma)

    return selected_feature_count, selected_nu, selected_gamma, tuning_table

# -----------------------------------------------------------------------------
# STEP 11: RUN THE FINAL MODEL
# -----------------------------------------------------------------------------

# Run the final model using the selected feature count and model settings, and evaluate one testing session.
def run_final_model(
    training_feature_table,
    testing_feature_table,
    participants_to_use,
    selected_features,
    model_nu,
    model_gamma,
    final_training_sessions,
    final_test_session,
    training_data_type="fixed normal",
    testing_data_type="fixed normal",
):
    print("Training and testing the final model...")
    print("Selected features used by final model:", len(selected_features))
    print("Selected nu:", model_nu)
    print("Selected gamma:", model_gamma)
    print("Training data type:", training_data_type)
    print("Testing data type:", testing_data_type)
    print("Training sessions:", final_training_sessions)
    print("Testing session:", final_test_session)

    return run_model_experiment(
        training_feature_table,
        testing_feature_table,
        participants_to_use,
        selected_features,
        training_sessions=final_training_sessions,
        testing_session=final_test_session,
        model_nu=model_nu,
        model_gamma=model_gamma,
        show_each_participant=True,
        training_data_type=training_data_type,
        testing_data_type=testing_data_type,
    )

# =============================================================================
# STEP 12: RUN ONE EXPERIMENT
# =============================================================================

def make_experiment_result_row(
    experiment_name,
    training_data_type,
    testing_data_type,
    tuning_training_sessions,
    tuning_validation_session,
    final_training_sessions,
    final_test_session,
    selected_features,
    selected_nu,
    selected_gamma,
    final_numbers,
):
    return {
        "experiment_name": experiment_name,
        "training_data_type": training_data_type,
        "testing_data_type": testing_data_type,
        "tuning_training_sessions": "+".join(str(session) for session in tuning_training_sessions),
        "tuning_validation_session": tuning_validation_session,
        "final_training_sessions": "+".join(str(session) for session in final_training_sessions),
        "final_test_session": final_test_session,
        "selected_feature_count": len(selected_features),
        "selected_nu": selected_nu,
        "selected_gamma": selected_gamma,
        "genuine_attempts": final_numbers["genuine_attempts"],
        "impostor_attempts": final_numbers["impostor_attempts"],
        "genuine_accepts": final_numbers["genuine_accepts"],
        "false_rejections": final_numbers["false_rejections"],
        "false_accepts": final_numbers["false_accepts"],
        "impostor_rejections": final_numbers["impostor_rejections"],
        "genuine_acceptance_rate": final_numbers["genuine_acceptance_rate"],
        "false_rejection_rate": final_numbers["false_rejection_rate"],
        "false_acceptance_rate": final_numbers["false_acceptance_rate"],
        "impostor_rejection_rate": final_numbers["impostor_rejection_rate"],
        "balanced_accuracy": final_numbers["balanced_accuracy"],
        "eer": final_numbers["eer"],
        "eer_threshold": final_numbers["eer_threshold"],
    }


def make_experiment_settings_table(
    experiment_name,
    training_data_type,
    testing_data_type,
    tuning_training_sessions,
    tuning_validation_session,
    final_training_sessions,
    final_test_sessions,
):
    settings = {
        "experiment_name": experiment_name,
        "training_data_type": training_data_type,
        "validation_data_type": training_data_type,
        "testing_data_type": testing_data_type,
        "tuning_training_sessions": session_list_text(tuning_training_sessions),
        "tuning_validation_session": tuning_validation_session,
        "final_training_sessions": session_list_text(final_training_sessions),
        "final_test_sessions": session_list_text(final_test_sessions),
    }

    settings_table = pd.DataFrame([settings])
    return settings_table


def prepare_data_once(output_folder):
    make_output_folder(output_folder)

    shared_output_folder = output_folder / "shared"
    make_output_folder(shared_output_folder)

    print("")
    print("=" * 70)
    print("PREPARING DATA")
    print("=" * 70)

    # Step 1. Read the raw clean_model_events_all.csv.
    raw_data = read_the_csv()
    
    # Step 2. Keep only the participants who completed all 5 required sessions.
    all_participants = sorted(raw_data["participant_code"].unique())
    participants_to_use = find_people_with_required_sessions(raw_data)
    excluded_people = save_participant_list(all_participants, participants_to_use, shared_output_folder)

    print("Required participant sessions:", REQUIRED_SESSIONS_FOR_PARTICIPANT)
    print("Participants with required sessions:", len(participants_to_use))
    print("Participants excluded:", len(excluded_people))

    included_participants_data = raw_data[raw_data["participant_code"].isin(participants_to_use)].copy()
    included_participants_data = included_participants_data.reset_index(drop=True)

    print("Rows after participant filter:", f"{len(included_participants_data):,}")

    # Step 3. Turn keyboard events into typing features for each sample.
    model_features = make_sample_feature_table(included_participants_data)
    model_features.to_csv(shared_output_folder / "sample_features.csv", index=False)

    free_windows = make_free_text_windows(included_participants_data)
    free_windows.to_csv(shared_output_folder / "free_text_window_features.csv", index=False)
    
    # Save a simple table showing how many feature rows each data type has.
    data_type_summary = make_data_type_summary(model_features, free_windows)
    data_type_summary.to_csv(shared_output_folder / "data_type_summary.csv", index=False)

    return {
        "raw_data": raw_data,
        "included_participants_data": included_participants_data,
        "model_features": model_features,
        "free_windows": free_windows,
        "data_type_summary": data_type_summary,
        "participants_to_use": participants_to_use,
        "excluded_people": excluded_people,
    }

# Run one experiment with the given settings passed in as arguments. 
# This function handles the entire process of tuning, training, and testing the model, as well as saving the results to the specified output folder.
def run_experiment(
    experiment_name,
    prepared_data,
    training_data_type,
    testing_data_type,
    tuning_training_sessions,
    tuning_validation_session,
    final_training_sessions,
    final_test_sessions,
    output_folder,
    model_settings_to_try=MODEL_SETTINGS_TO_TRY,
):
    experiments_output_folder = output_folder / "experiments"
    experiment_output_folder = experiments_output_folder / experiment_name
    participant_metrics_folder = experiment_output_folder / "participant_metrics"
    claim_scores_folder = experiment_output_folder / "claim_scores"
    model_details_folder = experiment_output_folder / "model_details"

    make_output_folder(experiments_output_folder)
    make_output_folder(experiment_output_folder)
    make_output_folder(participant_metrics_folder)
    make_output_folder(claim_scores_folder)
    make_output_folder(model_details_folder)

    raw_data = prepared_data["raw_data"]
    included_participants_data = prepared_data["included_participants_data"]
    model_features = prepared_data["model_features"]
    free_windows = prepared_data["free_windows"]
    participants_to_use = prepared_data["participants_to_use"]
    excluded_people = prepared_data["excluded_people"]

    # Choose the correct feature table for the training and testing data types.
    # For example, "fixed normal" uses sample_features, but "free normal" uses free_text_window_features.
    training_feature_table = get_feature_table_for_data_type(prepared_data, training_data_type)
    testing_feature_table = get_feature_table_for_data_type(prepared_data, testing_data_type)

    print("")
    print("=" * 70)
    print("RUNNING EXPERIMENT:", experiment_name)
    print("=" * 70)
    print("Training data type:", training_data_type)
    print("Testing data type:", testing_data_type)

    experiment_settings = make_experiment_settings_table(
        experiment_name,
        training_data_type,
        testing_data_type,
        tuning_training_sessions,
        tuning_validation_session,
        final_training_sessions,
        final_test_sessions,
    )
    experiment_settings.to_csv(experiment_output_folder / "experiment_settings.csv", index=False)

    # Step 4. Tuning phase.
    ranking_for_tuning = rank_features(training_feature_table, tuning_training_sessions, training_data_type)

    selected_feature_count, selected_nu, selected_gamma, tuning_table = tune_model_settings(
        training_feature_table,
        participants_to_use,
        ranking_for_tuning,
        tuning_training_sessions,
        tuning_validation_session,
        model_settings_to_try,
        training_data_type,
        experiment_name,
    )
    tuning_table.to_csv(experiment_output_folder / "model_tuning_validation.csv", index=False)

    # Step 5. Pick the final top feature using the final model training sessions.
    ranking = rank_features(training_feature_table, final_training_sessions, training_data_type)
    ranking.insert(1, "experiment_name", experiment_name)
    ranking.insert(2, "training_data_type", training_data_type)
    ranking.to_csv(experiment_output_folder / "feature_fisher_score_ranking.csv", index=False)
    
    selected_features = ranking["feature"].head(selected_feature_count).tolist()
    selected_feature_table = pd.DataFrame({
        "experiment_name": experiment_name,
        "training_data_type": training_data_type,
        "testing_data_type": testing_data_type,
        "feature": selected_features,
    })
    selected_feature_table.to_csv(experiment_output_folder / "selected_model_features.csv", index=False)

    # Check how much the tuning-stage and final feature lists agree.
    early_top_features = set(ranking_for_tuning["feature"].head(selected_feature_count))
    final_top_features = set(ranking["feature"].head(selected_feature_count))
    feature_overlap = len(early_top_features & final_top_features) / selected_feature_count
    print("Overlap between tuning-stage and final top feature count", selected_feature_count, "features:", percent(feature_overlap))
    
    # Step 6. Run the final model for each requested final test session.
    result_rows = []
 
    # Run the final model for each testing session specified
    for final_test_session in final_test_sessions:
        claims, model_details, final_numbers, participant_numbers = run_final_model(
            training_feature_table,
            testing_feature_table,
            participants_to_use,
            selected_features,
            selected_nu,
            selected_gamma,
            final_training_sessions,
            final_test_session,
            training_data_type,
            testing_data_type,
        )

        # Save every individual authentication attempt.
        # This is useful when we want to see the actual model scores behind the summary results.
        claims.to_csv(
            claim_scores_folder / ("session_" + str(final_test_session) + ".csv"),
            index=False,
        )

        # Save one row per participant model.
        # This records how many training samples and features each participant model used.
        model_details.to_csv(
            model_details_folder / ("session_" + str(final_test_session) + ".csv"),
            index=False,
        )

        readable_participant_numbers = make_readable_participant_metrics(participant_numbers)
        readable_participant_numbers.insert(0, "final_test_session", final_test_session)
        readable_participant_numbers.insert(0, "testing_data_type", testing_data_type)
        readable_participant_numbers.insert(0, "training_data_type", training_data_type)
        readable_participant_numbers.insert(0, "experiment_name", experiment_name)
        readable_participant_numbers.to_csv(
            participant_metrics_folder / ("session_" + str(final_test_session) + ".csv"),
            index=False,
        )

        result_rows.append(
            make_experiment_result_row(
                experiment_name,
                training_data_type,
                testing_data_type,
                tuning_training_sessions,
                tuning_validation_session,
                final_training_sessions,
                final_test_session,
                selected_features,
                selected_nu,
                selected_gamma,
                final_numbers,
            )
        )

    result_table = pd.DataFrame(result_rows)
    result_table.to_csv(experiment_output_folder / "experiment_results.csv", index=False)

    print("")
    print("Experiment done:", experiment_name)
    print("Outputs are in:")
    print(experiment_output_folder)

    return result_table
