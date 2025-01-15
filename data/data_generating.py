import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime, timedelta

# get the current script's directory
def get_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_datetime(start, end):
    # generate a random datetime between start and end, rounded to the nearest hour
    random_dt = start + timedelta(
        seconds=np.random.randint(0, int((end - start).total_seconds()))
    )

    # set minute, second, and microsecond to 0
    return random_dt.replace(minute=0, second=0, microsecond=0)


# set seed for reproducibility
np.random.seed(42)

# numbers of patients
n_patient = 100

# maximum number of admissions per patient
max_admission = 5

# maximum number of measurements per admission
max_entry = 3

# generate unique patient ids
patient_id = [str(uuid.uuid4()) for _ in range(n_patient)]
age = np.random.randint(18, 90, size=n_patient)
gender = np.random.choice([True, False], size=n_patient)

data = []

# per patient
for p_idx, p_id in enumerate(patient_id):

    # set number of admissions
    n_admissions = np.random.randint(1, max_admission + 1)

    # get unique admission ids for each admission
    admission_id = [str(uuid.uuid4()) for _ in range(n_admissions)]

    # per admission
    for ad_idx, ad_id in enumerate(admission_id):

        # set number of entries
        n_entries = np.random.randint(1, max_entry + 1)

        # set vitals
        hr = np.random.uniform(60, 140, size=n_entries).astype(int)
        bp1 = np.random.randint(90, 180, size=n_entries)
        bp2 = np.random.randint(60, 120, size=n_entries)
        temp = np.random.uniform(36.5, 39.5, size=n_entries).round(1)

        # set the time to event
        time_to_event = np.random.randint(1, 1000)

        # set the event marker with 30-70% ratio
        is_first = np.random.choice([True, False], p=[0.3, 0.7])

        # generate random datetime
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2022, 2, 1)
        date_time = [get_datetime(start_date, end_date) for _ in range(n_entries)]

        # assign to data
        for i in range(n_entries):
            entry = {
                "p_id": p_id,
                "admission_id": ad_id,
                "datetime": date_time[i],
                "hr": hr[i],
                "bp1": bp1[i],
                "bp2": bp2[i],
                "temp": temp[i],
                "age": age[p_idx],
                "gender": gender[p_idx],
                "time_to_event": time_to_event,
                "is_first": is_first,
            }
            data.append(entry)

df = pd.DataFrame(data)

# for creating avg_survival_probability for each unique admission
# REMOVE if not necessary
unique_admissions = df["admission_id"].unique()
random_probs = np.random.rand(len(unique_admissions))
admission_prob_map = dict(zip(unique_admissions, random_probs))
df["avg_survival_probability"] = df["admission_id"].map(admission_prob_map)

# set output directory
script_dir = get_dir()
output_file = os.path.join(script_dir, "example_data_with_prob.parquet")
# output_file = os.path.join(script_dir, "example_data_without_prob.parquet")

# save parquet file
df.to_parquet(output_file, engine="pyarrow", index=False)
