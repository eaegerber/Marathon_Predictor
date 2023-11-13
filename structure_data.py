
# structure_data.py: preprocess data from raw_data

import pandas as pd
from utils import str_to_int_time


def _read_data_with_year(path: str, yr: str):
    data = pd.read_csv(path)
    data["Year"] = str("20") + yr
    return data


def process_raw_data(
        yrs: list,
        store: bool = True,
        mins: bool = True,
        path: str = "processed_data/full_data.csv"
) -> pd.DataFrame:
    """Preprocess raw_data into single df"""
    data_list = [_read_data_with_year(f"raw_data/data{year}.csv", year) for year in yrs]
    data = pd.concat(data_list)

    cols = ['5K', '10K', '15K', '20K', 'Half', '25K', '30K', '35K', '40K', 'Official Time']
    for col in cols:
        data[col] = data[col].apply(str_to_int_time)

    data = data[['Name', 'Age', 'M/F'] + cols + ['Year']]
    data = data.dropna()

    for col in cols:
        data[col] = data[col].astype(int)
        if mins:
            data[col] = ((data[col] // 60) + 1).astype(int)  # convert to minutes

    data = data.rename({"Official Time": "Finish Net", "Half": "HALF"}, axis=1)

    if store:
        data.to_csv(path, index=False)
    return data


if __name__ == "__main__":
    years = ["09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "22", "23"]
    process_raw_data(yrs=years, store=True, mins=True, path="processed_data/full_data_mins.csv")
    process_raw_data(yrs=years, store=True, mins=False, path="processed_data/full_data_secs.csv")
