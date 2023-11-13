
# scrape_data.py: functionality to scrape old_data from website

import pandas as pd
import itertools
from bs4 import BeautifulSoup
import requests
import time
import random


state_id = 0
gender = 1
number_of_results = 40000
params = {
        'mode': 'results',
        'criteria': '',
        'StoredProcParamsOn': 'yes',
        'VarGenderID': 0,
        'VarBibNumber': '',
        'VarLastName': '',
        'VarFirstName': '',
        'VarStateID': 0,
        'VarCountryOfResID': 0,
        'VarCountryOfCtzID': 0,
        'VarReportingSegID': 1,
        'VarAwardsDivID': 0,
        'VarQualClassID': 0,
        'VarCity': '',
        'VarTargetCount': 40000,
        'records': 25,

    }


def scrape_data(year):
    link = f'http://registration.baa.org/20{year}/cf/Public/iframe_ResultsSearch.cfm'
    print(link)
    results = []
    for page_number, start in enumerate(itertools.count(1, 25)):

        # Don't hammer the server. Give it a sec between requests.
        time.sleep(random.random())

        print(f"Page {page_number + 1} of {number_of_results/25}")
        response = requests.post(link, params=params, data={'start': start})

        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find("table", attrs={"class": "tablegrid_table"})
        rows = table.findAll("tr")
        for row in rows:
            a = [t.text.strip() for t in row.findAll("td")][0:]
            # Don't store lines without raw_data
            if len(a) > 0 and a != [''] and a != ['', ''] and a != ['', '', '']:
                results.append(a)

        # No more pages!
        if 'Next 25 Records' not in response.text:
            break

    data = []
    for i, result in enumerate(results):
        if i % 4 == 0:
            data.append(results[i] + results[i+1][1:])

    columns = ['Bib', 'Name', 'Age', 'M/F', 'City', 'State', 'Country', 'Citizen', '',
               '5K', '10K', '15K', '20K', 'Half', '25K', '30K', '35K', '40K', 'Pace',
               'Proj Time', 'Official Time', 'Overall', 'Gender', 'Division']

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"raw_data/data{year}.csv", index=False)
    return df


if __name__ == "__main__":
    full_year_data = scrape_data("13")
