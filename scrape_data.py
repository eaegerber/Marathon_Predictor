
# scrape_data.py: functionality to scrape old_data from website

import numpy as np
import pandas as pd
import itertools
from bs4 import BeautifulSoup
import requests
import time
import random
import json

##### BOS < ####

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


def scrape_boston_data(year):
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
    df.to_csv(f"raw_data/boston/boston{year}.csv", index=False)
    return df

##### > BOS ####

##### NYC < ####

def get_runner_ids(eventCode="M2024", place1=48000, place2=48010):
    url = 'https://rmsprodapi.nyrr.org/api/v2/runners/finishers-filter'
    header = {"content-type": "application/json;charset=UTF-8"}
    j = {"eventCode": eventCode, "overallPlaceFrom": place1, "overallPlaceTo": place2,
         "sortColumn": "overallTime", "sortDescending":"false"
}
    response = requests.post(url, headers=header, json=j, allow_redirects=True)
    assert response.status_code == 200
    print(response)
    d = json.loads(response.text)
    print(d["totalItems"], len(d["items"]))
    print(d["items"][0]["runnerId"])
    return d

def get_splits(runner_id, sess):
    url = 'https://rmsprodapi.nyrr.org/api/v2/runners/resultDetails'
    header = {"content-type": "application/json;charset=UTF-8"}
    response = sess.post(url, headers=header, json={"runnerId": str(runner_id)}, allow_redirects=True, timeout=10)
    assert response.status_code == 200, f"Response Code = {response.status_code}"
    d = json.loads(response.text)
    assert d["details"], f"No data specified, d = {d}"
    return d


def load_nyc_data(last_runner=522, eventCode="M2024"):
    tables = []
    feats = ["runnerId", "bib", "firstName", "lastName", "age", "gender", "city", "countryCode"]

    for i in range(1, last_runner, 100):
        place1, place2 = i, i+99
        data = get_runner_ids(eventCode, place1, place2)
        table = pd.DataFrame(data['items'])[feats]
        tables.append(table)
        if (i - 1) % 1000 == 0:
            time.sleep(1)
            print("loaded", i - 1)

    return tables


def loop_attempts(a_dict, n, year=23):
    for i in range(n):
        a_dict = attempt_scrape_iteration(a_dict, year=year, itrn=i)
        num_left = len([k for k, v in a_dict.items() if v is None])
        print(f"{i}th cooldown, 30 seconds..........num left = {num_left}")
        pd.DataFrame([v for v in a_dict.values() if v is not None]).to_csv(f"raw_data/nyc/nyc_times{year}_.csv")
        if num_left == 0:
            print("done!!!!")
            break
        time.sleep(30)

    d_load = pd.read_csv(f"raw_data/nyc/nyc_names{year}_.csv")
    times = pd.DataFrame([v for v in a_dict.values() if v is not None])
    d_load.merge(times, left_on="runnerId", right_on="id").to_csv(f"raw_data/nyc/nyc{year}.csv")
    return a_dict

def attempt_scrape_iteration(adict, year=23, itrn=0):
    ids = [k for k, v in adict.items() if v is None]
    # failed = []
    session = requests.Session()
    for idx, idv in enumerate(ids):
        try:
            splits = get_splits(idv, session)['details']['splitResults']
            dt = {dct["splitCode"]: dct["time"] for dct in splits}
            ser = pd.Series(dt)
            ser["id"] = idv
            adict[idv] = ser

            if idx % 100 == 0:
                print("loaded", idx)
                pd.DataFrame([v for v in adict.values() if v is not None]).to_csv(f"raw_data/nyc/nyc_times{year}_{itrn}.csv")
                time.sleep(6.6)

            if idx % 1000 == 0:
                time.sleep(20.2)
            if idx % 44 == 0:
                time.sleep(4.4)
            if idx % 55 == 0:
                time.sleep(5.5)

            # if idx == 10:
            #     break

        except Exception as e:
            print("idx=", idx, ":", e)
            print("failed", idv)
            time.sleep(30)
            continue
    
    return adict

##### > NYC ####

##### CHI < ####


def get_links_chi(yr=23, pages=60):
    plinks = []
    for n in range(pages):
        url = f"https://chicago-history.r.mikatiming.com/2023/?page={n+1}&num_results=1000&event_main_group=20{yr}&pid=search&pidp=start"
        response = requests.post(url, allow_redirects=True)
        r = response.text.split("\n")
        # plinks += [line.split("idp=")[1].split("&amp")[0] for line in r if ("fullname" in line) and ("href" in line)]

        ids = [line.split("idp=")[1].split("&amp")[0] for line in r if ("fullname" in line) and ("href" in line)]
        fins = ["dash" not in line for line in r if ("inish" in line)][1:]
        # assert len(fins) == len(ids), f"Finishes {len(fins)} don't match up with names {len(ids)} on page {n}"
        if len(fins) != len(ids):
            print(f"Finishes {len(fins)} don't match up with names {len(ids)} on page {n}")
            plinks += ids
        else:
            plinks += [id for id, fin in zip(ids, fins) if fin]
        # for line in r:
        #     if ("fullname" in line) and ("href" in line):
        #         suffix = line.split('\"')[3].replace("&amp;", "&")
        #         plink = f"https://results.chicagomarathon.com/20{yr}/" + suffix
        #         # print(plink)
        #         plinks.append(plink)
    
        if n % 10 == 0:
            print("completed: ", n)
            time.sleep(2)

    pd.Series(plinks).to_csv(f"raw_data/chicago/chi_links{yr}.csv")
    return


def scrape_chi_data(yr=23):
    event_param = {24: "9TGG96382AC981", 23: "MAR_9TGG963812D", 22: "MAR_9TGG9638119", 21: "MAR_9TGG9638F1", 19: "MAR_999999107FA31100000000C9"}
    links_list = pd.read_csv(f"raw_data/chicago/ids{yr}.csv")['0'].values
    # np.random.shuffle(links_list)

    results = []
    for idx, person in enumerate(links_list):
      try:
        params = {"content" : "detail", "idp" : person, "lang" : "EN", "event" : event_param[yr],}
        response = requests.post("https://chicago-history.r.mikatiming.com/2023", params=params)
        ### for 2024 use response = requests.post("https://results.chicagomarathon.com/2024", params=params)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find("table", attrs={"class":"table table-condensed table-striped"})
        rows = table.findAll("tr")

        splits = [[t.text.strip() for t in row.findAll(["th", "td"])][2] for row in rows[1:]]
        rows2 = soup.findAll(name="table", attrs={"class":"table table-condensed"})[1]#.findAll("th")
        ### for 2024 change [1] to [0]
        info = [row.text.strip() for row in rows2.findAll(["th", "td"])][1::2]
        results.append(info + splits)
        if idx % 100 == 0:
           print(idx)

        if idx % 1000 == 0:
            mks = [[t.text.strip() for t in row.findAll(["th", "td"])][0] for row in rows[1:]]
            cols = [row.text.strip() for row in rows2.findAll(["th", "td"])][::2]
            pd.DataFrame(results, columns = cols + mks).to_csv(f"raw_data/chicago/chi{yr}.csv")

      except Exception as e:
          print('error', e)
          continue

    mks = [[t.text.strip() for t in row.findAll(["th", "td"])][0] for row in rows[1:]]
    cols = [row.text.strip() for row in rows2.findAll(["th", "td"])][::2]
    pd.DataFrame(results, columns = cols + mks).to_csv(f"raw_data/chicago/chi{yr}.csv")
    return

##### > CHI ####

if __name__ == "__main__":
    # full_year_data = scrape_boston_data("25")


    for year in [23, 19]:
      try:
        start = time.time()
        print("s", start)
        id_set = set(pd.read_csv(f"raw_data/nyc/nyc_names{year}.csv")["runnerId"])
        ids_left = id_set - set(pd.read_csv(f"raw_data/nyc/nyc_times{year}.csv")["id"])
        init_dict = {id: None for id in ids_left}
        print(len(ids_left))
        adding_dict = loop_attempts(init_dict, n=10, year=year)
        total = time.time() - start
        print('t', total)
      except Exception as e:
          print('error', e)
          continue


    # chi_conf = [(23, 61), (22, 52), (21, 34), (19, 56)]
    # for yr, page_num in chi_conf:
    #     start = time.time()
    #     print("start", yr, start)
    #     get_links_chi(yr=yr, pages=page_num)
    #     total = time.time() - start
    #     print('end', yr, total)

    # scrape_chi_data(23)

    pass
