import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import json
from lxml import etree
import re

Data = Path("../Data")
N = 1000

def peek_text(path, n=2000):
    with open(path, encoding="utf-8") as f:
        print(f.read(n))

def describe(df, name):
    print(f"\n==== {name} ====")
    print(df.shape)
    print(df.columns.tolist())
    print(df.head(3))
    df.info()



def load_boarding():
    return pd.read_csv(Data / "BoardingData.csv", nrows=None, sep=";", dtype=str, keep_default_na=False)

def load_ff_prof():
    with open(Data / "FrequentFlyerForum-Profiles.json", encoding="utf-8") as f:
        data = json.load(f)

        profiles = data['Forum Profiles']
        df = pd.json_normalize(profiles, sep='_', max_level=0)
        df = df.reset_index().rename(columns={"index": "ProfileID"})
        return df

def ff_flights(profiles):
    """Одна строка - это один рейс одного профиля"""
    df = profiles[["ProfileID", "NickName", "Registered Flights"]].explode("Registered Flights")
    df = df.reset_index(drop=True)
    flat = pd.json_normalize(df["Registered Flights"], sep="_")
    df = pd.concat([df.drop(columns='Registered Flights'), flat], axis=1)

    return df

def ff_documents(profiles):
    """Одна строка - это один документ одного профиля"""
    df = profiles[["ProfileID", "NickName", "Travel Documents"]].explode("Travel Documents")
    df = df.reset_index(drop=True)
    flat = pd.json_normalize(df["Travel Documents"], sep="_")
    df = pd.concat([df.drop(columns="Travel Documents"), flat], axis=1)
    return df


def ff_loyality(profiles):
    """Одна строка - это одна программа лояльности одного профиля"""
    df = profiles[["ProfileID", "NickName", "Loyality Programm"]].explode("Loyality Programm")
    df = df.reset_index(drop=True)
    flat = pd.json_normalize(df["Loyality Programm"], sep="_")
    df = pd.concat([df.drop(columns="Loyality Programm"), flat], axis=1)
    return df


def ff_realname(profiles):
    flat = pd.json_normalize(profiles["Real Name"], sep="_")
    flat.columns = ["RealName_" + c for c in flat.columns]   # префикс против коллизий
    return pd.concat([profiles[["ProfileID", "NickName"]].reset_index(drop=True),
                      flat.reset_index(drop=True)], axis=1)

def load_pointz(n=N):
    tree = etree.parse(Data / "PointzAggregator-AirlinesData.xml")
    root = tree.getroot()

    rows = []
    for user in root.findall("user"):
        uid = user.get("uid")

        name = user.find("name")
        first = name.get("first") if name is not None else None
        last = name.get("last")  if name is not None else None

        for card in user.findall("cards/card"):
            card_number = card.get("number")
            bonus = card.findtext("bonusprogramm")

            for act in card.findall("activities/activity"):
                rows.append({
                    "uid":           uid,
                    "first":         first,
                    "last":          last,
                    "card_number":   card_number,
                    "bonusprogramm": bonus,
                    "activity_type": act.get("type"),
                    "Code":          act.findtext("Code"),
                    "Date":          act.findtext("Date"),
                    "Departure":     act.findtext("Departure"),
                    "Arrival":       act.findtext("Arrival"),
                    "Fare":          act.findtext("Fare"),
                })

    return pd.DataFrame(rows)


def load_sirena(n=None):
    colspecs = [
        (0, 60),  # PaxName
        (60, 72),  # PaxBirthDate
        (72, 84),  # DepartDate
        (84, 96),  # DepartTime
        (96, 108),  # ArrivalDate
        (108, 120),  # ArrivalTime
        (120, 126),  # FlightCode
        (126, 130),  # Sh
        (130, 138),  # From
        (138, 144),  # Dest
        (144, 150),  # Code
        (150, 168),  # ETicket
        (168, 180),  # TravelDoc
        (180, 186),  # Seat
        (186, 192),  # Meal
        (192, 198),  # TrvCls
        (198, 204),  # Fare
        (204, 216),  # Baggage
        (216, 276),  # PaxAdditionalInfo
        (276, None),  # AgentInfo
    ]
    names = [
        "PaxName", "PaxBirthDate", "DepartDate", "DepartTime",
        "ArrivalDate", "ArrivalTime", "FlightCode", "Sh", "From",
        "Dest", "Code", "ETicket", "TravelDoc", "Seat", "Meal",
        "TrvCls", "Fare", "Baggage", "PaxAdditionalInfo", "AgentInfo",
    ]
    return pd.read_fwf(
        Data / "Sirena-export-fixed.tab",
        colspecs=colspecs,
        names=names,
        encoding="utf-8",
        skiprows=1,
        nrows=n,
        dtype=str,
    )



def parse_ff(s):
    if pd.isna(s):
        return None, None
    m = re.search(r"FF#([A-Z]{2})\s*(\d+)", s)
    if m:
        return m.group(1), m.group(2)
    return None, None




def load_skyteam():
    from yaml import CSafeLoader as Loader
    with open(Data / "SkyTeam-Exchange.yaml", encoding="utf-8") as f:
        data = yaml.load(f, Loader=Loader)

    rows = []
    for date, flights in data.items():
        for flight_code, info in flights.items():
            frm = info.get("FROM"); to = info.get("TO"); st = info.get("STATUS")
            for ff_key, ff_val in (info.get("FF") or {}).items():
                program, _, number = ff_key.partition(" ")
                rows.append((
                    date, flight_code, frm, to, st,
                    program, number,
                    ff_val.get("CLASS"), ff_val.get("FARE"),
                ))
    return pd.DataFrame(rows, columns=[
        "Date","FlightCode","From","To","Status",
        "FF_Program","FF_Number","Class","Fare"])

def parse_boarding_pass(path):
    df = pd.read_excel(path, header=None, dtype=str)
    if df.shape[0] < 13:
        return None   # пропускаем «поломанные» файлы
    return {
        "SourceFile":       path.name,
        "Sequence":         df.iloc[0, 7],
        "PaxTitle":         df.iloc[2, 0],
        "PaxName":          df.iloc[2, 1],
        "Class":            df.iloc[2, 7],
        "FlightCode":       df.iloc[4, 0],
        "FromCity":         df.iloc[4, 3],
        "ToCity":           df.iloc[4, 7],
        "FromAirport":      df.iloc[6, 3],
        "ToAirport":        df.iloc[6, 7],
        "Date":             df.iloc[8, 0],
        "Time":             df.iloc[8, 2],
        "CodeshareInfo":    df.iloc[8, 4],
        "Seat":             df.iloc[10, 7],
        "PNR":              df.iloc[12, 1],
        "TicketNumber":     df.iloc[12, 4],
    }




if __name__ == "__main__":

    # print("======= FILE BoardingData.csv " + "=" * 60)
    # boarding = load_boarding()
    # print(boarding.columns.tolist())
    # print(boarding.head(3).T)
    #
    # print("\n======= FILE FrequentFlyerForum-Profiles.json " + "=" * 40)
    # profiles = load_ff_prof()
    #
    # flights = ff_flights(profiles)
    # print(flights.columns.tolist())
    # print(flights.head(3).T)
    #
    # loyal = ff_loyality(profiles)
    # print(loyal.columns.tolist())
    # print(loyal.head(3).T)
    #
    # realname = ff_realname(profiles)
    # print(realname.columns.tolist())
    # print(realname.head(3).T)
    #
    # # docs = ff_documents(profiles)
    # # print(docs.columns.tolist())
    # # print(docs.head(3).T)
    # # print("Passports notna:", docs["Passports"].notna().sum())
    #
    # print("\n======= FILE PointzAggregator-AirlinesData.xml " + "=" * 40)
    # pointz = load_pointz()
    # print(pointz.columns.tolist())
    # print(pointz.head(3).T)
    #
    # print("\n======= FILE Sirena-export-fixed.tab " + "=" * 40)
    # sirena = load_sirena()
    # print(sirena.columns.tolist())
    # print(sirena.head(3).T)
    # sirena[["FF_Program", "FF_Number"]] = sirena["PaxAdditionalInfo"].apply(
    #     lambda x: pd.Series(parse_ff(x))
    # )
    # print(sirena[["PaxAdditionalInfo", "FF_Program", "FF_Number"]].head(10))
    #
    # print("\n======= FILE SkyTeam-Exchange.yaml " + "=" * 40)
    # sky = pd.read_pickle("skyteam.pkl")
    # print(sky.columns.tolist())
    # print(sky.head(3).T)
    #
    # print("\n======= FOLDER YourBoardingPassDotAero " + "=" * 40)
    # folder = Data / "YourBoardingPassDotAero"
    # files = sorted(folder.glob("*.xlsx"))
    # rows = []
    # for f in files:
    #     rec = parse_boarding_pass(f)
    #     if rec is not None:
    #         rows.append(rec)
    # bp = pd.DataFrame(rows)
    # print(bp.shape)
    # print(bp.columns.tolist())
    # print(bp.head(3).T)

    print(sum(1 for _ in open(Data / "BoardingData.csv", encoding="utf-8")) - 1)

