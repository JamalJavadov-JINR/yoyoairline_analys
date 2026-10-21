import re
import pandas as pd
from pathlib import Path
from Reformat import (load_skyteam, load_boarding, load_sirena,
                      load_ff_prof, load_pointz, parse_ff,
                      ff_loyality, ff_flights, ff_realname, ff_documents,
                      parse_boarding_pass)

Data = Path("../Data/")

DATE_FMT = "%Y-%m-%d"
TIME_FMT = "%H:%M"

"""
Единая схема событий:
source, first_name, last_name, middle_name, birth_date, sex,
doc_number, ticket_number, flight_number, flight_datetime,
from_iata, to_iata, from_city, to_city, pnr,
fare_class, fare_code, ff_program, ff_number, seat, meal, baggage,
profile_id (только forum)
"""


# ============================================================
#  1. BOARDING
# ============================================================

def to_boarding_events(boarding: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "first_name":    boarding["PassengerFirstName"].str.strip(),
        "last_name":     boarding["PassengerLastName"].str.strip(),
        "middle_name":   boarding["PassengerSecondName"].str.strip(),

        "birth_date": pd.to_datetime(
            boarding["PassengerBirthDate"], format="%m/%d/%Y", errors="coerce",
        ),

        "sex": boarding["PassengerSex"].map({"Male": "M", "Female": "F"}),

        "doc_number": boarding["PassengerDocument"].str.replace(" ", "", regex=False),
        "ticket_number": boarding["TicketNumber"].astype("string")
                           .replace({"Not presented": pd.NA}),

        "flight_number": boarding["FlightNumber"].str.strip(),

        "flight_date": pd.to_datetime(boarding["FlightDate"], errors="coerce"),
        "flight_time": pd.to_datetime(
            boarding["FlightTime"], format="%H:%M", errors="coerce",
        ).dt.strftime(TIME_FMT),

        "to_city": boarding["Destination"].str.strip(),
        "baggage": boarding["Baggage"],

        "pnr": boarding["BookingCode"].str.strip().replace("Not presented", pd.NA),
        "source": "boarding",
    })

    df["flight_datetime"] = pd.to_datetime(
        df["flight_date"].dt.strftime(DATE_FMT) + " " + df["flight_time"],
        format=f"{DATE_FMT} {TIME_FMT}", errors="coerce",
    )
    df = df.drop(columns=["flight_date", "flight_time"])
    return df


# ============================================================
#  2. SIRENA (транслитерация ГОСТ)
# ============================================================

GOST_52535_1_MAP = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E',
    'Ж': 'ZH', 'З': 'Z', 'И': 'I', 'Й': 'I', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'KH', 'Ц': 'TC', 'Ч': 'CH', 'Ш': 'SH', 'Щ': 'SHCH',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'IU', 'Я': 'IA',
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'i', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'tc', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'iu', 'я': 'ia',
}


def _transliterate_gost(text: str) -> str:
    return "".join(GOST_52535_1_MAP.get(ch, ch) for ch in text)


def split_pax_name(s):
    parts = s.split()
    last = _transliterate_gost(parts[0]) if len(parts) > 0 else None
    first = _transliterate_gost(parts[1]) if len(parts) > 1 else None
    middle = _transliterate_gost(parts[2]) if len(parts) > 2 else None
    return {"last_name": last, "first_name": first, "middle_name": middle}


def to_sirena_events(sirena: pd.DataFrame) -> pd.DataFrame:
    names = sirena["PaxName"].apply(split_pax_name).apply(pd.Series)

    depart_date = pd.to_datetime(sirena["DepartDate"], errors="coerce")
    depart_time = pd.to_datetime(sirena["DepartTime"], format="%H:%M", errors="coerce")
    flight_datetime = pd.to_datetime(
        depart_date.dt.strftime(DATE_FMT) + " " + depart_time.dt.strftime(TIME_FMT),
        format=f"{DATE_FMT} {TIME_FMT}", errors="coerce",
    )

    ff = sirena["PaxAdditionalInfo"].apply(lambda x: pd.Series(parse_ff(x)))
    ff.columns = ["ff_program", "ff_number"]

    return pd.DataFrame({
        "first_name":  names["first_name"],
        "last_name":   names["last_name"],
        "middle_name": names["middle_name"],

        "birth_date":  pd.to_datetime(sirena["PaxBirthDate"], errors="coerce"),
        "sex": pd.NA,

        "doc_number":  sirena["TravelDoc"].str.replace(" ", "", regex=False),
        "ticket_number": sirena["ETicket"].astype("string"),

        "flight_number": sirena["FlightCode"].str.strip(),
        "flight_datetime": flight_datetime,

        "from_iata": sirena["From"].str.strip(),
        "to_iata":   sirena["Dest"].str.strip(),
        "from_city": pd.NA,
        "to_city":   pd.NA,

        "fare_class": sirena["TrvCls"].str.strip(),
        "fare_code":  sirena["Fare"].str.strip(),

        "ff_program": ff["ff_program"],
        "ff_number":  ff["ff_number"],

        "seat": sirena["Seat"].astype("string").str.strip().replace("", pd.NA),
        "meal": sirena["Meal"].astype("string").str.strip().replace("", pd.NA),
        "baggage": sirena["Baggage"].str.strip().replace("", pd.NA),

        "pnr":    sirena["Code"].str.strip(),
        "source": "sirena",
    })


# ============================================================
#  3. POINTZ
# ============================================================

def parse_pointz_card(card_number):
    m = re.match(r"([A-Z]{2})\s*(\d+)", card_number.strip())
    if m:
        return m.group(1), m.group(2)
    return None, None


def to_pointz_events(pointz: pd.DataFrame) -> pd.DataFrame:
    pointz = pointz[pointz["activity_type"] == "Flight"].copy()
    ff = pointz["card_number"].apply(
        lambda x: pd.Series(parse_pointz_card(x), index=["ff_program", "ff_number"])
    )
    flight_date = pd.to_datetime(pointz["Date"], errors="coerce")

    return pd.DataFrame({
        "first_name":  pointz["first"].str.strip().str.upper(),
        "last_name":   pointz["last"].str.strip().str.upper(),
        "middle_name": pd.NA,
        "birth_date":  pd.NaT,
        "sex":         pd.NA,
        "doc_number":  pd.NA,
        "ticket_number": pd.NA,

        "flight_number": pointz["Code"].str.strip(),
        "flight_datetime": flight_date,   # времени нет -> 00:00
        "from_iata": pointz["Departure"].str.strip().str.upper(),
        "to_iata":   pointz["Arrival"].str.strip().str.upper(),
        "from_city": pd.NA,
        "to_city":   pd.NA,

        "fare_class": pd.NA,
        "fare_code":  pointz["Fare"].str.strip(),

        "ff_program": ff["ff_program"],
        "ff_number":  ff["ff_number"],

        "seat": pd.NA, "meal": pd.NA, "baggage": pd.NA,
        "pnr":  pd.NA,
        "pointz_uid": pointz["uid"],
        "source": "pointz",
    })


# ============================================================
#  4. SKYTEAM
# ============================================================

def to_skyteam_events(skyteam: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "first_name": pd.NA, "last_name": pd.NA, "middle_name": pd.NA,
        "birth_date": pd.NaT, "sex": pd.NA,
        "doc_number": pd.NA, "ticket_number": pd.NA,

        "flight_number":   skyteam["FlightCode"].str.strip(),
        "flight_datetime": pd.to_datetime(skyteam["Date"], errors="coerce"),
        "from_iata": skyteam["From"].str.strip().str.upper(),
        "to_iata":   skyteam["To"].str.strip().str.upper(),
        "from_city": pd.NA, "to_city": pd.NA,

        "fare_class": skyteam["Class"].str.strip(),
        "fare_code":  skyteam["Fare"].str.strip(),

        "ff_program": skyteam["FF_Program"].str.strip(),
        "ff_number":  skyteam["FF_Number"].astype("string"),

        "seat": pd.NA, "meal": pd.NA, "baggage": pd.NA,
        "pnr":  pd.NA,
        "source": "skyteam",
    })


# ============================================================
#  5. FORUM
# ============================================================

def to_forum_events(profiles: pd.DataFrame) -> pd.DataFrame:
    """Один рейс форума = одна строка. FF-карты сюда не подмешиваем (см. forum_loyalty_table)."""
    flights = ff_flights(profiles)
    realname = ff_realname(profiles)

    df = flights.merge(
        realname[["ProfileID", "RealName_First Name", "RealName_Last Name"]],
        on="ProfileID", how="left",
    )
    return pd.DataFrame({
        "first_name": df["RealName_First Name"].str.strip(),
        "last_name":  df["RealName_Last Name"].str.strip(),
        "middle_name": pd.NA,
        "birth_date": pd.NaT, "sex": pd.NA,
        "doc_number": pd.NA, "ticket_number": pd.NA,

        "flight_number": df["Flight"].str.strip(),
        "flight_datetime": pd.to_datetime(df["Date"], errors="coerce"),
        "from_iata": df["Departure_Airport"].str.strip().str.upper(),
        "to_iata":   df["Arrival_Airport"].str.strip().str.upper(),
        "from_city": df["Departure_City"].str.strip(),
        "to_city":   df["Arrival_City"].str.strip(),

        "fare_class": pd.NA, "fare_code": pd.NA,
        "ff_program": pd.NA, "ff_number": pd.NA,   # -> forum_loyalty_table

        "seat": pd.NA, "meal": pd.NA, "baggage": pd.NA,
        "pnr":  pd.NA,
        "source": "forum",

        "profile_id": df["ProfileID"],
    })


def forum_loyalty_table(profiles: pd.DataFrame) -> pd.DataFrame:
    loyal = ff_loyality(profiles)
    return pd.DataFrame({
        "profile_id": loyal["ProfileID"],
        "ff_program": loyal["programm"].str.strip().str.upper(),
        "ff_number":  loyal["Number"].astype("string").str.replace(r"\s+", "", regex=True),
    })


# ============================================================
#  6. BOARDINGPASS (xlsx)
# ============================================================

def boardingpass_events_from_folder(folder: Path) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.xlsx"))
    rows = []
    for f in files:
        rec = parse_boarding_pass(f)
        if rec is not None:
            rows.append(rec)
    return pd.DataFrame(rows)


SURNAME_SUFFIXES = (
    "OV", "EV", "IN", "YN",
    "OVA", "EVA", "INA", "YNA",
    "SKIY", "SKY", "SKAYA",
)


def _normalize_for_match(s: str) -> str:
    return s.upper().replace("'", "").replace("-", "")


def _looks_like_surname(word: str) -> bool:
    return word.upper().endswith(SURNAME_SUFFIXES)


def _split_pax_name_bp(name: str, known: set) -> tuple:
    if not isinstance(name, str):
        return pd.NA, pd.NA, pd.NA

    parts = name.strip().split()
    if not parts:
        return pd.NA, pd.NA, pd.NA

    if len(parts) == 1:
        return pd.NA, parts[0], pd.NA

    if len(parts) >= 3:
        # находим однобуквенный токен — это инициал отчества
        initial_idx = next((i for i, p in enumerate(parts) if len(p) == 1), None)
        if initial_idx is not None:
            middle = parts[initial_idx]
            rest = parts[:initial_idx] + parts[initial_idx + 1:]
        else:
            # на случай, если инициалов несколько слов или их нет вообще —
            # считаем middle всё, что между первым и последним
            middle = " ".join(parts[1:-1]) if len(parts) > 2 else pd.NA
            rest = [parts[0], parts[-1]]

        if len(rest) == 2:
            first, last = _resolve_two(rest[0], rest[1], known)
        else:
            first, last = rest[0], rest[-1]
        return first, last, middle

    # 2 слова
    w1, w2 = parts[0], parts[1]
    first, last = _resolve_two(w1, w2, known)
    return first, last, pd.NA


def _resolve_two(w1: str, w2: str, known: set) -> tuple:
    in1 = _normalize_for_match(w1) in known
    in2 = _normalize_for_match(w2) in known
    if in1 and not in2:
        return w1, w2
    if in2 and not in1:
        return w2, w1

    s1 = _looks_like_surname(w1)
    s2 = _looks_like_surname(w2)
    if s1 and not s2:
        return w2, w1
    if s2 and not s1:
        return w1, w2
    if s1:
        return w2, w1
    if s2:
        return w1, w2

    return w1, w2

def to_boardingpass_events(bp: pd.DataFrame, known_first_names: set) -> pd.DataFrame:
    parsed = bp["PaxName"].apply(lambda s: _split_pax_name_bp(s, known_first_names))
    first_name = parsed.str[0]
    last_name = parsed.str[1]
    middle_name = parsed.str[2]

    flight_date = pd.to_datetime(bp["Date"], errors="coerce")
    flight_time = pd.to_datetime(bp["Time"], format="%H:%M", errors="coerce").dt.strftime(TIME_FMT)
    flight_datetime = pd.to_datetime(
        flight_date.dt.strftime(DATE_FMT) + " " + flight_time,
        format=f"{DATE_FMT} {TIME_FMT}", errors="coerce",
    )

    return pd.DataFrame({
        "first_name": first_name, "last_name": last_name, "middle_name": middle_name,
        "birth_date": pd.NaT, "sex": pd.NA,
        "doc_number": pd.NA,
        "ticket_number": bp["TicketNumber"].astype("string").str.strip(),

        "flight_number":   bp["FlightCode"].astype("string").str.strip(),
        "flight_datetime": flight_datetime,
        "from_iata": bp["FromAirport"].astype("string").str.strip().str.upper(),
        "to_iata":   bp["ToAirport"].astype("string").str.strip().str.upper(),
        "from_city": bp["FromCity"].astype("string").str.strip(),
        "to_city":   bp["ToCity"].astype("string").str.strip(),

        "fare_class": bp["Class"].astype("string").str.strip(),
        "fare_code":  pd.NA,
        "ff_program": pd.NA, "ff_number": pd.NA,

        "seat": bp["Seat"].astype("string").str.strip().replace("", pd.NA),
        "meal": pd.NA, "baggage": pd.NA,

        "pnr": bp["PNR"].astype("string").str.strip(),
        "source": "boardingpass",

        "title": bp["PaxTitle"].astype("string").str.strip(),
    })


# ============================================================
#  7. СБОРКА EVENTS
# ============================================================

def build_events() -> pd.DataFrame:
    """Загружает все шесть источников и приводит их к общему контракту."""

    boarding = load_boarding()
    sirena = load_sirena()
    pointz = load_pointz()
    skyteam = pd.read_pickle("skyteam.pkl")
    profiles = load_ff_prof()
    bp_folder = Data / "YourBoardingPassDotAero"
    bp = boardingpass_events_from_folder(bp_folder)

    boarding_events = to_boarding_events(boarding)
    sirena_events = to_sirena_events(sirena)

    known_first_names = set(
        pd.concat([boarding_events["first_name"], sirena_events["first_name"]])
        .dropna().astype("string").apply(_normalize_for_match)
    )
    print(f"known_first_names: {len(known_first_names)}")

    pointz_events = to_pointz_events(pointz)
    skyteam_events = to_skyteam_events(skyteam)
    forum_events = to_forum_events(profiles)
    bp_events = to_boardingpass_events(bp, known_first_names)

    events = pd.concat(
        [boarding_events, sirena_events, pointz_events,
         skyteam_events, forum_events, bp_events],
        ignore_index=True, sort=False,
    )
    events["ff_number"] = events["ff_number"].astype("string")

    print("shape:", events.shape)
    print("\nsource value_counts:")
    print(events["source"].value_counts(dropna=False))
    print("\nNaN share per column:")
    print(events.isna().mean().round(2))

    return events

# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    events = build_events()

    # диагностика: проверяем, что имена boardingpass разобраны правильно
    bp_check = events[events["source"] == "boardingpass"]
    print("\nПроверка имён boardingpass (первые 10):")
    print(bp_check[["first_name", "last_name", "middle_name"]].head(10))

    events.to_parquet("Full_combined_dataset.parquet", index=False)
    print("Saved to Full_combined_dataset.parquet")


