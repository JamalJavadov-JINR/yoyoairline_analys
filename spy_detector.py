from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / "Full_combined_dataset_with_person_id.parquet"
OUT = HERE / "spy_output"


SCAN_COLS = ["last_name", "first_name", "middle_name", "birth_date",
             "doc_number", "ff_program", "ff_number"]

EVIDENCE_COLS = ["source", "last_name", "first_name", "middle_name", "birth_date",
                 "sex", "doc_number", "ticket_number", "ff_program", "ff_number",
                 "pnr", "flight_number", "flight_datetime", "from_iata", "to_iata",
                 "person_id"]


# --------------------------------------------------------------------------
# 1. Загрузка и ключи
# --------------------------------------------------------------------------
def load_scan() -> pd.DataFrame:

    df = pd.read_parquet(DATA, columns=SCAN_COLS)
    df["ff_program"] = df["ff_program"].astype("category")
    return df


def canon(s: pd.Series) -> pd.Series:

    out = s.astype(str).str.upper()

    out = out.str.replace(r"[^A-Z]", "", regex=True)
    for a, b in (("X", "KS"), ("IU", "YU"), ("IA", "YA"),
                 ("TC", "TS"), ("KH", "H")):
        out = out.str.replace(a, b, regex=False)
    out = out.str.replace("Y", "I", regex=False)
    out = out.str.replace(r"I+", "I", regex=True)
    return out.where(s.notna())


def add_keys(df: pd.DataFrame) -> pd.DataFrame:

    df["last_c"] = canon(df["last_name"])
    df["first_c"] = canon(df["first_name"])
    df["mid_init"] = df["middle_name"].astype(str).str.slice(0, 1).where(
        df["middle_name"].notna())

    df["ff_key"] = (df["ff_program"].astype(str) + " "
                    + df["ff_number"].astype(str)).where(df["ff_number"].notna())
    return df

def find_conflicts(df: pd.DataFrame, key: str) -> pd.DataFrame:

    named = df[df[key].notna() & df["last_name"].notna()]
    g = named.groupby(key, observed=True)

    stat = pd.DataFrame({
        "фамилий": g["last_c"].nunique(),
        "имён": g["first_c"].nunique(),
        "отчеств": g["mid_init"].nunique(),
        "дат_рождения": g["birth_date"].nunique(),
        "записей": g.size(),
    })
    stat["конфликт_ФИО"] = (stat[["фамилий", "имён", "отчеств"]] > 1).any(axis=1)
    stat["конфликт_даты"] = stat["дат_рождения"] > 1

    bad = stat[stat["конфликт_ФИО"] | stat["конфликт_даты"]].copy()
    bad.index.name = key

    print(f"\n  {key}: проверено значений {len(stat):,}, "
          f"конфликтных {len(bad):,}")
    if not bad.empty:
        print(bad.to_string())
    return bad


def collect_evidence(wanted: dict[str, set[str]]) -> dict[str, pd.DataFrame]:

    found: dict[str, list[pd.DataFrame]] = {k: [] for k in wanted}
    pf = pq.ParquetFile(DATA)

    for batch in pf.iter_batches(batch_size=200_000, columns=EVIDENCE_COLS):
        chunk = batch.to_pandas()
        chunk["ff_key"] = (chunk["ff_program"].astype(str) + " "
                           + chunk["ff_number"].astype(str)).where(
            chunk["ff_number"].notna())
        for key, values in wanted.items():
            hit = chunk[chunk[key].isin(values)]
            if not hit.empty:
                found[key].append(hit)

    return {k: (pd.concat(v).sort_values(["last_name", "flight_datetime"])
                if v else pd.DataFrame(columns=EVIDENCE_COLS))
            for k, v in found.items()}


def describe(records: pd.DataFrame, key: str, value: str) -> None:
    rec = records[records[key] == value]
    named = rec[rec["last_name"].notna()]

    print(f"\n{'=' * 78}\n  {key} = {value}\n{'=' * 78}")
    print(f"  записей всего: {len(rec)} | с ФИО: {len(named)} | "
          f"источники: {', '.join(sorted(rec['source'].dropna().unique()))}")

    people = (named.groupby(["last_name", "first_name"], observed=True)
                   .agg(отчества=("middle_name", lambda s: " / ".join(
                            sorted(set(s.dropna().astype(str)))) or "—"),
                        рождение=("birth_date", lambda s: " / ".join(
                            sorted({str(d)[:10] for d in s.dropna()})) or "—"),
                        записей=("source", "size"),
                        источники=("source", lambda s: ", ".join(
                            sorted(set(s.dropna().astype(str))))))
                   .reset_index())
    print("\n  Личности на одном идентификаторе:")
    print(people.to_string(index=False))

    traj = (rec.dropna(subset=["from_iata", "to_iata"])
               .assign(date=lambda d: d["flight_datetime"].dt.normalize())
               .sort_values(["date", "flight_number"])
               .drop_duplicates(["flight_number", "date", "from_iata", "to_iata"])
               .loc[:, ["last_name", "flight_number", "date",
                        "from_iata", "to_iata"]])
    if not traj.empty:
        traj["prev_to"] = traj["to_iata"].shift()
        traj["разрыв"] = (traj["prev_to"].notna()
                          & (traj["from_iata"] != traj["prev_to"]))
        print(f"\n  Перелёты (разрывов {int(traj['разрыв'].sum())} "
              f"из {len(traj)}):")
        print(traj.to_string(index=False))


def main() -> None:
    OUT.mkdir(exist_ok=True)

    print("1. Загрузка")
    df = add_keys(load_scan())
    print(f"  строк: {len(df):,} | с ФИО: {df['last_name'].notna().sum():,} | "
          f"фамилий: {df['last_c'].nunique():,}")

    print("\n2. Поиск идентификаторов, обслуживающих больше одного человека")
    conflicts = {key: find_conflicts(df, key)
                 for key in ("doc_number", "ff_key")}

    wanted = {k: set(v.index) for k, v in conflicts.items() if not v.empty}
    if not wanted:
        print("\nКонфликтов не найдено.")
        return

    del df

    print("\n3. Сбор сырых записей по находкам")
    evidence = collect_evidence(wanted)

    for key, values in wanted.items():
        for value in sorted(values):
            describe(evidence[key], key, value)

    print(f"\n4. Выгрузка в {OUT}")
    summary = []
    for key, stat in conflicts.items():
        if stat.empty:
            continue
        rows = evidence[key]
        for value in sorted(set(stat.index)):
            rec = rows[rows[key] == value]
            named = rec[rec["last_name"].notna()]
            people = sorted({f"{a} {b}" for a, b in
                             zip(named["last_name"], named["first_name"])})
            safe = str(value).replace(" ", "_").replace("/", "_")
            rec.to_csv(OUT / f"evidence_{safe}.csv", index=False,
                       encoding="utf-8-sig")
            summary.append({
                "идентификатор": key,
                "значение": value,
                "личностей": len(people),
                "дат_рождения": named["birth_date"].nunique(),
                "записей": len(rec),
                "фио": " ; ".join(people),
            })
    pd.DataFrame(summary).to_csv(OUT / "spies.csv", index=False,
                                 encoding="utf-8-sig")
    print(f"spies.csv — сводка ({len(summary)} находок)")
    print("evidence_*.csv — сырые записи по каждой находке")


if __name__ == "__main__":
    main()
