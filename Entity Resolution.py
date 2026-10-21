import re
import pandas as pd
from Reformat import load_ff_prof
from pipeline import forum_loyalty_table
import difflib


def build_person_id(events: pd.DataFrame, loyalty: pd.DataFrame = None) -> pd.DataFrame:
    events = events.copy().reset_index(drop=True)   # позиции строк = индексы в union-find

    events["doc_key"] = events["doc_number"].astype("string")
    events["ticket_key"] = events["ticket_number"].astype("string")
    events["ff_key"] = (
            events["ff_program"].astype("string").str.strip() + "_"
            + events["ff_number"].astype("string").str.replace(r"\s+", "", regex=True)
    ).where(events["ff_program"].notna() & events["ff_number"].notna())

    n = len(events)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # --- ключи, за которыми стоят разные люди ---
    uid_per_ff = events.groupby("ff_key")["pointz_uid"].nunique()
    bad_ff = set(uid_per_ff[uid_per_ff > 1].index)

    nb = events.groupby("doc_key")["birth_date"].nunique()
    bad_doc = set(nb[nb > 1].index)

    # конфликтный паспорт различаем по дате рождения; без даты рождения строка по паспорту не клеится
    m = events["doc_key"].isin(bad_doc)
    events.loc[m, "doc_key"] = (
            events.loc[m, "doc_key"] + "_" + events.loc[m, "birth_date"].dt.strftime("%Y%m%d")
    )

    pd.concat([
        pd.DataFrame({"key_type": "ff_key", "value": sorted(bad_ff)}),
        pd.DataFrame({"key_type": "doc_number", "value": sorted(bad_doc)}),
    ]).to_csv("shared_keys.csv", index=False)

    # 1) сильные ключи: паспорт, билет, FF-карта
    for key in ["doc_key", "ticket_key", "ff_key"]:
        for val, idx in events.groupby(key).indices.items():
            if key == "ff_key" and val in bad_ff:
                continue
            first = idx[0]
            for j in idx[1:]:
                union(first, j)

    # 1b) один пользователь Pointz = один человек
    if "pointz_uid" in events.columns:
        for idx in events.groupby("pointz_uid").indices.values():
            first = idx[0]
            for j in idx[1:]:
                union(first, j)

    # 2) форум: все рейсы одного профиля -> один человек
    prof_rows = {
        int(pid): idx
        for pid, idx in events.groupby("profile_id").indices.items()
    }
    for idx in prof_rows.values():
        for j in idx[1:]:
            union(idx[0], j)

    # 3) профиль -> его карты -> строки с такой же картой
    if loyalty is not None:
        ff_rows = events.groupby("ff_key").indices
        lo = loyalty.dropna(subset=["ff_program", "ff_number"]).copy()
        lo["ff_key"] = (
                lo["ff_program"].astype("string").str.strip() + "_"
                + lo["ff_number"].astype("string").str.replace(r"\s+", "", regex=True)
        )
        for pid, key in zip(lo["profile_id"], lo["ff_key"]):
            pid = int(pid)
            if key in bad_ff:
                continue
            if pid in prof_rows and key in ff_rows:
                union(prof_rows[pid][0], ff_rows[key][0])




    roots = [find(i) for i in range(n)]
    events["person_id"] = pd.factorize(pd.Series(roots))[0]
    events = events.drop(columns=["doc_key", "ticket_key", "ff_key"])
    return events


def canon(s):
    s = str(s).upper().replace("'", "")
    s = s.replace("X", "KS").replace("IU", "YU").replace("IA", "YA")
    s = s.replace("TC", "TS").replace("KH", "H").replace("Y", "I")
    return re.sub(r"I+", "I", s)


def report_person_id(events: pd.DataFrame) -> None:
    print("\nУникальных person_id:", events["person_id"].nunique())

    sizes = events.groupby("person_id").size()
    print("\nРаспределение размера компонент:")
    print(sizes.describe())
    print("\nТоп-15 крупнейших компонент:")
    print(sizes.sort_values(ascending=False).head(15))

    e = events.dropna(subset=["last_name"]).copy()
    e["last_canon"] = e["last_name"].map(canon)
    n_canon = e.groupby("person_id")["last_canon"].nunique()
    n_raw = e.groupby("person_id")["last_name"].nunique()
    print("\nКомпонент с >1 фамилией (сырой):", (n_raw > 1).sum())
    print("Компонент с >1 фамилией (после canon):", (n_canon > 1).sum())

    # форум
    f = events[events["source"] == "forum"]
    print(f"\nФорум: строк {len(f)}, уникальных person_id {f['person_id'].nunique()}, "
          f"профилей {f['profile_id'].nunique()}")

    src_per_person = events.groupby("person_id")["source"].agg(lambda s: set(s))
    forum_persons = src_per_person[src_per_person.map(lambda s: "forum" in s)]
    linked = forum_persons.map(lambda s: len(s - {"forum"}) > 0).sum()
    print(f"Людей с форумом: {len(forum_persons)}, из них связаны с другим источником: {linked}")



def min_sim(names):
    names = list(names)
    if len(names) < 2:
        return 1.0
    return min(difflib.SequenceMatcher(None, a, b).ratio()
               for i, a in enumerate(names) for b in names[i + 1:])

def shared_keys(events):
    e = events.copy()
    e["ff_key"] = (e["ff_program"].astype("string") + "_"
                   + e["ff_number"].astype("string")).where(e["ff_number"].notna())
    e["last_canon"] = e["last_name"].map(lambda x: canon(x) if pd.notna(x) else None)
    out = []
    for key in ["ff_key", "doc_number", "ticket_number"]:
        g = e.dropna(subset=[key]).groupby(key).agg(
            n_rec=(key, "size"),
            n_uid=("pointz_uid", "nunique"),
            n_last=("last_canon", "nunique"),
            n_birth=("birth_date", lambda s: s.dropna().nunique()),
        )
        bad = g[(g.n_uid > 1) | (g.n_last > 1) | (g.n_birth > 1)].reset_index()
        bad = bad.rename(columns={key: "value"})
        bad["key_type"] = key
        out.append(bad)
    return pd.concat(out, ignore_index=True)




if __name__ == "__main__":
    events = pd.read_parquet("../Notes/Лишка/Full_combined_dataset.parquet")
    loyalty = forum_loyalty_table(load_ff_prof())

    events = build_person_id(events, loyalty)
    events.to_parquet("Full_combined_dataset_with_person_id.parquet", index=False)
    report_person_id(events)
    #
    # e = events.dropna(subset=["last_name"]).copy()
    # e["last_canon"] = e["last_name"].map(canon)
    # g = e.groupby("person_id").agg(
    #     n_last=("last_canon", "nunique"),
    #     names=("last_canon", lambda s: sorted(set(s))),
    #     sources=("source", lambda s: sorted(set(s))),
    #     n_records=("source", "size"),
    # )
    # bad = g[g["n_last"] > 1].copy()
    # bad["min_sim"] = bad["names"].map(min_sim)
    #
    # print("всего:", len(bad))
    # print("похожие (>0.8):", (bad["min_sim"] > 0.8).sum())
    # print("разные (<=0.8):", (bad["min_sim"] <= 0.8).sum())
    # print(bad.sort_values("min_sim").head(20).to_string())
    #
    # cols = ["source", "last_name", "first_name", "birth_date", "doc_number",
    #         "ticket_number", "ff_program", "ff_number", "profile_id", "flight_number", "flight_datetime"]
    # for pid in bad.sort_values("min_sim").head(3).index:
    #     print(events[events["person_id"] == pid][cols].sort_values("flight_datetime").to_string())
    #
    # d = events["doc_number"].dropna()
    # bad_doc = d[~d.str.fullmatch(r"\d{10}")]
    # print("невалидных паспортов:", len(bad_doc), "из", len(d))
    # print(bad_doc.value_counts().head(10))
    #
    # sk = shared_keys(events)
    # print(sk["key_type"].value_counts())
    # print(sk.sort_values("n_rec", ascending=False).head(20).to_string())
    #
    # d = events.dropna(subset=["doc_number"]).copy()
    # d["doc_type"] = d["doc_number"].str.len().map({10: "internal", 9: "foreign"}).fillna("other")
    # print(d["doc_type"].value_counts())
    #
    # per = d.groupby(["person_id", "doc_type"])["doc_number"].nunique().unstack(fill_value=0)
    # print("людей с >1 внутренним паспортом:", (per["internal"] > 1).sum())
    # print("людей с >1 загранпаспортом:", (per["foreign"] > 1).sum())

    ff = events.dropna(subset=["ff_program", "ff_number"]).copy()
    ff["ff_key"] = ff["ff_program"].astype("string") + "_" + ff["ff_number"].astype("string")
    print(ff.groupby("person_id")["ff_key"].nunique().value_counts().sort_index())

    ff = events.dropna(subset=["ff_program", "ff_number"]).copy()
    cards = ff.groupby(["person_id", "ff_program"])["ff_number"].nunique()
    multi = cards[cards > 1]
    print("людей с >1 картой одной программы:", multi.index.get_level_values(0).nunique())
    print(multi.value_counts())

    d = events.dropna(subset=["doc_number"]).copy()
    d["doc_type"] = d["doc_number"].str.len().map({10: "internal", 9: "foreign"}).fillna("other")
    print(d["doc_type"].value_counts())

    per = d.groupby(["person_id", "doc_type"])["doc_number"].nunique().unstack(fill_value=0)
    print("людей с >1 внутренним паспортом:", (per.get("internal", 0) > 1).sum())
    print("людей с >1 загранпаспортом:", (per.get("foreign", 0) > 1).sum())