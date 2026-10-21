import csv
import re
import pymupdf

HEADER_RE = re.compile(
    r"(?:"
        r"(?P<kind1>FROM|TO):\s*"
        r"(?P<city1>.+?)\s*,\s*"
        r"(?P<country1>.+?)\s+"
        r"(?P<iata1>[A-Z]{3})"
    r"|"
        r"(?P<city2>.+?)\s*,\s*"
        r"(?P<country2>.+?)\s+"
        r"(?P<iata2>[A-Z]{3})\s*"
        r"(?P<kind2>FROM|TO):"
    r")"
)

ROW_RE = re.compile(
    r"^(?P<valid_from>\d{2} \w{3})\s*-\s*(?P<valid_to>\d{2} \w{3})\s+"
    r"(?P<days>[\d ]+?)\s+"
    r"(?P<dep>\d{2}:\d{2})\s+"
    r"(?P<arr>\d{2}:\d{2})(?P<next>\+1)?\s+"
    r"(?P<flight>[A-Z0-9]{2}\d+)\*?\s+"
    r"(?P<aircraft>[A-Z0-9]{3})\s+"
    r"(?P<travel>\d+H\d*M?)"
)

OPERATED_RE = re.compile(r"^Operated by:\s*(?P<op>.+?)\s*$")

FIELDS = [
    "page", "column", "raw_line",
    "from_city", "from_country", "from_iata",
    "to_city", "to_country", "to_iata",
    "valid_from", "valid_to", "days",
    "dep_time", "arr_time", "arr_next_day",
    "flight", "aircraft", "travel_time",
    "operated_by",
]


def split_page_text(page) -> tuple[str, str]:
    words = page.get_text("words")
    if not words:
        return "", ""

    mid_x = page.rect.width / 2
    left_words = [w for w in words if w[0] < mid_x]
    right_words = [w for w in words if w[0] >= mid_x]

    def join_lines(ws):
        if not ws:
            return ""
        ws = sorted(ws, key=lambda w: (round(w[1], 1), w[0]))
        lines = []
        current_top = None
        current_line = []
        for w in ws:
            top = w[1]
            if current_top is None or abs(top - current_top) <= 5:
                current_line.append(w)
                if current_top is None:
                    current_top = top
            else:
                current_line.sort(key=lambda x: x[0])
                lines.append(" ".join(x[4] for x in current_line))
                current_line = [w]
                current_top = top
        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append(" ".join(x[4] for x in current_line))
        return "\n".join(lines)

    return join_lines(left_words), join_lines(right_words)


def parse_half(text: str, page_no: int, column: str) -> list[dict]:
    rows: list[dict] = []
    from_ctx = None
    to_ctx = None
    pending = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = HEADER_RE.search(line)
        if m:
            if m.group("kind1"):
                kind = m.group("kind1")
                city = m.group("city1")
                country = m.group("country1")
                iata = m.group("iata1")
            else:
                kind = m.group("kind2")
                city = m.group("city2")
                country = m.group("country2")
                iata = m.group("iata2")
            ctx = {"city": city.strip(), "country": country.strip(), "iata": iata}
            if kind == "FROM":
                from_ctx = ctx
            else:
                to_ctx = ctx
            continue

        m = OPERATED_RE.match(line)
        if m and pending is not None:
            pending["operated_by"] = m.group("op")
            continue

        m = ROW_RE.match(line)
        if m and from_ctx and to_ctx:
            d = m.groupdict()
            pending = {
                "page": page_no,
                "column": column,
                "raw_line": line,
                "from_city": from_ctx["city"],
                "from_country": from_ctx["country"],
                "from_iata": from_ctx["iata"],
                "to_city": to_ctx["city"],
                "to_country": to_ctx["country"],
                "to_iata": to_ctx["iata"],
                "valid_from": d["valid_from"],
                "valid_to": d["valid_to"],
                "days": d["days"].replace(" ", ""),
                "dep_time": d["dep"],
                "arr_time": d["arr"],
                "arr_next_day": bool(d["next"]),
                "flight": d["flight"],
                "aircraft": d["aircraft"],
                "travel_time": d["travel"],
                "operated_by": "",
            }
            rows.append(pending)
            continue

    return rows


def parse_pdf_to_csv(pdf_path: str, out_csv: str) -> tuple[int, int]:
    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count
    total_rows = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for page_no, page in enumerate(doc, start=1):
            left_text, right_text = split_page_text(page)
            for col, text in (("left", left_text), ("right", right_text)):
                for row in parse_half(text, page_no, col):
                    writer.writerow(row)
                    total_rows += 1

    doc.close()
    return total_pages, total_rows


if __name__ == "__main__":
    pdf_path = "../Data/Skyteam_Timetable.pdf"
    out_csv = "flights_flat.csv"
    pages, rows = parse_pdf_to_csv(pdf_path, out_csv)
    print(f"\nСтраниц обработано: {pages}")
    print(f"Строк рейсов:       {rows}")
    print(f"Файл:               {out_csv}")