import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
import zipfile
from io import BytesIO

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NIQKI | Rainfall & Runoff Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL UI THEME
# ============================================================

st.markdown("""
<style>
.block-container { max-width: 1400px; padding-top: 2.2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(100,116,139,.18); }
section[data-testid="stSidebar"] > div { padding-top: 1.3rem; }
h1 { font-size: 2.35rem !important; font-weight: 700 !important; letter-spacing: -.025em; }
h2 { font-size: 1.55rem !important; font-weight: 650 !important; margin-top: 1.4rem !important; }
h3 { font-size: 1.15rem !important; font-weight: 650 !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: .25rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label { border-radius: 10px; padding: .42rem .65rem; transition: background .15s ease; }
.niqki-brand { padding: .2rem 0 1.2rem 0; }
.niqki-brand-mark { display:inline-flex; width:42px; height:42px; align-items:center; justify-content:center; border-radius:12px; background:rgba(14,116,144,.14); border:1px solid rgba(14,116,144,.25); font-size:1.35rem; margin-bottom:.65rem; }
.niqki-brand-title { font-size:1.15rem; font-weight:750; line-height:1.15; }
.niqki-brand-subtitle { color:#64748b; font-size:.78rem; margin-top:.25rem; }
.niqki-hero { padding:2rem 2.1rem; border-radius:18px; border:1px solid rgba(100,116,139,.20); background:linear-gradient(135deg,rgba(14,116,144,.12),rgba(30,41,59,.04)); margin-bottom:1.5rem; }
.niqki-eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:750; color:#0e7490; margin-bottom:.55rem; }
.niqki-hero-title { font-size:clamp(2rem,4vw,3.2rem); line-height:1.05; font-weight:800; letter-spacing:-.04em; margin-bottom:.8rem; }
.niqki-hero-text { max-width:820px; font-size:1.02rem; line-height:1.65; color:#64748b; }
.niqki-card { min-height:165px; padding:1.25rem; border-radius:14px; border:1px solid rgba(100,116,139,.20); background:rgba(255,255,255,.025); }
.niqki-card-icon { font-size:1.35rem; margin-bottom:.7rem; }
.niqki-card-title { font-weight:700; font-size:1.02rem; margin-bottom:.4rem; }
.niqki-card-text { color:#64748b; font-size:.9rem; line-height:1.5; }
.niqki-step { text-align:center; padding:.8rem .35rem; }
.niqki-step-number { width:34px; height:34px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:750; background:rgba(14,116,144,.12); border:1px solid rgba(14,116,144,.25); margin-bottom:.45rem; }
.niqki-step-label { font-size:.82rem; font-weight:650; }
.niqki-step-note { color:#64748b; font-size:.72rem; margin-top:.2rem; }
.niqki-footer { margin-top:2.5rem; padding-top:1rem; border-top:1px solid rgba(100,116,139,.18); color:#64748b; font-size:.76rem; }
div[data-testid="stFileUploader"] { border-radius:14px; }
.stButton > button, div[data-testid="stDownloadButton"] > button { border-radius:9px; font-weight:600; }
div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
hr { margin:1.4rem 0 !important; border-color:rgba(100,116,139,.18) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

DATE_KEYWORDS = [
    "date", "datum", "time", "zeit", "datetime",
    "date/time", "datum/zeit", "timestamp",
    "zeitstempel", "zeitangabe"
]

RAINFALL_KEYWORDS = [
    "rain", "rainfall", "precipitation", "precip",
    "niederschlag", "niederschlagsmenge",
    "regen", "regenmenge", "rain_mm",
    "precip_mm", "rain amount"
]

UNIT_KEYWORDS = [
    "mm", "mm/h", "millimeter", "millimetre"
]

MISSING_VALUES = [
    "", "-", "--", "---", "nan", "none", "null",
    "na", "n/a"
]

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "rainfall_data": None,
    "model_results": None,
    "model_parameters": None,
    "rainfall_events": None,
    "event_dry_period_hours": None,
    "file_signature": None,
    "traffic_data": None,
    "traffic_file_signature": None,
    "traffic_source_info": None,
    "pollutant_data": None,
    "traffic_parameters": None,
    "pollutant_reference_data": None,
    "pollutant_reference_source_info": None,
    "synchronized_data": None,
    "sync_summary": None,
    "bast_stations": None,
    "bast_signature": None,
    "bast_daily_data": None,
    "bast_selected_station": None,
    "bast_source_info": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def clean_column_names(df):
    df = df.copy()
    new_columns = []

    for i, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower() == "nan":
            name = f"Unnamed_{i + 1}"
        new_columns.append(name)

    df.columns = new_columns
    return df


def get_extension(filename):
    filename = str(filename).lower()
    return filename.rsplit(".", 1)[1] if "." in filename else ""


SUPPORTED_DATA_EXTENSIONS = ["csv", "txt", "dat", "xlsx", "xls"]


def get_zip_members(uploaded_file):
    """Return supported data files contained in a ZIP archive."""
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    try:
        archive = zipfile.ZipFile(BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded ZIP file is not a valid ZIP archive.") from exc

    members = []
    for info in archive.infolist():
        if info.is_dir():
            continue

        # Ignore macOS metadata and hidden/system files.
        name = info.filename.replace("\\", "/")
        base_name = name.rsplit("/", 1)[-1]

        if not base_name or base_name.startswith(".") or "__MACOSX" in name:
            continue

        extension = get_extension(base_name)
        if extension in SUPPORTED_DATA_EXTENSIONS:
            members.append(info.filename)

    return members


def extract_zip_member(uploaded_file, member_name):
    """Extract one supported data file from an uploaded ZIP into memory."""
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    try:
        archive = zipfile.ZipFile(BytesIO(raw))
        info = archive.getinfo(member_name)
        data = archive.read(info)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Could not extract the selected file from the ZIP archive.") from exc

    file_obj = BytesIO(data)
    file_obj.name = member_name.rsplit("/", 1)[-1]
    return file_obj


def prepare_uploaded_data_file(uploaded_file, selection_key, label):
    """
    Convert either a normal data file or a ZIP archive into a file-like object
    that the existing rainfall/traffic readers can process.
    """
    extension = get_extension(uploaded_file.name)

    if extension != "zip":
        return uploaded_file, extension, uploaded_file.name

    members = get_zip_members(uploaded_file)

    if not members:
        raise ValueError(
            f"No supported data files were found inside the ZIP archive. "
            f"Supported types: {', '.join('.' + x for x in SUPPORTED_DATA_EXTENSIONS)}."
        )

    selected_member = st.selectbox(
        f"Select the {label} file inside the ZIP archive",
        members,
        key=selection_key
    )

    extracted = extract_zip_member(uploaded_file, selected_member)
    return extracted, get_extension(extracted.name), selected_member


# ============================================================
# ENCODING / SEPARATOR / HEADER DETECTION
# ============================================================

def decode_text_file(uploaded_file):
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    raise ValueError("The file encoding could not be detected.")


def detect_separator(text):
    candidates = [";", ",", "\t", "|"]
    lines = [line for line in text.splitlines() if line.strip()]

    if not lines:
        return ";"

    sample = lines[:100]
    best_separator = ";"
    best_score = -1

    for separator in candidates:
        counts = []

        for line in sample:
            try:
                fields = next(csv.reader([line], delimiter=separator))
                counts.append(len(fields))
            except Exception:
                pass

        valid_counts = [x for x in counts if x > 1]

        if not valid_counts:
            continue

        score = len(valid_counts) * 10 + np.mean(valid_counts)

        if score > best_score:
            best_score = score
            best_separator = separator

    return best_separator


def score_header_line(line, separator):
    try:
        fields = next(csv.reader([line], delimiter=separator))
    except Exception:
        return -1

    if len(fields) < 2:
        return -1

    score = 0

    for field in fields:
        value = str(field).strip().lower()

        if any(keyword in value for keyword in DATE_KEYWORDS):
            score += 5

        if any(keyword in value for keyword in RAINFALL_KEYWORDS):
            score += 5

        if any(keyword in value for keyword in UNIT_KEYWORDS):
            score += 2

    return score


def detect_header_row(text, separator):
    lines = text.splitlines()
    best_row = 0
    best_score = -1

    for i in range(min(len(lines), 150)):
        if not lines[i].strip():
            continue

        score = score_header_line(lines[i], separator)

        if score > best_score:
            best_score = score
            best_row = i

    return best_row


# ============================================================
# DATE / NUMERIC PARSING
# ============================================================

def parse_dates(series):
    values = series.astype("string").str.strip()
    values = values.replace(MISSING_VALUES, pd.NA)

    try:
        return pd.to_datetime(
            values,
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )
    except Exception:
        return pd.to_datetime(
            values,
            errors="coerce",
            dayfirst=True
        )


def convert_numeric(series):
    values = series.astype("string").str.strip()
    values = values.replace(MISSING_VALUES, pd.NA)

    values = values.str.replace(" ", "", regex=False)
    values = values.str.replace(",", ".", regex=False)
    values = values.str.replace(r"[^0-9eE+\-.]", "", regex=True)

    return pd.to_numeric(values, errors="coerce")


def calculate_date_score(series):
    sample = series.dropna().head(500)

    if sample.empty:
        return 0.0

    parsed = parse_dates(sample)
    return float(parsed.notna().mean())


def calculate_numeric_score(series):
    sample = series.dropna().head(500)

    if sample.empty:
        return 0.0

    numeric = convert_numeric(sample)
    return float(numeric.notna().mean())


def keyword_score(column, keywords):
    name = str(column).strip().lower()
    return sum(keyword in name for keyword in keywords)


# ============================================================
# COLUMN ANALYSIS
# ============================================================

def analyse_columns(df):
    rows = []

    for column in df.columns:
        date_score = calculate_date_score(df[column])
        numeric_score = calculate_numeric_score(df[column])
        date_keyword = keyword_score(column, DATE_KEYWORDS)
        rainfall_keyword = keyword_score(column, RAINFALL_KEYWORDS)
        unit_keyword = keyword_score(column, UNIT_KEYWORDS)

        final_date_score = min(
            1.0,
            date_score * 0.75 + min(date_keyword, 2) * 0.125
        )

        final_rainfall_score = min(
            1.0,
            numeric_score * 0.50
            + min(rainfall_keyword, 2) * 0.35
            + min(unit_keyword, 2) * 0.15
        )

        rows.append({
            "Column": str(column),
            "Date Score": round(final_date_score, 3),
            "Rainfall Score": round(final_rainfall_score, 3),
            "Numeric Score": round(numeric_score, 3)
        })

    return pd.DataFrame(rows)


def suggest_date_column(analysis):
    if analysis.empty:
        return None

    row = analysis.loc[analysis["Date Score"].idxmax()]
    return row["Column"] if row["Date Score"] >= 0.40 else None


def suggest_rainfall_column(analysis):
    if analysis.empty:
        return None

    row = analysis.loc[analysis["Rainfall Score"].idxmax()]
    return row["Column"] if row["Rainfall Score"] >= 0.40 else None


# ============================================================
# FILE READING
# ============================================================

def read_text_file(uploaded_file):
    text, encoding = decode_text_file(uploaded_file)
    separator = detect_separator(text)
    header_row = detect_header_row(text, separator)

    uploaded_file.seek(0)

    df = pd.read_csv(
        uploaded_file,
        encoding=encoding,
        sep=separator,
        skiprows=header_row,
        header=0,
        engine="python",
        on_bad_lines="skip"
    )

    df = clean_column_names(df)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    return df, encoding, separator, header_row


def get_excel_sheets(uploaded_file):
    uploaded_file.seek(0)
    data = BytesIO(uploaded_file.read())
    excel = pd.ExcelFile(data)
    return excel.sheet_names


def read_excel_sheet(uploaded_file, sheet_name):
    uploaded_file.seek(0)
    data = BytesIO(uploaded_file.read())

    df = pd.read_excel(data, sheet_name=sheet_name)
    df = clean_column_names(df)

    return df.dropna(axis=0, how="all")


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_rainfall_data(df, date_column, rainfall_column):
    result = pd.DataFrame()

    result["Date/Time"] = parse_dates(df[date_column])
    result["Rainfall (mm)"] = convert_numeric(df[rainfall_column])

    result = result[result["Date/Time"].notna()].copy()
    result = result.sort_values("Date/Time")
    result = result.drop_duplicates(subset=["Date/Time"], keep="first")
    result = result.reset_index(drop=True)

    return result


# ============================================================
# TEMPORAL RESOLUTION / QUALITY
# ============================================================

def detect_temporal_resolution(data):
    """
    Detect the dominant temporal interval.

    Accepts either a DataFrame containing a ``Date/Time`` column or a
    pandas Series containing timestamps.

    Returns:
        (resolution_label, interval_minutes, irregular_percentage)
    """
    if isinstance(data, pd.DataFrame):
        if "Date/Time" not in data.columns:
            return "Unknown", None, 0.0
        dates = data["Date/Time"]
    else:
        dates = data

    dates = pd.to_datetime(dates, errors="coerce").dropna()
    dates = dates.sort_values().drop_duplicates()

    if len(dates) < 2:
        return "Unknown", None, 0.0

    differences = dates.diff().dropna().dt.total_seconds() / 60.0
    differences = differences[differences > 0]

    if differences.empty:
        return "Unknown", None, 0.0

    mode = differences.mode()
    interval = float(
        mode.iloc[0] if not mode.empty else differences.median()
    )

    known = {
        1: "1 minute",
        5: "5 minutes",
        10: "10 minutes",
        15: "15 minutes",
        30: "30 minutes",
        60: "Hourly",
        120: "2 hours",
        180: "3 hours",
        360: "6 hours",
        720: "12 hours",
        1440: "Daily",
        10080: "Weekly"
    }

    resolution = next(
        (
            label for minutes, label in known.items()
            if abs(interval - minutes) < 0.01
        ),
        f"{interval:.2f} minutes"
        if interval < 60
        else f"{interval / 60:.2f} hours"
    )

    irregular = int((abs(differences - interval) > 0.01).sum())
    irregular_percentage = irregular / len(differences) * 100.0

    return resolution, interval, irregular_percentage

def calculate_quality(df):
    total = len(df)
    missing = int(df["Rainfall (mm)"].isna().sum())
    duplicates = int(df["Date/Time"].duplicated().sum())
    negative = int((df["Rainfall (mm)"] < 0).sum())
    valid = total - missing

    completeness = valid / total * 100 if total > 0 else 0

    return {
        "total": total,
        "missing": missing,
        "duplicates": duplicates,
        "negative": negative,
        "valid": valid,
        "completeness": completeness
    }


def rainfall_statistics(df):
    # Always force the rainfall series to numeric before calculating
    # statistics. Some uploaded Excel files can otherwise leave a
    # datetime/NaT value in this column, which makes std() fail.
    rainfall = pd.to_numeric(
        df["Rainfall (mm)"], errors="coerce"
    ).dropna()

    if rainfall.empty:
        return None

    std_value = rainfall.std()
    if pd.isna(std_value):
        std_value = 0.0

    return {
        "total": float(rainfall.sum()),
        "maximum": float(rainfall.max()),
        "average": float(rainfall.mean()),
        "median": float(rainfall.median()),
        "minimum": float(rainfall.min()),
        "non_zero": int((rainfall > 0).sum()),
        "zero": int((rainfall == 0).sum()),
        "std": float(std_value)
    }


def default_dry_period(interval_minutes):
    if interval_minutes is None:
        return 6.0
    if interval_minutes <= 60:
        return 6.0
    if interval_minutes <= 360:
        return 12.0
    return 24.0


# ============================================================
# RAINFALL EVENT ANALYSIS
# ============================================================

def rainfall_events(df, dry_period_hours, interval_minutes):
    """
    Identify rainfall events using a dry-period separation threshold.

    This function is deliberately defensive because uploaded rainfall files can
    contain malformed interval metadata, NaT timestamps, mixed numeric/date
    values, or very large/negative interval values. Event duration is therefore
    calculated from ordinary numeric seconds rather than constructing a pandas
    Timedelta from an unchecked value.
    """
    required = {"Date/Time", "Rainfall (mm)"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    data = df[["Date/Time", "Rainfall (mm)"]].copy()

    # Robust timestamp conversion.
    data["Date/Time"] = pd.to_datetime(
        data["Date/Time"], errors="coerce"
    )

    # Robust rainfall conversion.
    data["Rainfall (mm)"] = pd.to_numeric(
        data["Rainfall (mm)"], errors="coerce"
    )

    data = data.dropna(subset=["Date/Time"])
    data = data.sort_values("Date/Time")
    data = data.drop_duplicates(subset=["Date/Time"], keep="first")
    data = data.reset_index(drop=True)

    # Only positive rainfall contributes to an event.
    positive = data[
        data["Rainfall (mm)"].notna()
        & np.isfinite(data["Rainfall (mm)"])
        & (data["Rainfall (mm)"] > 0)
    ].copy()

    if positive.empty:
        return pd.DataFrame()

    positive = positive.reset_index(drop=True)

    # Validate dry-period input independently.
    try:
        dry_hours = float(dry_period_hours)
    except (TypeError, ValueError):
        dry_hours = 24.0

    if not np.isfinite(dry_hours) or dry_hours < 0:
        dry_hours = 24.0

    # Keep the threshold within a practical range.
    dry_hours = min(max(dry_hours, 0.0), 8760.0)

    dry_seconds = dry_hours * 3600.0

    positive["Previous Rainfall Gap"] = positive["Date/Time"].diff()

    # ------------------------------------------------------------------
    # Determine a safe interval.
    #
    # Prefer the observed timestamp spacing because it is the most reliable
    # source for event-duration calculations. The supplied interval is used
    # only when it is a finite, positive, physically reasonable number.
    # ------------------------------------------------------------------
    observed_minutes = np.nan

    try:
        gaps_seconds = (
            positive["Date/Time"]
            .diff()
            .dt.total_seconds()
            .dropna()
        )
        gaps_seconds = gaps_seconds[
            np.isfinite(gaps_seconds) & (gaps_seconds > 0)
        ]

        if not gaps_seconds.empty:
            # Median is robust to occasional missing records/large gaps.
            observed_minutes = float(gaps_seconds.median() / 60.0)
    except Exception:
        observed_minutes = np.nan

    try:
        supplied_interval = float(interval_minutes)
    except (TypeError, ValueError):
        supplied_interval = np.nan

    # Accept supplied interval only if it is sensible.
    if (
        np.isfinite(supplied_interval)
        and supplied_interval > 0
        and supplied_interval <= 10080
    ):
        safe_interval = supplied_interval
    elif np.isfinite(observed_minutes) and 0 < observed_minutes <= 10080:
        safe_interval = observed_minutes
    else:
        # Last-resort fallback for a valid rainfall table with unusable
        # interval metadata.
        safe_interval = 1440.0

    # Final hard clamp. This guarantees that no enormous/negative value can
    # reach the duration calculation.
    safe_interval = float(np.clip(safe_interval, 0.001, 10080.0))
    interval_seconds = safe_interval * 60.0

    # Separate events after the selected dry period.
    gap_seconds = positive["Previous Rainfall Gap"].dt.total_seconds()
    positive["New Event"] = (
        gap_seconds > dry_seconds
    )

    positive.loc[0, "New Event"] = True

    positive["Event"] = (
        positive["New Event"].cumsum().astype(int)
    )

    event_rows = []

    for event_number, group in positive.groupby("Event"):
        group = group.sort_values("Date/Time")

        start = group["Date/Time"].iloc[0]
        end = group["Date/Time"].iloc[-1]

        total = float(group["Rainfall (mm)"].sum())
        maximum = float(group["Rainfall (mm)"].max())
        records = int(len(group))

        # Calculate duration using numeric seconds instead of
        # pd.Timedelta(minutes=<possibly malformed value>).
        try:
            span_seconds = max(
                0.0,
                float((end - start).total_seconds())
            )
        except Exception:
            span_seconds = 0.0

        duration_hours = (
            span_seconds + interval_seconds
        ) / 3600.0

        # Maximum interval rainfall converted to a rate.
        intensity = (
            maximum / (safe_interval / 60.0)
            if safe_interval > 0
            else np.nan
        )

        previous_gap = group["Previous Rainfall Gap"].iloc[0]
        antecedent = (
            np.nan
            if pd.isna(previous_gap)
            else max(
                0.0,
                float(previous_gap.total_seconds() / 3600.0)
            )
        )

        event_rows.append({
            "Event": int(event_number),
            "Start": start,
            "End": end,
            "Duration (hours)": duration_hours,
            "Total Rainfall (mm)": total,
            "Maximum Rainfall per Interval (mm)": maximum,
            "Average Rainfall Rate (mm/h)": intensity,
            "Rainfall Records": records,
            "Antecedent Dry Period (hours)": antecedent
        })

    return pd.DataFrame(event_rows)


# ============================================================
# MATHEMATICAL MODEL
# ============================================================

def run_mathematical_model(
    rainfall,
    area_ha,
    runoff_coefficient,
    pollutant_buildup,
    washoff_coefficient
):
    data = rainfall.copy()
    data = data.sort_values("Date/Time")

    data["Rainfall (mm)"] = (
        pd.to_numeric(
            data["Rainfall (mm)"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    data["Time Step (seconds)"] = (
        data["Date/Time"].diff().dt.total_seconds()
    )

    median_seconds = (
        data["Time Step (seconds)"].dropna().median()
    )

    if pd.isna(median_seconds):
        median_seconds = 3600.0

    data["Time Step (seconds)"] = (
        data["Time Step (seconds)"]
        .fillna(median_seconds)
        .clip(lower=1)
    )

    data["Effective Rainfall (mm)"] = (
        data["Rainfall (mm)"] * runoff_coefficient
    )

    # 1 mm over 1 hectare = 10 m³
    data["Runoff Volume (m³)"] = (
        data["Effective Rainfall (mm)"]
        * area_ha
        * 10
    )

    data["Runoff Flow (m³/s)"] = (
        data["Runoff Volume (m³)"]
        / data["Time Step (seconds)"]
    )

    data["Runoff Flow (L/s)"] = (
        data["Runoff Flow (m³/s)"] * 1000
    )

    data["Wash-off Fraction"] = (
        1
        - np.exp(
            -washoff_coefficient
            * data["Rainfall (mm)"]
        )
    ).clip(0, 1)

    available_mass = pollutant_buildup * area_ha

    data["Available Pollutant Mass (kg)"] = available_mass

    data["Pollutant Load (kg)"] = (
        available_mass
        * data["Wash-off Fraction"]
    )

    data["Pollutant Concentration (mg/L)"] = np.where(
        data["Runoff Volume (m³)"] > 0,
        data["Pollutant Load (kg)"]
        / data["Runoff Volume (m³)"]
        * 1000,
        0
    )

    data["Cumulative Runoff (m³)"] = (
        data["Runoff Volume (m³)"].cumsum()
    )

    data["Cumulative Pollutant Load (kg)"] = (
        data["Pollutant Load (kg)"].cumsum()
    )

    return data


# ============================================================
# HELPER: EVENT-LEVEL MODEL RESULTS
# ============================================================

def calculate_event_model_results(results, events):
    if results is None or events is None or events.empty:
        return pd.DataFrame()

    rows = []

    for _, event in events.iterrows():
        start = event["Start"]
        end = event["End"]

        subset = results[
            (results["Date/Time"] >= start)
            & (results["Date/Time"] <= end)
        ].copy()

        if subset.empty:
            continue

        rows.append({
            "Event": int(event["Event"]),
            "Start": start,
            "End": end,
            "Rainfall (mm)": float(subset["Rainfall (mm)"].sum()),
            "Runoff Volume (m³)": float(subset["Runoff Volume (m³)"].sum()),
            "Peak Flow (m³/s)": float(subset["Runoff Flow (m³/s)"].max()),
            "Pollutant Load (kg)": float(subset["Pollutant Load (kg)"].sum())
        })

    return pd.DataFrame(rows)


def format_event_table(df):
    output = df.copy()

    for col in ["Start", "End"]:
        if col in output.columns:
            output[col] = pd.to_datetime(
                output[col], errors="coerce"
            ).dt.strftime("%d.%m.%Y %H:%M")

    return output



# ============================================================
# TRAFFIC / POLLUTANT FUNCTIONS
# ============================================================

TRAFFIC_DATE_KEYWORDS = [
    "date", "datum", "time", "zeit", "datetime", "date/time",
    "datum/zeit", "timestamp", "zeitstempel", "zeitangabe"
]

TRAFFIC_COUNT_KEYWORDS = [
    "traffic", "vehicle", "vehicles", "count", "volume",
    "fahrzeug", "fahrzeuge", "verkehr", "verkehrsstärke",
    "verkehrsmenge", "kfz", "vehicles/day", "veh/day"
]


def analyse_traffic_columns(df):
    rows = []

    for column in df.columns:
        date_score = calculate_date_score(df[column])
        numeric_score = calculate_numeric_score(df[column])
        name = str(column).strip().lower()

        date_keyword = sum(k in name for k in TRAFFIC_DATE_KEYWORDS)
        traffic_keyword = sum(k in name for k in TRAFFIC_COUNT_KEYWORDS)

        date_final = min(
            1.0,
            date_score * 0.75 + min(date_keyword, 2) * 0.125
        )
        traffic_final = min(
            1.0,
            numeric_score * 0.60 + min(traffic_keyword, 2) * 0.40
        )

        rows.append({
            "Column": str(column),
            "Date Score": round(date_final, 3),
            "Traffic Score": round(traffic_final, 3),
            "Numeric Score": round(numeric_score, 3)
        })

    return pd.DataFrame(rows)


def suggest_traffic_date_column(analysis):
    if analysis.empty:
        return None
    row = analysis.loc[analysis["Date Score"].idxmax()]
    return row["Column"] if row["Date Score"] >= 0.40 else None


def suggest_traffic_count_column(analysis):
    if analysis.empty:
        return None
    row = analysis.loc[analysis["Traffic Score"].idxmax()]
    return row["Column"] if row["Traffic Score"] >= 0.40 else None


def read_swmm_external_inflow_file(uploaded_file):
    """
    Read the user's existing SWMM external pollutant inflow DAT file.

    Expected data rows:
        MM/DD/YYYY HH:MM MASS_KG_PER_DAY

    Header/comment lines beginning with ';' are ignored.
    The function intentionally returns pollutant mass data, not traffic counts.
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    if isinstance(raw, bytes):
        text = None
        for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError("Could not decode the SWMM inflow DAT file.")
    else:
        text = str(raw)

    lines = text.splitlines()

    # Confirm that this really looks like an SWMM external inflow file.
    header_text = "\n".join(lines[:20]).lower()
    looks_like_inflow = (
        "external inflow" in header_text
        or "mass_inflow" in header_text
        or "mass inflow" in header_text
    )

    if not looks_like_inflow:
        raise ValueError(
            "This file does not appear to be an SWMM external inflow file."
        )

    records = []

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith(";"):
            continue

        # Ignore obvious header rows.
        if re.search(r"[A-Za-z]", stripped):
            continue

        parts = re.split(r"\s+", stripped)

        if len(parts) < 3:
            continue

        date_text = parts[0]
        time_text = parts[1]
        mass_text = parts[2]

        dt = pd.to_datetime(
            f"{date_text} {time_text}",
            errors="coerce",
            dayfirst=False
        )

        mass = pd.to_numeric(
            str(mass_text).replace(",", "."),
            errors="coerce"
        )

        if pd.isna(dt) or pd.isna(mass):
            continue

        records.append({
            "Date/Time": dt,
            "Mass Inflow (kg/day)": float(mass)
        })

    if not records:
        raise ValueError(
            "No valid SWMM external inflow records were found."
        )

    result = pd.DataFrame(records)
    result = result.sort_values("Date/Time")
    result = result.drop_duplicates(
        subset=["Date/Time"],
        keep="last"
    ).reset_index(drop=True)

    return result


def read_traffic_file(uploaded_file, sheet_name=None):
    extension = get_extension(uploaded_file.name)

    if extension in ["xlsx", "xls"]:
        sheets = get_excel_sheets(uploaded_file)
        if sheet_name is None:
            sheet_name = sheets[0]
        df = read_excel_sheet(uploaded_file, sheet_name)
        return df, f"Excel sheet: {sheet_name}"

    df, encoding, separator, header_row = read_text_file(uploaded_file)
    return df, f"{encoding}; separator={repr(separator)}; header row={header_row + 1}"


def prepare_traffic_data(df, date_column, count_column):
    data = pd.DataFrame()

    data["Date/Time"] = parse_dates(df[date_column])
    data["Traffic Count"] = convert_numeric(df[count_column])

    data = data.dropna(subset=["Date/Time"])
    data["Traffic Count"] = data["Traffic Count"].clip(lower=0)
    data = data.sort_values("Date/Time")
    data = data.drop_duplicates(subset=["Date/Time"], keep="last")
    data = data.reset_index(drop=True)

    return data


def calculate_pollutant_loading(
    traffic_data,
    emission_factor_g_per_vehicle_day,
    scale_factor=1.0
):
    data = traffic_data.copy()

    data["Traffic Count"] = pd.to_numeric(
        data["Traffic Count"], errors="coerce"
    ).fillna(0).clip(lower=0)

    # Transparent parameterized calculation:
    # kg/day = vehicles/day × emission factor (g/vehicle/day)
    #          × scale factor ÷ 1000.
    data["Mass Inflow (kg/day)"] = (
        data["Traffic Count"]
        * float(emission_factor_g_per_vehicle_day)
        * float(scale_factor)
        / 1000.0
    )

    return data


def generate_swmm_inflow_dat(
    pollutant_data,
    title="NIQKI SWMM External Inflow Time Series"
):
    lines = [
        "; SWMM 5 External Inflow Time Series File",
        f"; Source: {title}",
        "; Date       Time    Mass_Inflow_(kg/day)"
    ]

    for _, row in pollutant_data.iterrows():
        dt = pd.to_datetime(row["Date/Time"], errors="coerce")
        mass = pd.to_numeric(
            row["Mass Inflow (kg/day)"], errors="coerce"
        )

        if pd.isna(dt) or pd.isna(mass):
            continue

        lines.append(
            f"{dt.strftime('%m/%d/%Y')} "
            f"{dt.strftime('%H:%M')} "
            f"{float(mass):.6f}"
        )

    return "\n".join(lines) + "\n"


def traffic_summary(data):
    if data is None or data.empty:
        return {}

    return {
        "records": int(len(data)),
        "total_vehicles": float(data["Traffic Count"].sum()),
        "average": float(data["Traffic Count"].mean()),
        "maximum": float(data["Traffic Count"].max()),
        "minimum": float(data["Traffic Count"].min()),
        "start": data["Date/Time"].min(),
        "end": data["Date/Time"].max()
    }


# ============================================================
# BASt / mFUND TRAFFIC DATA
# ============================================================

BAST_REQUIRED_COLUMNS = [
    "Zst", "Datum", "Stunde",
    "KFZ_R1", "KFZ_R2",
    "Pkw_R1", "Pkw_R2",
    "Lfw_R1", "Lfw_R2",
    "Lkw_R1", "Lkw_R2",
    "Lzg_R1", "Lzg_R2",
    "Sat_R1", "Sat_R2",
    "Bus_R1", "Bus_R2"
]

BAST_EMISSION_DEFAULTS = {
    "EF_PKW_mg_veh_km": 90.0,
    "EF_LKW_mg_veh_km": 800.0,
    "EF_BUS_mg_veh_km": 700.0,
    "K_drive": 1.3,
    "K_road": 1.1,
    "eta_verge": 0.40,
    "eta_trap": 0.30,
    "k_loss": 0.05,
    "C1_washoff": 0.05,
    "C2_washoff": 1.3,
}

def _zip_member_name(uploaded_file):
    uploaded_file.seek(0)
    with zipfile.ZipFile(uploaded_file) as archive:
        candidates = [
            info.filename for info in archive.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith((".txt", ".csv", ".tsv"))
            and not info.filename.replace("\\", "/").split("/")[-1].startswith(".")
        ]
    if not candidates:
        raise ValueError("No TXT/CSV/TSV traffic file was found inside the ZIP archive.")
    return candidates[0]

def _open_bast_source(uploaded_file):
    """
    Return a ZipFile member stream for a ZIP or the uploaded file itself.
    The caller is responsible for closing the returned archive when needed.
    """
    ext = get_extension(uploaded_file.name)
    if ext == "zip":
        uploaded_file.seek(0)
        archive = zipfile.ZipFile(uploaded_file)
        member = _zip_member_name(uploaded_file)
        return archive, archive.open(member, "r"), member
    uploaded_file.seek(0)
    return None, uploaded_file, uploaded_file.name

def is_bast_mfund_header(uploaded_file):
    try:
        archive, stream, member = _open_bast_source(uploaded_file)
        try:
            sample = pd.read_csv(
                stream,
                sep=";",
                dtype=str,
                nrows=3,
                usecols=lambda c: str(c).strip() in BAST_REQUIRED_COLUMNS
            )
        finally:
            if archive is not None:
                archive.close()
        cols = {str(c).strip() for c in sample.columns}
        return {"Zst", "Datum", "Stunde"}.issubset(cols)
    except Exception:
        return False

def bast_list_stations(uploaded_file, chunksize=250_000):
    """
    Memory-safe station discovery. Only the Zst column is read from the
    potentially 1+ GB decompressed BASt text file.
    """
    archive, stream, member = _open_bast_source(uploaded_file)
    stations = set()
    try:
        for chunk in pd.read_csv(
            stream,
            sep=";",
            dtype={"Zst": "string"},
            usecols=["Zst"],
            chunksize=chunksize,
            low_memory=True,
            on_bad_lines="skip"
        ):
            values = (
                chunk["Zst"]
                .astype("string")
                .str.strip()
                .dropna()
            )
            stations.update(v for v in values.tolist() if str(v).strip())
    finally:
        if archive is not None:
            archive.close()
    return sorted(stations)

def bast_process_station(uploaded_file, selected_zst, chunksize=200_000):
    """
    Reproduce the user's working roadandrain.py traffic aggregation without
    loading the complete 1.44 GB TXT file into memory.

    The original method:
      - filters by Zst
      - sums both traffic directions
      - groups vehicle classes
      - parses Datum as YYMMDD
      - aggregates to daily values
    """
    archive, stream, member = _open_bast_source(uploaded_file)

    available_columns = None
    daily_parts = []

    usecols = [
        c for c in BAST_REQUIRED_COLUMNS
        if c in BAST_REQUIRED_COLUMNS
    ]

    try:
        for chunk in pd.read_csv(
            stream,
            sep=";",
            dtype=str,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=True,
            on_bad_lines="skip"
        ):
            chunk.columns = [str(c).strip() for c in chunk.columns]

            if "Zst" not in chunk.columns:
                continue

            site = chunk[
                chunk["Zst"].astype(str).str.strip() == str(selected_zst).strip()
            ].copy()

            if site.empty:
                continue

            # Exact vehicle groups from the user's original script.
            def num_sum(names):
                total = pd.Series(0.0, index=site.index)
                for name in names:
                    if name in site.columns:
                        total = total + pd.to_numeric(
                            site[name], errors="coerce"
                        ).fillna(0.0)
                return total

            site["PKW_Van"] = num_sum([
                "Pkw_R1", "Pkw_R2", "Lfw_R1", "Lfw_R2"
            ])
            site["LKW"] = num_sum([
                "Lkw_R1", "Lkw_R2",
                "Lzg_R1", "Lzg_R2",
                "Sat_R1", "Sat_R2"
            ])
            site["Bus"] = num_sum([
                "Bus_R1", "Bus_R2"
            ])
            site["Traffic Count"] = num_sum([
                "KFZ_R1", "KFZ_R2"
            ])

            site["Date/Time"] = pd.to_datetime(
                site["Datum"].astype(str).str.strip(),
                format="%y%m%d",
                errors="coerce"
            )

            site = site.dropna(subset=["Date/Time"])

            if site.empty:
                continue

            part = (
                site.groupby("Date/Time", as_index=False)[
                    ["PKW_Van", "LKW", "Bus", "Traffic Count"]
                ].sum()
            )
            daily_parts.append(part)

    finally:
        if archive is not None:
            archive.close()

    if not daily_parts:
        return pd.DataFrame(
            columns=[
                "Date/Time", "PKW_Van", "LKW", "Bus", "Traffic Count"
            ]
        )

    daily = (
        pd.concat(daily_parts, ignore_index=True)
        .groupby("Date/Time", as_index=False)[
            ["PKW_Van", "LKW", "Bus", "Traffic Count"]
        ].sum()
        .sort_values("Date/Time")
        .reset_index(drop=True)
    )

    daily["Station Zst"] = str(selected_zst)
    return daily

def bast_calculate_twp_hourly(
    traffic_daily,
    road_length_km,
    ef_pkw=90.0,
    ef_lkw=800.0,
    ef_bus=700.0,
    k_drive=1.3,
    k_road=1.1
):
    """
    Exact emission-factor calculation used in the user's working script.

    Result is daily generation because the original script sums hourly
    generation across all hourly observations for each day.
    """
    data = traffic_daily.copy()

    data["TWP_Hourly_kg"] = (
        (
            data["PKW_Van"] * float(ef_pkw)
            + data["LKW"] * float(ef_lkw)
            + data["Bus"] * float(ef_bus)
        )
        * float(road_length_km)
        * float(k_drive)
        * float(k_road)
        / 1e6
    )

    return data

def bast_daily_washoff(
    merged_daily,
    k_loss=0.05,
    c1_washoff=0.05,
    c2_washoff=1.3,
    eta_verge=0.40,
    eta_trap=0.30
):
    """
    Exact daily buildup/wash-off formulation from roadandrain.py.
    """
    data = merged_daily.copy().sort_values("Date/Time").reset_index(drop=True)

    if data.empty:
        data["Sewer_Mass_kg"] = []
        return data

    B = float(data["M_gen_daily_kg"].iloc[0]) / float(k_loss)
    sewer_mass = []

    for _, row in data.iterrows():
        m_gen = float(row["M_gen_daily_kg"])
        day_rain = float(row.get("Rainfall_mm", row.get("Rainfall (mm)", 0.0)))

        b_before = (
            B * np.exp(-float(k_loss) * 1.0)
            + (m_gen / float(k_loss))
            * (1.0 - np.exp(-float(k_loss) * 1.0))
        )

        if day_rain > 0:
            q = day_rain / 2.0
            w_rate = float(c1_washoff) * (q ** float(c2_washoff)) * b_before
            mass_washed = min(w_rate * 2.0, b_before)
        else:
            mass_washed = 0.0

        B = b_before - mass_washed
        mass_to_sewer = (
            mass_washed
            * (1.0 - float(eta_verge))
            * (1.0 - float(eta_trap))
        )
        sewer_mass.append(mass_to_sewer)

    data["Sewer_Mass_kg"] = sewer_mass
    return data



# ============================================================
# GENERIC TRAFFIC FILE HELPERS
# ============================================================

def traffic_detect_text_source(uploaded_file):
    """
    Inspect only the beginning of a TXT/CSV/DAT/ZIP member.
    This avoids loading a large traffic file into memory.
    """
    ext = get_extension(uploaded_file.name)
    archive = None
    stream = None
    member = None

    try:
        if ext == "zip":
            uploaded_file.seek(0)
            archive = zipfile.ZipFile(uploaded_file)
            members = [
                i.filename for i in archive.infolist()
                if not i.is_dir()
                and i.filename.lower().endswith(
                    (".txt", ".csv", ".tsv", ".dat")
                )
            ]
            if not members:
                raise ValueError("No supported TXT/CSV/TSV/DAT file found in ZIP.")
            member = members[0]
            stream = archive.open(member, "r")
        else:
            uploaded_file.seek(0)
            stream = uploaded_file

        raw = stream.read(256 * 1024)
        if isinstance(raw, str):
            text = raw
        else:
            text = None
            for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                raise ValueError("Could not decode the traffic file preview.")

        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("The traffic file appears to be empty.")

        # Find the first plausible header line. BASt files normally use ';'.
        header_line = next(
            (line for line in lines if not line.lstrip().startswith(";")),
            lines[0]
        )

        candidates = [";", "\t", ",", "|"]
        sep = max(
            candidates,
            key=lambda s: header_line.count(s)
        )
        if header_line.count(sep) == 0:
            sep = r"\s+"

        return {
            "extension": ext,
            "member": member,
            "separator": sep,
            "encoding": enc if 'enc' in locals() else "utf-8"
        }
    finally:
        if archive is not None:
            archive.close()

def traffic_text_columns(uploaded_file, separator, encoding):
    """Read only the header from a text/ZIP traffic source."""
    ext = get_extension(uploaded_file.name)
    archive = None
    stream = None
    try:
        if ext == "zip":
            uploaded_file.seek(0)
            archive = zipfile.ZipFile(uploaded_file)
            member = _zip_member_name(uploaded_file)
            stream = archive.open(member, "r")
        else:
            uploaded_file.seek(0)
            stream = uploaded_file

        header = pd.read_csv(
            stream,
            sep=separator,
            encoding=encoding,
            nrows=0,
            engine="python"
        )
        return [str(c).strip() for c in header.columns]
    finally:
        if archive is not None:
            archive.close()

def generic_traffic_process(
    uploaded_file,
    date_column,
    count_column,
    separator,
    encoding,
    station_column=None,
    station_value=None,
    chunksize=200_000
):
    """
    Generic, memory-safe traffic processor.

    Works for ordinary traffic datasets and large delimited files.
    If a station column/value is supplied, only that station is retained.
    """
    ext = get_extension(uploaded_file.name)

    if ext in ["xlsx", "xls"]:
        raw = read_excel_sheet(uploaded_file, None)
        if station_column and station_value is not None:
            raw = raw[
                raw[station_column].astype(str).str.strip()
                == str(station_value).strip()
            ]
        return prepare_traffic_data(raw, date_column, count_column)

    archive = None
    stream = None
    if ext == "zip":
        uploaded_file.seek(0)
        archive = zipfile.ZipFile(uploaded_file)
        member = _zip_member_name(uploaded_file)
        stream = archive.open(member, "r")
    else:
        uploaded_file.seek(0)
        stream = uploaded_file

    parts = []
    usecols = [date_column, count_column]
    if station_column and station_column not in usecols:
        usecols.append(station_column)

    try:
        for chunk in pd.read_csv(
            stream,
            sep=separator,
            encoding=encoding,
            dtype=str,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=True,
            engine="python",
            on_bad_lines="skip"
        ):
            chunk.columns = [str(c).strip() for c in chunk.columns]

            if station_column and station_value is not None:
                chunk = chunk[
                    chunk[station_column].astype(str).str.strip()
                    == str(station_value).strip()
                ]

            if chunk.empty:
                continue

            prepared = prepare_traffic_data(
                chunk, date_column, count_column
            )
            if not prepared.empty:
                parts.append(prepared)

    finally:
        if archive is not None:
            archive.close()

    if not parts:
        return pd.DataFrame(columns=["Date/Time", "Traffic Count"])

    result = (
        pd.concat(parts, ignore_index=True)
        .groupby("Date/Time", as_index=False)["Traffic Count"]
        .sum()
        .sort_values("Date/Time")
        .reset_index(drop=True)
    )
    return result

def generic_traffic_is_bast(columns):
    normalized = {str(c).strip().lower() for c in columns}
    return {
        "zst", "datum", "stunde", "kfz_r1", "kfz_r2"
    }.issubset(normalized)

def generic_traffic_station_values(
    uploaded_file, station_column, separator, encoding, chunksize=250_000
):
    """Return unique station values without loading the full source."""
    ext = get_extension(uploaded_file.name)
    archive = None
    stream = None
    values = set()

    try:
        if ext == "zip":
            uploaded_file.seek(0)
            archive = zipfile.ZipFile(uploaded_file)
            member = _zip_member_name(uploaded_file)
            stream = archive.open(member, "r")
        else:
            uploaded_file.seek(0)
            stream = uploaded_file

        for chunk in pd.read_csv(
            stream,
            sep=separator,
            encoding=encoding,
            dtype={station_column: str},
            usecols=[station_column],
            chunksize=chunksize,
            low_memory=True,
            engine="python",
            on_bad_lines="skip"
        ):
            values.update(
                v for v in
                chunk[station_column].astype(str).str.strip().dropna().tolist()
                if v and v.lower() != "nan"
            )
    finally:
        if archive is not None:
            archive.close()

    return sorted(values)


# ============================================================
# SIDEBAR / NAVIGATION
# ============================================================

st.sidebar.markdown("""
<div class="niqki-brand">
    <div class="niqki-brand-mark">💧</div>
    <div class="niqki-brand-title">NIQKI</div>
    <div class="niqki-brand-subtitle">Rainfall &amp; Runoff Analysis</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "APPLICATION",
    [
        "Home",
        "Methodology",
        "Rainfall Data",
        "Traffic Data",
        "Data Synchronization",
        "Pollutant Loading",
        "Model Parameters",
        "Simulation",
        "Results",
        "SWMM"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Engineering analysis workflow")
if st.session_state.get("rainfall_data") is not None:
    st.sidebar.success("Rainfall dataset loaded")
else:
    st.sidebar.info("No rainfall dataset loaded")

if st.session_state.get("traffic_data") is not None:
    st.sidebar.success("Traffic dataset loaded")
else:
    st.sidebar.info("No traffic dataset loaded")

if st.session_state.get("synchronized_data") is not None:
    st.sidebar.success("Rainfall–traffic data synchronized")
else:
    st.sidebar.info("Rainfall–traffic synchronization pending")

st.sidebar.caption("NIQKI Web Application · v2.0")

# ============================================================
# HOME
# ============================================================

if page == "Home":
    st.markdown("""
    <div class="niqki-hero">
        <div class="niqki-eyebrow">NIQKI · Engineering Analysis Platform</div>
        <div class="niqki-hero-title">Rainfall &amp; Runoff Analysis</div>
        <div class="niqki-hero-text">
            A structured workflow for rainfall data preparation, quality control,
            event analysis, simplified runoff and pollutant calculations, and
            preparation of rainfall input files for EPA SWMM.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What this application does")
    c1, c2, c3 = st.columns(3, gap="medium")
    cards = [
        ("🌧️", "Rainfall data", "Import CSV, TXT, DAT or Excel rainfall datasets, detect columns and standardize the time series."),
        ("🚗", "Traffic & pollutants", "Analyze traffic data and convert a user-defined, transparent emission factor into a pollutant mass time series."),
        ("🔗", "SWMM preparation", "Prepare rainfall and pollutant external input files for connection to an EPA SWMM model."),
    ]
    for col, (icon, title, text) in zip((c1, c2, c3), cards):
        with col:
            st.markdown(f"""
            <div class="niqki-card">
                <div class="niqki-card-icon">{icon}</div>
                <div class="niqki-card-title">{title}</div>
                <div class="niqki-card-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Workflow")
    steps = [
        ("01", "Upload", "Load rainfall data"),
        ("02", "Validate", "Check structure & quality"),
        ("03", "Analyze", "Statistics & events"),
        ("04", "Traffic", "Analyze traffic data"),
        ("05", "Pollutants", "Calculate mass loading"),
        ("06", "SWMM", "Prepare external inputs"),
    ]
    cols = st.columns(6, gap="small")
    for col, (num, label, note) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="niqki-step">
                <div class="niqki-step-number">{num}</div>
                <div class="niqki-step-label">{label}</div>
                <div class="niqki-step-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Get started")
    if st.session_state.get("rainfall_data") is None:
        st.info("Go to **Rainfall Data** in the navigation panel to upload your first rainfall dataset.")
    else:
        st.success("A rainfall dataset is currently loaded. Continue with the analysis or SWMM preparation pages.")

    st.markdown("""
    <div class="niqki-footer">
        NIQKI Rainfall &amp; Runoff Analysis · Data preparation and engineering calculations ·
        EPA SWMM simulation is performed externally.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RAINFALL DATA
# ============================================================


elif page == "Methodology":
    st.title("NIQKI Methodology")
    st.write(
        "This page documents the engineering workflow implemented in the "
        "NIQKI application. It describes how rainfall and traffic data are "
        "prepared, synchronized, used for pollutant loading calculations, "
        "and converted into external inputs for EPA SWMM."
    )

    st.divider()

    # ------------------------------------------------------------
    # 1. PROJECT OBJECTIVE
    # ------------------------------------------------------------
    st.header("1. Project Objective")
    st.write(
        "The NIQKI application provides a structured workflow for analysing "
        "rainfall and traffic data and preparing model inputs for runoff and "
        "road-related pollutant investigations. The application separates "
        "data preparation, quality control, temporal synchronization, "
        "pollutant loading and SWMM input preparation."
    )

    st.info(
        "The application is designed to be data-source independent. "
        "BASt/mFUND is one supported traffic-data structure; it is not a "
        "requirement of the overall workflow."
    )

    # ------------------------------------------------------------
    # 2. INPUT DATA
    # ------------------------------------------------------------
    st.header("2. Input Data")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rainfall data")
        st.markdown(
            """
            - Date/time
            - Rainfall depth
            - Recording interval
            - Station information, where available
            - Optional metadata
            """
        )
    with c2:
        st.subheader("Traffic data")
        st.markdown(
            """
            - Date/time
            - Traffic volume
            - Optional vehicle classes
            - Optional station/location identifier
            - Optional road-segment information
            """
        )

    st.caption(
        "Supported source formats are handled through the Rainfall Data and "
        "Traffic Data modules. ZIP archives can contain the source data file."
    )

    # ------------------------------------------------------------
    # 3. RAINFALL PROCESSING
    # ------------------------------------------------------------
    st.header("3. Rainfall Data Processing")

    st.markdown(
        """
        The rainfall workflow converts heterogeneous source files into a
        standardized time series with the following core fields:

        **Date/Time** and **Rainfall (mm)**.
        """
    )

    st.markdown(
        """
        **Processing sequence**

        1. Upload rainfall data.
        2. Detect file encoding and separator where applicable.
        3. Identify metadata/header rows.
        4. Detect date/time and rainfall columns.
        5. Convert date/time values to a consistent timestamp format.
        6. Convert rainfall values to numeric values.
        7. Check missing, negative and invalid observations.
        8. Standardize the dataset for subsequent analysis.
        """
    )

    st.latex(r"R_{\mathrm{total}}=\sum_{i=1}^{n} R_i")

    st.caption(
        "Rainfall totals are calculated from the standardized rainfall "
        "observations. Missing observations are not automatically assumed "
        "to represent zero rainfall."
    )

    # ------------------------------------------------------------
    # 4. RAINFALL EVENT ANALYSIS
    # ------------------------------------------------------------
    st.header("4. Rainfall Event Analysis")

    st.write(
        "Rainfall events are identified from consecutive rainfall observations "
        "using a dry-period threshold. The threshold is selected with respect "
        "to the temporal resolution of the rainfall dataset."
    )

    st.markdown(
        """
        An event is characterized by:

        - Event start
        - Event end
        - Event duration
        - Total event rainfall
        - Maximum interval rainfall
        - Interval-average intensity
        - Number of rainfall records
        - Antecedent dry period
        """
    )

    st.info(
        "Event separation is resolution-aware. For example, daily rainfall "
        "data require a different interpretation of dry periods than "
        "minute-resolution rainfall data."
    )

    # ------------------------------------------------------------
    # 5. TRAFFIC PROCESSING
    # ------------------------------------------------------------
    st.header("5. Traffic Data Processing")

    st.write(
        "Traffic data are processed independently from rainfall. The "
        "application detects the structure of the uploaded dataset and "
        "standardizes the available traffic information."
    )

    st.markdown(
        """
        **Generic workflow**

        1. Upload the traffic dataset.
        2. Detect or select the date/time field.
        3. Detect or select the traffic-volume field.
        4. Identify optional vehicle-class or station fields.
        5. Convert traffic values to numeric values.
        6. Validate timestamps and traffic values.
        7. Create the standardized traffic time series.
        """
    )

    st.info(
        "When vehicle-class information is available, the pollutant-loading "
        "module can use class-specific emission factors. When it is not "
        "available, a generic traffic-based scenario can be used."
    )

    # ------------------------------------------------------------
    # 6. RAINFALL–TRAFFIC SYNCHRONIZATION
    # ------------------------------------------------------------
    st.header("6. Rainfall–Traffic Time Synchronization")

    st.write(
        "Rainfall and traffic datasets must be compared over a common "
        "analysis period before they are combined. The application therefore "
        "identifies the temporal overlap between the two datasets."
    )

    st.markdown(
        """
        **Synchronization sequence**

        1. Determine rainfall start and end timestamps.
        2. Determine traffic start and end timestamps.
        3. Calculate the common overlap period.
        4. Compare temporal resolutions.
        5. Aggregate the datasets to a compatible analysis interval.
        6. Create the synchronized rainfall–traffic dataset.
        7. Report missing aligned values and the final analysis period.
        """
    )

    st.latex(
        r"t_{\mathrm{start,common}}="
        r"\max(t_{\mathrm{start,rain}},t_{\mathrm{start,traffic}})"
    )
    st.latex(
        r"t_{\mathrm{end,common}}="
        r"\min(t_{\mathrm{end,rain}},t_{\mathrm{end,traffic}})"
    )

    st.warning(
        "The synchronized period is based on the actual overlap of the "
        "uploaded datasets. The application does not silently extend one "
        "dataset beyond the period covered by the other."
    )

    # ------------------------------------------------------------
    # 7. POLLUTANT LOADING
    # ------------------------------------------------------------
    st.header("7. Traffic-Based Pollutant Loading")

    st.write(
        "Traffic data can be converted into a pollutant mass time series "
        "using user-defined emission factors and road characteristics. "
        "The calculation is transparent so that the assumptions can be "
        "reviewed and changed."
    )

    st.markdown(
        """
        For a generic traffic dataset, the conceptual calculation is:

        **Traffic volume × emission factor × road length × correction factors**
        """
    )

    st.latex(
        r"M_{\mathrm{pollutant}}="
        r"\frac{N_{\mathrm{vehicles}}\;EF\;L\;K}{10^6}"
    )

    st.caption(
        "The exact units and correction factors depend on the selected "
        "calculation setup. The application displays the active parameters "
        "in the Pollutant Loading module."
    )

    # ------------------------------------------------------------
    # 8. VEHICLE-CLASS METHOD
    # ------------------------------------------------------------
    st.header("8. Vehicle-Class Emission Method")

    st.write(
        "If the traffic dataset contains separate vehicle classes, "
        "class-specific emission factors can be applied."
    )

    st.markdown(
        """
        The implemented NIQKI vehicle-class workflow supports categories such as:

        - Passenger cars / light commercial vehicles
        - Heavy goods vehicles
        - Buses

        The corresponding emission factors are user-configurable in the
        Pollutant Loading module.
        """
    )

    st.latex(
        r"M_{\mathrm{TWP}}="
        r"\frac{\sum_j N_j\,EF_j\,L\,K_{\mathrm{drive}}\,K_{\mathrm{road}}}"
        r"{10^6}"
    )

    st.caption(
        "This formulation is applicable when the uploaded traffic dataset "
        "contains the corresponding vehicle-class counts."
    )

    # ------------------------------------------------------------
    # 9. BUILDUP / WASH-OFF
    # ------------------------------------------------------------
    st.header("9. Pollutant Buildup and Wash-off")

    st.write(
        "The pollutant-loading workflow represents the accumulation of "
        "traffic-derived pollutant mass on the contributing road surface "
        "and its subsequent removal during rainfall."
    )

    st.markdown(
        """
        The implemented daily buildup/wash-off workflow includes:

        - Pollutant generation
        - Dry-period buildup
        - Rainfall-driven wash-off
        - Verge retention
        - Gully/trap retention
        - Resulting mass transferred to the sewer system
        """
    )

    st.latex(
        r"B_{\mathrm{before}}="
        r"B_{\mathrm{previous}}e^{-k_{\mathrm{loss}}\Delta t}"
        r"+\frac{M_{\mathrm{gen}}}{k_{\mathrm{loss}}}"
        r"\left(1-e^{-k_{\mathrm{loss}}\Delta t}\right)"
    )

    st.latex(
        r"M_{\mathrm{washed}}="
        r"\min\left(C_1 q^{C_2}B_{\mathrm{before}}\Delta t,"
        r"B_{\mathrm{before}}\right)"
    )

    st.latex(
        r"M_{\mathrm{sewer}}="
        r"M_{\mathrm{washed}}(1-\eta_{\mathrm{verge}})"
        r"(1-\eta_{\mathrm{trap}})"
    )

    st.caption(
        "The active parameter values are displayed in the Pollutant Loading "
        "module when the calculation is performed."
    )

    # ------------------------------------------------------------
    # 10. RUNOFF MODEL
    # ------------------------------------------------------------
    st.header("10. Runoff Calculation")

    st.write(
        "The simplified runoff module estimates effective rainfall and "
        "runoff volume using the selected model parameters."
    )

    st.latex(
        r"P_{\mathrm{effective}}="
        r"P_{\mathrm{total}}(1-C)"
    )

    st.latex(
        r"V_{\mathrm{runoff}}="
        r"P_{\mathrm{effective}}\,A"
    )

    st.caption(
        "The simplified runoff calculation is an engineering screening "
        "calculation. It should not be interpreted as a replacement for "
        "a full hydrodynamic SWMM simulation."
    )

    # ------------------------------------------------------------
    # 11. SWMM PREPARATION
    # ------------------------------------------------------------
    st.header("11. EPA SWMM Input Preparation")

    st.write(
        "The SWMM module prepares external rainfall and pollutant input "
        "files for connection to an EPA SWMM model."
    )

    st.markdown(
        """
        **Rainfall input**

        The application prepares a Rain Gage external time series containing:

        `Station ID · Year · Month · Day · Hour · Minute · Rainfall`

        **Pollutant input**

        The pollutant workflow prepares an external mass-inflow time series
        containing:

        `Date · Time · Mass Inflow (kg/day)`
        """
    )

    st.info(
        "The rainfall `.dat` file and pollutant-inflow `.dat` file are "
        "different SWMM inputs. A pollutant inflow DAT is not a traffic "
        "dataset and should not be uploaded as raw traffic data."
    )

    # ------------------------------------------------------------
    # 12. QUALITY CONTROL
    # ------------------------------------------------------------
    st.header("12. Quality Control and Validation")

    st.markdown(
        """
        Before model preparation, the application checks the consistency
        of the uploaded data. Typical checks include:

        - Date/time values are valid
        - Time series are ordered
        - Required columns are available
        - Rainfall values are non-negative
        - Traffic values are valid numeric values
        - Recording interval can be determined
        - Rainfall and traffic periods overlap
        - Missing values are reported
        - Generated SWMM records follow the required structure
        """
    )

    st.success(
        "The application reports unresolved missing rainfall values as "
        "a review condition rather than silently converting them to zero."
    )

    # ------------------------------------------------------------
    # 13. ASSUMPTIONS / LIMITATIONS
    # ------------------------------------------------------------
    st.header("13. Assumptions and Limitations")

    st.markdown(
        """
        **Important interpretation points**

        1. The quality of the output depends on the quality and temporal
           coverage of the uploaded source datasets.

        2. Rainfall and traffic data must have a valid common analysis period
           before combined calculations are performed.

        3. Emission factors are model parameters and should be documented
           for the selected study/application scenario.

        4. Missing rainfall observations are not equivalent to zero rainfall.

        5. The simplified runoff calculation is not a full hydrodynamic
           simulation.

        6. The SWMM module prepares external input files; it does not replace
           the user's EPA SWMM model setup and independent model validation.

        7. Traffic-source formats are not hard-coded as a scientific
           requirement. Source-specific parsing is used only where the
           structure of a dataset requires it.
        """
    )

    # ------------------------------------------------------------
    # 14. COMPLETE WORKFLOW
    # ------------------------------------------------------------
    st.header("14. Complete NIQKI Workflow")

    st.markdown(
        """
        **Step 1 — Upload**

        Rainfall and traffic datasets are imported.

        **Step 2 — Validate**

        File structure, timestamps, values and data quality are checked.

        **Step 3 — Analyze**

        Rainfall statistics, events and traffic characteristics are derived.

        **Step 4 — Synchronize**

        The common rainfall–traffic analysis period is established.

        **Step 5 — Calculate pollutant loading**

        Traffic information is converted into pollutant generation and
        buildup/wash-off results using the selected model parameters.

        **Step 6 — Model**

        Runoff and pollutant results are prepared for model analysis.

        **Step 7 — SWMM preparation**

        External rainfall and pollutant input files are generated for
        connection to EPA SWMM.
        """
    )

    st.divider()

    st.success(
        "Methodology reference: use the individual application modules "
        "to inspect the actual input data, active parameters, calculated "
        "results and generated SWMM files."
    )


elif page == "Rainfall Data":
    st.title("Rainfall Data")

    st.write(
        """
        Upload rainfall data from different sources. The application
        automatically attempts to detect the file encoding, separator,
        header row, date/time column and rainfall column.
        """
    )

    st.divider()
    st.subheader("1. Upload Rainfall Dataset")

    uploaded_file = st.file_uploader(
        "Choose a rainfall file",
        type=["csv", "txt", "dat", "xlsx", "xls", "zip"],
        help=(
            "You can upload a single rainfall file or a ZIP archive "
            "containing CSV, TXT, DAT or Excel rainfall data."
        )
    )

    if uploaded_file is None:
        st.info("Please upload a rainfall file or ZIP archive.")

    else:
        try:
            selected_file, extension, source_name = prepare_uploaded_data_file(
                uploaded_file,
                "rainfall_zip_member",
                "rainfall"
            )
        except Exception as exc:
            st.error(f"Could not read the uploaded file/archive: {exc}")
            st.stop()

        signature = (
            uploaded_file.name,
            uploaded_file.size,
            source_name
        )

        if st.session_state.file_signature != signature:
            st.session_state.rainfall_data = None
            st.session_state.model_results = None
            st.session_state.model_parameters = None
            st.session_state.rainfall_events = None
            st.session_state.file_signature = signature

        try:
            extension = extension

            if extension in ["xlsx", "xls"]:
                sheets = get_excel_sheets(selected_file)

                if len(sheets) > 1:
                    sheet_name = st.selectbox("Select worksheet", sheets)
                else:
                    sheet_name = sheets[0]

                df = read_excel_sheet(selected_file, sheet_name)
                encoding_used = "Excel"
                separator_used = "Not applicable"
                header_row = None

            else:
                (
                    df,
                    encoding_used,
                    separator_used,
                    header_row
                ) = read_text_file(selected_file)

            st.subheader("2. File Information")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("File Type", extension.upper())

            with c2:
                st.metric("Rows", f"{len(df):,}")

            with c3:
                st.metric("Columns", f"{len(df.columns):,}")

            with c4:
                st.metric(
                    "File Size",
                    f"{selected_file.size / 1024:.1f} KB"
                )

            if get_extension(uploaded_file.name) == "zip":
                st.info(
                    f"ZIP archive selected: **{uploaded_file.name}** → "
                    f"using **{source_name}**"
                )

            if extension not in ["xlsx", "xls"]:
                st.write(f"**Encoding:** {encoding_used}")

                separator_display = (
                    "Tab" if separator_used == "\t"
                    else separator_used
                )

                st.write(
                    f"**Detected separator:** {separator_display}"
                )

                st.write(
                    f"**Detected header row:** {header_row + 1}"
                )

            st.subheader("3. Imported Data Preview")

            st.dataframe(
                df.head(20),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("4. Automatic Column Analysis")

            analysis = analyse_columns(df)

            st.dataframe(
                analysis,
                use_container_width=True,
                hide_index=True
            )

            suggested_date = suggest_date_column(analysis)
            suggested_rainfall = suggest_rainfall_column(analysis)
            columns = list(df.columns)

            st.subheader("5. Identify Rainfall Variables")

            date_index = (
                columns.index(suggested_date)
                if suggested_date in columns
                else 0
            )

            date_column = st.selectbox(
                "Date / Time column",
                columns,
                index=date_index
            )

            if suggested_rainfall in columns:
                rainfall_index = columns.index(suggested_rainfall)
            else:
                other_columns = [
                    i for i, col in enumerate(columns)
                    if col != date_column
                ]
                rainfall_index = other_columns[0] if other_columns else 0

            rainfall_column = st.selectbox(
                "Rainfall column",
                columns,
                index=rainfall_index
            )

            st.write("**Selected data preview:**")

            selected_preview = pd.DataFrame({
                "Date/Time": df[date_column].head(10),
                "Rainfall": df[rainfall_column].head(10)
            })

            st.dataframe(
                selected_preview,
                use_container_width=True,
                hide_index=True
            )

            if st.button(
                "Confirm and Process Rainfall Data",
                type="primary"
            ):
                standardized = standardize_rainfall_data(
                    df,
                    date_column,
                    rainfall_column
                )

                if standardized.empty:
                    st.error(
                        "No valid date/time records could be detected."
                    )
                else:
                    st.session_state.rainfall_data = standardized
                    st.session_state.model_results = None
                    st.session_state.rainfall_events = None

                    st.success(
                        "Rainfall data standardized successfully."
                    )

            standardized = st.session_state.rainfall_data

            if standardized is not None:
                st.subheader("6. Standardized Rainfall Data")

                st.dataframe(
                    standardized.head(20),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("7. Data Quality Check")

                quality = calculate_quality(standardized)

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric("Records", f"{quality['total']:,}")

                with c2:
                    st.metric(
                        "Available Rainfall",
                        f"{quality['valid']:,}"
                    )

                with c3:
                    st.metric(
                        "Missing Rainfall",
                        f"{quality['missing']:,}"
                    )

                with c4:
                    st.metric(
                        "Completeness",
                        f"{quality['completeness']:.2f}%"
                    )

                if quality["duplicates"] > 0:
                    st.warning(
                        f"{quality['duplicates']:,} duplicate timestamps detected."
                    )

                if quality["negative"] > 0:
                    st.warning(
                        f"{quality['negative']:,} negative rainfall values detected."
                    )

                st.subheader("8. Temporal Resolution")

                (
                    resolution,
                    interval_minutes,
                    irregular
                ) = detect_temporal_resolution(standardized)

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Detected Resolution", resolution)

                with c2:
                    if interval_minutes is None:
                        interval_text = "Unknown"
                    elif interval_minutes < 60:
                        interval_text = f"{interval_minutes:g} minutes"
                    else:
                        interval_text = f"{interval_minutes / 60:g} hours"

                    st.metric("Typical Interval", interval_text)

                with c3:
                    st.metric(
                        "Irregular Intervals",
                        f"{irregular:.2f}%"
                    )

                st.subheader("9. Rainfall Statistics")

                stats = rainfall_statistics(standardized)

                if stats:
                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.metric(
                            "Total Rainfall",
                            f"{stats['total']:.2f} mm"
                        )

                    with c2:
                        st.metric(
                            "Maximum",
                            f"{stats['maximum']:.3f} mm"
                        )

                    with c3:
                        st.metric(
                            "Average",
                            f"{stats['average']:.4f} mm"
                        )

                    with c4:
                        st.metric(
                            "Non-zero Records",
                            f"{stats['non_zero']:,}"
                        )

                st.subheader("10. Rainfall Time Series")

                chart_data = (
                    standardized[["Date/Time", "Rainfall (mm)"]]
                    .set_index("Date/Time")
                )

                if len(chart_data) > 200000:
                    chart_data = chart_data.resample("1h").sum(min_count=1)
                elif len(chart_data) > 50000:
                    chart_data = chart_data.resample("15min").sum(min_count=1)

                st.line_chart(
                    chart_data,
                    use_container_width=True
                )

                st.subheader("11. Rainfall Event Analysis")

                default_dry = default_dry_period(interval_minutes)

                dry_period = st.number_input(
                    "Minimum dry period between rainfall events (hours)",
                    min_value=1.0,
                    max_value=720.0,
                    value=float(default_dry),
                    step=1.0
                )

                st.info(
                    f"A new rainfall event begins after more than "
                    f"{dry_period:g} hours without positive rainfall."
                )

                events = rainfall_events(
                    standardized,
                    dry_period,
                    interval_minutes
                )

                # Keep the event settings in session state so the
                # Results page can reproduce the event analysis even
                # after navigation or a Streamlit rerun.
                st.session_state.rainfall_events = events
                st.session_state.event_dry_period_hours = float(dry_period)

                if not events.empty:
                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.metric(
                            "Number of Events",
                            f"{len(events):,}"
                        )

                    with c2:
                        st.metric(
                            "Largest Event",
                            f"{events['Total Rainfall (mm)'].max():.2f} mm"
                        )

                    with c3:
                        st.metric(
                            "Longest Event",
                            f"{events['Duration (hours)'].max():.2f} h"
                        )

                    with c4:
                        st.metric(
                            "Maximum Interval",
                            f"{events['Maximum Rainfall per Interval (mm)'].max():.3f} mm"
                        )

                    st.dataframe(
                        format_event_table(events),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No positive-rainfall events were identified.")

                st.subheader("12. Download Standardized Data")

                download_df = standardized.copy()
                download_df["Date/Time"] = (
                    download_df["Date/Time"]
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                )

                csv_data = (
                    download_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    "Download Standardized Rainfall CSV",
                    data=csv_data,
                    file_name="standardized_rainfall.csv",
                    mime="text/csv",
                    key="download_standardized_rainfall"
                )

        except Exception as error:
            st.error("Could not process the uploaded rainfall file.")
            st.exception(error)


# ============================================================
# TRAFFIC DATA
# ============================================================

elif page == "Traffic Data":
    st.title("Traffic Data")
    st.write(
        "Upload a traffic dataset in CSV, TXT, DAT, Excel or ZIP format. "
        "The application automatically inspects the structure and lets you "
        "select the date/time and traffic-count fields. Large delimited files "
        "are processed in chunks."
    )

    st.divider()

    input_type = st.radio(
        "Input type",
        [
            "Automatic traffic dataset",
            "Existing SWMM pollutant inflow DAT"
        ],
        horizontal=True,
        key="traffic_input_type"
    )

    if input_type == "Existing SWMM pollutant inflow DAT":
        st.subheader("Upload Existing SWMM Pollutant Inflow")

        reference_file = st.file_uploader(
            "Choose an existing SWMM pollutant inflow DAT file",
            type=["dat", "txt", "zip"],
            key="pollutant_reference_uploader",
        )

        if reference_file is not None:
            try:
                reference_input, _, reference_source = prepare_uploaded_data_file(
                    reference_file,
                    "pollutant_zip_member",
                    "SWMM pollutant inflow"
                )
                reference = read_swmm_external_inflow_file(reference_input)
                st.session_state.pollutant_reference_data = reference
                st.session_state.pollutant_reference_source_info = reference_source

                st.success(
                    f"SWMM pollutant inflow read successfully · "
                    f"{len(reference):,} records"
                )
                st.dataframe(
                    reference.head(1000),
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    "Download parsed pollutant CSV",
                    reference.to_csv(index=False).encode("utf-8"),
                    "parsed_swmm_pollutant_inflow.csv",
                    "text/csv",
                    key="download_reference_pollutant_csv"
                )
            except Exception as exc:
                st.error(f"Could not process the pollutant inflow file: {exc}")

    else:
        st.subheader("1. Upload Traffic Dataset")

        traffic_file = st.file_uploader(
            "Choose a traffic dataset",
            type=["csv", "txt", "dat", "xlsx", "xls", "zip"],
            key="traffic_generic_auto_uploader",
            help="ZIP files may contain large TXT/CSV/DAT traffic datasets."
        )

        if traffic_file is not None:
            try:
                ext = get_extension(traffic_file.name)

                if ext in ["xlsx", "xls"]:
                    sheets = get_excel_sheets(traffic_file)
                    sheet_name = st.selectbox(
                        "Excel sheet",
                        sheets,
                        key="traffic_auto_excel_sheet"
                    )
                    raw_traffic = read_excel_sheet(traffic_file, sheet_name)
                    columns = [str(c).strip() for c in raw_traffic.columns]
                    analysis = analyse_traffic_columns(raw_traffic)

                    source_info = f"Excel sheet: {sheet_name}"
                    station_column = None
                    station_value = None

                else:
                    preview_info = traffic_detect_text_source(traffic_file)
                    separator = preview_info["separator"]
                    encoding = preview_info["encoding"]

                    if ext == "zip":
                        st.success(
                            f"ZIP detected · source file: **{preview_info['member']}**"
                        )
                    else:
                        st.success("Delimited traffic file detected.")

                    columns = traffic_text_columns(
                        traffic_file, separator, encoding
                    )

                    # Read only a small preview for scoring/column selection.
                    traffic_file.seek(0)
                    archive_preview = None
                    if ext == "zip":
                        archive_preview = zipfile.ZipFile(traffic_file)
                        member = _zip_member_name(traffic_file)
                        stream_preview = archive_preview.open(member, "r")
                    else:
                        stream_preview = traffic_file

                    try:
                        preview = pd.read_csv(
                            stream_preview,
                            sep=separator,
                            encoding=encoding,
                            dtype=str,
                            nrows=2000,
                            engine="python",
                            on_bad_lines="skip"
                        )
                    finally:
                        if archive_preview is not None:
                            archive_preview.close()

                    preview.columns = [str(c).strip() for c in preview.columns]
                    analysis = analyse_traffic_columns(preview)
                    source_info = (
                        f"{encoding}; separator={repr(separator)}"
                    )
                    station_column = None
                    station_value = None

                st.subheader("2. Automatic Structure Detection")

                is_bast = generic_traffic_is_bast(columns)

                if is_bast:
                    st.info(
                        "A BASt/mFUND-style structure was detected. "
                        "This is treated as one supported traffic format, "
                        "not as a requirement. You can select the station "
                        "and the traffic-count field like any other dataset."
                    )

                    if "Zst" in columns:
                        station_column = st.selectbox(
                            "Optional counting-station field",
                            ["None", "Zst"],
                            key="traffic_station_column"
                        )
                        if station_column == "None":
                            station_column = None
                        else:
                            with st.spinner(
                                "Finding available station values..."
                            ):
                                stations = generic_traffic_station_values(
                                    traffic_file,
                                    station_column,
                                    separator,
                                    encoding
                                )

                            if stations:
                                station_value = st.selectbox(
                                    "Counting station",
                                    stations,
                                    key="traffic_station_value"
                                )
                            else:
                                st.warning(
                                    "No station values were found; all records "
                                    "will be considered."
                                )
                                station_column = None

                if not analysis.empty:
                    st.dataframe(
                        analysis.sort_values(
                            ["Date Score", "Traffic Score"],
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                columns = list(columns)
                date_suggestion = suggest_traffic_date_column(analysis)
                count_suggestion = suggest_traffic_count_column(analysis)

                # For BASt, prefer the real date/hour fields if available.
                if is_bast and "Datum" in columns:
                    date_col = "Datum"
                    date_is_combined = False
                else:
                    date_col = st.selectbox(
                        "Date/time column",
                        columns,
                        index=columns.index(date_suggestion)
                        if date_suggestion in columns else 0,
                        key="traffic_auto_date_column"
                    )
                    date_is_combined = True

                if is_bast:
                    # BASt date is YYMMDD and Stunde is a separate hour field.
                    hour_col = st.selectbox(
                        "Hour column (optional)",
                        ["None"] + [c for c in columns if c.lower() in {
                            "stunde", "hour", "hh", "zeit"
                        }],
                        key="traffic_auto_hour_column"
                    )
                    count_options = [
                        c for c in columns
                        if c.lower() in {
                            "kfz_r1", "kfz_r2", "traffic", "traffic_count",
                            "vehicles", "vehicle_count"
                        }
                    ]
                    if count_options:
                        default_count = (
                            "KFZ_R1"
                            if "KFZ_R1" in count_options
                            else count_options[0]
                        )
                    else:
                        default_count = count_suggestion

                    count_col = st.selectbox(
                        "Traffic-count column",
                        columns,
                        index=columns.index(default_count)
                        if default_count in columns else 0,
                        key="traffic_auto_count_column"
                    )

                    st.caption(
                        "For this source, the selected count column is used "
                        "as supplied. Both directions can be represented by "
                        "selecting an already aggregated KFZ field or by "
                        "processing each direction separately."
                    )

                    process_bast_like = st.checkbox(
                        "Combine KFZ_R1 + KFZ_R2 as total traffic",
                        value=True,
                        key="traffic_combine_kfz"
                    )
                else:
                    hour_col = "None"
                    count_col = st.selectbox(
                        "Traffic-count column",
                        columns,
                        index=columns.index(count_suggestion)
                        if count_suggestion in columns else 0,
                        key="traffic_auto_count_column_generic"
                    )
                    process_bast_like = False

                process_label = (
                    "Process selected traffic dataset"
                )

                if st.button(
                    process_label,
                    type="primary",
                    key="process_traffic_generic"
                ):
                    if ext in ["xlsx", "xls"]:
                        traffic = prepare_traffic_data(
                            raw_traffic, date_col, count_col
                        )
                    elif is_bast:
                        # Build a generic, memory-safe selected-column processor.
                        # For YYMMDD + hour, create a combined timestamp from
                        # only the required columns.
                        if date_col == "Datum" and hour_col != "None":
                            # Read selected columns in chunks.
                            selected_cols = [date_col, hour_col, count_col]
                            if station_column:
                                selected_cols.append(station_column)

                            archive = None
                            try:
                                if ext == "zip":
                                    traffic_file.seek(0)
                                    archive = zipfile.ZipFile(traffic_file)
                                    member = _zip_member_name(traffic_file)
                                    stream = archive.open(member, "r")
                                else:
                                    traffic_file.seek(0)
                                    stream = traffic_file

                                parts = []
                                for chunk in pd.read_csv(
                                    stream,
                                    sep=separator,
                                    encoding=encoding,
                                    dtype=str,
                                    usecols=list(dict.fromkeys(selected_cols)),
                                    chunksize=200_000,
                                    low_memory=True,
                                    engine="python",
                                    on_bad_lines="skip"
                                ):
                                    chunk.columns = [
                                        str(c).strip() for c in chunk.columns
                                    ]

                                    if station_column and station_value is not None:
                                        chunk = chunk[
                                            chunk[station_column].astype(str).str.strip()
                                            == str(station_value).strip()
                                        ]

                                    if chunk.empty:
                                        continue

                                    dates = pd.to_datetime(
                                        chunk[date_col].astype(str).str.strip(),
                                        format="%y%m%d",
                                        errors="coerce"
                                    )
                                    hours = pd.to_numeric(
                                        chunk[hour_col], errors="coerce"
                                    ).fillna(0).clip(0, 23)

                                    prepared = pd.DataFrame({
                                        "Date/Time": dates
                                        + pd.to_timedelta(hours, unit="h"),
                                        "Traffic Count": pd.to_numeric(
                                            chunk[count_col], errors="coerce"
                                        )
                                    }).dropna(subset=["Date/Time"])

                                    prepared["Traffic Count"] = (
                                        prepared["Traffic Count"]
                                        .clip(lower=0)
                                    )

                                    # Optional sum of the two standard BASt
                                    # directions when requested and available.
                                    if (
                                        process_bast_like
                                        and count_col == "KFZ_R1"
                                        and "KFZ_R2" in columns
                                    ):
                                        # A separate pass would be needed for R2;
                                        # do not silently pretend it is included.
                                        st.warning(
                                            "KFZ_R1 was selected. The app does not "
                                            "silently add KFZ_R2 unless both are read. "
                                            "Select an already combined traffic field "
                                            "or use KFZ_R1/KFZ_R2 explicitly."
                                        )

                                    parts.append(prepared)

                                if parts:
                                    traffic = (
                                        pd.concat(parts, ignore_index=True)
                                        .groupby(
                                            "Date/Time", as_index=False
                                        )["Traffic Count"]
                                        .sum()
                                        .sort_values("Date/Time")
                                        .reset_index(drop=True)
                                    )
                                else:
                                    traffic = pd.DataFrame(
                                        columns=["Date/Time", "Traffic Count"]
                                    )
                            finally:
                                if archive is not None:
                                    archive.close()
                        else:
                            traffic = generic_traffic_process(
                                traffic_file,
                                date_col,
                                count_col,
                                separator,
                                encoding,
                                station_column,
                                station_value
                            )
                    else:
                        traffic = generic_traffic_process(
                            traffic_file,
                            date_col,
                            count_col,
                            separator,
                            encoding,
                            station_column,
                            station_value
                        )

                    if traffic.empty:
                        st.error(
                            "No valid traffic records could be created. "
                            "Check the selected columns and station filter."
                        )
                    else:
                        st.session_state.traffic_data = traffic
                        st.session_state.traffic_source_info = source_info

                        summary = traffic_summary(traffic)

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Records", f"{summary['records']:,}")
                        c2.metric(
                            "Total vehicles",
                            f"{summary['total_vehicles']:,.0f}"
                        )
                        c3.metric(
                            "Average / record",
                            f"{summary['average']:,.1f}"
                        )
                        c4.metric(
                            "Maximum / record",
                            f"{summary['maximum']:,.0f}"
                        )

                        st.write(
                            f"**Period:** "
                            f"{summary['start']:%d.%m.%Y %H:%M} → "
                            f"{summary['end']:%d.%m.%Y %H:%M}"
                        )

                        st.line_chart(
                            traffic.set_index("Date/Time")[["Traffic Count"]]
                        )

                        st.dataframe(
                            traffic.head(1000),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download standardized traffic CSV",
                            traffic.to_csv(index=False).encode("utf-8"),
                            "NIQKI_standardized_traffic.csv",
                            "text/csv",
                            key="download_generic_traffic_csv"
                        )

                        st.success(
                            "Traffic data successfully processed. "
                            "The dataset is now available for synchronization."
                        )

            except Exception as exc:
                st.error(
                    f"Could not inspect/process the traffic file: {exc}"
                )



elif page == "Data Synchronization":
    st.title("Rainfall–Traffic Data Synchronization")
    st.write(
        "Establish the common analysis period between rainfall and traffic "
        "before applying the NIQKI TWP buildup/wash-off model."
    )

    rainfall_sync = st.session_state.get("rainfall_data")
    traffic_sync = st.session_state.get("traffic_data")

    if rainfall_sync is None or rainfall_sync.empty:
        st.warning("Upload and process rainfall data first.")
    elif traffic_sync is None or traffic_sync.empty:
        st.warning("Upload and process traffic data first.")
    else:
        rain = rainfall_sync.copy()
        traf = traffic_sync.copy()

        rain["Date/Time"] = pd.to_datetime(rain["Date/Time"], errors="coerce")
        rain["Rainfall (mm)"] = pd.to_numeric(
            rain["Rainfall (mm)"], errors="coerce"
        )
        traf["Date/Time"] = pd.to_datetime(traf["Date/Time"], errors="coerce")

        rain = rain.dropna(subset=["Date/Time"]).sort_values("Date/Time")
        traf = traf.dropna(subset=["Date/Time"]).sort_values("Date/Time")

        if rain.empty or traf.empty:
            st.error("One of the datasets has no valid timestamps.")
        else:
            rain_start, rain_end = rain["Date/Time"].min(), rain["Date/Time"].max()
            traf_start, traf_end = traf["Date/Time"].min(), traf["Date/Time"].max()

            overlap_start = max(rain_start, traf_start)
            overlap_end = min(rain_end, traf_end)

            st.subheader("1. Source Period Comparison")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rainfall**")
                st.write(f"{rain_start:%d.%m.%Y %H:%M} → {rain_end:%d.%m.%Y %H:%M}")
                st.write(f"Records: **{len(rain):,}**")
            with c2:
                st.markdown("**Traffic**")
                st.write(f"{traf_start:%d.%m.%Y %H:%M} → {traf_end:%d.%m.%Y %H:%M}")
                st.write(f"Records: **{len(traf):,}**")

            if overlap_start > overlap_end:
                st.error(
                    "There is no common analysis period. The rainfall and traffic "
                    "datasets cannot be synchronized as supplied."
                )
            else:
                overlap_days = (
                    overlap_end - overlap_start
                ).total_seconds() / 86400.0

                st.success(
                    f"Common analysis period found: **{overlap_start:%d.%m.%Y} → "
                    f"{overlap_end:%d.%m.%Y} ({overlap_days:.1f} days)**"
                )

                st.subheader("2. Daily Synchronization")

                st.info(
                    "The original NIQKI traffic-to-TWP model is a daily model: "
                    "BASt hourly traffic observations are aggregated to daily "
                    "vehicle totals, while rainfall is aggregated to daily rainfall "
                    "depth before the buildup/wash-off calculation."
                )

                # Always use daily rainfall for compatibility with the original model.
                rain_common = rain[
                    (rain["Date/Time"] >= overlap_start) &
                    (rain["Date/Time"] <= overlap_end)
                ].copy()
                traf_common = traf[
                    (traf["Date/Time"] >= overlap_start) &
                    (traf["Date/Time"] <= overlap_end)
                ].copy()

                rain_daily = (
                    rain_common.assign(
                        Date=pd.to_datetime(rain_common["Date/Time"]).dt.normalize()
                    )
                    .groupby("Date", as_index=False)["Rainfall (mm)"]
                    .sum(min_count=1)
                    .rename(columns={"Date": "Date/Time"})
                )

                traffic_cols = [
                    c for c in
                    ["Traffic Count", "PKW_Van", "LKW", "Bus"]
                    if c in traf_common.columns
                ]

                if not traffic_cols:
                    traf_daily = (
                        traf_common.assign(
                            Date=pd.to_datetime(traf_common["Date/Time"]).dt.normalize()
                        )
                        .groupby("Date", as_index=False)["Traffic Count"]
                        .sum(min_count=1)
                        .rename(columns={"Date": "Date/Time"})
                    )
                else:
                    traf_daily = (
                        traf_common.assign(
                            Date=pd.to_datetime(traf_common["Date/Time"]).dt.normalize()
                        )
                        .groupby("Date", as_index=False)[traffic_cols]
                        .sum(min_count=1)
                        .rename(columns={"Date": "Date/Time"})
                    )

                synchronized = pd.merge(
                    rain_daily,
                    traf_daily,
                    on="Date/Time",
                    how="inner"
                ).sort_values("Date/Time").reset_index(drop=True)

                missing_rain = int(synchronized["Rainfall (mm)"].isna().sum())
                traffic_count_missing = (
                    int(synchronized["Traffic Count"].isna().sum())
                    if "Traffic Count" in synchronized.columns else 0
                )

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Common days", f"{len(synchronized):,}")
                r2.metric("Start", synchronized["Date/Time"].min().strftime("%d.%m.%Y") if not synchronized.empty else "—")
                r3.metric("End", synchronized["Date/Time"].max().strftime("%d.%m.%Y") if not synchronized.empty else "—")
                r4.metric("Missing aligned values", f"{missing_rain + traffic_count_missing:,}")

                if synchronized.empty:
                    st.error("No common daily records were produced.")
                else:
                    st.dataframe(
                        synchronized.head(1000),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.session_state.synchronized_data = synchronized
                    st.session_state.sync_summary = {
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "overlap_days": overlap_days,
                        "records": len(synchronized),
                        "mode": "Daily — original NIQKI TWP model compatibility"
                    }

                    st.download_button(
                        "Download synchronized rainfall–traffic CSV",
                        synchronized.to_csv(index=False).encode("utf-8"),
                        "NIQKI_synchronized_rainfall_traffic_daily.csv",
                        "text/csv",
                        key="download_synchronized_daily_csv"
                    )

                    st.success(
                        "Daily synchronization is complete. The synchronized dataset "
                        "is ready for the original NIQKI TWP buildup/wash-off model."
                    )



elif page == "Pollutant Loading":
    st.title("Pollutant Loading")
    st.write(
        "Convert the synchronized traffic dataset into pollutant mass loading. "
        "The calculation is generic: emission factors, road length and model "
        "parameters are user-configurable rather than tied to one traffic provider."
    )

    traffic = st.session_state.get("traffic_data")
    rainfall = st.session_state.get("rainfall_data")
    synchronized = st.session_state.get("synchronized_data")

    if traffic is None or traffic.empty:
        st.warning("Upload and process traffic data first.")
    elif rainfall is None or rainfall.empty:
        st.warning("Upload and process rainfall data first.")
    else:
        st.subheader("1. Traffic-to-Pollutant Parameters")

        method = st.radio(
            "Calculation method",
            [
                "Generic traffic emission factor",
                "Vehicle-class emission factors"
            ],
            horizontal=True,
            key="pollutant_method"
        )

        road_length = st.number_input(
            "Road segment length (km)",
            min_value=0.1,
            max_value=100.0,
            value=2.0,
            step=0.1,
            key="generic_road_length"
        )

        if method == "Generic traffic emission factor":
            emission_factor = st.number_input(
                "Emission factor (g pollutant / vehicle / km)",
                min_value=0.0,
                value=0.09,
                step=0.01,
                key="generic_emission_factor"
            )
            driving_factor = st.number_input(
                "Driving dynamics factor",
                min_value=0.0,
                value=1.3,
                step=0.05,
                key="generic_driving_factor"
            )
            surface_factor = st.number_input(
                "Road surface factor",
                min_value=0.0,
                value=1.1,
                step=0.05,
                key="generic_surface_factor"
            )

            st.latex(
                r"M_{\mathrm{generation}}="
                r"\frac{N_{\mathrm{vehicles}}\;EF\;L\;"
                r"K_{\mathrm{drive}}\;K_{\mathrm{road}}}{1000}"
            )

            if st.button(
                "Calculate pollutant generation",
                type="primary",
                key="calculate_generic_pollutant"
            ):
                calc = traffic.copy()
                calc["Traffic Count"] = pd.to_numeric(
                    calc["Traffic Count"], errors="coerce"
                ).fillna(0).clip(lower=0)

                calc["M_gen_daily_kg"] = (
                    calc["Traffic Count"]
                    * float(emission_factor)
                    * float(road_length)
                    * float(driving_factor)
                    * float(surface_factor)
                    / 1000.0
                )

                traffic_daily = (
                    calc.assign(
                        Date=calc["Date/Time"].dt.normalize()
                    )
                    .groupby("Date", as_index=False)["M_gen_daily_kg"]
                    .sum()
                    .rename(columns={"Date": "Date/Time"})
                )

        else:
            if not all(
                c in traffic.columns
                for c in ["PKW_Van", "LKW", "Bus"]
            ):
                st.warning(
                    "Vehicle-class fields are not present in this traffic dataset. "
                    "Use the generic emission-factor method, or upload a dataset "
                    "that contains separate vehicle classes."
                )
                traffic_daily = None
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    ef_pkw = st.number_input(
                        "PKW/LFW (mg/veh/km)",
                        min_value=0.0, value=90.0, step=1.0,
                        key="class_ef_pkw"
                    )
                with c2:
                    ef_lkw = st.number_input(
                        "LKW/LZG/SAT (mg/veh/km)",
                        min_value=0.0, value=800.0, step=10.0,
                        key="class_ef_lkw"
                    )
                with c3:
                    ef_bus = st.number_input(
                        "Bus (mg/veh/km)",
                        min_value=0.0, value=700.0, step=10.0,
                        key="class_ef_bus"
                    )

                driving_factor = st.number_input(
                    "Driving dynamics factor",
                    min_value=0.0, value=1.3, step=0.05,
                    key="class_driving_factor"
                )
                surface_factor = st.number_input(
                    "Road surface factor",
                    min_value=0.0, value=1.1, step=0.05,
                    key="class_surface_factor"
                )

                st.latex(
                    r"M_{\mathrm{generation}}="
                    r"\frac{(N_{PKW/LFW}EF_{PKW}"
                    r"+N_{LKW/LZG/SAT}EF_{LKW}"
                    r"+N_{Bus}EF_{Bus})LK_{drive}K_{road}}{10^6}"
                )

                if st.button(
                    "Calculate pollutant generation",
                    type="primary",
                    key="calculate_class_pollutant"
                ):
                    calc = traffic.copy()
                    for col in ["PKW_Van", "LKW", "Bus"]:
                        calc[col] = pd.to_numeric(
                            calc[col], errors="coerce"
                        ).fillna(0).clip(lower=0)

                    calc["M_generation"] = (
                        (
                            calc["PKW_Van"] * float(ef_pkw)
                            + calc["LKW"] * float(ef_lkw)
                            + calc["Bus"] * float(ef_bus)
                        )
                        * float(road_length)
                        * float(driving_factor)
                        * float(surface_factor)
                        / 1e6
                    )

                    traffic_daily = (
                        calc.assign(
                            Date=calc["Date/Time"].dt.normalize()
                        )
                        .groupby("Date", as_index=False)["M_generation"]
                        .sum()
                        .rename(
                            columns={
                                "Date": "Date/Time",
                                "M_generation": "M_gen_daily_kg"
                            }
                        )
                    )

        if "traffic_daily" in locals() and traffic_daily is not None and not traffic_daily.empty:
            # Daily rainfall, then common date intersection.
            rain = rainfall.copy()
            rain["Date/Time"] = pd.to_datetime(
                rain["Date/Time"], errors="coerce"
            )
            rain["Rainfall (mm)"] = pd.to_numeric(
                rain["Rainfall (mm)"], errors="coerce"
            ).fillna(0.0)

            rain_daily = (
                rain.assign(
                    Date=rain["Date/Time"].dt.normalize()
                )
                .groupby("Date", as_index=False)["Rainfall (mm)"]
                .sum()
                .rename(columns={"Date": "Date/Time"})
            )

            df_sim = pd.merge(
                rain_daily,
                traffic_daily,
                on="Date/Time",
                how="inner"
            ).sort_values("Date/Time").reset_index(drop=True)

            if df_sim.empty:
                st.error(
                    "Rainfall and traffic do not share a common daily period."
                )
            else:
                st.subheader("2. Buildup / Wash-off Parameters")

                b1, b2, b3 = st.columns(3)
                with b1:
                    k_loss = st.number_input(
                        "Buildup loss rate k_loss (1/day)",
                        min_value=0.0001, value=0.05, step=0.005,
                        key="generic_k_loss"
                    )
                    c1_washoff = st.number_input(
                        "Wash-off coefficient C1",
                        min_value=0.0, value=0.05, step=0.01,
                        key="generic_c1"
                    )
                with b2:
                    c2_washoff = st.number_input(
                        "Wash-off exponent C2",
                        min_value=0.0, value=1.3, step=0.1,
                        key="generic_c2"
                    )
                    eta_verge = st.number_input(
                        "Verge retention η",
                        min_value=0.0, max_value=1.0,
                        value=0.40, step=0.05,
                        key="generic_eta_verge"
                    )
                with b3:
                    eta_trap = st.number_input(
                        "Gully/trap retention η",
                        min_value=0.0, max_value=1.0,
                        value=0.30, step=0.05,
                        key="generic_eta_trap"
                    )

                result = bast_daily_washoff(
                    df_sim,
                    k_loss=k_loss,
                    c1_washoff=c1_washoff,
                    c2_washoff=c2_washoff,
                    eta_verge=eta_verge,
                    eta_trap=eta_trap
                )

                st.session_state.pollutant_data = result
                st.session_state.traffic_parameters = {
                    "method": method,
                    "road_length_km": float(road_length),
                    "k_loss": float(k_loss),
                    "C1_washoff": float(c1_washoff),
                    "C2_washoff": float(c2_washoff),
                    "eta_verge": float(eta_verge),
                    "eta_trap": float(eta_trap)
                }

                st.subheader("3. Results")

                total_generated = float(
                    result["M_gen_daily_kg"].sum()
                )
                total_sewer = float(
                    result["Sewer_Mass_kg"].sum()
                )

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Total generated pollutant",
                    f"{total_generated:,.4f} kg"
                )
                c2.metric(
                    "Total sewer discharge",
                    f"{total_sewer:,.4f} kg"
                )
                c3.metric(
                    "Maximum daily sewer mass",
                    f"{result['Sewer_Mass_kg'].max():,.4f} kg/day"
                )

                st.dataframe(
                    result.head(1000),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download pollutant results CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    "NIQKI_pollutant_loading_results.csv",
                    "text/csv",
                    key="download_generic_pollutant_results"
                )

                dat_df = result[
                    ["Date/Time", "Sewer_Mass_kg"]
                ].rename(
                    columns={"Sewer_Mass_kg": "Mass Inflow (kg/day)"}
                )

                dat_text = generate_swmm_inflow_dat(
                    dat_df,
                    title="NIQKI generic traffic pollutant loading"
                )

                st.subheader("4. SWMM Pollutant Inflow DAT")
                st.code(
                    "\n".join(dat_text.splitlines()[:15]),
                    language="text"
                )
                st.download_button(
                    "Download pollutant SWMM inflow DAT",
                    dat_text.encode("utf-8"),
                    "twp_swmm_inflow.dat",
                    "text/plain",
                    key="download_generic_pollutant_dat"
                )



elif page == "Model Parameters":
    st.title("Mathematical Model Parameters")

    rainfall = st.session_state.rainfall_data

    if rainfall is None:
        st.warning("Please process a rainfall dataset first.")
    else:
        st.success("Rainfall dataset is ready for modelling.")
        st.divider()

        st.subheader("1. Catchment Parameters")

        col1, col2 = st.columns(2)

        with col1:
            area_ha = st.number_input(
                "Catchment area (ha)",
                min_value=0.001,
                value=10.0,
                step=0.5
            )

        with col2:
            runoff_coefficient = st.number_input(
                "Runoff coefficient C",
                min_value=0.0,
                max_value=1.0,
                value=0.70,
                step=0.05
            )

        st.caption(
            "Effective rainfall = Rainfall × Runoff coefficient"
        )

        st.subheader("2. Pollutant Parameters")

        pollutant_type = st.selectbox(
            "Pollutant / pollutant proxy",
            [
                "Tyre abrasion / microplastic proxy",
                "Suspended solids",
                "Generic pollutant",
                "User-defined pollutant"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:
            pollutant_buildup = st.number_input(
                "Surface pollutant buildup (kg/ha)",
                min_value=0.0,
                value=0.10,
                step=0.01,
                format="%.4f"
            )

        with col2:
            washoff_coefficient = st.number_input(
                "Wash-off coefficient (1/mm)",
                min_value=0.0,
                value=0.10,
                step=0.01,
                format="%.4f"
            )

        st.divider()

        st.subheader("3. Mathematical Model")

        st.write("Effective rainfall:")
        st.latex(r"P_e = C \times P")

        st.write("Runoff volume:")
        st.latex(r"V = P_e \times A \times 10")

        st.write("Runoff flow:")
        st.latex(r"Q = \frac{V}{\Delta t}")

        st.write("Wash-off fraction:")
        st.latex(r"f_w = 1-e^{-k_wP}")

        st.write("Pollutant load:")
        st.latex(r"M = B \times A \times f_w")

        st.warning(
            """
            The pollutant buildup and wash-off values are configurable
            example inputs. Replace them with project-specific values
            after the NIQKI methodology and literature parameters are
            confirmed.
            """
        )

        if st.button(
            "Save Model Parameters",
            type="primary"
        ):
            st.session_state.model_parameters = {
                "area_ha": area_ha,
                "runoff_coefficient": runoff_coefficient,
                "pollutant_type": pollutant_type,
                "pollutant_buildup": pollutant_buildup,
                "washoff_coefficient": washoff_coefficient
            }

            st.success("Model parameters saved successfully.")


# ============================================================
# SIMULATION
# ============================================================

elif page == "Simulation":
    st.title("Rainfall → Runoff → Pollutant Simulation")

    rainfall = st.session_state.rainfall_data
    parameters = st.session_state.model_parameters

    if rainfall is None:
        st.warning("Please process rainfall data first.")

    elif parameters is None:
        st.warning(
            "Please configure and save the model parameters first."
        )

    else:
        st.success(
            "Rainfall data and model parameters are ready."
        )

        st.divider()
        st.subheader("Current Model Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Catchment Area",
                f"{parameters['area_ha']:.2f} ha"
            )

        with col2:
            st.metric(
                "Runoff Coefficient",
                f"{parameters['runoff_coefficient']:.2f}"
            )

        with col3:
            st.metric(
                "Pollutant",
                parameters["pollutant_type"]
            )

        st.divider()
        st.subheader("Model Workflow")

        st.code(
            """
Rainfall
    ↓
Effective Rainfall
    ↓
Runoff Volume
    ↓
Runoff Flow
    ↓
Pollutant Wash-off
    ↓
Pollutant Load
"""
        )

        if st.button(
            "Run Mathematical Model",
            type="primary"
        ):
            with st.spinner("Running mathematical model..."):
                try:
                    results = run_mathematical_model(
                        rainfall,
                        parameters["area_ha"],
                        parameters["runoff_coefficient"],
                        parameters["pollutant_buildup"],
                        parameters["washoff_coefficient"]
                    )

                    st.session_state.model_results = results

                    st.success(
                        "Mathematical model completed successfully."
                    )

                except Exception as error:
                    st.error(
                        "The mathematical model could not be completed."
                    )
                    st.exception(error)

        results = st.session_state.model_results

        if results is not None:
            st.divider()
            st.subheader("Simulation Summary")

            total_rainfall = results["Rainfall (mm)"].sum()
            total_effective = results["Effective Rainfall (mm)"].sum()
            total_runoff = results["Runoff Volume (m³)"].sum()
            maximum_flow = results["Runoff Flow (m³/s)"].max()
            total_pollutant = results["Pollutant Load (kg)"].sum()

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "Total Rainfall",
                    f"{total_rainfall:.2f} mm"
                )

            with col2:
                st.metric(
                    "Effective Rainfall",
                    f"{total_effective:.2f} mm"
                )

            with col3:
                st.metric(
                    "Runoff Volume",
                    f"{total_runoff:.2f} m³"
                )

            with col4:
                st.metric(
                    "Maximum Flow",
                    f"{maximum_flow:.4f} m³/s"
                )

            with col5:
                st.metric(
                    "Pollutant Load",
                    f"{total_pollutant:.4f} kg"
                )


# ============================================================
# IMPROVED RESULTS PAGE
# ============================================================

elif page == "Results":
    st.title("Model Results")

    results = st.session_state.model_results
    events = st.session_state.rainfall_events

    if results is None:
        st.info(
            "Run the mathematical model first."
        )

    else:
        st.success(
            "Model results are available for analysis."
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader("1. Overall Model Summary")

        total_rainfall = float(results["Rainfall (mm)"].sum())
        total_effective = float(results["Effective Rainfall (mm)"].sum())
        total_runoff = float(results["Runoff Volume (m³)"].sum())
        peak_flow = float(results["Runoff Flow (m³/s)"].max())
        total_pollutant = float(results["Pollutant Load (kg)"].sum())

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric("Total Rainfall", f"{total_rainfall:.2f} mm")

        with c2:
            st.metric("Effective Rainfall", f"{total_effective:.2f} mm")

        with c3:
            st.metric("Runoff Volume", f"{total_runoff:.2f} m³")

        with c4:
            st.metric("Peak Calculated Flow", f"{peak_flow:.4f} m³/s")

        with c5:
            st.metric("Pollutant Load", f"{total_pollutant:.4f} kg")

        st.caption(
            """
            Note: the calculated runoff flow is a time-step-average
            flow derived from runoff volume divided by the detected
            rainfall interval. It is not a hydrodynamic SWMM hydrograph.
            """
        )

        st.divider()

        # ----------------------------------------------------
        # RAINFALL VS RUNOFF
        # ----------------------------------------------------

        st.subheader("2. Rainfall and Runoff Comparison")

        comparison = results[
            [
                "Date/Time",
                "Rainfall (mm)",
                "Effective Rainfall (mm)",
                "Runoff Flow (m³/s)"
            ]
        ].copy()

        comparison = comparison.set_index("Date/Time")

        if len(comparison) > 200000:
            comparison = comparison.resample("1h").agg({
                "Rainfall (mm)": "sum",
                "Effective Rainfall (mm)": "sum",
                "Runoff Flow (m³/s)": "mean"
            })
        elif len(comparison) > 50000:
            comparison = comparison.resample("15min").agg({
                "Rainfall (mm)": "sum",
                "Effective Rainfall (mm)": "sum",
                "Runoff Flow (m³/s)": "mean"
            })

        st.line_chart(
            comparison,
            use_container_width=True
        )

        st.caption(
            "Rainfall and calculated runoff response over the selected time period."
        )

        # ----------------------------------------------------
        # RUNOFF FLOW
        # ----------------------------------------------------

        st.subheader("3. Calculated Runoff Flow")

        flow = results[
            [
                "Date/Time",
                "Runoff Flow (m³/s)"
            ]
        ].set_index("Date/Time")

        if len(flow) > 200000:
            flow = flow.resample("1h").mean()
        elif len(flow) > 50000:
            flow = flow.resample("15min").mean()

        st.line_chart(
            flow,
            use_container_width=True
        )

        # ----------------------------------------------------
        # CUMULATIVE RUNOFF
        # ----------------------------------------------------

        st.subheader("4. Cumulative Runoff")

        cumulative = results[
            [
                "Date/Time",
                "Cumulative Runoff (m³)"
            ]
        ].set_index("Date/Time")

        if len(cumulative) > 200000:
            cumulative = cumulative.resample("1h").last()
        elif len(cumulative) > 50000:
            cumulative = cumulative.resample("15min").last()

        st.line_chart(
            cumulative,
            use_container_width=True
        )

        # ----------------------------------------------------
        # EVENT ANALYSIS
        # ----------------------------------------------------

        st.subheader("5. Event-Based Results")

        # Rebuild the event table automatically if it is missing.
        # This makes the Results page independent of page-navigation
        # order and robust to Streamlit reruns.
        if events is None or events.empty:
            source_rainfall = st.session_state.rainfall_data

            if source_rainfall is not None and not source_rainfall.empty:
                _, interval_minutes, _ = detect_temporal_resolution(
                    source_rainfall
                )

                saved_dry_period = (
                    st.session_state.event_dry_period_hours
                )

                if saved_dry_period is None:
                    saved_dry_period = default_dry_period(
                        interval_minutes
                    )

                events = rainfall_events(
                    source_rainfall,
                    float(saved_dry_period),
                    interval_minutes
                )

                st.session_state.rainfall_events = events
                st.session_state.event_dry_period_hours = float(
                    saved_dry_period
                )

        if events is None or events.empty:
            st.warning(
                "No rainfall events could be identified from the processed "
                "rainfall dataset."
            )

            st.write(
                "Go to **Rainfall Data → Rainfall Event Analysis** if you "
                "want to change the dry-period threshold."
            )

        else:
            saved_dry_period = st.session_state.event_dry_period_hours

            if saved_dry_period is not None:
                st.info(
                    f"Event separation threshold: "
                    f"{saved_dry_period:g} hours without positive rainfall."
                )

            event_results = calculate_event_model_results(
                results,
                events
            )

            if event_results.empty:
                st.warning(
                    "Rainfall events were detected, but no matching model "
                    "results were found for those events."
                )

            else:
                # Event selector
                event_numbers = event_results["Event"].astype(int).tolist()

                selected_event = st.selectbox(
                    "Select a rainfall event",
                    event_numbers,
                    format_func=lambda x: f"Event {x}"
                )

                selected = event_results[
                    event_results["Event"] == selected_event
                ].iloc[0]

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "Event Rainfall",
                        f"{selected['Rainfall (mm)']:.2f} mm"
                    )

                with c2:
                    st.metric(
                        "Event Runoff",
                        f"{selected['Runoff Volume (m³)']:.2f} m³"
                    )

                with c3:
                    st.metric(
                        "Peak Flow",
                        f"{selected['Peak Flow (m³/s)']:.4f} m³/s"
                    )

                with c4:
                    st.metric(
                        "Pollutant Load",
                        f"{selected['Pollutant Load (kg)']:.4f} kg"
                    )

                selected_start = pd.to_datetime(selected["Start"])
                selected_end = pd.to_datetime(selected["End"])

                event_timeseries = results[
                    (results["Date/Time"] >= selected_start)
                    & (results["Date/Time"] <= selected_end)
                ].copy()

                st.write(
                    f"**Event period:** "
                    f"{selected_start.strftime('%d.%m.%Y %H:%M')} → "
                    f"{selected_end.strftime('%d.%m.%Y %H:%M')}"
                )

                # Separate event rainfall and runoff plots so the
                # small runoff-flow values are not hidden by the
                # rainfall scale.
                event_rainfall = event_timeseries[
                    ["Date/Time", "Rainfall (mm)"]
                ].set_index("Date/Time")

                event_flow = event_timeseries[
                    ["Date/Time", "Runoff Flow (m³/s)"]
                ].set_index("Date/Time")

                st.write("**Event rainfall:**")
                st.line_chart(
                    event_rainfall,
                    use_container_width=True
                )

                st.write("**Event runoff flow:**")
                st.line_chart(
                    event_flow,
                    use_container_width=True
                )

                st.write("**All event-level results:**")

                display_events = event_results.copy()

                for col in ["Start", "End"]:
                    display_events[col] = pd.to_datetime(
                        display_events[col],
                        errors="coerce"
                    ).dt.strftime("%d.%m.%Y %H:%M")

                numeric_cols = [
                    "Rainfall (mm)",
                    "Runoff Volume (m³)",
                    "Peak Flow (m³/s)",
                    "Pollutant Load (kg)"
                ]

                for col in numeric_cols:
                    if col in display_events.columns:
                        display_events[col] = display_events[col].round(4)

                st.dataframe(
                    display_events,
                    use_container_width=True,
                    hide_index=True
                )

                event_csv = (
                    event_results
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    "Download Event Results CSV",
                    data=event_csv,
                    file_name="NIQKI_event_results.csv",
                    mime="text/csv",
                    key="download_event_results"
                )

        # ----------------------------------------------------
        # POLLUTANT LOAD
        # ----------------------------------------------------

        st.subheader("6. Pollutant Load")

        pollutant = results[
            [
                "Date/Time",
                "Pollutant Load (kg)"
            ]
        ].set_index("Date/Time")

        if len(pollutant) > 200000:
            pollutant = pollutant.resample("1h").sum()
        elif len(pollutant) > 50000:
            pollutant = pollutant.resample("15min").sum()

        st.bar_chart(
            pollutant,
            use_container_width=True
        )

        # ----------------------------------------------------
        # CUMULATIVE POLLUTANT
        # ----------------------------------------------------

        st.subheader("7. Cumulative Pollutant Load")

        cumulative_pollutant = results[
            [
                "Date/Time",
                "Cumulative Pollutant Load (kg)"
            ]
        ].set_index("Date/Time")

        if len(cumulative_pollutant) > 200000:
            cumulative_pollutant = (
                cumulative_pollutant.resample("1h").last()
            )
        elif len(cumulative_pollutant) > 50000:
            cumulative_pollutant = (
                cumulative_pollutant.resample("15min").last()
            )

        st.line_chart(
            cumulative_pollutant,
            use_container_width=True
        )

        # ----------------------------------------------------
        # OUTPUT DATA
        # ----------------------------------------------------

        st.subheader("8. Model Output Data")

        st.dataframe(
            results.head(100),
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DOWNLOAD FULL RESULTS
        # ----------------------------------------------------

        st.subheader("9. Download Model Results")

        output = results.copy()

        output["Date/Time"] = (
            output["Date/Time"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        output_csv = (
            output
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Full Model Results CSV",
            data=output_csv,
            file_name="NIQKI_model_results.csv",
            mime="text/csv",
            key="download_full_model_results"
        )

# ============================================================
# SWMM PAGE — OPTION A: RAINFALL INPUT PREPARATION
# ============================================================

elif page == "SWMM":
    st.title("SWMM Rainfall Preparation")

    st.write(
        """
        Prepare a standardized rainfall file for use with an EPA SWMM
        Rain Gage. This module does not run SWMM. It creates the
        external rainfall file that can be selected in the SWMM Rain
        Gage editor.
        """
    )

    rainfall = st.session_state.rainfall_data

    if rainfall is None or rainfall.empty:
        st.warning(
            "Please process a rainfall dataset first on the Rainfall Data page."
        )
    else:
        st.success("Standardized rainfall data is available.")

        # --------------------------------------------------------
        # 1. SOURCE RAINFALL INFORMATION
        # --------------------------------------------------------

        st.subheader("1. Source Rainfall Information")

        resolution, interval_minutes, irregular = detect_temporal_resolution(
            rainfall
        )
        quality = calculate_quality(rainfall)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Records", f"{len(rainfall):,}")

        with c2:
            st.metric("Time Resolution", resolution)

        with c3:
            st.metric("Missing Rainfall", f"{quality['missing']:,}")

        with c4:
            st.metric("Completeness", f"{quality['completeness']:.2f}%")

        st.write(
            f"**Period:** "
            f"{rainfall['Date/Time'].min().strftime('%d.%m.%Y %H:%M:%S')} "
            f"→ "
            f"{rainfall['Date/Time'].max().strftime('%d.%m.%Y %H:%M:%S')}"
        )

        st.divider()

        # --------------------------------------------------------
        # 2. SWMM RAIN GAGE SETTINGS
        # --------------------------------------------------------

        st.subheader("2. SWMM Rain Gage Settings")

        col1, col2 = st.columns(2)

        with col1:
            gage_name = st.text_input(
                "SWMM Rain Gage / Station ID",
                value="RAIN_GAGE_01",
                help=(
                    "This becomes the station ID in the external SWMM "
                    "rainfall file. Use letters, numbers, underscore or hyphen."
                ),
                key="swmm_gage_name"
            )

            rainfall_unit = st.selectbox(
                "Rainfall Unit",
                ["mm", "in"],
                index=0,
                key="swmm_rainfall_unit"
            )

        with col2:
            station_name = st.text_input(
                "Station / Source Description",
                value="Rainfall Station",
                key="swmm_station_description"
            )

            if interval_minutes is not None:
                detected_interval_text = (
                    f"{interval_minutes:g} minutes"
                )
            else:
                detected_interval_text = "Unknown"

            st.info(
                f"Detected recording interval: **{detected_interval_text}**"
            )

        # --------------------------------------------------------
        # 3. OUTPUT INTERVAL
        # --------------------------------------------------------

        st.subheader("3. SWMM Recording Interval")

        interval_options = {
            "Use detected interval": interval_minutes,
            "1 minute": 1,
            "5 minutes": 5,
            "10 minutes": 10,
            "15 minutes": 15,
            "30 minutes": 30,
            "1 hour": 60,
            "2 hours": 120,
            "3 hours": 180,
            "6 hours": 360,
            "12 hours": 720,
            "24 hours": 1440
        }

        recording_interval = st.selectbox(
            "Recording Interval",
            list(interval_options.keys()),
            index=0,
            key="swmm_recording_interval"
        )

        output_interval = interval_options[recording_interval]

        if output_interval is None or output_interval <= 0:
            st.error(
                "The rainfall interval could not be determined. "
                "Please select an interval manually."
            )
            st.stop()

        st.info(
            f"SWMM recording interval: **{output_interval:g} minutes** "
            f"({output_interval / 60:g} hours)"
        )

        # For rainfall depths measured over each recording interval,
        # VOLUME is the appropriate SWMM rain format.
        rain_format = "VOLUME"

        st.write(
            f"**SWMM Rain Format:** `{rain_format}` — "
            "rainfall depth accumulated during each recording interval."
        )

        # --------------------------------------------------------
        # 4. MISSING DATA TREATMENT
        # --------------------------------------------------------

        st.subheader("4. Missing Rainfall Treatment")

        missing_options = [
            "Keep missing values as missing",
            "Replace missing values with 0.00"
        ]

        missing_treatment = st.selectbox(
            "Treatment of missing rainfall values",
            missing_options,
            index=0,
            key="swmm_missing_treatment"
        )

        if quality["missing"] > 0:
            st.warning(
                f"This dataset contains {quality['missing']:,} missing "
                "rainfall observations."
            )

            st.caption(
                "Important: an external SWMM rainfall file does not have "
                "a dedicated missing-value field. If a missing observation "
                "is converted to zero, SWMM will treat it as zero rainfall. "
                "Only do this when zero is scientifically justified."
            )

        # --------------------------------------------------------
        # 5. PREPARE REGULAR TIME SERIES
        # --------------------------------------------------------

        st.subheader("5. Prepare SWMM Rainfall Data")

        prepared = rainfall.copy()

        prepared["Date/Time"] = pd.to_datetime(
            prepared["Date/Time"],
            errors="coerce"
        )

        prepared["Rainfall (mm)"] = pd.to_numeric(
            prepared["Rainfall (mm)"],
            errors="coerce"
        )

        prepared = prepared.dropna(
            subset=["Date/Time"]
        )

        prepared = (
            prepared
            .sort_values("Date/Time")
            .drop_duplicates(
                subset=["Date/Time"],
                keep="first"
            )
            .set_index("Date/Time")
        )

        # Resample to the selected SWMM interval.
        rule = f"{int(output_interval)}min"

        # Rainfall values represent depth accumulated during each
        # source interval, therefore aggregation to a longer interval
        # is by summation.
        prepared = prepared.resample(rule).sum(min_count=1)

        prepared = prepared.reset_index()

        if missing_treatment == "Replace missing values with 0.00":
            prepared["Rainfall (mm)"] = (
                prepared["Rainfall (mm)"].fillna(0.0)
            )

        # Remove physically invalid negative rainfall.
        prepared.loc[
            prepared["Rainfall (mm)"] < 0,
            "Rainfall (mm)"
        ] = np.nan

        # Convert to requested SWMM unit.
        if rainfall_unit == "in":
            prepared["Rainfall"] = (
                prepared["Rainfall (mm)"] / 25.4
            )
        else:
            prepared["Rainfall"] = (
                prepared["Rainfall (mm)"]
            )

        prepared["Rainfall"] = pd.to_numeric(
            prepared["Rainfall"],
            errors="coerce"
        )

        # --------------------------------------------------------
        # 6. PREPARED DATA CHECK
        # --------------------------------------------------------

        st.subheader("6. Prepared SWMM Data Check")

        prepared_missing = int(
            prepared["Rainfall"].isna().sum()
        )

        prepared_records = len(prepared)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Prepared Records",
                f"{prepared_records:,}"
            )

        with c2:
            st.metric(
                "Output Interval",
                f"{output_interval:g} min"
            )

        with c3:
            st.metric(
                "Missing Values",
                f"{prepared_missing:,}"
            )

        with c4:
            total_prepared = prepared["Rainfall"].sum(
                skipna=True
            )

            st.metric(
                f"Total Rainfall ({rainfall_unit})",
                f"{total_prepared:.3f}"
            )

        if prepared_missing > 0:
            st.warning(
                f"{prepared_missing:,} values remain missing. "
                "These cannot be represented as a special missing value "
                "in the standard SWMM external rainfall format."
            )

        st.dataframe(
            prepared[
                ["Date/Time", "Rainfall"]
            ].head(50),
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------------
        # 7. CREATE STANDARD SWMM EXTERNAL RAINFALL FILE
        # --------------------------------------------------------

        st.subheader("7. SWMM External Rainfall File")

        st.write(
            """
            The generated `.dat` file uses SWMM's standard user-prepared
            rainfall-file structure:

            `StationID Year Month Day Hour Minute Rainfall`
            """
        )

        st.code(
            "RAIN_GAGE_01 2023 1 18 0 0 0.17\n"
            "RAIN_GAGE_01 2023 1 19 0 0 12.52\n"
            "RAIN_GAGE_01 2023 1 20 0 0 0.73",
            language="text"
        )

        # Sanitize station ID.
        clean_gage_name = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            gage_name.strip()
        )

        if not clean_gage_name:
            clean_gage_name = "RAIN_GAGE_01"

        # Standard SWMM rainfall-file format:
        # station year month day hour minute non-zero precipitation
        #
        # Zero rainfall records are not required in this format.
        # Missing values are NOT silently written as zero unless the
        # user explicitly selected that treatment.
        dat_lines = []

        export_data = prepared.copy()

        if missing_treatment == "Replace missing values with 0.00":
            export_data["Rainfall"] = (
                export_data["Rainfall"].fillna(0.0)
            )

        for _, row in export_data.iterrows():

            value = row["Rainfall"]

            if pd.isna(value):
                continue

            value = float(value)

            # Standard SWMM user-prepared rainfall files contain
            # non-zero precipitation readings.
            if value <= 0:
                continue

            dt = row["Date/Time"]

            dat_lines.append(
                f"{clean_gage_name} "
                f"{dt.year} "
                f"{dt.month} "
                f"{dt.day} "
                f"{dt.hour} "
                f"{dt.minute} "
                f"{value:.6f}".rstrip("0").rstrip(".")
            )

        dat_content = (
            "\n".join(dat_lines) + "\n"
            if dat_lines
            else ""
        ).encode("utf-8")

        st.write(
            f"**Station ID:** `{clean_gage_name}`  \n"
            f"**Rain units:** `{rainfall_unit.upper()}`  \n"
            f"**Rain format:** `{rain_format}`  \n"
            f"**Recording interval:** `{output_interval:g} minutes`  \n"
            f"**Non-zero rainfall records exported:** `{len(dat_lines):,}`"
        )

        if dat_lines:
            st.subheader("SWMM File Preview")

            st.code(
                "\n".join(dat_lines[:20]),
                language="text"
            )

            st.caption(
                "The preview shows the first 20 non-zero rainfall records."
            )

            st.download_button(
                "Download SWMM Rainfall DAT",
                data=dat_content,
                file_name=f"{clean_gage_name}_rainfall.dat",
                mime="text/plain",
                key="swmm_external_rainfall_dat_download"
            )

        else:
            st.error(
                "No non-zero rainfall records are available for export."
            )

        # --------------------------------------------------------
        # 8. SWMM-READY CSV FOR INSPECTION
        # --------------------------------------------------------

        st.subheader("8. Download Standardized CSV")

        csv_output = prepared[
            ["Date/Time", "Rainfall"]
        ].copy()

        csv_output["Date/Time"] = (
            csv_output["Date/Time"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        csv_output = csv_output.rename(
            columns={
                "Rainfall": f"Rainfall ({rainfall_unit})"
            }
        )

        csv_content = (
            csv_output
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Prepared Rainfall CSV",
            data=csv_content,
            file_name=f"{clean_gage_name}_prepared_rainfall.csv",
            mime="text/csv",
            key="swmm_prepared_csv_download"
        )

        # --------------------------------------------------------
        # 9. SWMM RAIN GAGE SETUP
        # --------------------------------------------------------

        st.subheader("9. SWMM Rain Gage Setup")

        st.write(
            """
            In EPA SWMM, create or edit the Rain Gage and use the
            external-file data source. The station ID must match the
            first field of the generated `.dat` file.
            """
        )

        st.code(
            f"""Rain Gage: {clean_gage_name}
Data Source: FILE
Rain Format: VOLUME
Rain Interval: {output_interval / 60:g} hours
Rain Units: {rainfall_unit.upper()}
Data File: {clean_gage_name}_rainfall.dat
Station ID: {clean_gage_name}""",
            language="text"
        )

        st.info(
            """
            The web application stops here by design. It prepares and
            validates the rainfall input but does not execute SWMM.
            """
        )

        # --------------------------------------------------------
        # 10. VALIDATION CHECKLIST
        # --------------------------------------------------------

        st.subheader("10. SWMM Import Checklist")

        checks = [
            (
                "Rainfall column identified",
                True
            ),
            (
                "Date/time values valid",
                bool(prepared["Date/Time"].notna().all())
            ),
            (
                "Rainfall values non-negative",
                bool(
                    (
                        prepared["Rainfall"].dropna() >= 0
                    ).all()
                )
            ),
            (
                "Recording interval defined",
                bool(output_interval > 0)
            ),
            (
                "Rainfall unit defined",
                rainfall_unit in ["mm", "in"]
            ),
            (
                "Station ID defined",
                bool(clean_gage_name)
            ),
            (
                "Prepared records available",
                bool(len(prepared) > 0)
            ),
            (
                "No unresolved missing values",
                prepared_missing == 0
            ),
            (
                "SWMM DAT records generated",
                len(dat_lines) > 0
            )
        ]

        checklist_df = pd.DataFrame(
            {
                "Check": [item[0] for item in checks],
                "Status": [
                    "PASS" if item[1] else "REVIEW"
                    for item in checks
                ]
            }
        )

        st.dataframe(
            checklist_df,
            use_container_width=True,
            hide_index=True
        )

        if all(item[1] for item in checks):
            st.success(
                "SWMM rainfall preparation passed all validation checks."
            )
        else:
            st.warning(
                "Review the items marked REVIEW before importing the "
                "rainfall file into SWMM."
            )

        st.caption(
            f"Source: {station_name}"
        )
