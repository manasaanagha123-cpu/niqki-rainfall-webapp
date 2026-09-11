import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
from io import BytesIO

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NIQKI Rainfall & Runoff Web Model",
    page_icon="💧",
    layout="wide"
)

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
    "file_signature": None
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

def detect_temporal_resolution(df):
    dates = (
        df["Date/Time"]
        .dropna()
        .sort_values()
        .drop_duplicates()
    )

    if len(dates) < 2:
        return "Unknown", None, 0.0

    differences = dates.diff().dropna().dt.total_seconds() / 60
    differences = differences[differences > 0]

    if differences.empty:
        return "Unknown", None, 0.0

    mode = differences.mode()
    interval = float(mode.iloc[0]) if not mode.empty else float(differences.median())

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
        (label for minutes, label in known.items() if abs(interval - minutes) < 0.01),
        f"{interval:.2f} minutes" if interval < 60 else f"{interval / 60:.2f} hours"
    )

    irregular = (abs(differences - interval) > 0.01).sum()
    irregular_percentage = irregular / len(differences) * 100

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
# SIDEBAR
# ============================================================

st.sidebar.title("NIQKI Web Model")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Rainfall Data",
        "Model Parameters",
        "Simulation",
        "Results",
        "SWMM"
    ]
)

# ============================================================
# HOME
# ============================================================

if page == "Home":
    st.title("NIQKI Rainfall & Runoff Web Model")

    st.subheader(
        "Rainfall Data Processing, Event Analysis and Mathematical Modelling"
    )

    st.write(
        """
        This web application provides a flexible workflow for importing
        rainfall data from different sources, automatically identifying
        relevant columns, standardizing the time series, checking data
        quality, separating rainfall events and performing a configurable
        rainfall-runoff-pollutant calculation.
        """
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Rainfall")
        st.write("Import and standardize rainfall datasets from different sources.")

    with c2:
        st.subheader("Runoff")
        st.write("Calculate effective rainfall, runoff volume and flow.")

    with c3:
        st.subheader("Pollutants")
        st.write("Estimate wash-off and pollutant load indicators.")

    st.divider()

    st.info("Start with Rainfall Data and upload your rainfall file.")


# ============================================================
# RAINFALL DATA
# ============================================================

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
        type=["csv", "txt", "dat", "xlsx", "xls"]
    )

    if uploaded_file is None:
        st.info("Please upload a rainfall file.")

    else:
        signature = (uploaded_file.name, uploaded_file.size)

        if st.session_state.file_signature != signature:
            st.session_state.rainfall_data = None
            st.session_state.model_results = None
            st.session_state.model_parameters = None
            st.session_state.rainfall_events = None
            st.session_state.file_signature = signature

        try:
            extension = get_extension(uploaded_file.name)

            if extension in ["xlsx", "xls"]:
                sheets = get_excel_sheets(uploaded_file)

                if len(sheets) > 1:
                    sheet_name = st.selectbox("Select worksheet", sheets)
                else:
                    sheet_name = sheets[0]

                df = read_excel_sheet(uploaded_file, sheet_name)
                encoding_used = "Excel"
                separator_used = "Not applicable"
                header_row = None

            else:
                (
                    df,
                    encoding_used,
                    separator_used,
                    header_row
                ) = read_text_file(uploaded_file)

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
                    f"{uploaded_file.size / 1024:.1f} KB"
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
# MODEL PARAMETERS
# ============================================================

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
