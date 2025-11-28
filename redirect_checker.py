import streamlit as st
import pandas as pd
import requests
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

# ======================
# CONFIG
# ======================
DB_PATH = "redirect_monitor.db"

# How often to check BEFORE redirect is OK (in minutes)
PENDING_CHECK_INTERVAL_MIN = 30

# How often to check AFTER redirect is OK (in minutes)
OK_CHECK_INTERVAL_MIN = 60 * 24  # once per day


# ======================
# DB HELPERS
# ======================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS redirects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client TEXT NOT NULL,
            old_url TEXT NOT NULL,
            new_url TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            last_checked TEXT,
            created_at TEXT NOT NULL,
            details TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_redirect(client: str, old_url: str, new_url: str):
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO redirects (client, old_url, new_url, status, last_checked, created_at, details)
        VALUES (?, ?, ?, 'pending', NULL, ?, '')
        """,
        (client.strip(), old_url.strip(), new_url.strip(), now),
    )
    conn.commit()
    conn.close()


def load_redirects() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM redirects", conn)
    conn.close()
    return df


def update_redirect(row_id: int, status: str, details: str):
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE redirects SET status = ?, last_checked = ?, details = ? WHERE id = ?",
        (status, now, details, row_id),
    )
    conn.commit()
    conn.close()


def delete_redirect(row_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM redirects WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


# ======================
# REDIRECT CHECKING
# ======================
def normalize_url(url: str) -> str:
    """Very light normalization: strip trailing slashes and spaces."""
    if not url:
        return url
    return url.strip().rstrip("/")


def check_single_redirect(old_url: str, new_url: str) -> Tuple[str, str]:
    """
    Check if old_url redirects to new_url.
    Returns (status, details).
    status: 'ok', 'mismatch', 'error'
    """
    try:
        resp = requests.get(
            old_url,
            allow_redirects=True,
            timeout=10,
            headers={"User-Agent": "RedirectMonitor/1.0"},
        )
        final_url = resp.url
        norm_final = normalize_url(final_url)
        norm_target = normalize_url(new_url)

        if norm_final == norm_target:
            status = "ok"
            details = f"Redirect OK. Final URL: {final_url} (HTTP {resp.status_code})"
        else:
            status = "mismatch"
            details = (
                f"Redirect mismatch. Expected: {new_url}, "
                f"got: {final_url} (HTTP {resp.status_code})"
            )
    except Exception as e:
        status = "error"
        details = f"Request failed: {e}"

    return status, details


def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def get_interval_minutes(status: str) -> int:
    if status == "ok":
        return OK_CHECK_INTERVAL_MIN
    return PENDING_CHECK_INTERVAL_MIN


def should_check_row(row: pd.Series, force: bool = False) -> bool:
    if force:
        return True

    last_checked = parse_ts(row.get("last_checked"))
    interval_min = get_interval_minutes(row.get("status", "pending"))

    if last_checked is None:
        return True

    now = datetime.now(timezone.utc)
    next_allowed = last_checked + timedelta(minutes=interval_min)
    return now >= next_allowed


def compute_next_check(row: pd.Series) -> Optional[datetime]:
    last_checked = parse_ts(row.get("last_checked"))
    if last_checked is None:
        return None
    interval_min = get_interval_minutes(row.get("status", "pending"))
    return last_checked + timedelta(minutes=interval_min)


# ======================
# BULK IMPORT HELPERS
# ======================
def find_old_new_columns(df: pd.DataFrame):
    """
    Try to detect OLD and NEW URL columns.
    Priority:
      1) Columns explicitly named like 'OLD URL' / 'NEW URL'
      2) Else: first two columns
    Returns (old_col_name, new_col_name) or (None, None) if not possible.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}

    candidates_old = ["old url", "old", "source", "from"]
    candidates_new = ["new url", "new", "target", "to"]

    old_col = None
    new_col = None

    for key in candidates_old:
        if key in cols_lower:
            old_col = cols_lower[key]
            break

    for key in candidates_new:
        if key in cols_lower:
            new_col = cols_lower[key]
            break

    # Fallback: first two columns
    if old_col is None or new_col is None:
        if len(df.columns) >= 2:
            old_col = df.columns[0]
            new_col = df.columns[1]
        else:
            return None, None

    return old_col, new_col


# ======================
# STREAMLIT APP
# ======================
def main():
    st.set_page_config(
        page_title="Redirect Monitor",
        page_icon="🔁",
        layout="wide",
    )

    init_db()
    df_all = load_redirects()
    clients = sorted(df_all["client"].unique()) if not df_all.empty else []

    # --- SIDEBAR NAV ---
    st.sidebar.subheader("Navigation")
    if clients:
        view = st.sidebar.selectbox(
            "Choose client dashboard",
            ["Overview (all clients)"] + clients,
        )
    else:
        view = "No clients yet"
        st.sidebar.info("No clients yet – create the first redirect below.")

    if st.sidebar.button("🔁 Reload page"):
        st.rerun()

    st.title("🔁 Redirect Monitor")
    st.caption(
        "Track OLD → NEW URL redirects per client. "
        "Each client has its own dashboard; redirects that are OK are checked once per day."
    )

    # ======================
    # FIRST REDIRECT (NO CLIENTS YET)
    # ======================
    if df_all.empty:
        st.header("Create first redirect / client")
        with st.form("first_redirect_form"):
            client = st.text_input("Client name", "")
            old_url = st.text_input("OLD URL (source)", "")
            new_url = st.text_input("NEW URL (target)", "")
            submitted = st.form_submit_button("Add")
            if submitted:
                if not client or not old_url or not new_url:
                    st.error("Please fill in client, OLD URL and NEW URL.")
                else:
                    insert_redirect(client, old_url, new_url)
                    st.success("Redirect mapping added.")
                    st.rerun()
        return

    # ======================
    # OVERVIEW DASHBOARD
    # ======================
    if view == "Overview (all clients)":
        st.header("📊 Overview – all clients")

        total_redirects = len(df_all)
        total_clients = len(clients)
        total_ok = (df_all["status"] == "ok").sum()
        total_pending = (df_all["status"] != "ok").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clients", total_clients)
        c2.metric("Total redirects", total_redirects)
        c3.metric("OK redirects", int(total_ok))
        c4.metric("Pending / error", int(total_pending))

        st.subheader("Redirects by client")
        summary = (
            df_all.groupby("client")["id"]
            .count()
            .reset_index()
            .rename(columns={"id": "Redirect count"})
        )
        st.dataframe(summary, use_container_width=True)

        st.subheader("All redirects (read-only)")
        df_display = df_all.copy()

        next_checks = []
        last_checked_pretty = []
        for _, r in df_display.iterrows():
            lc = parse_ts(r.get("last_checked"))
            if lc:
                last_checked_pretty.append(
                    lc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                )
            else:
                last_checked_pretty.append("Never")

            nc = compute_next_check(r)
            if nc:
                next_checks.append(
                    nc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                )
            else:
                next_checks.append("As soon as possible")

        df_display["last_checked"] = last_checked_pretty
        df_display["next_check"] = next_checks
        df_display["check_interval_min"] = df_display["status"].apply(
            get_interval_minutes
        )

        df_display = df_display[
            [
                "id",
                "client",
                "old_url",
                "new_url",
                "status",
                "last_checked",
                "next_check",
                "check_interval_min",
                "details",
            ]
        ]
        df_display = df_display.rename(
            columns={
                "id": "ID",
                "client": "Client",
                "old_url": "OLD URL",
                "new_url": "NEW URL",
                "status": "Status",
                "last_checked": "Last checked",
                "next_check": "Next check",
                "check_interval_min": "Interval (min)",
                "details": "Details",
            }
        )

        st.dataframe(df_display, use_container_width=True)
        st.info("To add or manage redirects, open a specific client dashboard from the sidebar.")
        return

    elif view == "No clients yet":
        return

    # ======================
    # PER-CLIENT DASHBOARD
    # ======================
    selected_client = view
    df_client = df_all[df_all["client"] == selected_client].copy()

    st.header(f"👤 Client dashboard – {selected_client}")

    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Redirects", len(df_client))
    col_m2.metric("OK", int((df_client["status"] == "ok").sum()))
    col_m3.metric("Pending / error", int((df_client["status"] != "ok").sum()))

    # ---- Add single redirect ----
    st.subheader("➕ Add redirect for this client")
    with st.form("add_redirect_for_client"):
        st.text_input("Client", value=selected_client, disabled=True)
        col1, col2 = st.columns(2)
        with col1:
            old_url = st.text_input("OLD URL (source)", key="old_single_client")
        with col2:
            new_url = st.text_input("NEW URL (target)", key="new_single_client")

        submitted = st.form_submit_button("Add redirect")
        if submitted:
            if not old_url or not new_url:
                st.error("Please fill in both OLD URL and NEW URL.")
            else:
                insert_redirect(selected_client, old_url, new_url)
                st.success("Redirect mapping added.")
                st.rerun()

    # ---- Bulk upload for this client ----
    st.subheader("📥 Bulk upload for this client (Excel / CSV)")
    with st.form("bulk_upload_client_form"):
        st.text_input("Client", value=selected_client, disabled=True, key="bulk_client_name")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file with two columns: OLD URL, NEW URL",
            type=["csv", "xlsx", "xls"],
            key="bulk_file_client",
        )
        bulk_submit = st.form_submit_button("Import file for this client")

        if bulk_submit:
            if uploaded_file is None:
                st.error("Please upload a CSV or Excel file.")
            else:
                try:
                    name = uploaded_file.name.lower()
                    if name.endswith(".csv"):
                        df_file = pd.read_csv(uploaded_file)
                    else:
                        df_file = pd.read_excel(uploaded_file)

                    if df_file.empty:
                        st.error("The uploaded file is empty.")
                    else:
                        old_col, new_col = find_old_new_columns(df_file)
                        if old_col is None or new_col is None:
                            st.error(
                                "Could not detect OLD/NEW URL columns. "
                                "Make sure the file has at least two columns, "
                                "ideally named 'OLD URL' and 'NEW URL'."
                            )
                        else:
                            st.write("Detected columns:")
                            st.write(f"• OLD URL column: **{old_col}**")
                            st.write(f"• NEW URL column: **{new_col}**")
                            st.write("Preview of data (first 10 rows):")
                            st.dataframe(
                                df_file[[old_col, new_col]].head(10),
                                use_container_width=True,
                            )

                            imported = 0
                            for _, row in df_file.iterrows():
                                old_val = str(row.get(old_col, "")).strip()
                                new_val = str(row.get(new_col, "")).strip()
                                if (
                                    old_val
                                    and new_val
                                    and old_val.lower() != "nan"
                                    and new_val.lower() != "nan"
                                ):
                                    insert_redirect(selected_client, old_val, new_val)
                                    imported += 1

                            st.success(
                                f"Imported {imported} redirect pair(s) for client '{selected_client}'."
                            )
                            st.rerun()

                except Exception as e:
                    st.error(f"Failed to read file: {e}")

    # ---- Check redirects for this client ----
    st.subheader("🔍 Check redirects for this client")
    colA, colB = st.columns(2)
    with colA:
        check_due = st.button("Check due redirects for this client now")
    with colB:
        check_all = st.button("Force check ALL for this client now")

    if check_due or check_all:
        base_df = df_client

        # Determine which rows to check
        rows_to_check: List[pd.Series] = []
        for _, r in base_df.iterrows():
            if should_check_row(r, force=check_all):
                rows_to_check.append(r)

        total_to_check = len(rows_to_check)

        if total_to_check == 0:
            st.info("No redirects are due for checking right now for this client.")
        else:
            st.write(f"Will check **{total_to_check}** redirect(s) for **{selected_client}**.")
            progress_bar = st.progress(0)
            current_text = st.empty()
            log_box = st.empty()

            log_lines: List[str] = []
            updated_count = 0

            for i, row in enumerate(rows_to_check, start=1):
                current_text.markdown(
                    f"**Checking {i}/{total_to_check}:** {row['old_url']} → {row['new_url']}"
                )

                status, details = check_single_redirect(row["old_url"], row["new_url"])
                update_redirect(row["id"], status, details)
                updated_count += 1

                log_lines.append(
                    f"{i}/{total_to_check} | {row['old_url']} → {row['new_url']} | status: {status}"
                )
                log_box.text("\n".join(log_lines[-50:]))

                progress_bar.progress(int(i / total_to_check * 100))

            current_text.markdown("✅ Finished checking redirects for this client.")
            st.success(f"Checked {updated_count} redirect(s).")

            # Reload client data
            df_all = load_redirects()
            df_client = df_all[df_all["client"] == selected_client].copy()

    # ---- Redirect table for this client ----
    st.subheader("📋 Redirect list for this client")
    display_df = df_client.copy()

    next_checks = []
    last_checked_pretty = []
    for _, r in display_df.iterrows():
        lc = parse_ts(r.get("last_checked"))
        if lc:
            last_checked_pretty.append(
                lc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            )
        else:
            last_checked_pretty.append("Never")

        nc = compute_next_check(r)
        if nc:
            next_checks.append(
                nc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            )
        else:
            next_checks.append("As soon as possible")

    display_df["last_checked"] = last_checked_pretty
    display_df["next_check"] = next_checks
    display_df["check_interval_min"] = display_df["status"].apply(
        get_interval_minutes
    )

    display_df = display_df[
        [
            "id",
            "old_url",
            "new_url",
            "status",
            "last_checked",
            "next_check",
            "check_interval_min",
            "details",
        ]
    ]
    display_df = display_df.rename(
        columns={
            "id": "ID",
            "old_url": "OLD URL",
            "new_url": "NEW URL",
            "status": "Status",
            "last_checked": "Last checked",
            "next_check": "Next check",
            "check_interval_min": "Interval (min)",
            "details": "Details",
        }
    )

    st.dataframe(display_df, use_container_width=True)

    # ---- Delete redirect by ID (any client) ----
    st.subheader("🗑️ Delete redirect entry (for any client, by ID)")
    delete_id = st.number_input("ID to delete", min_value=0, value=0, step=1)
    if st.button("Delete by ID"):
        if delete_id == 0:
            st.warning("Enter a valid ID (non-zero).")
        else:
            if int(delete_id) in df_all["id"].values:
                delete_redirect(int(delete_id))
                st.success(f"Deleted redirect with ID {int(delete_id)}.")
                st.rerun()
            else:
                st.error("ID not found in current data.")


if __name__ == "__main__":
    main()
