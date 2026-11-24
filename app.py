import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="ハナハナ高設定予測AI", layout="wide")

SPREADSHEET_ID = "1mcVZi74Z06ANRWKV6cMvOqonZqq9xVsOa1dEA26obWk"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# ---------------------------------------------------------
# 1. Googleスプレッドシート接続・読み書き
# ---------------------------------------------------------
@st.cache_resource
def get_worksheet():
    if os.path.exists("secrets.json"):
        creds = Credentials.from_service_account_file("secrets.json", scopes=SCOPES)
    elif "gcp_service_account" in st.secrets:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    else:
        st.error("認証キー（secrets.json または Streamlit Secrets）が見つかりません。")
        st.stop()
    client = gspread.authorize(creds)
    sh = client.open_by_key(SPREADSHEET_ID)
    return sh.sheet1

def load_db_from_sheets():
    ws = get_worksheet()
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame(columns=["date", "unit", "diff"])
    df = pd.DataFrame(records)
    df["date"] = df["date"].astype(str)
    df["unit"] = pd.to_numeric(df["unit"], errors="coerce").astype(int)
    df["diff"] = pd.to_numeric(df["diff"], errors="coerce").astype(int)
    df = df[df["date"] >= "2026-09-02"]
    return df.sort_values(["date", "unit"]).reset_index(drop=True)

def save_db_to_sheets(df_long):
    ws = get_worksheet()
    df_clean = df_long[["date", "unit", "diff"]].sort_values(["date", "unit"]).copy()
    rows = [["date", "unit", "diff"]] + df_clean.astype(str).values.tolist()
    ws.clear()
    ws.update(rows, value_input_option="USER_ENTERED")

# ---------------------------------------------------------
# 2. Excel / CSV 初期データ変換
# ---------------------------------------------------------
def parse_excel_date(col_val, base_year=2026):
    if isinstance(col_val, (datetime, pd.Timestamp)):
        return col_val.strftime("%Y-%m-%d")
    s = str(col_val).strip()
    m = re.search(r"(\d{1,2})[月/](\d{1,2})", s)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        return f"{base_year:04d}-{month:02d}-{day:02d}"
    return None

def load_initial_file(uploaded_file, base_year=2026):
    if uploaded_file.name.lower().endswith(".csv"):
        try:
            df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, encoding="cp932")
    else:
        df_raw = pd.read_excel(uploaded_file, engine="openpyxl")

    unit_col = df_raw.columns[0]
    df_raw = df_raw[pd.to_numeric(df_raw[unit_col], errors="coerce").notna()].copy()
    df_raw[unit_col] = df_raw[unit_col].astype(int)
    
    records = []
    current_year = base_year
    prev_month = None

    for col in df_raw.columns[1:]:
        date_str = parse_excel_date(col, current_year)
        if not date_str:
            continue
        month = int(date_str.split("-")[1])
        if prev_month is not None and prev_month == 12 and month == 1:
            current_year += 1
            date_str = parse_excel_date(col, current_year)
        prev_month = month

        vals = pd.to_numeric(df_raw[col], errors="coerce")
        for unit, diff in zip(df_raw[unit_col], vals):
            if pd.notna(diff):
                records.append({
                    "date": date_str,
                    "unit": int(unit),
                    "diff": int(diff)
                })
                
    df_long = pd.DataFrame(records)
    df_long = df_long[df_long["date"] >= "2026-09-02"]
    return df_long.drop_duplicates(subset=["date", "unit"]).sort_values(["date", "unit"])

def export_matrix_excel(df_long):
    pivot = df_long.pivot(index="unit", columns="date", values="diff")
    pivot.columns = [f"{int(c.split('-')[1])}月{int(c.split('-')[2])}日" for c in pivot.columns]
    pivot.index.name = "台番"
    return pivot.reset_index()

# ---------------------------------------------------------
# 3. 特徴量エンジニアリング ＆ 予測
# ---------------------------------------------------------
def create_features(df_long, target_threshold=1500, event_digits=None):
    if event_digits is None:
        event_digits = []
        
    df = df_long.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["unit", "date_dt"]).reset_index(drop=True)
    df["is_high"] = (df["diff"] >= target_threshold).astype(int)
    
    g = df.groupby("unit")
    df["prev1_diff"] = g["diff"].shift(1)
    df["prev2_diff"] = g["diff"].shift(2)
    df["prev3_diff"] = g["diff"].shift(3)
    df["prev1_high"] = g["is_high"].shift(1)
    df["prev2_high"] = g["is_high"].shift(2)
    
    df["sum_3d"] = g["diff"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    df["sum_7d"] = g["diff"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    df["mean_7d"] = g["diff"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    
    def calc_neg_streak(series):
        s = series.shift(1)
        streaks = []
        cnt = 0
        for v in s:
            if pd.isna(v):
                streaks.append(np.nan)
            elif v < 0:
                cnt += 1
                streaks.append(cnt)
            else:
                cnt = 0
                streaks.append(0)
        return streaks

    df["neg_streak"] = g["diff"].transform(calc_neg_streak)
    
    def calc_days_since_high(series):
        s = series.shift(1)
        days = []
        cnt = 7
        for v in s:
            if pd.isna(v):
                days.append(np.nan)
            elif v == 1:
                cnt = 1
                days.append(cnt)
            else:
                cnt += 1
                days.append(cnt)
        return days

    df["days_since_high"] = g["is_high"].transform(calc_days_since_high)

    daily_stats = df.groupby("date_dt").agg(
        island_mean_diff=("diff", "mean"),
        island_high_rate=("is_high", "mean")
    ).shift(1).reset_index()
    df = pd.merge(df, daily_stats, on="date_dt", how="left")

    prev_day_map = df[["date_dt", "unit", "prev1_diff", "prev1_high"]].copy()
    left_neighbor = prev_day_map.copy()
    left_neighbor["unit"] = left_neighbor["unit"] + 1
    left_neighbor = left_neighbor.rename(columns={"prev1_diff": "left_prev1_diff", "prev1_high": "left_prev1_high"})
    
    right_neighbor = prev_day_map.copy()
    right_neighbor["unit"] = right_neighbor["unit"] - 1
    right_neighbor = right_neighbor.rename(columns={"prev1_diff": "right_prev1_diff", "prev1_high": "right_prev1_high"})
    
    df = pd.merge(df, left_neighbor, on=["date_dt", "unit"], how="left")
    df = pd.merge(df, right_neighbor, on=["date_dt", "unit"], how="left")
    df["neighbor_prev1_high_cnt"] = df["left_prev1_high"].fillna(0) + df["right_prev1_high"].fillna(0)

    df["unit_last_digit"] = df["unit"] % 10
    min_u, max_u = df["unit"].min(), df["unit"].max()
    df["is_corner"] = df["unit"].isin([min_u, max_u]).astype(int)
    
    df["day_of_week"] = df["date_dt"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_num"] = df["date_dt"].dt.day
    df["day_last_digit"] = df["day_num"] % 10
    df["is_zorome"] = df["day_num"].isin([11, 22]).astype(int)
    df["is_event_day"] = df["day_last_digit"].isin(event_digits).astype(int)
    df["is_unit_day_match"] = (df["unit_last_digit"] == df["day_last_digit"]).astype(int)

    feature_cols = [
        "unit", "unit_last_digit", "is_corner",
        "prev1_diff", "prev2_diff", "prev3_diff",
        "prev1_high", "prev2_high",
        "sum_3d", "sum_7d", "mean_7d",
        "neg_streak", "days_since_high",
        "island_mean_diff", "island_high_rate",
        "left_prev1_diff", "right_prev1_diff", "neighbor_prev1_high_cnt",
        "day_of_week", "is_weekend", "day_last_digit", "is_zorome", "is_event_day", "is_unit_day_match"
    ]
    
    return df, feature_cols

def train_and_predict(df_long, target_date_str, target_threshold=1500, event_digits=None):
    units = sorted(df_long["unit"].unique())
    future_rows = pd.DataFrame({
        "date": [target_date_str] * len(units),
        "unit": units,
        "diff": [0] * len(units)
    })
    
    df_past = df_long[df_long["date"] < target_date_str].copy()
    df_combined = pd.concat([df_past, future_rows], ignore_index=True)
    df_feat, feature_cols = create_features(df_combined, target_threshold, event_digits)
    
    min_train_date = df_feat["date_dt"].min() + timedelta(days=3)
    train_df = df_feat[(df_feat["date"] < target_date_str) & (df_feat["date_dt"] >= min_train_date)].copy()
    pred_df = df_feat[df_feat["date"] == target_date_str].copy()
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["is_high"]
    X_pred = pred_df[feature_cols].fillna(0)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_pred)[:, 1] * 100
    pred_df["prob"] = np.round(probs, 1)
    
    reasons = []
    for _, row in pred_df.iterrows():
        r = []
        if row["prev1_high"] == 1:
            r.append("前日高設定挙動(据え置き候補)")
        elif row["prev1_diff"] <= -1000:
            r.append(f"前日大きく凹み({int(row['prev1_diff'])}枚)")
        if row["neg_streak"] >= 2:
            r.append(f"{int(row['neg_streak'])}日連続マイナス")
        if row["neighbor_prev1_high_cnt"] >= 1:
            r.append("前日隣が高設定挙動(並び/スライド)")
        if row["is_event_day"] == 1:
            r.append("特定日")
        if row["is_unit_day_match"] == 1:
            r.append("日付末尾一致")
        if row["days_since_high"] >= 6:
            r.append(f"{int(row['days_since_high'])}日間不発")
        reasons.append(" / ".join(r) if r else "総合バランス評価")
        
    pred_df["reason"] = reasons
    
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return pred_df.sort_values("prob", ascending=False), imp_df, len(train_df)

# ---------------------------------------------------------
# 4. UI 構築
# ---------------------------------------------------------
st.title("🌺 ハナハナ 高設定予測AI（クラウド連携版）")

df_db = load_db_from_sheets()

with st.sidebar:
    st.header("⚙️ 設定・データ管理")
    uploaded_file = st.file_uploader("初回ファイル読込 (.csv / .xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if st.button("スプレッドシートに初期データを書き込む"):
            with st.spinner("Googleスプレッドシートへ書き込み中..."):
                df_init = load_initial_file(uploaded_file)
                save_db_to_sheets(df_init)
                st.success(f"書込完了！({df_init['date'].nunique()}日分 × {df_init['unit'].nunique()}台)")
                st.rerun()

    st.divider()
    target_threshold = st.slider("高設定とみなす差枚ライン (枚)", 1000, 3000, 1500, step=100)
    event_digits = st.multiselect("店舗の特定日（日付の末尾）", list(range(10)), default=[])
    
    if not df_db.empty:
        st.info(f"現在の蓄積データ: **{df_db['date'].nunique()}日分**\n\n最新日: **{df_db['date'].max()}**")
        excel_df = export_matrix_excel(df_db)
        csv_bytes = excel_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 蓄積データをCSVでバックアップ",
            data=csv_bytes,
            file_name="hanahana_updated.csv",
            mime="text/csv"
        )

if df_db.empty:
    st.warning("👈 左のサイドバーから、初期データ（.csv または .xlsx）を読み込んでスプレッドシートに保存してください。")
    st.stop()

latest_date = pd.to_datetime(df_db["date"].max())
default_next_date = (latest_date + timedelta(days=1)).date()

tab1, tab2, tab3 = st.tabs(["🎯 明日の狙い台予測", "📝 本日の差枚入力・追加学習", "📊 店舗傾向分析"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        pred_date = st.date_input("予測したい日付を選択", value=default_next_date)
    
    pred_date_str = pred_date.strftime("%Y-%m-%d")
    res_df, imp_df, sample_cnt = train_and_predict(df_db, pred_date_str, target_threshold, event_digits)
    
    st.subheader(f"📅 {pred_date_str} の高設定期待度ランキング（学習データ: {sample_cnt}件）")
    display_cols = ["unit", "prob", "prev1_diff", "prev2_diff", "sum_3d", "neg_streak", "reason"]
    show_df = res_df[display_cols].copy()
    show_df.columns = ["台番", "高設定期待度(%)", "前日差枚", "2日前差枚", "3日間累計", "連続マイナス", "AIの注目ポイント"]
    
    st.dataframe(
        show_df.style.background_gradient(subset=["高設定期待度(%)"], cmap="Reds"),
        use_container_width=True,
        height=600
    )
    
    with st.expander("🧠 AIが現在重視している要素（特徴量重要度トップ10）"):
        jp_names = {
            "prev1_diff": "前日の差枚", "prev2_diff": "2日前の差枚", "prev3_diff": "3日前の差枚",
            "sum_3d": "直近3日間累計差枚", "sum_7d": "直近7日間累計差枚", "mean_7d": "直近7日間平均差枚",
            "neg_streak": "連続マイナス日数", "days_since_high": "前回高設定からの経過日数",
            "island_mean_diff": "前日の島全体平均差枚", "island_high_rate": "前日の島全体高設定率",
            "left_prev1_diff": "左隣の昨日差枚", "right_prev1_diff": "右隣の昨日差枚",
            "neighbor_prev1_high_cnt": "両隣の昨日高設定数", "unit": "台番",
            "unit_last_digit": "台番末尾", "day_of_week": "曜日", "day_last_digit": "日付末尾",
            "is_unit_day_match": "日付と台番の末尾一致", "is_event_day": "特定日フラグ"
        }
        top_imp = imp_df.head(10).copy()
        top_imp["要素"] = top_imp["feature"].map(lambda x: jp_names.get(x, x))
        st.bar_chart(top_imp.set_index("要素")["importance"])

with tab2:
    st.subheader("📝 新しい日の差枚を追加してAIをアップデート")
    input_date = st.date_input("追加する日付", value=default_next_date, key="input_date")
    input_date_str = input_date.strftime("%Y-%m-%d")
    units = sorted(df_db["unit"].unique())
    
    st.markdown("**方法①：36台分の差枚を縦一列でコピペ一括入力（一番ラクです）**")
    st.caption(f"台番 {units[0]} 〜 {units[-1]} の順に、1行に1台ずつ差枚を貼り付けてください（全{len(units)}行）。")
    paste_text = st.text_area("差枚リストを貼り付け", height=150, placeholder="-1200\n2500\n-400\n...")
    
    init_vals = [0] * len(units)
    if paste_text.strip():
        lines = [line.strip().replace(",", "") for line in paste_text.strip().splitlines() if line.strip()]
        for i in range(min(len(lines), len(units))):
            try:
                init_vals[i] = int(float(lines[i]))
            except ValueError:
                pass

    input_df = pd.DataFrame({"台番": units, "差枚": init_vals})
    st.markdown("**方法②：下の表を直接編集・確認して登録**")
    edited_df = st.data_editor(input_df, use_container_width=True, height=400)
    
    if st.button("💾 スプレッドシートに保存して再学習する", type="primary"):
        with st.spinner("スプレッドシートを更新中..."):
            new_records = pd.DataFrame({
                "date": [input_date_str] * len(edited_df),
                "unit": edited_df["台番"].astype(int),
                "diff": edited_df["差枚"].astype(int)
            })
            df_updated = df_db[df_db["date"] != input_date_str].copy()
            df_updated = pd.concat([df_updated, new_records], ignore_index=True)
            save_db_to_sheets(df_updated)
            st.success(f"✅ {input_date_str} のデータをスプレッドシートに保存しました！")
            st.rerun()

with tab3:
    st.subheader("📊 蓄積データから見る店舗のクセ")
    df_analysis = df_db.copy()
    df_analysis["is_high"] = (df_analysis["diff"] >= target_threshold).astype(int)
    df_analysis["unit_last_digit"] = df_analysis["unit"] % 10
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**台番別の平均差枚 ＆ 高設定ライン達成率**")
        unit_summary = df_analysis.groupby("unit").agg(
            平均差枚=("diff", "mean"),
            高設定達成率=("is_high", lambda x: round(x.mean() * 100, 1)),
            データ日数=("diff", "count")
        ).round(1)
        st.dataframe(unit_summary, use_container_width=True, height=400)
        
    with c2:
        st.markdown("**台番末尾別の高設定達成率 (%)**")
        digit_summary = df_analysis.groupby("unit_last_digit")["is_high"].mean() * 100
        st.bar_chart(digit_summary)
