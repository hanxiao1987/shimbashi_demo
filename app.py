"""
新橋駅周辺 滞在者行動分析 & DOOH ターゲティング デモ
Plateau建物属性 × 擬似GPS × 時間帯別可視化
"""

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────
SHIMBASHI_LAT = 35.6659
SHIMBASHI_LON = 139.7575
RANDOM_SEED   = 42

# 時間帯ラベル
TIME_BANDS = {
    "深夜 (0-4時)":        (0,  4),
    "早朝 (5-6時)":        (5,  6),
    "朝ラッシュ (7-9時)":  (7,  9),
    "午前 (10-11時)":      (10, 11),
    "ランチ (12-13時)":    (12, 13),
    "午後 (14-16時)":      (14, 16),
    "夕方ラッシュ (17-19時)": (17, 19),
    "夜 (20-22時)":        (20, 22),
    "深夜帯 (23時-)":      (23, 23),
}

# ─────────────────────────────────────────────────────────────────────────────
# DOOH マッピング（建物用途 → 滞在目的 → 広告訴求）
# ─────────────────────────────────────────────────────────────────────────────
DOOH_MAP = {
    "居酒屋・バー": {
        "purpose":      "飲み会・会食",
        "audience":     "飲食目的ビジネス層",
        "dooh_cat":     "アルコール飲料・グルメサービス",
        "brands":       ["サントリー ザ・プレミアム・モルツ", "キリン 一番搾り",
                         "アサヒ スーパードライ", "ぐるなび", "ホットペッパーグルメ"],
        "reason":       "飲み会後の帰宅途中 or 入店検討中の時間帯に訴求効果が高い",
        "color":        "#e74c3c",
        "icon":         "🍺",
    },
    "レストラン": {
        "purpose":      "食事・グルメ",
        "audience":     "グルメ・ランチ層",
        "dooh_cat":     "フードデリバリー・飲食チェーン",
        "brands":       ["Uber Eats", "出前館", "ぐるなび", "食べログ", "CoCo壱番屋"],
        "reason":       "ランチ・夕食前後の滞在中に次回来店・デリバリー利用を促進",
        "color":        "#e67e22",
        "icon":         "🍽️",
    },
    "カフェ": {
        "purpose":      "休憩・カジュアル商談",
        "audience":     "ビジネスカジュアル・学習層",
        "dooh_cat":     "コーヒー・嗜好品・モバイル決済",
        "brands":       ["スターバックス", "ネスカフェ ゴールドブレンド",
                         "ジョージア", "PayPay", "楽天ペイ"],
        "reason":       "休憩中の情報感度が高く、飲料・決済サービスの認知向上に最適",
        "color":        "#8e44ad",
        "icon":         "☕",
    },
    "オフィスビル": {
        "purpose":      "仕事・商談・会議",
        "audience":     "ビジネスパーソン層",
        "dooh_cat":     "ビジネスSaaS・転職・金融",
        "brands":       ["リクルートエージェント", "マイナビ転職", "freee",
                         "Sansan", "SmartHR"],
        "reason":       "就業中・通勤導線でのビジネス系サービス認知に高い効果",
        "color":        "#2980b9",
        "icon":         "💼",
    },
    "コンビニ": {
        "purpose":      "日用品・食品購入",
        "audience":     "生活者・通勤者層",
        "dooh_cat":     "日用品FMCG・飲料・スナック",
        "brands":       ["P&G ジョイ", "明治 チョコレート",
                         "キューピー", "サントリー 天然水", "ローソン"],
        "reason":       "短時間滞在で購買意欲が高い; 衝動購買を即時促進",
        "color":        "#27ae60",
        "icon":         "🏪",
    },
    "ショッピング": {
        "purpose":      "買い物・ショッピング",
        "audience":     "消費者・ファッション層",
        "dooh_cat":     "ファッション・EC・クーポン",
        "brands":       ["ZOZOTOWN", "楽天市場", "Amazon",
                         "ユニクロ", "GU"],
        "reason":       "購買行動中の層にEC・店舗誘引広告が高コンバージョン",
        "color":        "#f39c12",
        "icon":         "🛍️",
    },
    "ホテル": {
        "purpose":      "宿泊・出張",
        "audience":     "ビジネストラベラー層",
        "dooh_cat":     "旅行・交通・宿泊予約",
        "brands":       ["JR東海 新幹線", "ANA", "楽天トラベル",
                         "じゃらん", "東横INN"],
        "reason":       "出張・旅行者に移動・宿泊サービスの次回利用訴求が有効",
        "color":        "#16a085",
        "icon":         "🏨",
    },
    "銀行・ATM": {
        "purpose":      "金融手続き・資産管理",
        "audience":     "金融関心・資産形成層",
        "dooh_cat":     "金融サービス・投資・FinTech",
        "brands":       ["三井住友銀行", "楽天証券", "SBI証券",
                         "PayPay", "d払い"],
        "reason":       "金融機関滞在中は金融商品の検討意欲が最も高まるタイミング",
        "color":        "#2c3e50",
        "icon":         "🏦",
    },
    "医療・クリニック": {
        "purpose":      "通院・健康管理",
        "audience":     "健康関心・シニア層",
        "dooh_cat":     "医薬品・サプリメント・健康食品",
        "brands":       ["大正製薬 リポビタンD", "DHC",
                         "ロート製薬", "アリナミン", "キューピーコーワ"],
        "reason":       "健康意識が高まっているタイミングで医薬品・健康訴求が最適",
        "color":        "#c0392b",
        "icon":         "🏥",
    },
    "駅・交通施設": {
        "purpose":      "通勤・移動・乗換",
        "audience":     "通勤者・移動者層（全属性）",
        "dooh_cat":     "交通・コンビニ・スマホアプリ",
        "brands":       ["Suica", "モバイルSuica", "Yahoo!カーナビ",
                         "Googleマップ", "NAVITIME"],
        "reason":       "最大リーチ、全層へのブランド認知向上と交通・アプリ系に最適",
        "color":        "#7f8c8d",
        "icon":         "🚉",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 建物マスタ（新橋駅周辺 ~40棟）
# ─────────────────────────────────────────────────────────────────────────────
BUILDINGS_RAW = [
    # 居酒屋・バー
    {"id":"B001","name":"炭火焼鳥 新橋屋",           "lat":35.6655,"lon":139.7588,"usage":"居酒屋・バー"},
    {"id":"B002","name":"大衆酒場 のんき",            "lat":35.6651,"lon":139.7583,"usage":"居酒屋・バー"},
    {"id":"B003","name":"居酒屋 春よ来い",            "lat":35.6658,"lon":139.7581,"usage":"居酒屋・バー"},
    {"id":"B004","name":"立ち飲み 角打ち番長",        "lat":35.6653,"lon":139.7591,"usage":"居酒屋・バー"},
    {"id":"B005","name":"もつ焼き でんがな",          "lat":35.6647,"lon":139.7587,"usage":"居酒屋・バー"},
    {"id":"B006","name":"ビアホール LION",             "lat":35.6665,"lon":139.7579,"usage":"居酒屋・バー"},
    {"id":"B007","name":"個室居酒屋 和楽",            "lat":35.6669,"lon":139.7573,"usage":"居酒屋・バー"},
    {"id":"B008","name":"赤ちょうちん 二番街",        "lat":35.6644,"lon":139.7584,"usage":"居酒屋・バー"},
    # レストラン
    {"id":"B009","name":"ラーメン 蔦 新橋店",        "lat":35.6662,"lon":139.7568,"usage":"レストラン"},
    {"id":"B010","name":"焼肉 和牛TOKYO",             "lat":35.6650,"lon":139.7566,"usage":"レストラン"},
    {"id":"B011","name":"寿司 銀座やまと 新橋",      "lat":35.6672,"lon":139.7581,"usage":"レストラン"},
    {"id":"B012","name":"洋食 グリル アンジュ",      "lat":35.6648,"lon":139.7571,"usage":"レストラン"},
    {"id":"B013","name":"中華 龍雲閣",               "lat":35.6640,"lon":139.7576,"usage":"レストラン"},
    # カフェ
    {"id":"B014","name":"スターバックス SL広場店",   "lat":35.6660,"lon":139.7573,"usage":"カフェ"},
    {"id":"B015","name":"ドトール 新橋西口店",       "lat":35.6655,"lon":139.7568,"usage":"カフェ"},
    {"id":"B016","name":"タリーズ 新橋店",           "lat":35.6645,"lon":139.7572,"usage":"カフェ"},
    # オフィスビル
    {"id":"B017","name":"新橋駅前ビル1号館",         "lat":35.6664,"lon":139.7585,"usage":"オフィスビル"},
    {"id":"B018","name":"新橋駅前ビル2号館",         "lat":35.6666,"lon":139.7582,"usage":"オフィスビル"},
    {"id":"B019","name":"汐留シオサイト オフィス棟", "lat":35.6638,"lon":139.7558,"usage":"オフィスビル"},
    {"id":"B020","name":"電通本社ビル",              "lat":35.6634,"lon":139.7556,"usage":"オフィスビル"},
    {"id":"B021","name":"日本テレビタワー",          "lat":35.6628,"lon":139.7561,"usage":"オフィスビル"},
    {"id":"B022","name":"新橋東洋ビル",              "lat":35.6675,"lon":139.7576,"usage":"オフィスビル"},
    {"id":"B023","name":"ニュー新橋ビル",            "lat":35.6662,"lon":139.7578,"usage":"オフィスビル"},
    # コンビニ
    {"id":"B024","name":"セブン-イレブン 新橋駅前店","lat":35.6658,"lon":139.7575,"usage":"コンビニ"},
    {"id":"B025","name":"ローソン SL広場店",         "lat":35.6657,"lon":139.7578,"usage":"コンビニ"},
    {"id":"B026","name":"ファミリーマート 新橋西口店","lat":35.6661,"lon":139.7570,"usage":"コンビニ"},
    # ショッピング
    {"id":"B027","name":"マツモトキヨシ 新橋店",     "lat":35.6653,"lon":139.7577,"usage":"ショッピング"},
    {"id":"B028","name":"ユニクロ 新橋店",           "lat":35.6648,"lon":139.7579,"usage":"ショッピング"},
    {"id":"B029","name":"GU 新橋店",                 "lat":35.6646,"lon":139.7581,"usage":"ショッピング"},
    # ホテル
    {"id":"B030","name":"アパホテル 新橋御成門",     "lat":35.6680,"lon":139.7565,"usage":"ホテル"},
    {"id":"B031","name":"東横INN 新橋駅前",          "lat":35.6670,"lon":139.7562,"usage":"ホテル"},
    {"id":"B032","name":"コートヤード by Marriott",  "lat":35.6642,"lon":139.7565,"usage":"ホテル"},
    # 銀行・ATM
    {"id":"B033","name":"三井住友銀行 新橋支店",     "lat":35.6667,"lon":139.7574,"usage":"銀行・ATM"},
    {"id":"B034","name":"みずほ銀行 新橋支店",       "lat":35.6660,"lon":139.7581,"usage":"銀行・ATM"},
    {"id":"B035","name":"ゆうちょ ATMコーナー",      "lat":35.6656,"lon":139.7573,"usage":"銀行・ATM"},
    # 医療・クリニック
    {"id":"B036","name":"新橋内科クリニック",        "lat":35.6673,"lon":139.7568,"usage":"医療・クリニック"},
    {"id":"B037","name":"新橋整形外科",              "lat":35.6650,"lon":139.7562,"usage":"医療・クリニック"},
    {"id":"B038","name":"銀座歯科クリニック",        "lat":35.6643,"lon":139.7573,"usage":"医療・クリニック"},
    # 駅（センターエリア）
    {"id":"STATION","name":"新橋駅",                 "lat":SHIMBASHI_LAT,"lon":SHIMBASHI_LON,"usage":"駅・交通施設"},
]

# ─────────────────────────────────────────────────────────────────────────────
# メッシュエンコード (JIS X 0410 / 15桁 10次メッシュ)
# ─────────────────────────────────────────────────────────────────────────────
def encode_mesh10(lat: float, lon: float) -> str:
    p = int(lat * 1.5)
    q = int(lon - 100.0)
    lat_rem = lat - p / 1.5
    lon_rem = lon - (q + 100.0)
    lat_sz, lon_sz = 2.0 / 3.0, 1.0
    code = f"{p:02d}{q:02d}"
    lat_sz /= 8;  lon_sz /= 8
    r2 = min(int(lat_rem / lat_sz), 7); c2 = min(int(lon_rem / lon_sz), 7)
    lat_rem -= r2 * lat_sz;  lon_rem -= c2 * lon_sz
    code += f"{r2}{c2}"
    lat_sz /= 10; lon_sz /= 10
    r3 = min(int(lat_rem / lat_sz), 9); c3 = min(int(lon_rem / lon_sz), 9)
    lat_rem -= r3 * lat_sz;  lon_rem -= c3 * lon_sz
    code += f"{r3}{c3}"
    for _ in range(7):
        lat_sz /= 2; lon_sz /= 2
        eps = 1e-12
        n = lat_rem >= lat_sz - eps
        e = lon_rem >= lon_sz - eps
        d = 4 if (n and e) else 3 if n else 2 if e else 1
        if n: lat_rem -= lat_sz
        if e: lon_rem -= lon_sz
        code += str(d)
    return code


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2点間の距離をメートルで返す"""
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# 建物 DataFrame 生成
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_buildings_df() -> pd.DataFrame:
    rows = []
    for b in BUILDINGS_RAW:
        d = DOOH_MAP.get(b["usage"], DOOH_MAP["駅・交通施設"])
        rows.append({
            "building_id":   b["id"],
            "name":          b["name"],
            "lat":           b["lat"],
            "lon":           b["lon"],
            "usage":         b["usage"],
            "purpose":       d["purpose"],
            "audience":      d["audience"],
            "dooh_cat":      d["dooh_cat"],
            "brands":        " / ".join(d["brands"][:3]),
            "reason":        d["reason"],
            "color":         d["color"],
            "icon":          d["icon"],
            "mesh_code":     encode_mesh10(b["lat"], b["lon"]),
            "dist_from_center": haversine_m(b["lat"], b["lon"], SHIMBASHI_LAT, SHIMBASHI_LON),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 擬似 GPS データ生成
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_gps_df() -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)
    bdf = build_buildings_df()
    buildings = {r.building_id: r for r in bdf.itertuples()}

    # 建物ごとのピーク時間帯と滞在時間分布
    usage_params = {
        "居酒屋・バー":    {"peak_h": [18, 19, 20, 21], "dur": (60, 180),  "n_weight": 4},
        "レストラン":      {"peak_h": [12, 13, 18, 19], "dur": (30,  90),  "n_weight": 3},
        "カフェ":          {"peak_h": [8, 9, 10, 13],   "dur": (20,  60),  "n_weight": 2},
        "オフィスビル":    {"peak_h": list(range(9,19)), "dur": (120,480),  "n_weight": 5},
        "コンビニ":        {"peak_h": [8, 12, 13, 18],  "dur": (5,   15),  "n_weight": 3},
        "ショッピング":    {"peak_h": [11, 12, 13, 14, 15], "dur": (20, 60), "n_weight": 2},
        "ホテル":          {"peak_h": [9, 10, 20, 21],  "dur": (30, 120),  "n_weight": 1},
        "銀行・ATM":       {"peak_h": [11, 12, 14, 15], "dur": (10,  30),  "n_weight": 1},
        "医療・クリニック":{"peak_h": [10, 11, 14, 15], "dur": (30,  90),  "n_weight": 1},
        "駅・交通施設":    {"peak_h": [7,8,12,13,17,18,19,20], "dur": (5, 30), "n_weight": 10},
    }

    # 建物リスト（STATION除く）
    non_station = [b for b in BUILDINGS_RAW if b["id"] != "STATION"]
    station_b   = buildings["STATION"]

    base_date = datetime(2024, 3, 1)
    records = []
    n_members = 2000

    for m_idx in range(n_members):
        member_id = f"M{m_idx+1:05d}"
        n_days = rng.randint(1, 7)
        visit_days = sorted(rng.sample(range(7), min(n_days, 7)))

        for day_offset in visit_days:
            date = base_date + timedelta(days=day_offset)

            # ① 必ず新橋駅を通過（センターエリア検出）
            peak_h  = rng.choice([7, 8, 17, 18, 19])
            dt_sta  = date + timedelta(hours=peak_h, minutes=rng.randint(0, 59))
            dur_sta = rng.uniform(5, 25)
            records.append({
                "member_id":       member_id,
                "building_id":     "STATION",
                "mesh_code":       encode_mesh10(
                    station_b.lat + np_rng.uniform(-0.0003, 0.0003),
                    station_b.lon + np_rng.uniform(-0.0003, 0.0003),
                ),
                "stay_datetime":   dt_sta,
                "stay_duration_min": round(dur_sta, 1),
            })

            # ② 周辺建物を 1〜4 か所訪問
            n_visits = rng.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]
            visited  = rng.sample(non_station, min(n_visits, len(non_station)))
            cursor_dt = dt_sta + timedelta(minutes=dur_sta + rng.randint(5, 30))

            for bld in visited:
                usage  = bld["usage"]
                params = usage_params.get(usage, usage_params["駅・交通施設"])
                hour   = rng.choice(params["peak_h"])
                visit_dt = date + timedelta(
                    hours=hour,
                    minutes=rng.randint(0, 59),
                )
                dur_min = rng.uniform(*params["dur"])
                records.append({
                    "member_id":       member_id,
                    "building_id":     bld["id"],
                    "mesh_code":       encode_mesh10(
                        bld["lat"] + np_rng.uniform(-0.00005, 0.00005),
                        bld["lon"] + np_rng.uniform(-0.00005, 0.00005),
                    ),
                    "stay_datetime":   visit_dt,
                    "stay_duration_min": round(dur_min, 1),
                })

    df = pd.DataFrame(records)
    df["stay_datetime"] = pd.to_datetime(df["stay_datetime"])
    df["hour"] = df["stay_datetime"].dt.hour
    df["date"] = df["stay_datetime"].dt.date
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 分析パイプライン
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(
    gps_df: pd.DataFrame,
    bdf: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    center_radius_m: float,
    surround_radius_m: float,
    min_stay_min: float,
    hour_range: tuple,
    selected_dates: list,
) -> pd.DataFrame:

    # 時間帯・日付フィルタ
    h_start, h_end = hour_range
    mask = (
        (gps_df["hour"] >= h_start) &
        (gps_df["hour"] <= h_end)
    )
    if selected_dates:
        mask &= gps_df["date"].isin(selected_dates)
    gps_f = gps_df[mask].copy()

    # 各建物と中心の距離を計算
    bdf = bdf.copy()
    bdf["dist_from_center"] = bdf.apply(
        lambda r: haversine_m(r["lat"], r["lon"], center_lat, center_lon), axis=1
    )

    # ① センターエリア内の建物 ID
    center_bldg_ids = set(
        bdf[bdf["dist_from_center"] <= center_radius_m]["building_id"]
    )

    # ② センターエリアで検出されたメンバー
    center_members = set(
        gps_f[gps_f["building_id"].isin(center_bldg_ids)]["member_id"]
    )

    # ③ 周辺エリア内の建物
    surround_bldg_ids = set(
        bdf[
            (bdf["dist_from_center"] <= surround_radius_m) &
            (~bdf["building_id"].isin(center_bldg_ids))
        ]["building_id"]
    )

    # ④ センターメンバーの周辺滞在 & 最低滞在時間フィルタ
    result = gps_f[
        (gps_f["member_id"].isin(center_members)) &
        (gps_f["building_id"].isin(surround_bldg_ids)) &
        (gps_f["stay_duration_min"] >= min_stay_min)
    ].copy()

    # ⑤ 建物情報を JOIN
    result = result.merge(bdf, on="building_id", how="left")
    return result, len(center_members)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="新橋 滞在行動分析 & DOOH",
    page_icon="📍",
    layout="wide",
)

st.title("📍 新橋駅周辺 滞在行動分析 & DOOH ターゲティング")
st.caption("Plateau建物属性 × 擬似GPSデータ（15桁メッシュ）× 時間帯別可視化")

# ── データ読み込み ────────────────────────────────────────────────────────────
with st.spinner("データ生成中..."):
    bdf     = build_buildings_df()
    gps_df  = generate_gps_df()

# ── サイドバー ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 分析パラメータ")
    st.divider()

    st.subheader("① センターエリア")
    c_lat = st.number_input("中心緯度", value=SHIMBASHI_LAT, format="%.6f")
    c_lon = st.number_input("中心経度", value=SHIMBASHI_LON, format="%.6f")
    c_radius = st.slider("検出半径 (m)", 50, 500, 200, step=25)

    st.divider()
    st.subheader("② 周辺エリア")
    s_radius = st.slider("周辺範囲 (m)", 200, 2000, 1000, step=100)

    st.divider()
    st.subheader("③ 滞在条件")
    min_stay = st.slider("最低滞在時間 (分)", 5, 120, 15, step=5)

    st.divider()
    st.subheader("④ 時間帯")
    h_range = st.slider("対象時間帯", 0, 23, (0, 23))

    st.divider()
    st.subheader("⑤ 期間")
    all_dates = sorted(gps_df["date"].unique())
    sel_dates = st.multiselect(
        "日付（複数選択可）",
        options=all_dates,
        default=all_dates,
        format_func=lambda d: d.strftime("%m/%d"),
    )

    st.divider()
    run_btn = st.button("▶ 分析実行", type="primary", use_container_width=True)

# ── 分析実行 ──────────────────────────────────────────────────────────────────
if "result_df" not in st.session_state or run_btn:
    result, n_center = run_analysis(
        gps_df, bdf,
        c_lat, c_lon, c_radius, s_radius,
        min_stay, h_range, sel_dates,
    )
    st.session_state["result_df"]  = result
    st.session_state["n_center"]   = n_center

result   = st.session_state["result_df"]
n_center = st.session_state["n_center"]

# ── KPI ───────────────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("センターエリア検出人数",  f"{n_center:,} 人")
k2.metric("周辺滞在レコード数",      f"{len(result):,} 件")
k3.metric("ユニーク滞在者数",        f"{result['member_id'].nunique():,} 人" if not result.empty else "0 人")
k4.metric("対象建物数",              f"{result['building_id'].nunique():,} 棟" if not result.empty else "0 棟")

if result.empty:
    st.warning("条件に合うデータがありません。パラメータを調整してください。")
    st.stop()

# ── 集計 ──────────────────────────────────────────────────────────────────────
# 建物ごとの滞在ユニーク人数
bldg_agg = (
    result.groupby(["building_id", "name", "lat", "lon",
                    "usage", "purpose", "color", "icon",
                    "dooh_cat", "brands"])
    ["member_id"].nunique().reset_index()
    .rename(columns={"member_id": "visitors"})
    .sort_values("visitors", ascending=False)
)

# 目的別集計
purpose_agg = (
    result.groupby("purpose")["member_id"].nunique()
    .reset_index().rename(columns={"member_id": "visitors"})
    .sort_values("visitors", ascending=False)
)

# 時間帯×目的別集計
hour_agg = (
    result.groupby(["hour", "purpose"])["member_id"].nunique()
    .reset_index().rename(columns={"member_id": "visitors"})
)

# ── 地図 ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🗺️ 目的別滞在者数マップ")
st.caption("● サイズ = 滞在者数　● 色 = 滞在目的")

fig_map = go.Figure()

# センターエリア円
angles = np.linspace(0, 2 * math.pi, 90)
lat_sc, lon_sc = 111320.0, 111320.0 * math.cos(math.radians(c_lat))
circle_lats = [c_lat + (c_radius / lat_sc) * math.sin(a) for a in angles]
circle_lons = [c_lon + (c_radius / lon_sc) * math.cos(a) for a in angles]
fig_map.add_trace(go.Scattermapbox(
    lat=circle_lats + [circle_lats[0]],
    lon=circle_lons + [circle_lons[0]],
    mode="lines",
    line=dict(color="rgba(255,100,0,0.8)", width=2),
    fill="toself", fillcolor="rgba(255,140,0,0.10)",
    name=f"センターエリア ({c_radius}m)",
    hoverinfo="skip",
))

# 周辺エリア円
s_lats = [c_lat + (s_radius / lat_sc) * math.sin(a) for a in angles]
s_lons = [c_lon + (s_radius / lon_sc) * math.cos(a) for a in angles]
fig_map.add_trace(go.Scattermapbox(
    lat=s_lats + [s_lats[0]],
    lon=s_lons + [s_lons[0]],
    mode="lines",
    line=dict(color="rgba(100,100,255,0.4)", width=1.5, dash="dot"),
    fill="toself", fillcolor="rgba(100,100,255,0.04)",
    name=f"周辺エリア ({s_radius}m)",
    hoverinfo="skip",
))

# 建物プロット
for _, row in bldg_agg.iterrows():
    dooh_info = DOOH_MAP.get(row["usage"], DOOH_MAP["駅・交通施設"])
    fig_map.add_trace(go.Scattermapbox(
        lat=[row["lat"]], lon=[row["lon"]],
        mode="markers",
        marker=dict(
            size=max(10, min(50, row["visitors"] * 2)),
            color=row["color"],
            opacity=0.85,
        ),
        name=f"{row['icon']} {row['purpose']}",
        text=[
            f"<b>{row['name']}</b><br>"
            f"用途: {row['usage']}<br>"
            f"目的: {row['purpose']}<br>"
            f"滞在者: {row['visitors']:,} 人<br>"
            f"DOOH: {row['dooh_cat']}<br>"
            f"推奨ブランド: {row['brands']}"
        ],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

# センター点
fig_map.add_trace(go.Scattermapbox(
    lat=[c_lat], lon=[c_lon],
    mode="markers+text",
    marker=dict(size=18, color="#ff6600", symbol="star"),
    text=["新橋駅"], textposition="top right",
    name="新橋駅（検出中心）",
))

fig_map.update_layout(
    mapbox=dict(style="open-street-map",
                center=dict(lat=c_lat, lon=c_lon), zoom=15),
    height=500,
    margin=dict(r=0, t=0, l=0, b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.88)"),
)
st.plotly_chart(fig_map, use_container_width=True)

# ── グラフ (円 + 棒) ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 目的別滞在者内訳 & 時間帯別推移")

col1, col2 = st.columns([1, 2])

with col1:
    # 円グラフ
    purpose_colors = [
        DOOH_MAP.get(
            next((b["usage"] for b in BUILDINGS_RAW if
                  DOOH_MAP.get(b["usage"], {}).get("purpose") == p), None),
            {"color": "#aaa"}
        )["color"]
        for p in purpose_agg["purpose"]
    ]
    fig_pie = px.pie(
        purpose_agg, values="visitors", names="purpose",
        title="目的別 滞在者割合",
        color_discrete_sequence=[
            DOOH_MAP.get(
                next((u for u, v in DOOH_MAP.items() if v["purpose"] == p), "駅・交通施設"),
                {"color": "#aaa"}
            )["color"]
            for p in purpose_agg["purpose"]
        ],
        hole=0.38,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                          textfont_size=10)
    fig_pie.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10),
                          showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # 時間帯別積み上げ棒グラフ
    purpose_color_map = {
        v["purpose"]: v["color"] for v in DOOH_MAP.values()
    }
    fig_bar = px.bar(
        hour_agg, x="hour", y="visitors", color="purpose",
        title="時間帯別 × 目的別 滞在者数",
        labels={"hour": "時間帯 (時)", "visitors": "滞在者数 (人)", "purpose": "目的"},
        color_discrete_map=purpose_color_map,
        barmode="stack",
    )
    fig_bar.update_layout(
        height=380,
        margin=dict(t=40, b=10, l=10, r=10),
        xaxis=dict(dtick=1, range=[-0.5, 23.5]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── DOOH 推奨テーブル ─────────────────────────────────────────────────────────
st.divider()
st.subheader("📺 DOOH ターゲティング推奨（滞在目的 × 広告訴求）")
st.caption("滞在者数の多い順 | 建物属性からDOOH配信カテゴリ・ブランドを自動提案")

# 目的別集計 + DOOH情報
dooh_table_rows = []
for _, row in purpose_agg.iterrows():
    usage = next(
        (u for u, v in DOOH_MAP.items() if v["purpose"] == row["purpose"]),
        "駅・交通施設"
    )
    d = DOOH_MAP[usage]
    dooh_table_rows.append({
        " ": d["icon"],
        "滞在目的":       row["purpose"],
        "滞在者数":       f"{row['visitors']:,} 人",
        "ターゲット層":   d["audience"],
        "DOOH訴求カテゴリ": d["dooh_cat"],
        "推奨ブランド例": " / ".join(d["brands"]),
        "選定理由":       d["reason"],
    })

dooh_df = pd.DataFrame(dooh_table_rows)
st.dataframe(dooh_df, use_container_width=True, hide_index=True)

# ── 建物別 Top10 ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("🏢 滞在者数 Top 10 建物")
top10 = bldg_agg.head(10)[["icon", "name", "usage", "purpose",
                             "visitors", "dooh_cat", "brands"]].copy()
top10.columns = [" ", "建物名", "用途", "滞在目的", "滞在者数", "DOOH訴求", "推奨ブランド"]
top10["滞在者数"] = top10["滞在者数"].map(lambda x: f"{x:,} 人")
st.dataframe(top10, use_container_width=True, hide_index=True)

# ── データプレビュー ──────────────────────────────────────────────────────────
with st.expander("📄 生データプレビュー（擬似GPS × 建物JOIN）"):
    cols_show = ["member_id", "mesh_code", "stay_datetime",
                 "stay_duration_min", "name", "usage", "purpose"]
    st.dataframe(result[cols_show].head(200), use_container_width=True)
    csv = result[cols_show].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 全データCSVダウンロード", csv,
                       "shimbashi_stay_analysis.csv", "text/csv")
