"""
新橋駅周辺 擬似GPSデータ生成スクリプト
実行: python generate_sample_gps.py
出力: sample_gps_data.csv

GPS データ仕様:
  member_id         : メンバーID (M00001〜M02000)
  lat               : 緯度 (WGS84)
  lon               : 経度 (WGS84)
  stay_datetime     : 滞在日時
  stay_duration_min : 滞在時間 (分)
"""

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SHIMBASHI_LAT = 35.6659
SHIMBASHI_LON = 139.7575
RANDOM_SEED = 42

# 新橋周辺の滞在ロケーション（緯度・経度・用途）
LOCATIONS = [
    # 新橋駅周辺（センターエリア）
    {"lat": 35.6659, "lon": 139.7575, "usage": "駅・交通施設",  "name": "新橋駅"},
    {"lat": 35.6660, "lon": 139.7572, "usage": "駅・交通施設",  "name": "新橋駅 SL広場"},
    {"lat": 35.6657, "lon": 139.7578, "usage": "駅・交通施設",  "name": "新橋駅 東口"},
    # 居酒屋・バー
    {"lat": 35.6655, "lon": 139.7588, "usage": "居酒屋・バー",  "name": "炭火焼鳥 新橋屋"},
    {"lat": 35.6651, "lon": 139.7583, "usage": "居酒屋・バー",  "name": "大衆酒場 のんき"},
    {"lat": 35.6658, "lon": 139.7581, "usage": "居酒屋・バー",  "name": "居酒屋 春よ来い"},
    {"lat": 35.6653, "lon": 139.7591, "usage": "居酒屋・バー",  "name": "立ち飲み 角打ち番長"},
    {"lat": 35.6647, "lon": 139.7587, "usage": "居酒屋・バー",  "name": "もつ焼き でんがな"},
    {"lat": 35.6665, "lon": 139.7579, "usage": "居酒屋・バー",  "name": "ビアホール LION"},
    {"lat": 35.6669, "lon": 139.7573, "usage": "居酒屋・バー",  "name": "個室居酒屋 和楽"},
    {"lat": 35.6644, "lon": 139.7584, "usage": "居酒屋・バー",  "name": "赤ちょうちん 二番街"},
    # レストラン
    {"lat": 35.6662, "lon": 139.7568, "usage": "レストラン",    "name": "ラーメン 蔦 新橋店"},
    {"lat": 35.6650, "lon": 139.7566, "usage": "レストラン",    "name": "焼肉 和牛TOKYO"},
    {"lat": 35.6672, "lon": 139.7581, "usage": "レストラン",    "name": "寿司 銀座やまと 新橋"},
    {"lat": 35.6648, "lon": 139.7571, "usage": "レストラン",    "name": "洋食 グリル アンジュ"},
    {"lat": 35.6640, "lon": 139.7576, "usage": "レストラン",    "name": "中華 龍雲閣"},
    # カフェ
    {"lat": 35.6660, "lon": 139.7573, "usage": "カフェ",        "name": "スターバックス SL広場店"},
    {"lat": 35.6655, "lon": 139.7568, "usage": "カフェ",        "name": "ドトール 新橋西口店"},
    {"lat": 35.6645, "lon": 139.7572, "usage": "カフェ",        "name": "タリーズ 新橋店"},
    # オフィスビル
    {"lat": 35.6664, "lon": 139.7585, "usage": "オフィスビル",  "name": "新橋駅前ビル1号館"},
    {"lat": 35.6666, "lon": 139.7582, "usage": "オフィスビル",  "name": "新橋駅前ビル2号館"},
    {"lat": 35.6638, "lon": 139.7558, "usage": "オフィスビル",  "name": "汐留シオサイト オフィス棟"},
    {"lat": 35.6634, "lon": 139.7556, "usage": "オフィスビル",  "name": "電通本社ビル"},
    {"lat": 35.6628, "lon": 139.7561, "usage": "オフィスビル",  "name": "日本テレビタワー"},
    {"lat": 35.6675, "lon": 139.7576, "usage": "オフィスビル",  "name": "新橋東洋ビル"},
    {"lat": 35.6662, "lon": 139.7578, "usage": "オフィスビル",  "name": "ニュー新橋ビル"},
    # コンビニ
    {"lat": 35.6658, "lon": 139.7575, "usage": "コンビニ",      "name": "セブン-イレブン 新橋駅前店"},
    {"lat": 35.6657, "lon": 139.7578, "usage": "コンビニ",      "name": "ローソン SL広場店"},
    {"lat": 35.6661, "lon": 139.7570, "usage": "コンビニ",      "name": "ファミリーマート 新橋西口店"},
    # ショッピング
    {"lat": 35.6653, "lon": 139.7577, "usage": "ショッピング",  "name": "マツモトキヨシ 新橋店"},
    {"lat": 35.6648, "lon": 139.7579, "usage": "ショッピング",  "name": "ユニクロ 新橋店"},
    # ホテル
    {"lat": 35.6680, "lon": 139.7565, "usage": "ホテル",        "name": "アパホテル 新橋御成門"},
    {"lat": 35.6670, "lon": 139.7562, "usage": "ホテル",        "name": "東横INN 新橋駅前"},
    # 銀行・ATM
    {"lat": 35.6667, "lon": 139.7574, "usage": "銀行・ATM",     "name": "三井住友銀行 新橋支店"},
    {"lat": 35.6660, "lon": 139.7581, "usage": "銀行・ATM",     "name": "みずほ銀行 新橋支店"},
    # 医療
    {"lat": 35.6673, "lon": 139.7568, "usage": "医療・クリニック", "name": "新橋内科クリニック"},
    {"lat": 35.6650, "lon": 139.7562, "usage": "医療・クリニック", "name": "新橋整形外科"},
]

# 用途別ピーク時間帯・滞在時間
USAGE_PARAMS = {
    "駅・交通施設":   {"peak_h": [7, 8, 12, 13, 17, 18, 19, 20], "dur": (3,  20),  "weight": 10},
    "居酒屋・バー":   {"peak_h": [18, 19, 20, 21, 22],            "dur": (60, 180), "weight": 4},
    "レストラン":     {"peak_h": [12, 13, 18, 19],                 "dur": (30, 90),  "weight": 3},
    "カフェ":         {"peak_h": [8, 9, 10, 13, 14],               "dur": (20, 60),  "weight": 2},
    "オフィスビル":   {"peak_h": list(range(9, 19)),               "dur": (120,480), "weight": 5},
    "コンビニ":       {"peak_h": [7, 8, 12, 13, 18, 19],           "dur": (3, 15),   "weight": 3},
    "ショッピング":   {"peak_h": [11, 12, 13, 14, 15, 16],         "dur": (20, 60),  "weight": 2},
    "ホテル":         {"peak_h": [8, 9, 20, 21, 22],               "dur": (30, 120), "weight": 1},
    "銀行・ATM":      {"peak_h": [10, 11, 12, 14, 15],             "dur": (5, 30),   "weight": 1},
    "医療・クリニック": {"peak_h": [10, 11, 14, 15],               "dur": (30, 90),  "weight": 1},
}


def generate_gps_data(n_members: int = 2000, n_days: int = 7,
                      base_date_str: str = "2024-03-01") -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)
    base_date = datetime.strptime(base_date_str, "%Y-%m-%d")

    station_locs = [l for l in LOCATIONS if l["usage"] == "駅・交通施設"]
    other_locs   = [l for l in LOCATIONS if l["usage"] != "駅・交通施設"]
    usage_weights = [USAGE_PARAMS.get(l["usage"], USAGE_PARAMS["コンビニ"])["weight"] for l in other_locs]

    records = []

    for m_idx in range(n_members):
        member_id = f"M{m_idx+1:05d}"
        visit_days = sorted(rng.sample(range(n_days), rng.randint(1, min(5, n_days))))

        for day_offset in visit_days:
            date = base_date + timedelta(days=day_offset)

            # ① 新橋駅を必ず通過
            sta = rng.choice(station_locs)
            peak_h = rng.choice([7, 8, 17, 18, 19])
            dt_sta = date + timedelta(hours=peak_h, minutes=rng.randint(0, 59))
            dur_sta = rng.uniform(3, 20)
            jlat = sta["lat"] + np_rng.uniform(-0.0003, 0.0003)
            jlon = sta["lon"] + np_rng.uniform(-0.0003, 0.0003)
            records.append({
                "member_id":         member_id,
                "lat":               round(jlat, 6),
                "lon":               round(jlon, 6),
                "stay_datetime":     dt_sta.strftime("%Y-%m-%d %H:%M:%S"),
                "stay_duration_min": round(dur_sta, 1),
            })

            # ② 周辺ロケーションを 1〜4 か所訪問
            n_visits = rng.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]
            visited = rng.choices(other_locs, weights=usage_weights, k=n_visits)

            for loc in visited:
                params = USAGE_PARAMS.get(loc["usage"], USAGE_PARAMS["コンビニ"])
                hour = rng.choice(params["peak_h"])
                visit_dt = date + timedelta(hours=hour, minutes=rng.randint(0, 59))
                dur_min = rng.uniform(*params["dur"])
                jlat = loc["lat"] + np_rng.uniform(-0.00008, 0.00008)
                jlon = loc["lon"] + np_rng.uniform(-0.00008, 0.00008)
                records.append({
                    "member_id":         member_id,
                    "lat":               round(jlat, 6),
                    "lon":               round(jlon, 6),
                    "stay_datetime":     visit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "stay_duration_min": round(dur_min, 1),
                })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("擬似 GPS データを生成中...")
    df = generate_gps_data(n_members=2000, n_days=7)
    output_path = "sample_gps_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    size_kb = df.memory_usage(deep=True).sum() / 1024
    print(f"✅ 生成完了: {len(df):,} レコード / {df['member_id'].nunique():,} メンバー")
    print(f"   期間: {df['stay_datetime'].min()} 〜 {df['stay_datetime'].max()}")
    print(f"   緯度範囲: {df['lat'].min():.4f} 〜 {df['lat'].max():.4f}")
    print(f"   経度範囲: {df['lon'].min():.4f} 〜 {df['lon'].max():.4f}")
    print(f"   ファイル: {output_path}")
    print(f"   メモリ使用量: {size_kb:.0f} KB")
