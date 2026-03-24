"""
特定エリア周辺 滞在行動分析 & DOOH ターゲティング
Plateau CityGML × 擬似GPS × 時間帯別可視化
"""
import io
import math
import random
import re
import struct
import zlib
import urllib.request
import json as _json
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point, Polygon, box as shapely_box
from pyproj import Transformer, CRS
from lxml import etree

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DOOH マッピング（建物用途 → 滞在目的 → 広告訴求）
# ─────────────────────────────────────────────────────────────────────────────
DOOH_MAP = {
    "居酒屋・バー": {
        "purpose": "飲み会・会食", "audience": "飲食目的ビジネス層",
        "dooh_cat": "アルコール飲料・グルメサービス",
        "brands": ["サントリー ザ・プレミアム・モルツ", "キリン 一番搾り", "アサヒ スーパードライ", "ぐるなび", "ホットペッパーグルメ"],
        "reason": "飲み会後の帰宅途中 or 入店検討中の時間帯に訴求効果が高い",
        "color": "#e74c3c", "icon": "🍺",
    },
    "レストラン": {
        "purpose": "食事・グルメ", "audience": "グルメ・ランチ層",
        "dooh_cat": "フードデリバリー・飲食チェーン",
        "brands": ["Uber Eats", "出前館", "ぐるなび", "食べログ", "CoCo壱番屋"],
        "reason": "ランチ・夕食前後の滞在中に次回来店・デリバリー利用を促進",
        "color": "#e67e22", "icon": "🍽️",
    },
    "カフェ": {
        "purpose": "休憩・カジュアル商談", "audience": "ビジネスカジュアル・学習層",
        "dooh_cat": "コーヒー・嗜好品・モバイル決済",
        "brands": ["スターバックス", "ネスカフェ ゴールドブレンド", "ジョージア", "PayPay", "楽天ペイ"],
        "reason": "休憩中の情報感度が高く、飲料・決済サービスの認知向上に最適",
        "color": "#8e44ad", "icon": "☕",
    },
    "オフィスビル": {
        "purpose": "仕事・商談・会議", "audience": "ビジネスパーソン層",
        "dooh_cat": "ビジネスSaaS・転職・金融",
        "brands": ["リクルートエージェント", "マイナビ転職", "freee", "Sansan", "SmartHR"],
        "reason": "就業中・通勤導線でのビジネス系サービス認知に高い効果",
        "color": "#2980b9", "icon": "💼",
    },
    "コンビニ": {
        "purpose": "日用品・食品購入", "audience": "生活者・通勤者層",
        "dooh_cat": "日用品FMCG・飲料・スナック",
        "brands": ["P&G ジョイ", "明治 チョコレート", "キューピー", "サントリー 天然水", "ローソン"],
        "reason": "短時間滞在で購買意欲が高い; 衝動購買を即時促進",
        "color": "#27ae60", "icon": "🏪",
    },
    "ショッピング": {
        "purpose": "買い物・ショッピング", "audience": "消費者・ファッション層",
        "dooh_cat": "ファッション・EC・クーポン",
        "brands": ["ZOZOTOWN", "楽天市場", "Amazon", "ユニクロ", "GU"],
        "reason": "購買行動中の層にEC・店舗誘引広告が高コンバージョン",
        "color": "#f39c12", "icon": "🛍️",
    },
    "ホテル": {
        "purpose": "宿泊・出張", "audience": "ビジネストラベラー層",
        "dooh_cat": "旅行・交通・宿泊予約",
        "brands": ["JR東海 新幹線", "ANA", "楽天トラベル", "じゃらん", "東横INN"],
        "reason": "出張・旅行者に移動・宿泊サービスの次回利用訴求が有効",
        "color": "#16a085", "icon": "🏨",
    },
    "銀行・ATM": {
        "purpose": "金融手続き・資産管理", "audience": "金融関心・資産形成層",
        "dooh_cat": "金融サービス・投資・FinTech",
        "brands": ["三井住友銀行", "楽天証券", "SBI証券", "PayPay", "d払い"],
        "reason": "金融機関滞在中は金融商品の検討意欲が最も高まるタイミング",
        "color": "#2c3e50", "icon": "🏦",
    },
    "医療・クリニック": {
        "purpose": "通院・健康管理", "audience": "健康関心・シニア層",
        "dooh_cat": "医薬品・サプリメント・健康食品",
        "brands": ["大正製薬 リポビタンD", "DHC", "ロート製薬", "アリナミン", "キューピーコーワ"],
        "reason": "健康意識が高まっているタイミングで医薬品・健康訴求が最適",
        "color": "#c0392b", "icon": "🏥",
    },
    "駅・交通施設": {
        "purpose": "通勤・移動・乗換", "audience": "通勤者・移動者層（全属性）",
        "dooh_cat": "交通・コンビニ・スマホアプリ",
        "brands": ["Suica", "モバイルSuica", "Yahoo!カーナビ", "Googleマップ", "NAVITIME"],
        "reason": "最大リーチ、全層へのブランド認知向上と交通・アプリ系に最適",
        "color": "#7f8c8d", "icon": "🚉",
    },
    "住宅": {
        "purpose": "居住・生活", "audience": "居住者・生活者層",
        "dooh_cat": "住宅・不動産・生活サービス",
        "brands": ["SUUMO", "アットホーム", "UR賃貸住宅", "LIFULL HOME'S", "楽天不動産"],
        "reason": "居住者層への生活サービス・不動産訴求に有効",
        "color": "#95a5a6", "icon": "🏠",
    },
}

# Plateau 建物用途コード → DOOH カテゴリ名
PLATEAU_USAGE_MAP = {
    "401": "住宅", "402": "住宅", "403": "住宅", "411": "オフィスビル",
    "412": "オフィスビル", "413": "ショッピング", "414": "ショッピング",
    "415": "ホテル", "416": "レストラン", "417": "ショッピング",
    "418": "医療・クリニック", "419": "オフィスビル", "420": "オフィスビル",
    "421": "ショッピング", "422": "ショッピング", "510": "駅・交通施設",
    "431": "居酒屋・バー", "432": "カフェ",
}

# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def encode_mesh10(lat: float, lon: float) -> str:
    p = int(lat * 1.5); q = int(lon - 100.0)
    lat_rem = lat - p / 1.5; lon_rem = lon - (q + 100.0)
    lat_sz, lon_sz = 2.0/3.0, 1.0
    code = f"{p:02d}{q:02d}"
    lat_sz /= 8; lon_sz /= 8
    r2 = min(int(lat_rem/lat_sz), 7); c2 = min(int(lon_rem/lon_sz), 7)
    lat_rem -= r2*lat_sz; lon_rem -= c2*lon_sz; code += f"{r2}{c2}"
    lat_sz /= 10; lon_sz /= 10
    r3 = min(int(lat_rem/lat_sz), 9); c3 = min(int(lon_rem/lon_sz), 9)
    lat_rem -= r3*lat_sz; lon_rem -= c3*lon_sz; code += f"{r3}{c3}"
    for _ in range(7):
        lat_sz /= 2; lon_sz /= 2; eps = 1e-12
        n = lat_rem >= lat_sz - eps; e = lon_rem >= lon_sz - eps
        d = 4 if (n and e) else 3 if n else 2 if e else 1
        if n: lat_rem -= lat_sz
        if e: lon_rem -= lon_sz
        code += str(d)
    return code


# ─────────────────────────────────────────────────────────────────────────────
# CityGML パーサー
# ─────────────────────────────────────────────────────────────────────────────
_GML_NS   = "http://www.opengis.net/gml"
_BLDG_NS  = "http://www.opengis.net/citygml/building/1.0"
_BLDG_NS2 = "http://www.opengis.net/citygml/building/2.0"


def _detect_crs(root) -> str:
    srs = root.get("srsName", "")
    if not srs:
        for el in root.iter():
            srs = el.get("srsName", "")
            if srs: break
    m = re.search(r"EPSG/\d+/(\d+)", srs, re.IGNORECASE)
    if m: return f"EPSG:{m.group(1)}"
    m = re.search(r"crs:EPSG:[^:]*:(\d+)", srs, re.IGNORECASE)
    if m: return f"EPSG:{m.group(1)}"
    m = re.search(r"epsg\.xml#(\d+)", srs, re.IGNORECASE)
    if m: return f"EPSG:{m.group(1)}"
    m = re.search(r"EPSG[:/](\d{4,})", srs, re.IGNORECASE)
    if m: return f"EPSG:{m.group(1)}"
    return "EPSG:6668"


def _detect_swap_xy(src_crs: str) -> bool:
    try:
        crs_obj = CRS(src_crs)
        return crs_obj.axis_info[0].direction.lower() in ("north", "south")
    except Exception:
        m = re.search(r"(\d{4,5})$", src_crs)
        if m: return int(m.group(1)) in (4326, 6668, 6697, 4019, 4612)
        return False


def _parse_pos_list(text: str, dim: int = 3) -> list:
    vals = [float(v) for v in text.split()]
    return [tuple(vals[i:i+dim]) for i in range(0, len(vals)-dim+1, dim)]


def _polygon_from_pos_list(el, dim: int = 3, swap_xy: bool = False) -> Optional[Polygon]:
    ns = _GML_NS
    ring = el.find(f".//{{{ns}}}LinearRing")
    if ring is None: return None
    pos_el = ring.find(f"{{{ns}}}posList")
    if pos_el is None or not pos_el.text: return None
    pts = _parse_pos_list(pos_el.text, dim)
    if len(pts) < 3: return None
    return Polygon([(p[1], p[0]) for p in pts] if swap_xy else [(p[0], p[1]) for p in pts])


def parse_citygml(file_bytes: bytes) -> gpd.GeoDataFrame:
    root = etree.fromstring(file_bytes)
    src_crs = _detect_crs(root)
    swap_xy = _detect_swap_xy(src_crs)
    to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    dim = 3
    for el in root.iter(f"{{{_GML_NS}}}posList"):
        sd = el.get("srsDimension")
        if sd: dim = int(sd); break

    bldg_ns = _BLDG_NS
    buildings = root.findall(f".//{{{bldg_ns}}}Building")
    if not buildings:
        bldg_ns = _BLDG_NS2
        buildings = root.findall(f".//{{{bldg_ns}}}Building")

    rows = []
    for bldg in buildings:
        h_el = bldg.find(f".//{{{bldg_ns}}}measuredHeight")
        height = float(h_el.text) if h_el is not None and h_el.text else 0.0
        # 用途コード取得
        usage_el = bldg.find(f".//{{{bldg_ns}}}usage")
        usage_code = usage_el.text.strip() if usage_el is not None and usage_el.text else ""

        footprint = None
        fp_el = bldg.find(f".//{{{bldg_ns}}}lod0FootPrint")
        if fp_el is not None:
            poly_el = fp_el.find(f".//{{{_GML_NS}}}Polygon")
            if poly_el is not None:
                footprint = _polygon_from_pos_list(poly_el, dim=2, swap_xy=swap_xy)
        if footprint is None:
            solid_el = bldg.find(f".//{{{bldg_ns}}}lod1Solid") or bldg.find(f".//{{{_GML_NS}}}Solid")
            if solid_el is not None:
                candidates = []
                for poly_el in solid_el.findall(f".//{{{_GML_NS}}}Polygon"):
                    pos_el = poly_el.find(f".//{{{_GML_NS}}}posList")
                    if pos_el is None or not pos_el.text: continue
                    pts = _parse_pos_list(pos_el.text, dim)
                    if len(pts) < 3: continue
                    z_vals = [p[2] for p in pts] if dim >= 3 else [0]
                    candidates.append((min(z_vals), pts))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    pts = candidates[0][1]
                    footprint = Polygon([(p[1], p[0]) for p in pts] if swap_xy else [(p[0], p[1]) for p in pts])

        if footprint is None or not footprint.is_valid or footprint.is_empty:
            continue
        try:
            xcoords, ycoords = to_wgs84.transform(
                [c[0] for c in footprint.exterior.coords],
                [c[1] for c in footprint.exterior.coords],
            )
            rows.append({"height": height, "usage_code": usage_code, "geometry": Polygon(zip(xcoords, ycoords))})
        except Exception:
            continue

    if not rows:
        return gpd.GeoDataFrame(columns=["height", "usage_code", "geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf = gdf[gdf["height"] > 0].reset_index(drop=True)
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Plateau 取得
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_plateau_catalog() -> dict:
    catalog = {}
    rows_per_page, start = 100, 0
    while True:
        url = (f"https://www.geospatial.jp/ckan/api/3/action/package_search"
               f"?fq=tags:PLATEAU&rows={rows_per_page}&start={start}")
        with urllib.request.urlopen(url, timeout=20) as r:
            data = _json.loads(r.read())
        results = data["result"]["results"]
        total = data["result"]["count"]
        for item in results:
            name = item.get("name", "")
            m = re.match(r"^plateau-(\d{5})-.*-(\d{4})$", name)
            if m:
                muni_cd = m.group(1); year = int(m.group(2))
                if not catalog.get(muni_cd) or int(catalog[muni_cd].split("-")[-1]) < year:
                    catalog[muni_cd] = name
        start += rows_per_page
        if start >= total: break
    return catalog


def _gsi_reverse_geocode(lat: float, lon: float) -> Optional[str]:
    url = (f"https://mreversegeocoder.gsi.go.jp/reverse-geocoder/"
           f"LonLatToAddress?lat={lat}&lon={lon}")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = _json.loads(r.read())
        return data["results"]["muniCd"]
    except Exception:
        return None


def _get_plateau_zip_url(dataset_id: str) -> Optional[str]:
    url = f"https://www.geospatial.jp/ckan/api/3/action/package_show?id={dataset_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = _json.loads(r.read())
        resources = data["result"]["resources"]
        v3_url = fallback_url = None
        for res in resources:
            name = res.get("name", ""); rurl = res.get("url", "")
            if "CityGML" in name and rurl.endswith(".zip"):
                if "v3" in name or "v3" in rurl: v3_url = rurl
                elif fallback_url is None: fallback_url = rurl
        return v3_url or fallback_url
    except Exception:
        return None


def _read_zip_cd(zip_url: str) -> dict:
    req = urllib.request.Request(zip_url, headers={"Range": "bytes=-65536"})
    with urllib.request.urlopen(req, timeout=30) as r:
        tail = r.read()
    sig = b"PK\x05\x06"
    pos = tail.rfind(sig)
    if pos == -1: raise ValueError("ZIP EOCD が見つかりません")
    eocd = tail[pos:]
    cd_size   = struct.unpack_from("<I", eocd, 12)[0]
    cd_offset = struct.unpack_from("<I", eocd, 16)[0]
    req2 = urllib.request.Request(zip_url, headers={"Range": f"bytes={cd_offset}-{cd_offset+cd_size-1}"})
    with urllib.request.urlopen(req2, timeout=30) as r:
        cd_data = r.read()
    files = {}; offset = 0
    while offset + 46 <= len(cd_data):
        if cd_data[offset:offset+4] != b"PK\x01\x02": break
        method      = struct.unpack_from("<H", cd_data, offset+10)[0]
        comp_size   = struct.unpack_from("<I", cd_data, offset+20)[0]
        fname_len   = struct.unpack_from("<H", cd_data, offset+28)[0]
        extra_len   = struct.unpack_from("<H", cd_data, offset+30)[0]
        comment_len = struct.unpack_from("<H", cd_data, offset+32)[0]
        local_off   = struct.unpack_from("<I", cd_data, offset+42)[0]
        fname = cd_data[offset+46:offset+46+fname_len].decode("utf-8", errors="replace")
        if "bldg" in fname and fname.endswith(".gml"):
            files[fname.split("/")[-1]] = (local_off, comp_size, method)
        offset += 46 + fname_len + extra_len + comment_len
    return files


def _extract_gml_from_zip(zip_url: str, local_off: int, comp_size: int, method: int) -> bytes:
    lh_req = urllib.request.Request(zip_url, headers={"Range": f"bytes={local_off}-{local_off+29}"})
    with urllib.request.urlopen(lh_req, timeout=30) as r:
        lh = r.read()
    lh_fname_len = struct.unpack_from("<H", lh, 26)[0]
    lh_extra_len = struct.unpack_from("<H", lh, 28)[0]
    data_start = local_off + 30 + lh_fname_len + lh_extra_len
    data_req = urllib.request.Request(zip_url, headers={"Range": f"bytes={data_start}-{data_start+comp_size-1}"})
    with urllib.request.urlopen(data_req, timeout=120) as r:
        comp_data = r.read()
    return zlib.decompress(comp_data, -15) if method == 8 else comp_data


def fetch_plateau_for_area(center_lat: float, center_lon: float,
                            radius_m: float, log_box) -> Optional[gpd.GeoDataFrame]:
    """指定エリア（円）の Plateau 建物データを取得"""
    logs = []
    def log(msg):
        logs.append(msg)
        log_box.markdown("\n\n".join(logs))

    lat_sc = 111320.0
    lon_sc = 111320.0 * math.cos(math.radians(center_lat))
    dlat = radius_m / lat_sc
    dlon = radius_m / lon_sc
    area_circle = Point(center_lon, center_lat).buffer(max(dlat, dlon) * 1.05)

    log("📋 Plateau カタログを取得中...")
    try:
        catalog = _fetch_plateau_catalog()
    except Exception as e:
        log(f"❌ カタログ取得エラー: {e}"); return None
    log(f"✅ カタログ取得完了（{len(catalog)} 市区町村）")

    log("📍 エリアの市区町村を特定中...")
    muni_cd = _gsi_reverse_geocode(center_lat, center_lon)
    if not muni_cd:
        log("❌ 市区町村コードを取得できませんでした"); return None
    log(f"✅ 市区町村コード: {muni_cd}")

    lat_sz_3 = (2.0/3.0) / 8 / 10
    lon_sz_3 = 1.0 / 8 / 10
    prefixes = set()
    la = math.floor((center_lat - dlat) / lat_sz_3) * lat_sz_3
    while la <= center_lat + dlat + lat_sz_3:
        lo = math.floor((center_lon - dlon) / lon_sz_3) * lon_sz_3
        while lo <= center_lon + dlon + lon_sz_3:
            if area_circle.intersects(shapely_box(lo, la, lo + lon_sz_3, la + lat_sz_3)):
                code = encode_mesh10(la + lat_sz_3 / 2, lo + lon_sz_3 / 2)
                prefixes.add(code[:8])
            lo += lon_sz_3
        la += lat_sz_3
    log(f"✅ 対象 3 次メッシュ: {len(prefixes)} タイル")

    dataset_id = catalog.get(muni_cd)
    if not dataset_id:
        log(f"⚠️ 市区町村 {muni_cd} の Plateau データが見つかりません（対応エリア外）"); return None
    log(f"🔍 ZIP URL を取得中...")
    zip_url = _get_plateau_zip_url(dataset_id)
    if not zip_url:
        log("⚠️ ZIP URL が取得できませんでした"); return None

    log("📦 ZIP インデックスを解析中...")
    try:
        cd = _read_zip_cd(zip_url)
    except Exception as e:
        log(f"❌ ZIP 解析エラー: {e}"); return None

    needed = {f: info for f, info in cd.items() if any(f.startswith(p) for p in prefixes)}
    if not needed:
        log("⚠️ 対象メッシュの GML が見つかりませんでした"); return None

    log(f"⬇️ {len(needed)} 個の GML をダウンロード中...")
    all_gdfs = []
    for fname, (local_off, comp_size, method) in needed.items():
        log(f"　`{fname}` ({comp_size//1024:,} KB)...")
        try:
            gml_bytes = _extract_gml_from_zip(zip_url, local_off, comp_size, method)
            gdf = parse_citygml(gml_bytes)
            if not gdf.empty:
                all_gdfs.append(gdf)
                log(f"　✅ {len(gdf):,} 棟")
            else:
                log(f"　⚠️ データなし")
        except Exception as e:
            log(f"　❌ {e}")

    if not all_gdfs:
        log("❌ 建物データを取得できませんでした"); return None

    combined = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
    before = len(combined)
    combined = combined[combined.geometry.intersects(area_circle)].reset_index(drop=True)
    log(f"✂️ エリア外除去: {before:,} → {len(combined):,} 棟")
    log(f"\n✅ **取得完了: {len(combined):,} 棟**")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 建物分類・DOOH エンリッチ
# ─────────────────────────────────────────────────────────────────────────────
def _classify_usage(usage_code: str, height: float, lon: float, lat: float) -> str:
    if usage_code in PLATEAU_USAGE_MAP:
        return PLATEAU_USAGE_MAP[usage_code]
    h = hash((round(lon, 5), round(lat, 5))) % 100
    if height > 60: return "オフィスビル"
    elif height > 30: return ["オフィスビル", "ホテル", "ショッピング"][h % 3]
    elif height > 15: return ["レストラン", "オフィスビル", "カフェ", "居酒屋・バー"][h % 4]
    elif height > 5:  return ["居酒屋・バー", "コンビニ", "カフェ", "銀行・ATM"][h % 4]
    else:             return ["コンビニ", "居酒屋・バー"][h % 2]


def enrich_buildings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    centroids = gdf.geometry.centroid
    gdf["centroid_lat"] = centroids.y
    gdf["centroid_lon"] = centroids.x
    gdf["usage"] = gdf.apply(
        lambda r: _classify_usage(r["usage_code"], r["height"], r["centroid_lon"], r["centroid_lat"]),
        axis=1,
    )
    for col in ["purpose", "audience", "dooh_cat", "reason", "color", "icon"]:
        gdf[col] = gdf["usage"].map(lambda u: DOOH_MAP.get(u, DOOH_MAP["駅・交通施設"])[col])
    gdf["brands"] = gdf["usage"].map(
        lambda u: " / ".join(DOOH_MAP.get(u, DOOH_MAP["駅・交通施設"])["brands"][:3])
    )
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# 分析パイプライン
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(
    gps_df: pd.DataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    center_lat: float, center_lon: float,
    center_radius_m: float, surround_radius_m: float,
    min_stay_min: float,
    hour_range: tuple,
    selected_dates: list,
    join_threshold_m: float = 50.0,
) -> tuple:
    # 時間帯・日付フィルタ
    h_start, h_end = hour_range
    mask = (gps_df["hour"] >= h_start) & (gps_df["hour"] <= h_end)
    if selected_dates:
        mask &= gps_df["date"].isin(selected_dates)
    gps_f = gps_df[mask].copy()

    # 各GPSレコードと中心点の距離
    gps_f["dist_center"] = gps_f.apply(
        lambda r: haversine_m(r["lat"], r["lon"], center_lat, center_lon), axis=1
    )

    # センターエリア内メンバー検出
    center_members = set(gps_f[gps_f["dist_center"] <= center_radius_m]["member_id"])
    if not center_members:
        return pd.DataFrame(), 0

    # 周辺エリアの GPS ポイント（センターメンバー × 周辺範囲 × 最低滞在時間）
    surround_gps = gps_f[
        (gps_f["member_id"].isin(center_members)) &
        (gps_f["dist_center"] <= surround_radius_m) &
        (gps_f["stay_duration_min"] >= min_stay_min)
    ].copy()

    if surround_gps.empty:
        return pd.DataFrame(), len(center_members)

    # 建物重心への空間結合（sindex で高速化）
    bldg = buildings_gdf.copy()
    bldg = bldg.reset_index(drop=True)
    bldg_centroids = bldg.geometry.centroid
    bldg_centroid_gdf = gpd.GeoDataFrame(
        {"bldg_row_idx": bldg.index}, geometry=bldg_centroids, crs="EPSG:4326"
    )
    sindex = bldg_centroid_gdf.sindex

    lat_sc = 111320.0
    lon_sc = 111320.0 * math.cos(math.radians(center_lat))
    thresh_deg = join_threshold_m / min(lat_sc, lon_sc)

    matched_rows = []
    for _, gps_row in surround_gps.iterrows():
        pt = Point(gps_row["lon"], gps_row["lat"])
        candidates = list(sindex.intersection(pt.buffer(thresh_deg).bounds))
        if not candidates:
            continue
        best_idx, best_dist = None, float("inf")
        for ci in candidates:
            cgeom = bldg_centroid_gdf.geometry.iloc[ci]
            d = haversine_m(gps_row["lat"], gps_row["lon"], cgeom.y, cgeom.x)
            if d < best_dist and d < join_threshold_m:
                best_dist = d; best_idx = ci
        if best_idx is not None:
            b = bldg.iloc[best_idx]
            matched_rows.append({
                **gps_row.to_dict(),
                "bldg_height":   b["height"],
                "usage":         b["usage"],
                "purpose":       b["purpose"],
                "dooh_cat":      b["dooh_cat"],
                "brands":        b["brands"],
                "reason":        b["reason"],
                "color":         b["color"],
                "icon":          b["icon"],
                "centroid_lat":  b["centroid_lat"],
                "centroid_lon":  b["centroid_lon"],
                "dist_to_bldg":  best_dist,
            })

    if not matched_rows:
        return pd.DataFrame(), len(center_members)

    result = pd.DataFrame(matched_rows)
    return result, len(center_members)


# ─────────────────────────────────────────────────────────────────────────────
# GPS CSV 読み込み・バリデーション
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"member_id", "lat", "lon", "stay_datetime", "stay_duration_min"}


def load_gps_csv(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSV 読み込みエラー: {e}"); return None
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"必要なカラムがありません: {missing}"); return None
    df["stay_datetime"] = pd.to_datetime(df["stay_datetime"], errors="coerce")
    df = df.dropna(subset=["stay_datetime", "lat", "lon"])
    df["hour"] = df["stay_datetime"].dt.hour
    df["date"] = df["stay_datetime"].dt.date
    df["stay_duration_min"] = pd.to_numeric(df["stay_duration_min"], errors="coerce").fillna(0)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="特定エリア周辺 滞在行動分析", page_icon="📍", layout="wide")

st.title("📍 特定エリア周辺 滞在行動分析 & DOOH ターゲティング")
st.caption("Plateau CityGML（建物データ） × GPS 滞在データ（CSV） × 時間帯別可視化")

# ── セッション初期化 ──────────────────────────────────────────────────────────
for key, default in [
    ("buildings_gdf", None),
    ("fetch_done", False),
    ("last_fetch_params", None),
    ("gps_df", None),
    ("result_df", None),
    ("n_center", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── サイドバー ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ エリア設定")
    st.divider()

    st.subheader("① 中心点")
    c_lat = st.number_input("緯度", value=35.6659, format="%.6f", step=0.0001)
    c_lon = st.number_input("経度", value=139.7575, format="%.6f", step=0.0001)

    st.divider()
    st.subheader("② センターエリア半径")
    c_radius = st.slider("検出半径 (m)", 50, 500, 200, step=25,
                         help="この範囲内に来たメンバーを分析対象とします")

    st.divider()
    st.subheader("③ 周辺エリア半径")
    s_radius = st.slider("周辺範囲 (m)", 200, 2000, 800, step=100,
                         help="センターメンバーの周辺滞在を分析する範囲")

    st.divider()
    fetch_btn = st.button("🏗️ 建物データ取得 (Plateau)", type="primary",
                           use_container_width=True)

    if st.session_state["fetch_done"]:
        bldg_count = len(st.session_state["buildings_gdf"]) if st.session_state["buildings_gdf"] is not None else 0
        st.success(f"✅ 建物取得済: {bldg_count:,} 棟")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: 建物データ取得
# ─────────────────────────────────────────────────────────────────────────────
st.header("① 建物データ取得")

fetch_params = (c_lat, c_lon, s_radius)
if fetch_btn:
    st.session_state["fetch_done"] = False
    st.session_state["buildings_gdf"] = None

    log_area = st.empty()
    with st.spinner("Plateau から建物データを取得中..."):
        gdf = fetch_plateau_for_area(c_lat, c_lon, s_radius, log_area)

    if gdf is not None and not gdf.empty:
        gdf = enrich_buildings(gdf)
        st.session_state["buildings_gdf"] = gdf
        st.session_state["fetch_done"] = True
        st.session_state["last_fetch_params"] = fetch_params
        st.session_state["result_df"] = None
    else:
        st.error("建物データの取得に失敗しました。エリアを変更して再試行してください。")

if not st.session_state["fetch_done"]:
    st.info("サイドバーで中心点と半径を設定し、「建物データ取得 (Plateau)」ボタンを押してください。")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: エリア確認マップ
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("② エリア確認マップ")

buildings_gdf: gpd.GeoDataFrame = st.session_state["buildings_gdf"]
bldg_count = len(buildings_gdf)
st.caption(f"取得建物: **{bldg_count:,} 棟** | センター半径: {c_radius} m | 周辺範囲: {s_radius} m")

fig_check = go.Figure()
angles = np.linspace(0, 2 * math.pi, 90)
lat_sc_v = 111320.0
lon_sc_v = 111320.0 * math.cos(math.radians(c_lat))

# センターエリア円
c_lats = [c_lat + (c_radius / lat_sc_v) * math.sin(a) for a in angles]
c_lons = [c_lon + (c_radius / lon_sc_v) * math.cos(a) for a in angles]
fig_check.add_trace(go.Scattermapbox(
    lat=c_lats + [c_lats[0]], lon=c_lons + [c_lons[0]], mode="lines",
    line=dict(color="rgba(255,100,0,0.9)", width=2),
    fill="toself", fillcolor="rgba(255,140,0,0.12)",
    name=f"センターエリア ({c_radius}m)", hoverinfo="skip",
))

# 周辺エリア円
s_lats = [c_lat + (s_radius / lat_sc_v) * math.sin(a) for a in angles]
s_lons = [c_lon + (s_radius / lon_sc_v) * math.cos(a) for a in angles]
fig_check.add_trace(go.Scattermapbox(
    lat=s_lats + [s_lats[0]], lon=s_lons + [s_lons[0]], mode="lines",
    line=dict(color="rgba(50,100,255,0.5)", width=1.5, dash="dot"),
    fill="toself", fillcolor="rgba(50,100,255,0.04)",
    name=f"周辺エリア ({s_radius}m)", hoverinfo="skip",
))

# 建物プロット（用途別色）
for usage, grp in buildings_gdf.groupby("usage"):
    d = DOOH_MAP.get(usage, DOOH_MAP["駅・交通施設"])
    fig_check.add_trace(go.Scattermapbox(
        lat=grp["centroid_lat"], lon=grp["centroid_lon"],
        mode="markers",
        marker=dict(size=6, color=d["color"], opacity=0.75),
        name=f"{d['icon']} {usage}",
        text=[f"<b>{usage}</b><br>高さ: {h:.1f}m" for h in grp["height"]],
        hovertemplate="%{text}<extra></extra>",
    ))

# 中心点
fig_check.add_trace(go.Scattermapbox(
    lat=[c_lat], lon=[c_lon], mode="markers+text",
    marker=dict(size=16, color="#ff6600", symbol="star"),
    text=["中心点"], textposition="top right",
    name="中心点", hoverinfo="skip",
))

fig_check.update_layout(
    mapbox=dict(style="open-street-map", center=dict(lat=c_lat, lon=c_lon), zoom=15),
    height=500, margin=dict(r=0, t=0, l=0, b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.88)"),
)
st.plotly_chart(fig_check, use_container_width=True)

# 建物用途内訳
usage_summary = buildings_gdf["usage"].value_counts().reset_index()
usage_summary.columns = ["用途", "棟数"]
st.dataframe(usage_summary, use_container_width=False, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: GPS データアップロード & 分析
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("③ GPS データアップロード & 分析実行")

with st.expander("📄 GPS CSV の仕様（クリックで確認）", expanded=False):
    st.markdown("""
**必須カラム:**

| カラム名 | 型 | 説明 |
|---|---|---|
| `member_id` | 文字列 | メンバー識別子 (例: M00001) |
| `lat` | 数値 | 緯度 (WGS84) |
| `lon` | 数値 | 経度 (WGS84) |
| `stay_datetime` | 日時 | 滞在日時 (例: 2024-03-01 08:30:00) |
| `stay_duration_min` | 数値 | 滞在時間 (分) |

サンプルデータは `generate_sample_gps.py` で生成できます。
""")

uploaded = st.file_uploader("GPS データ CSV をアップロード", type=["csv"])

if uploaded:
    gps_df = load_gps_csv(uploaded)
    if gps_df is not None:
        st.session_state["gps_df"] = gps_df
        st.success(f"✅ {len(gps_df):,} レコード読み込み完了（メンバー: {gps_df['member_id'].nunique():,} 人）")

if st.session_state["gps_df"] is None:
    st.info("GPS データ CSV をアップロードしてください。")
    st.stop()

gps_df: pd.DataFrame = st.session_state["gps_df"]

# 分析パラメータ
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    min_stay = st.slider("最低滞在時間 (分)", 1, 120, 10, step=1)
with col_p2:
    h_range = st.slider("対象時間帯", 0, 23, (0, 23))
with col_p3:
    all_dates = sorted(gps_df["date"].unique())
    sel_dates = st.multiselect(
        "日付フィルタ", options=all_dates, default=all_dates,
        format_func=lambda d: d.strftime("%m/%d") if hasattr(d, "strftime") else str(d),
    )

run_btn = st.button("▶ 分析実行", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("分析中..."):
        result, n_center = run_analysis(
            gps_df, buildings_gdf,
            c_lat, c_lon, c_radius, s_radius,
            min_stay, h_range, sel_dates,
        )
    st.session_state["result_df"] = result
    st.session_state["n_center"] = n_center

if st.session_state["result_df"] is None:
    st.stop()

result: pd.DataFrame = st.session_state["result_df"]
n_center: int = st.session_state["n_center"]

# ─────────────────────────────────────────────────────────────────────────────
# 分析結果
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("④ 分析結果")

k1, k2, k3, k4 = st.columns(4)
k1.metric("センターエリア検出人数", f"{n_center:,} 人")
k2.metric("周辺滞在レコード数",     f"{len(result):,} 件" if not result.empty else "0 件")
k3.metric("ユニーク滞在者数",       f"{result['member_id'].nunique():,} 人" if not result.empty else "0 人")
k4.metric("マッチ建物数",           f"{result.groupby(['centroid_lat','centroid_lon']).ngroups:,} 棟" if not result.empty else "0 棟")

if result.empty:
    st.warning("条件に合うデータがありません。パラメータを調整してください。")
    st.stop()

# 建物ごとに集計
bldg_agg = (
    result.groupby(["centroid_lat", "centroid_lon", "usage", "purpose",
                    "color", "icon", "dooh_cat", "brands"])
    ["member_id"].nunique().reset_index()
    .rename(columns={"member_id": "visitors"})
    .sort_values("visitors", ascending=False)
)

purpose_agg = (
    result.groupby("purpose")["member_id"].nunique()
    .reset_index().rename(columns={"member_id": "visitors"})
    .sort_values("visitors", ascending=False)
)

hour_agg = (
    result.groupby(["hour", "purpose"])["member_id"].nunique()
    .reset_index().rename(columns={"member_id": "visitors"})
)

# 結果マップ
st.subheader("🗺️ 目的別滞在者数マップ")
fig_res = go.Figure()

fig_res.add_trace(go.Scattermapbox(
    lat=c_lats + [c_lats[0]], lon=c_lons + [c_lons[0]], mode="lines",
    line=dict(color="rgba(255,100,0,0.8)", width=2),
    fill="toself", fillcolor="rgba(255,140,0,0.10)",
    name=f"センターエリア ({c_radius}m)", hoverinfo="skip",
))
fig_res.add_trace(go.Scattermapbox(
    lat=s_lats + [s_lats[0]], lon=s_lons + [s_lons[0]], mode="lines",
    line=dict(color="rgba(100,100,255,0.4)", width=1.5, dash="dot"),
    fill="toself", fillcolor="rgba(100,100,255,0.04)",
    name=f"周辺エリア ({s_radius}m)", hoverinfo="skip",
))

for _, row in bldg_agg.iterrows():
    fig_res.add_trace(go.Scattermapbox(
        lat=[row["centroid_lat"]], lon=[row["centroid_lon"]], mode="markers",
        marker=dict(size=max(10, min(50, row["visitors"] * 3)), color=row["color"], opacity=0.85),
        name=f"{row['icon']} {row['purpose']}",
        text=[f"<b>{row['usage']}</b><br>目的: {row['purpose']}<br>滞在者: {row['visitors']:,} 人<br>DOOH: {row['dooh_cat']}<br>推奨: {row['brands']}"],
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))

fig_res.add_trace(go.Scattermapbox(
    lat=[c_lat], lon=[c_lon], mode="markers+text",
    marker=dict(size=16, color="#ff6600", symbol="star"),
    text=["中心点"], textposition="top right", name="中心点",
))
fig_res.update_layout(
    mapbox=dict(style="open-street-map", center=dict(lat=c_lat, lon=c_lon), zoom=15),
    height=500, margin=dict(r=0, t=0, l=0, b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.88)"),
)
st.plotly_chart(fig_res, use_container_width=True)

# グラフ
st.subheader("📊 目的別滞在者内訳 & 時間帯別推移")
col1, col2 = st.columns([1, 2])

with col1:
    fig_pie = px.pie(
        purpose_agg, values="visitors", names="purpose",
        title="目的別 滞在者割合",
        color="purpose",
        color_discrete_map={v["purpose"]: v["color"] for v in DOOH_MAP.values()},
        hole=0.38,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
    fig_pie.update_layout(height=380, margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    purpose_color_map = {v["purpose"]: v["color"] for v in DOOH_MAP.values()}
    fig_bar = px.bar(
        hour_agg, x="hour", y="visitors", color="purpose",
        title="時間帯別 × 目的別 滞在者数",
        labels={"hour": "時間帯 (時)", "visitors": "滞在者数 (人)", "purpose": "目的"},
        color_discrete_map=purpose_color_map, barmode="stack",
    )
    fig_bar.update_layout(
        height=380, margin=dict(t=40,b=10,l=10,r=10),
        xaxis=dict(dtick=1, range=[-0.5, 23.5]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# DOOH 推奨テーブル
st.divider()
st.subheader("📺 DOOH ターゲティング推奨")
dooh_rows = []
for _, row in purpose_agg.iterrows():
    usage = next((u for u, v in DOOH_MAP.items() if v["purpose"] == row["purpose"]), "駅・交通施設")
    d = DOOH_MAP[usage]
    dooh_rows.append({
        " ": d["icon"], "滞在目的": row["purpose"],
        "滞在者数": f"{row['visitors']:,} 人",
        "ターゲット層": d["audience"],
        "DOOH訴求カテゴリ": d["dooh_cat"],
        "推奨ブランド例": " / ".join(d["brands"]),
        "選定理由": d["reason"],
    })
st.dataframe(pd.DataFrame(dooh_rows), use_container_width=True, hide_index=True)

# 生データ & ダウンロード
st.divider()
st.subheader("📥 分析結果ダウンロード")
dl_df = result[["member_id", "lat", "lon", "stay_datetime", "stay_duration_min",
                 "usage", "purpose", "dooh_cat", "dist_to_bldg"]].copy()
st.download_button(
    "⬇️ 分析結果 CSV ダウンロード",
    data=dl_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="analysis_result.csv", mime="text/csv",
)
with st.expander("生データプレビュー"):
    st.dataframe(result.head(200), use_container_width=True)
