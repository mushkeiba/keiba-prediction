# 地方競馬 予測API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime
import os
from pathlib import Path

# プロジェクトのルートディレクトリを取得
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="地方競馬予測API",
    description="AIが予測する地方競馬の3着以内予測",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 競馬場設定 ==========
TRACKS = {
    "44": {"name": "大井", "model": "models/model_ohi.pkl", "emoji": "🏟️"},
    "45": {"name": "川崎", "model": "models/model_kawasaki.pkl", "emoji": "🌊"},
    "43": {"name": "船橋", "model": "models/model_funabashi.pkl", "emoji": "⚓"},
    "42": {"name": "浦和", "model": "models/model_urawa.pkl", "emoji": "🌸"},
    "30": {"name": "門別", "model": "models/model_monbetsu.pkl", "emoji": "🐴"},
    "35": {"name": "盛岡", "model": "models/model_morioka.pkl", "emoji": "⛰️"},
    "36": {"name": "水沢", "model": "models/model_mizusawa.pkl", "emoji": "💧"},
    "46": {"name": "金沢", "model": "models/model_kanazawa.pkl", "emoji": "✨"},
    "47": {"name": "笠松", "model": "models/model_kasamatsu.pkl", "emoji": "🎋"},
    "48": {"name": "名古屋", "model": "models/model_nagoya.pkl", "emoji": "🏯"},
    "50": {"name": "園田", "model": "models/model_sonoda.pkl", "emoji": "🌳"},
    "51": {"name": "姫路", "model": "models/model_himeji.pkl", "emoji": "🏰"},
    "54": {"name": "高知", "model": "models/model_kochi.pkl", "emoji": "🐋"},
    "55": {"name": "佐賀", "model": "models/model_saga.pkl", "emoji": "🎋"},
}

# モデルキャッシュ
model_cache = {}

# 旧モデル名との互換性
MODEL_ALIASES = {
    "models/model_ohi.pkl": ["models/model_ohi.pkl", "model_v2.pkl"],
}


# ========== スクレイパー ==========
class NARScraper:
    BASE_URL = "https://nar.netkeiba.com"
    DB_URL = "https://db.netkeiba.com"

    def __init__(self, track_code, delay=0.5):
        self.track_code = track_code
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.horse_cache = {}
        self.jockey_cache = {}

    def _fetch(self, url, encoding='EUC-JP'):
        time.sleep(self.delay)
        r = self.session.get(url)
        r.encoding = encoding
        return BeautifulSoup(r.text, 'lxml')

    def get_race_list_by_date(self, date: str) -> list:
        url = f"{self.BASE_URL}/top/race_list_sub.html?kaisai_date={date}"
        try:
            soup = self._fetch(url, encoding='UTF-8')
            ids = []
            for a in soup.find_all('a', href=True):
                m = re.search(r'race_id=(\d+)', a['href'])
                if m:
                    race_id = m.group(1)
                    if len(race_id) >= 6 and race_id[4:6] == self.track_code:
                        ids.append(race_id)
            return list(set(ids))
        except:
            return []

    def get_race_data(self, race_id: str):
        url = f"{self.BASE_URL}/race/shutuba.html?race_id={race_id}"
        try:
            soup = self._fetch(url)
            info = {'race_id': race_id}

            # レース名
            nm = soup.find('h1', class_='RaceName')
            if nm:
                info['race_name'] = nm.get_text(strip=True)

            # 発走時刻
            rd = soup.find('div', class_='RaceData01')
            if rd:
                text = rd.get_text()
                tm = re.search(r'(\d{1,2}):(\d{2})', text)
                if tm:
                    info['start_time'] = f"{tm.group(1)}:{tm.group(2)}"
                dm = re.search(r'(\d{3,4})m', text)
                if dm:
                    info['distance'] = int(dm.group(1))

            # テーブル取得
            table = soup.find('table', class_='ShutubaTable')
            if not table:
                table = soup.find('table', class_='RaceTable01')
            if not table:
                for t in soup.find_all('table'):
                    if t.find('a', href=re.compile(r'/horse/')):
                        table = t
                        break
            if not table:
                return None

            rows = []
            for tr in table.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 4:
                    continue

                data = info.copy()

                bracket_text = tds[0].get_text(strip=True)
                if bracket_text.isdigit():
                    data['bracket'] = int(bracket_text)
                umaban_text = tds[1].get_text(strip=True)
                if umaban_text.isdigit():
                    data['horse_number'] = int(umaban_text)

                horse_link = tr.find('a', href=re.compile(r'/horse/\d+'))
                if horse_link:
                    data['horse_name'] = horse_link.get_text(strip=True)
                    m = re.search(r'/horse/(\d+)', horse_link['href'])
                    if m:
                        data['horse_id'] = m.group(1)

                jockey_link = tr.find('a', href=re.compile(r'/jockey/'))
                if jockey_link:
                    data['jockey_name'] = jockey_link.get_text(strip=True)
                    m = re.search(r'/jockey/(?:result/recent/)?([a-zA-Z0-9]+)', jockey_link['href'])
                    if m:
                        data['jockey_id'] = m.group(1)

                for td in tds:
                    text = td.get_text(strip=True)
                    if re.match(r'^[牡牝セ]\d$', text):
                        data['sex'] = text[0]
                        data['age'] = int(text[1])
                    if re.match(r'^\d{2}(\.\d)?$', text):
                        w = float(text)
                        if 45 <= w <= 65 and 'weight_carried' not in data:
                            data['weight_carried'] = w

                if data.get('horse_name'):
                    rows.append(data)

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df['field_size'] = len(df)
            return df
        except Exception as e:
            print(f'Error: {e}')

    def get_odds(self, race_id: str, horse_names: list = None) -> dict:
        """単勝オッズを取得（複数ソースから試行）"""
        odds_dict = {}

        # 1. オッズページから取得（開催中のレース用）
        url = f"{self.BASE_URL}/odds/odds_get_form.html?race_id={race_id}&type=b1"
        try:
            soup = self._fetch(url, encoding='UTF-8')

            # オッズページの馬名を取得して、正しいレースか確認
            page_horse_names = []
            for tr in soup.find_all('tr'):
                horse_link = tr.find('a', href=re.compile(r'/horse/'))
                if horse_link:
                    page_horse_names.append(horse_link.get_text(strip=True))

            # 馬名リストが渡されていて、ページの馬名と一致するか確認
            is_correct_race = True
            if horse_names and page_horse_names:
                # 少なくとも1頭が一致するか確認
                is_correct_race = any(name in page_horse_names for name in horse_names[:3])

            if is_correct_race and page_horse_names:
                # オッズテーブルから取得
                for tr in soup.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) >= 4:
                        try:
                            umaban_text = tds[1].get_text(strip=True)
                            if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                                umaban = int(umaban_text)
                                for td in reversed(tds):
                                    text = td.get_text(strip=True)
                                    odds_match = re.match(r'^(\d+\.?\d*)$', text)
                                    if odds_match:
                                        val = float(odds_match.group(1))
                                        if 1.0 <= val <= 500:
                                            odds_dict[umaban] = val
                                            break
                        except (ValueError, IndexError):
                            continue

                if odds_dict:
                    return odds_dict
        except Exception as e:
            print(f'Odds page error: {e}')

        # オッズページから取得できなかった場合、スマホ版結果ページから全馬のオッズを取得
        # スマホ版は結果テーブルに全馬の単勝オッズが含まれている
        sp_result_url = f"https://nar.sp.netkeiba.com/race/race_result.html?race_id={race_id}"
        try:
            soup = self._fetch(sp_result_url, encoding='UTF-8')

            # スマホ版の結果テーブルからオッズを取得
            # テーブル内の各行から馬番とオッズを抽出
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 8:
                    try:
                        # 馬番を探す（通常は最初の方のtd）
                        umaban = None
                        odds_val = None

                        for i, td in enumerate(tds):
                            text = td.get_text(strip=True)
                            # 馬番（1-18の数字、通常2桁以下）
                            if text.isdigit() and 1 <= int(text) <= 18 and umaban is None:
                                # 着順ではなく馬番かを確認（着順は1から始まる小さい数字）
                                # class属性やdata属性で判別できる場合もある
                                td_class = td.get('class', [])
                                if 'Umaban' in str(td_class) or i >= 1:
                                    umaban = int(text)

                        # オッズを探す（小数点を含む数字）
                        for td in tds:
                            text = td.get_text(strip=True)
                            # オッズパターン: "1.5" or "29.8" など
                            odds_match = re.match(r'^(\d+\.\d+)$', text)
                            if odds_match:
                                val = float(odds_match.group(1))
                                if 1.0 <= val <= 999.9:
                                    odds_val = val
                                    break

                        if umaban and odds_val:
                            odds_dict[umaban] = odds_val

                    except (ValueError, IndexError):
                        continue

            if odds_dict:
                return odds_dict

        except Exception as e:
            print(f'SP result page error: {e}')

        # 最後の手段: PC版結果ページの払戻金から勝ち馬のみ取得
        result_url = f"{self.BASE_URL}/race/result.html?race_id={race_id}"
        try:
            soup = self._fetch(result_url)
            payout_table = soup.find('table', class_='Payout_Detail_Table')
            if payout_table:
                for tr in payout_table.find_all('tr'):
                    th = tr.find('th')
                    if th and '単勝' in th.get_text():
                        tds = tr.find_all('td')
                        if len(tds) >= 2:
                            umaban_text = tds[0].get_text(strip=True)
                            payout_text = tds[1].get_text(strip=True)
                            if umaban_text.isdigit():
                                umaban = int(umaban_text)
                                payout_match = re.search(r'([\d,]+)', payout_text)
                                if payout_match:
                                    payout = int(payout_match.group(1).replace(',', ''))
                                    odds_dict[umaban] = payout / 100
            return odds_dict
        except Exception as e:
            print(f'Result page error: {e}')
            return {}

    def get_horse_history(self, horse_id: str):
        if horse_id in self.horse_cache:
            return self.horse_cache[horse_id]

        url = f"{self.DB_URL}/horse/ajax_horse_results.html?id={horse_id}"
        try:
            time.sleep(self.delay)
            r = self.session.get(url)
            r.encoding = 'EUC-JP'
            soup = BeautifulSoup(r.text, 'lxml')

            results = []
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 6:
                    continue
                for td in tds[3:7]:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 20:
                        results.append(int(t))
                        break
                if len(results) >= 20:
                    break

            stats = self._calc_stats(results)
            self.horse_cache[horse_id] = stats
            return stats
        except:
            return self._empty_stats()

    def get_jockey_stats(self, jockey_id: str):
        if jockey_id in self.jockey_cache:
            return self.jockey_cache[jockey_id]

        url = f"{self.DB_URL}/jockey/{jockey_id}/"
        try:
            soup = self._fetch(url)
            text = soup.get_text()
            stats = {'jockey_win_rate': 0, 'jockey_place_rate': 0, 'jockey_show_rate': 0}

            m = re.search(r'勝率[：:\s]*(\d+\.?\d*)', text)
            if m:
                stats['jockey_win_rate'] = float(m.group(1)) / 100
            m = re.search(r'連対率[：:\s]*(\d+\.?\d*)', text)
            if m:
                stats['jockey_place_rate'] = float(m.group(1)) / 100
            m = re.search(r'複勝率[：:\s]*(\d+\.?\d*)', text)
            if m:
                stats['jockey_show_rate'] = float(m.group(1)) / 100

            self.jockey_cache[jockey_id] = stats
            return stats
        except:
            return {'jockey_win_rate': 0, 'jockey_place_rate': 0, 'jockey_show_rate': 0}

    def _calc_stats(self, ranks):
        if not ranks:
            return self._empty_stats()
        total = len(ranks)
        wins = sum(1 for r in ranks if r == 1)
        place = sum(1 for r in ranks if r <= 2)
        show = sum(1 for r in ranks if r <= 3)
        recent = ranks[:5]
        r_total = len(recent)
        return {
            'horse_runs': total,
            'horse_win_rate': wins / total,
            'horse_place_rate': place / total,
            'horse_show_rate': show / total,
            'horse_avg_rank': np.mean(ranks),
            'horse_recent_win_rate': sum(1 for r in recent if r == 1) / r_total if r_total else 0,
            'horse_recent_show_rate': sum(1 for r in recent if r <= 3) / r_total if r_total else 0,
            'horse_recent_avg_rank': np.mean(recent) if recent else 10,
            'last_rank': ranks[0] if ranks else 10
        }

    def _empty_stats(self):
        return {
            'horse_runs': 0, 'horse_win_rate': 0, 'horse_place_rate': 0,
            'horse_show_rate': 0, 'horse_avg_rank': 10,
            'horse_recent_win_rate': 0, 'horse_recent_show_rate': 0,
            'horse_recent_avg_rank': 10, 'last_rank': 10
        }

    def enrich_data(self, df):
        df = df.copy()
        if 'horse_id' in df.columns:
            horse_data = []
            for hid in df['horse_id'].dropna().unique():
                stats = self.get_horse_history(str(hid))
                stats['horse_id'] = hid
                horse_data.append(stats)
            if horse_data:
                hdf = pd.DataFrame(horse_data)
                df['horse_id'] = df['horse_id'].astype(str)
                hdf['horse_id'] = hdf['horse_id'].astype(str)
                df = df.merge(hdf, on='horse_id', how='left')

        if 'jockey_id' in df.columns:
            jockey_data = []
            for jid in df['jockey_id'].dropna().unique():
                stats = self.get_jockey_stats(str(jid))
                stats['jockey_id'] = jid
                jockey_data.append(stats)
            if jockey_data:
                jdf = pd.DataFrame(jockey_data)
                df['jockey_id'] = df['jockey_id'].astype(str)
                jdf['jockey_id'] = jdf['jockey_id'].astype(str)
                df = df.merge(jdf, on='jockey_id', how='left')
        return df


# ========== 前処理 ==========
class Processor:
    def __init__(self):
        self.features = [
            'horse_runs', 'horse_win_rate', 'horse_place_rate', 'horse_show_rate',
            'horse_avg_rank', 'horse_recent_win_rate', 'horse_recent_show_rate',
            'horse_recent_avg_rank', 'last_rank',
            'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
            'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
            'sex_encoded', 'track_encoded', 'field_size', 'weight_diff'
        ]

    def process(self, df):
        df = df.copy()
        num_cols = ['bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'sex' in df.columns:
            df['sex_encoded'] = df['sex'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        else:
            df['sex_encoded'] = 0

        df['track_encoded'] = 0

        if 'weight_carried' in df.columns and 'race_id' in df.columns:
            df['weight_diff'] = df.groupby('race_id')['weight_carried'].transform(lambda x: x - x.mean())
        else:
            df['weight_diff'] = 0

        if 'field_size' not in df.columns:
            if 'race_id' in df.columns:
                df['field_size'] = df.groupby('race_id')['race_id'].transform('count')
            else:
                df['field_size'] = 12

        for f in self.features:
            if f not in df.columns:
                df[f] = 0
        return df

    def get_features(self):
        return self.features


# ========== モデル読み込み ==========
def load_model(track_code: str):
    if track_code in model_cache:
        return model_cache[track_code]

    if track_code not in TRACKS:
        return None, None

    model_name = TRACKS[track_code]['model']

    # エイリアスをチェック（旧モデル名との互換性）
    paths_to_try = [model_name]
    if model_name in MODEL_ALIASES:
        paths_to_try = MODEL_ALIASES[model_name]

    for model_name in paths_to_try:
        model_path = BASE_DIR / model_name
        if model_path.exists():
            with open(model_path, 'rb') as f:
                d = pickle.load(f)
            model_cache[track_code] = (d['model'], d['features'])
            return d['model'], d['features']
    return None, None


# ========== APIエンドポイント ==========

@app.get("/")
def root():
    return {"message": "地方競馬予測API", "version": "1.0.0"}


@app.get("/api/tracks")
def get_tracks():
    """利用可能な競馬場一覧を取得"""
    tracks = []
    for code, info in TRACKS.items():
        model_name = info['model']
        # エイリアスも含めてチェック
        paths_to_check = [model_name]
        if model_name in MODEL_ALIASES:
            paths_to_check = MODEL_ALIASES[model_name]
        model_exists = any((BASE_DIR / p).exists() for p in paths_to_check)
        tracks.append({
            "code": code,
            "name": info['name'],
            "emoji": info['emoji'],
            "model_available": model_exists
        })
    return {"tracks": tracks}


class PredictRequest(BaseModel):
    track_code: str
    date: str  # YYYY-MM-DD形式


class PredictionResult(BaseModel):
    rank: int
    number: int
    name: str
    jockey: str
    prob: float
    win_rate: float
    show_rate: float


class RaceResult(BaseModel):
    id: str
    name: str
    distance: int
    time: str
    predictions: list[PredictionResult]


@app.post("/api/predict")
def predict(request: PredictRequest):
    """予測を実行"""
    track_code = request.track_code
    date_str = request.date.replace("-", "")

    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    model, model_features = load_model(track_code)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"{TRACKS[track_code]['name']}のモデルがありません"
        )

    scraper = NARScraper(track_code, delay=0.3)
    processor = Processor()

    # レース一覧取得
    race_ids = scraper.get_race_list_by_date(date_str)
    if not race_ids:
        return {"races": [], "message": "レースが見つかりません"}

    results = []
    for rid in sorted(race_ids):
        df = scraper.get_race_data(rid)
        if df is None:
            continue

        df = scraper.enrich_data(df)
        df = processor.process(df)

        # オッズ取得（馬名リストを渡して正しいレースか確認）
        horse_names = df['horse_name'].tolist() if 'horse_name' in df.columns else []
        odds_dict = scraper.get_odds(rid, horse_names)

        # 予測
        X = df[model_features].fillna(-1)
        df['prob'] = model.predict(X)
        df['pred_rank'] = df['prob'].rank(ascending=False, method='min').astype(int)
        df = df.sort_values('prob', ascending=False)

        # レース番号抽出
        race_num = rid[-2:]
        race_name = df['race_name'].iloc[0] if 'race_name' in df.columns else f"{race_num}R"
        distance = int(df['distance'].iloc[0]) if 'distance' in df.columns else 0
        start_time = df['start_time'].iloc[0] if 'start_time' in df.columns else ""

        predictions = []
        for i, (_, row) in enumerate(df.iterrows()):  # 全馬を返す
            horse_num = int(row['horse_number']) if pd.notna(row.get('horse_number')) else 0
            odds = odds_dict.get(horse_num, 0)
            prob = float(row['prob'])

            # 勝率・複勝率を取得（0-1の範囲に正規化）
            raw_win_rate = float(row.get('horse_win_rate', 0))
            raw_show_rate = float(row.get('horse_show_rate', 0))

            # 値が1より大きい場合は既にパーセンテージなので100で割る
            if raw_win_rate > 1:
                raw_win_rate = raw_win_rate / 100
            if raw_show_rate > 1:
                raw_show_rate = raw_show_rate / 100

            # 0-100%の範囲にクランプ
            win_rate = min(max(raw_win_rate * 100, 0), 100)
            show_rate = min(max(raw_show_rate * 100, 0), 100)

            # 妙味計算: 予測確率 × オッズ > 1 なら妙味あり
            # 例: 予測30% × オッズ5.0 = 1.5 → 期待値プラス
            expected_value = prob * odds if odds > 0 else 0
            is_value = expected_value > 1.0  # 期待値1以上なら妙味あり

            predictions.append({
                "rank": i + 1,
                "number": horse_num,
                "name": row.get('horse_name', '不明'),
                "jockey": row.get('jockey_name', '不明'),
                "prob": round(prob, 3),
                "win_rate": round(win_rate, 1),
                "show_rate": round(show_rate, 1),
                "odds": odds,
                "expected_value": round(expected_value, 2),
                "is_value": is_value
            })

        results.append({
            "id": race_num,
            "name": race_name,
            "distance": distance,
            "time": start_time,
            "field_size": len(df),  # 出走頭数を追加
            "predictions": predictions
        })

    return {
        "track": {
            "code": track_code,
            "name": TRACKS[track_code]['name'],
            "emoji": TRACKS[track_code]['emoji']
        },
        "date": request.date,
        "races": results
    }


class RaceListRequest(BaseModel):
    track_code: str
    date: str


@app.post("/api/races")
def get_race_list(request: RaceListRequest):
    """レース一覧を取得（軽量）"""
    track_code = request.track_code
    date_str = request.date.replace("-", "")

    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    scraper = NARScraper(track_code, delay=0.3)
    race_ids = scraper.get_race_list_by_date(date_str)

    return {
        "track": TRACKS[track_code],
        "race_ids": sorted(race_ids)
    }


class SingleRaceRequest(BaseModel):
    race_id: str
    track_code: str


@app.post("/api/predict/race")
def predict_single_race(request: SingleRaceRequest):
    """単一レースの予測"""
    race_id = request.race_id
    track_code = request.track_code

    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    model, model_features = load_model(track_code)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"{TRACKS[track_code]['name']}のモデルがありません"
        )

    scraper = NARScraper(track_code, delay=0.3)
    processor = Processor()

    df = scraper.get_race_data(race_id)
    if df is None:
        raise HTTPException(status_code=404, detail="レースデータが取得できません")

    df = scraper.enrich_data(df)
    df = processor.process(df)

    # オッズ取得（馬名リストを渡して正しいレースか確認）
    horse_names = df['horse_name'].tolist() if 'horse_name' in df.columns else []
    odds_dict = scraper.get_odds(race_id, horse_names)

    # 予測
    X = df[model_features].fillna(-1)
    df['prob'] = model.predict(X)
    df['pred_rank'] = df['prob'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values('prob', ascending=False)

    # レース情報
    race_num = race_id[-2:]
    race_name = df['race_name'].iloc[0] if 'race_name' in df.columns else f"{race_num}R"
    distance = int(df['distance'].iloc[0]) if 'distance' in df.columns else 0
    start_time = df['start_time'].iloc[0] if 'start_time' in df.columns else ""

    predictions = []
    for i, (_, row) in enumerate(df.iterrows()):  # 全馬を返す
        horse_num = int(row['horse_number']) if pd.notna(row.get('horse_number')) else 0
        odds = odds_dict.get(horse_num, 0)
        prob = float(row['prob'])

        # 勝率・複勝率を取得（0-1の範囲に正規化）
        raw_win_rate = float(row.get('horse_win_rate', 0))
        raw_show_rate = float(row.get('horse_show_rate', 0))

        # 値が1より大きい場合は既にパーセンテージなので100で割る
        if raw_win_rate > 1:
            raw_win_rate = raw_win_rate / 100
        if raw_show_rate > 1:
            raw_show_rate = raw_show_rate / 100

        # 0-100%の範囲にクランプ
        win_rate = min(max(raw_win_rate * 100, 0), 100)
        show_rate = min(max(raw_show_rate * 100, 0), 100)

        expected_value = prob * odds if odds > 0 else 0
        is_value = expected_value > 1.0

        predictions.append({
            "rank": i + 1,
            "number": horse_num,
            "name": row.get('horse_name', '不明'),
            "jockey": row.get('jockey_name', '不明'),
            "prob": round(prob, 3),
            "win_rate": round(win_rate, 1),
            "show_rate": round(show_rate, 1),
            "odds": odds,
            "expected_value": round(expected_value, 2),
            "is_value": is_value
        })

    return {
        "id": race_num,
        "name": race_name,
        "distance": distance,
        "time": start_time,
        "field_size": len(df),
        "predictions": predictions
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
