# 地方競馬 予測API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
import json
from pathlib import Path
from collections import defaultdict
import asyncio

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

# ========== 予測ログ保存 ==========
def save_prediction_log(race_id: str, track_code: str, predictions: list, metadata: dict = None):
    """予測結果をJSONに保存（後で結果と照合するため）"""
    try:
        # 日付を抽出（race_idから）
        date_str = race_id[:4] + "-" + race_id[6:8] + "-" + race_id[8:10]

        # ディレクトリ作成
        log_dir = BASE_DIR / "prediction_logs" / date_str
        log_dir.mkdir(parents=True, exist_ok=True)

        # ログデータ作成
        log_data = {
            "race_id": race_id,
            "track_code": track_code,
            "track_name": TRACKS.get(track_code, {}).get("name", "不明"),
            "predicted_at": datetime.now().isoformat(),
            "predictions": predictions,
            "metadata": metadata or {}
        }

        # レースごとにファイル保存
        log_file = log_dir / f"{race_id}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"Prediction log saved: {log_file}")
    except Exception as e:
        print(f"Failed to save prediction log: {e}")

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
                rd_text = rd.get_text()
                tm = re.search(r'(\d{1,2}):(\d{2})', rd_text)
                if tm:
                    info['start_time'] = f"{tm.group(1)}:{tm.group(2)}"
                dm = re.search(r'(\d{3,4})m', rd_text)
                if dm:
                    info['distance'] = int(dm.group(1))

                # 馬場状態を抽出（良/稍重/重/不良）
                track_cond_match = re.search(r'[ダ芝].*?[:：]\s*(良|稍重|重|不良)', rd_text)
                if track_cond_match:
                    info['track_condition'] = track_cond_match.group(1)
                else:
                    info['track_condition'] = '良'

                # 天気を抽出（晴/曇/雨/小雨/雪）
                weather_match = re.search(r'天気[:：]\s*(晴|曇|雨|小雨|雪)', rd_text)
                if weather_match:
                    info['weather'] = weather_match.group(1)
                else:
                    info['weather'] = '晴'

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

                # 調教師を抽出
                trainer_link = tr.find('a', href=re.compile(r'/trainer/'))
                if trainer_link:
                    data['trainer_name'] = trainer_link.get_text(strip=True)
                    m = re.search(r'/trainer/(?:result/recent/)?([a-zA-Z0-9]+)', trainer_link['href'])
                    if m:
                        data['trainer_id'] = m.group(1)

                # 馬体重を抽出（例: 450(+4), 448(-2), 452）
                for td in tds:
                    weight_text = td.get_text(strip=True)
                    weight_match = re.match(r'^(\d{3,4})(?:\(([+-]?\d+)\))?$', weight_text)
                    if weight_match and 300 <= int(weight_match.group(1)) <= 600:
                        data['horse_weight'] = int(weight_match.group(1))
                        if weight_match.group(2):
                            data['weight_change'] = int(weight_match.group(2))
                        else:
                            data['weight_change'] = 0
                        break

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

    def get_all_odds(self, race_id: str) -> dict:
        """単勝・複勝オッズを一括取得（API呼び出し最小化）"""
        result = {'win': {}, 'place': {}}

        # 1. 出馬表ページから単勝オッズを取得（1リクエスト目）
        shutuba_url = f"{self.BASE_URL}/race/shutuba.html?race_id={race_id}"
        try:
            soup = self._fetch(shutuba_url)
            table = soup.find('table', class_='ShutubaTable')
            if not table:
                table = soup.find('table', class_='RaceTable01')

            if table:
                for tr in table.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) >= 2:
                        umaban = None
                        odds_val = None

                        for i, td in enumerate(tds[:3]):
                            td_class = ' '.join(td.get('class', []))
                            text = td.get_text(strip=True)
                            if 'Umaban' in td_class or (i == 1 and text.isdigit()):
                                if text.isdigit() and 1 <= int(text) <= 18:
                                    umaban = int(text)
                                    break

                        for td in tds:
                            td_class = ' '.join(td.get('class', []))
                            if 'Popular' in td_class or 'Odds' in td_class or 'odds' in td_class.lower():
                                text = td.get_text(strip=True)
                                odds_match = re.search(r'(\d+\.?\d*)', text)
                                if odds_match:
                                    val = float(odds_match.group(1))
                                    if 1.0 <= val <= 999.9:
                                        odds_val = val
                                        break

                        if umaban and odds_val:
                            result['win'][umaban] = odds_val
        except Exception as e:
            print(f'Win odds error: {e}')

        # 2. 複勝オッズページから取得（2リクエスト目）
        place_url = f"{self.BASE_URL}/odds/odds_get_form.html?type=b2&race_id={race_id}"
        try:
            soup = self._fetch(place_url)
            tables = soup.find_all('table')
            if len(tables) >= 2:
                table = tables[1]
                for tr in table.find_all('tr'):
                    tds = tr.find_all('td')
                    # td構造: [枠番, 馬番, 空, 馬名, オッズ]
                    if len(tds) >= 5:
                        umaban_text = tds[1].get_text(strip=True)  # td[1]が馬番
                        if umaban_text.isdigit():
                            umaban = int(umaban_text)
                            # オッズは最後のtd
                            odds_text = tds[-1].get_text(strip=True)
                            # 「1.4 - 2.6」形式をパース
                            odds_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', odds_text)
                            if odds_match:
                                min_odds = float(odds_match.group(1))
                                max_odds = float(odds_match.group(2))
                                result['place'][umaban] = {
                                    'min': min_odds,
                                    'max': max_odds,
                                    'avg': round((min_odds + max_odds) / 2, 2)
                                }
                            else:
                                # 単一の数値の場合
                                single_match = re.search(r'(\d+\.?\d*)', odds_text)
                                if single_match:
                                    odds_val = float(single_match.group(1))
                                    result['place'][umaban] = {
                                        'min': odds_val,
                                        'max': odds_val,
                                        'avg': odds_val
                                    }
        except Exception as e:
            print(f'Place odds error: {e}')

        return result

    def get_odds(self, race_id: str, horse_names: list = None) -> dict:
        """単勝オッズを取得（出馬表の予想オッズ列から）"""
        odds_dict = {}

        # 1. 出馬表ページから予想オッズを取得（最も確実）
        shutuba_url = f"{self.BASE_URL}/race/shutuba.html?race_id={race_id}"
        try:
            soup = self._fetch(shutuba_url)
            table = soup.find('table', class_='ShutubaTable')
            if not table:
                table = soup.find('table', class_='RaceTable01')

            if table:
                for tr in table.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) >= 2:
                        # 馬番は通常2番目のtd（1番目は枠番）
                        umaban = None
                        odds_val = None

                        # 馬番を取得（Umabanクラスまたは2番目のtd）
                        for i, td in enumerate(tds[:3]):
                            td_class = ' '.join(td.get('class', []))
                            text = td.get_text(strip=True)
                            if 'Umaban' in td_class or (i == 1 and text.isdigit()):
                                if text.isdigit() and 1 <= int(text) <= 18:
                                    umaban = int(text)
                                    break

                        # 予想オッズを取得（Popular列、通常は後ろの方のtd）
                        for td in tds:
                            td_class = ' '.join(td.get('class', []))
                            # Popularクラスまたはodds関連のクラスを持つセル
                            if 'Popular' in td_class or 'Odds' in td_class or 'odds' in td_class.lower():
                                text = td.get_text(strip=True)
                                odds_match = re.search(r'(\d+\.?\d*)', text)
                                if odds_match:
                                    val = float(odds_match.group(1))
                                    if 1.0 <= val <= 999.9:
                                        odds_val = val
                                        break

                        if umaban and odds_val:
                            odds_dict[umaban] = odds_val

            if odds_dict:
                print(f"DEBUG shutuba odds: {odds_dict}")
                return odds_dict
        except Exception as e:
            print(f'Shutuba odds error: {e}')

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
            'sex_encoded', 'track_encoded', 'field_size', 'weight_diff',
            # 新特徴量
            'track_condition_encoded', 'weather_encoded',
            'trainer_encoded', 'horse_weight', 'horse_weight_change'
        ]

    def process(self, df):
        df = df.copy()
        num_cols = ['bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
                    'horse_weight', 'weight_change']
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

        # 馬場状態エンコーディング（良=0, 稍重=1, 重=2, 不良=3）
        if 'track_condition' in df.columns:
            df['track_condition_encoded'] = df['track_condition'].map(
                {'良': 0, '稍重': 1, '重': 2, '不良': 3}
            ).fillna(0)
        else:
            df['track_condition_encoded'] = 0

        # 天気エンコーディング（晴=0, 曇=1, 小雨=2, 雨=3, 雪=4）
        if 'weather' in df.columns:
            df['weather_encoded'] = df['weather'].map(
                {'晴': 0, '曇': 1, '小雨': 2, '雨': 3, '雪': 4}
            ).fillna(0)
        else:
            df['weather_encoded'] = 0

        # 調教師エンコーディング（ハッシュベース）
        if 'trainer_id' in df.columns:
            df['trainer_encoded'] = df['trainer_id'].apply(
                lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0
            )
        else:
            df['trainer_encoded'] = 0

        # 馬体重（欠損は450kgで補完）
        if 'horse_weight' in df.columns:
            df['horse_weight'] = df['horse_weight'].fillna(450)
        else:
            df['horse_weight'] = 450

        # 馬体重増減
        if 'weight_change' in df.columns:
            df['horse_weight_change'] = df['weight_change'].fillna(0)
        else:
            df['horse_weight_change'] = 0

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

        # オッズ取得（単勝・複勝を一括取得）
        all_odds = scraper.get_all_odds(rid)
        win_odds_dict = all_odds.get('win', {})
        place_odds_dict = all_odds.get('place', {})

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
            win_odds = win_odds_dict.get(horse_num, 0)
            place_odds_data = place_odds_dict.get(horse_num, {})
            place_odds = place_odds_data.get('avg', 0) if place_odds_data else 0
            place_odds_min = place_odds_data.get('min', 0) if place_odds_data else 0
            place_odds_max = place_odds_data.get('max', 0) if place_odds_data else 0
            prob = float(row['prob'])

            # 勝率・複勝率を取得（0-1の範囲であるべき）
            raw_win_rate = float(row.get('horse_win_rate') or 0)
            raw_show_rate = float(row.get('horse_show_rate') or 0)

            win_rate = raw_win_rate * 100
            show_rate = raw_show_rate * 100

            # 期待値計算（複勝オッズ × AI確率）
            # 複勝オッズがあればそれを使用、なければ単勝/3で推定
            effective_place_odds = place_odds if place_odds > 0 else (win_odds / 3 if win_odds > 0 else 0)
            expected_value = prob * effective_place_odds if effective_place_odds > 0 else 0

            # 妙味判定: 期待値 > 1.0 なら黒字期待
            is_value = expected_value > 1.0

            predictions.append({
                "rank": i + 1,
                "number": horse_num,
                "name": row.get('horse_name', '不明'),
                "jockey": row.get('jockey_name', '不明'),
                "prob": round(prob, 3),
                "win_rate": round(win_rate, 1),
                "show_rate": round(show_rate, 1),
                "odds": win_odds,
                "place_odds": place_odds,
                "place_odds_min": place_odds_min,
                "place_odds_max": place_odds_max,
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

    # オッズ取得（単勝・複勝を一括取得）
    all_odds = scraper.get_all_odds(race_id)
    win_odds_dict = all_odds.get('win', {})
    place_odds_dict = all_odds.get('place', {})

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
        win_odds = win_odds_dict.get(horse_num, 0)
        place_odds_data = place_odds_dict.get(horse_num, {})
        place_odds = place_odds_data.get('avg', 0) if place_odds_data else 0
        place_odds_min = place_odds_data.get('min', 0) if place_odds_data else 0
        place_odds_max = place_odds_data.get('max', 0) if place_odds_data else 0
        prob = float(row['prob'])

        # 勝率・複勝率を取得（0-1の範囲であるべき）
        raw_win_rate = float(row.get('horse_win_rate') or 0)
        raw_show_rate = float(row.get('horse_show_rate') or 0)

        win_rate = raw_win_rate * 100
        show_rate = raw_show_rate * 100

        # 期待値計算（複勝オッズ × AI確率）
        effective_place_odds = place_odds if place_odds > 0 else (win_odds / 3 if win_odds > 0 else 0)
        expected_value = prob * effective_place_odds if effective_place_odds > 0 else 0
        is_value = expected_value > 1.0

        predictions.append({
            "rank": i + 1,
            "number": horse_num,
            "name": row.get('horse_name', '不明'),
            "jockey": row.get('jockey_name', '不明'),
            "prob": round(prob, 3),
            "win_rate": round(win_rate, 1),
            "show_rate": round(show_rate, 1),
            "odds": win_odds,
            "place_odds": place_odds,
            "place_odds_min": place_odds_min,
            "place_odds_max": place_odds_max,
            "expected_value": round(expected_value, 2),
            "is_value": is_value
        })

    # 予測ログを保存（誤答分析用）
    metadata = {
        "race_name": race_name,
        "distance": distance,
        "track_condition": df['track_condition'].iloc[0] if 'track_condition' in df.columns else "不明",
        "weather": df['weather'].iloc[0] if 'weather' in df.columns else "不明",
        "field_size": len(df)
    }
    save_prediction_log(race_id, track_code, predictions, metadata)

    return {
        "id": race_num,
        "name": race_name,
        "distance": distance,
        "time": start_time,
        "field_size": len(df),
        "predictions": predictions
    }


# ========== 軽量オッズ取得API ==========

class OddsRequest(BaseModel):
    race_id: str
    track_code: str


def get_race_result(race_id: str) -> list:
    """レース結果（着順）を取得"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        time.sleep(0.2)
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        r = session.get(url, timeout=10)
        r.encoding = 'EUC-JP'
        soup = BeautifulSoup(r.text, 'lxml')

        results = []
        table = soup.find('table', class_='RaceTable01')
        if not table:
            table = soup.find('table', class_='Result_Table')
        if not table:
            return []

        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 3:
                continue

            rank_text = tds[0].get_text(strip=True)
            if not rank_text.isdigit():
                continue
            rank = int(rank_text)

            # 馬番を取得（tds[2]が馬番、tds[1]は枠番）
            horse_num = None
            if len(tds) >= 3:
                umaban_text = tds[2].get_text(strip=True)
                if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                    horse_num = int(umaban_text)

            if horse_num:
                results.append({"rank": rank, "number": horse_num})

        return sorted(results, key=lambda x: x["rank"])[:3]  # TOP3のみ
    except:
        return []


@app.post("/api/odds")
def get_odds_only(request: OddsRequest):
    """オッズと結果を取得（レース終了時は結果も含む）"""
    race_id = request.race_id
    track_code = request.track_code

    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    scraper = NARScraper(track_code, delay=0.2)
    odds_dict = scraper.get_odds(race_id)

    # 結果も取得（終了していれば返る、まだなら空）
    result = get_race_result(race_id)

    return {
        "race_id": race_id,
        "odds": odds_dict,
        "result": result if result else None  # 終了していなければnull
    }


# ========== 事前計算済み予測取得API ==========

@app.get("/api/predictions/{date}/{track_code}")
def get_precomputed_predictions(date: str, track_code: str):
    """事前計算済みの予測JSONを取得"""
    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    predictions_file = BASE_DIR / "predictions" / date / f"{track_code}.json"

    if not predictions_file.exists():
        raise HTTPException(status_code=404, detail="予測データがありません")

    import json
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


@app.get("/api/predictions/{date}")
def list_available_predictions(date: str):
    """指定日の利用可能な予測一覧"""
    predictions_dir = BASE_DIR / "predictions" / date

    if not predictions_dir.exists():
        return {"date": date, "tracks": []}

    available = []
    for f in predictions_dir.glob("*.json"):
        track_code = f.stem
        if track_code in TRACKS:
            available.append({
                "code": track_code,
                "name": TRACKS[track_code]['name'],
                "emoji": TRACKS[track_code]['emoji']
            })

    return {"date": date, "tracks": available}


# ========== 精度評価API ==========

@app.get("/api/accuracy/{date}/{track_code}")
def get_accuracy(date: str, track_code: str):
    """指定日・競馬場の精度データを取得"""
    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    accuracy_file = BASE_DIR / "accuracy" / date / f"{track_code}.json"

    if not accuracy_file.exists():
        raise HTTPException(status_code=404, detail="精度データがありません")

    import json
    with open(accuracy_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


@app.get("/api/accuracy/{date}")
def get_daily_accuracy(date: str):
    """指定日の全体精度サマリーを取得"""
    summary_file = BASE_DIR / "accuracy" / date / "summary.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="精度データがありません")

    import json
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


@app.get("/api/accuracy")
def get_accuracy_history():
    """過去の精度データ一覧を取得"""
    accuracy_dir = BASE_DIR / "accuracy"

    if not accuracy_dir.exists():
        return {"dates": []}

    dates = []
    for d in sorted(accuracy_dir.iterdir(), reverse=True):
        if d.is_dir():
            summary_file = d / "summary.json"
            if summary_file.exists():
                import json
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                dates.append(summary)

    return {"history": dates[:30]}  # 直近30日分


# ========== モデル情報API ==========

@app.get("/api/models/{track_code}")
def get_model_info(track_code: str):
    """モデルのメタデータを取得"""
    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    model_path = TRACKS[track_code]['model']
    meta_path = model_path.replace('.pkl', '_meta.json')
    meta_file = BASE_DIR / meta_path

    if meta_file.exists():
        import json
        with open(meta_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # メタデータJSONがない場合、pklから読み込み試行
    model_file = BASE_DIR / model_path
    if model_file.exists():
        try:
            with open(model_file, 'rb') as f:
                data = pickle.load(f)
            if 'metadata' in data:
                return data['metadata']
        except:
            pass

    raise HTTPException(status_code=404, detail="モデル情報がありません")


@app.get("/api/models")
def get_all_models_info():
    """全モデルの情報を取得"""
    models_info = []

    for code, info in TRACKS.items():
        model_path = info['model']
        meta_path = model_path.replace('.pkl', '_meta.json')
        meta_file = BASE_DIR / meta_path
        model_file = BASE_DIR / model_path

        model_data = {
            "code": code,
            "name": info['name'],
            "emoji": info['emoji'],
            "model_exists": model_file.exists(),
            "metadata": None
        }

        if meta_file.exists():
            try:
                import json
                with open(meta_file, 'r', encoding='utf-8') as f:
                    model_data["metadata"] = json.load(f)
            except:
                pass

        models_info.append(model_data)

    return {"models": models_info}


# ========== 誤答分析API（SSE対応） ==========

def analyze_get_race_result(race_id: str) -> list:
    """レース結果（着順）を取得（分析用）"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        r = session.get(url, timeout=10)
        r.encoding = 'EUC-JP'
        soup = BeautifulSoup(r.text, 'lxml')

        results = []
        table = soup.find('table', class_='RaceTable01')
        if not table:
            table = soup.find('table', class_='Result_Table')
        if not table:
            return []

        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 3:
                continue

            rank_text = tds[0].get_text(strip=True)
            if not rank_text.isdigit():
                continue
            rank = int(rank_text)

            umaban_text = tds[2].get_text(strip=True)
            if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                horse_num = int(umaban_text)
                results.append({"rank": rank, "number": horse_num})

        return sorted(results, key=lambda x: x["rank"])
    except Exception as e:
        print(f"Error fetching result for {race_id}: {e}")
        return []


def compare_prediction(prediction_log: dict, result: list) -> dict:
    """予測と結果を照合"""
    if not result:
        return None

    predictions = prediction_log["predictions"]
    metadata = prediction_log.get("metadata", {})

    pred_top3 = [p["number"] for p in predictions[:3]]
    pred_1st = predictions[0]["number"] if predictions else None
    actual_top3 = [r["number"] for r in result[:3]]
    actual_1st = result[0]["number"] if result else None

    win_hit = (pred_1st == actual_1st)
    show_hit = (pred_1st in actual_top3)

    # 予測1位の馬が実際に何着だったか
    pred_1st_actual_rank = None
    for r in result:
        if r["number"] == pred_1st:
            pred_1st_actual_rank = r["rank"]
            break

    # エラータイプ分類
    error_type = None
    if not show_hit:
        if pred_1st_actual_rank is None:
            error_type = "出走取消"
        elif pred_1st_actual_rank >= 10:
            error_type = "大外れ(10着以下)"
        elif pred_1st_actual_rank >= 6:
            error_type = "中外れ(6-9着)"
        elif pred_1st_actual_rank >= 4:
            error_type = "惜しい(4-5着)"

    return {
        "race_id": prediction_log["race_id"],
        "track_name": prediction_log.get("track_name", "不明"),
        "race_name": metadata.get("race_name", "不明"),
        "pred_1st": pred_1st,
        "actual_1st": actual_1st,
        "win_hit": win_hit,
        "show_hit": show_hit,
        "pred_1st_actual_rank": pred_1st_actual_rank,
        "error_type": error_type,
        "metadata": metadata
    }


async def analyze_stream(date: str):
    """分析をストリーミングで実行"""
    log_dir = BASE_DIR / "prediction_logs" / date

    if not log_dir.exists():
        yield f"data: {json.dumps({'type': 'error', 'message': '予測ログがありません'})}\n\n"
        return

    log_files = list(log_dir.glob("*.json"))
    total = len(log_files)

    if total == 0:
        yield f"data: {json.dumps({'type': 'error', 'message': '予測ログがありません'})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

    comparisons = []
    for i, log_file in enumerate(log_files):
        with open(log_file, 'r', encoding='utf-8') as f:
            prediction_log = json.load(f)

        race_id = prediction_log["race_id"]

        # 進捗を送信
        yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total, 'race_id': race_id})}\n\n"

        # 結果を取得
        result = analyze_get_race_result(race_id)
        if result:
            comparison = compare_prediction(prediction_log, result)
            if comparison:
                comparisons.append(comparison)

        # 少し待機（サーバー負荷軽減）
        await asyncio.sleep(0.3)

    # 集計
    if not comparisons:
        yield f"data: {json.dumps({'type': 'error', 'message': '照合できるデータがありません'})}\n\n"
        return

    # 統計を計算
    total_races = len(comparisons)
    win_hits = sum(1 for c in comparisons if c["win_hit"])
    show_hits = sum(1 for c in comparisons if c["show_hit"])

    # 馬場状態別
    by_track_condition = defaultdict(lambda: {"total": 0, "show_hits": 0})
    # 天気別
    by_weather = defaultdict(lambda: {"total": 0, "show_hits": 0})
    # 距離別
    by_distance = defaultdict(lambda: {"total": 0, "show_hits": 0})
    # エラータイプ
    error_types = defaultdict(int)

    for c in comparisons:
        meta = c.get("metadata", {})

        # 馬場状態
        track_cond = meta.get("track_condition", "不明")
        by_track_condition[track_cond]["total"] += 1
        if c["show_hit"]:
            by_track_condition[track_cond]["show_hits"] += 1

        # 天気
        weather = meta.get("weather", "不明")
        by_weather[weather]["total"] += 1
        if c["show_hit"]:
            by_weather[weather]["show_hits"] += 1

        # 距離
        distance = meta.get("distance", 0)
        if distance < 1400:
            dist_cat = "短距離(<1400m)"
        elif distance < 1800:
            dist_cat = "中距離(1400-1800m)"
        else:
            dist_cat = "長距離(>1800m)"
        by_distance[dist_cat]["total"] += 1
        if c["show_hit"]:
            by_distance[dist_cat]["show_hits"] += 1

        # エラータイプ
        if c.get("error_type"):
            error_types[c["error_type"]] += 1

    # 結果を送信
    result_data = {
        "type": "result",
        "date": date,
        "summary": {
            "total_races": total_races,
            "win_hits": win_hits,
            "win_rate": round(win_hits / total_races * 100, 1) if total_races > 0 else 0,
            "show_hits": show_hits,
            "show_rate": round(show_hits / total_races * 100, 1) if total_races > 0 else 0
        },
        "by_track_condition": {k: v for k, v in by_track_condition.items()},
        "by_weather": {k: v for k, v in by_weather.items()},
        "by_distance": {k: v for k, v in by_distance.items()},
        "error_types": dict(error_types),
        "details": comparisons
    }

    yield f"data: {json.dumps(result_data, ensure_ascii=False)}\n\n"

    # 結果をファイルに保存
    output_dir = BASE_DIR / "analysis_reports" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    yield f"data: {json.dumps({'type': 'complete', 'saved_to': str(output_file)})}\n\n"


@app.get("/api/analyze/{date}")
def analyze_predictions(date: str):
    """予測の誤答分析を実行（通常API版）"""
    log_dir = BASE_DIR / "prediction_logs" / date

    if not log_dir.exists():
        raise HTTPException(status_code=404, detail="予測ログがありません")

    log_files = list(log_dir.glob("*.json"))
    total = len(log_files)

    if total == 0:
        raise HTTPException(status_code=404, detail="予測ログがありません")

    comparisons = []
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            prediction_log = json.load(f)

        race_id = prediction_log["race_id"]

        # 結果を取得
        result = analyze_get_race_result(race_id)
        if result:
            comparison = compare_prediction(prediction_log, result)
            if comparison:
                comparisons.append(comparison)

        # サーバー負荷軽減
        time.sleep(0.3)

    if not comparisons:
        raise HTTPException(status_code=404, detail="照合できるデータがありません")

    # 統計を計算
    total_races = len(comparisons)
    win_hits = sum(1 for c in comparisons if c["win_hit"])
    show_hits = sum(1 for c in comparisons if c["show_hit"])

    # 馬場状態別
    by_track_condition = defaultdict(lambda: {"total": 0, "show_hits": 0})
    by_weather = defaultdict(lambda: {"total": 0, "show_hits": 0})
    by_distance = defaultdict(lambda: {"total": 0, "show_hits": 0})
    error_types = defaultdict(int)

    for c in comparisons:
        meta = c.get("metadata", {})

        track_cond = meta.get("track_condition", "不明")
        by_track_condition[track_cond]["total"] += 1
        if c["show_hit"]:
            by_track_condition[track_cond]["show_hits"] += 1

        weather = meta.get("weather", "不明")
        by_weather[weather]["total"] += 1
        if c["show_hit"]:
            by_weather[weather]["show_hits"] += 1

        distance = meta.get("distance", 0)
        if distance < 1400:
            dist_cat = "短距離(<1400m)"
        elif distance < 1800:
            dist_cat = "中距離(1400-1800m)"
        else:
            dist_cat = "長距離(>1800m)"
        by_distance[dist_cat]["total"] += 1
        if c["show_hit"]:
            by_distance[dist_cat]["show_hits"] += 1

        if c.get("error_type"):
            error_types[c["error_type"]] += 1

    result_data = {
        "date": date,
        "summary": {
            "total_races": total_races,
            "win_hits": win_hits,
            "win_rate": round(win_hits / total_races * 100, 1) if total_races > 0 else 0,
            "show_hits": show_hits,
            "show_rate": round(show_hits / total_races * 100, 1) if total_races > 0 else 0
        },
        "by_track_condition": {k: v for k, v in by_track_condition.items()},
        "by_weather": {k: v for k, v in by_weather.items()},
        "by_distance": {k: v for k, v in by_distance.items()},
        "error_types": dict(error_types),
        "details": comparisons
    }

    # 結果をファイルに保存
    output_dir = BASE_DIR / "analysis_reports" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return result_data


@app.get("/api/analysis/{date}")
def get_analysis_report(date: str):
    """保存済みの分析レポートを取得"""
    report_file = BASE_DIR / "analysis_reports" / date / "report.json"

    if not report_file.exists():
        raise HTTPException(status_code=404, detail="分析レポートがありません")

    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
