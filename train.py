"""
地方競馬モデル学習スクリプト
GitHub Actionsから自動実行される

使い方:
  python train.py <競馬場名> <モード>

  モード:
    init   - 初回モデル作成（3年分取得）
    update - モデル再学習（差分のみ取得）

  例:
    python train.py 大井 init    # 初回: 3年分取得してモデル作成
    python train.py 大井 update  # 再学習: 差分のみ取得して再学習
"""

import os
import sys
import time
import re
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# ========== 設定 ==========
TRACKS = {
    "大井": {"code": "44", "model": "models/model_ohi.pkl", "data": "data/races_ohi.csv"},
    "川崎": {"code": "45", "model": "models/model_kawasaki.pkl", "data": "data/races_kawasaki.csv"},
    "船橋": {"code": "43", "model": "models/model_funabashi.pkl", "data": "data/races_funabashi.csv"},
    "浦和": {"code": "42", "model": "models/model_urawa.pkl", "data": "data/races_urawa.csv"},
    "門別": {"code": "30", "model": "models/model_monbetsu.pkl", "data": "data/races_monbetsu.csv"},
    "盛岡": {"code": "35", "model": "models/model_morioka.pkl", "data": "data/races_morioka.csv"},
    "水沢": {"code": "36", "model": "models/model_mizusawa.pkl", "data": "data/races_mizusawa.csv"},
    "金沢": {"code": "46", "model": "models/model_kanazawa.pkl", "data": "data/races_kanazawa.csv"},
    "笠松": {"code": "47", "model": "models/model_kasamatsu.pkl", "data": "data/races_kasamatsu.csv"},
    "名古屋": {"code": "48", "model": "models/model_nagoya.pkl", "data": "data/races_nagoya.csv"},
    "園田": {"code": "50", "model": "models/model_sonoda.pkl", "data": "data/races_sonoda.csv"},
    "姫路": {"code": "51", "model": "models/model_himeji.pkl", "data": "data/races_himeji.csv"},
    "高知": {"code": "54", "model": "models/model_kochi.pkl", "data": "data/races_kochi.csv"},
    "佐賀": {"code": "55", "model": "models/model_saga.pkl", "data": "data/races_saga.csv"},
}

INIT_DAYS = 1095  # 初回取得日数（3年分）
DELAY = 0.5  # リクエスト間隔


# ========== スクレイパー ==========
class NARScraper:
    BASE_URL = "https://nar.netkeiba.com"
    DB_URL = "https://db.netkeiba.com"

    def __init__(self, track_code, delay=0.5):
        self.track_code = track_code
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.session.verify = False  # SSL証明書検証をスキップ
        self.horse_cache = {}
        self.jockey_cache = {}
        self.pedigree_cache = {}
        # SSL警告を抑制
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _fetch(self, url, encoding='EUC-JP'):
        time.sleep(self.delay)
        r = self.session.get(url, timeout=30)
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
        except Exception as e:
            print(f"  レース一覧取得エラー: {e}")
            return []

    def get_race_data(self, race_id: str):
        url = f"{self.BASE_URL}/race/result.html?race_id={race_id}"
        try:
            soup = self._fetch(url)
            info = {'race_id': race_id}

            # 日付を抽出（race_idから）
            # race_id形式: YYYYJJMMDDNN (年+競馬場+月日+レース番号)
            # 例: 202544123001 → 2025年12月30日、大井(44)、01R
            if len(race_id) >= 10:
                info['race_date'] = race_id[:4] + race_id[6:10]  # YYYY + MMDD

            nm = soup.find('h1', class_='RaceName')
            if nm:
                info['race_name'] = nm.get_text(strip=True)

            rd = soup.find('div', class_='RaceData01')
            if rd:
                rd_text = rd.get_text()
                dm = re.search(r'(\d{3,4})m', rd_text)
                if dm:
                    info['distance'] = int(dm.group(1))

                # 馬場状態を抽出（良/稍重/重/不良）
                track_cond_match = re.search(r'[ダ芝].*?[:：]\s*(良|稍重|重|不良)', rd_text)
                if track_cond_match:
                    info['track_condition'] = track_cond_match.group(1)
                else:
                    info['track_condition'] = '良'  # デフォルト

                # 天気を抽出（晴/曇/雨/小雨/雪）
                weather_match = re.search(r'天気[:：]\s*(晴|曇|雨|小雨|雪)', rd_text)
                if weather_match:
                    info['weather'] = weather_match.group(1)
                else:
                    info['weather'] = '晴'  # デフォルト

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

                # 着順・枠番・馬番
                rank_text = tds[0].get_text(strip=True)
                if rank_text.isdigit():
                    data['rank'] = int(rank_text)
                bracket_text = tds[1].get_text(strip=True)
                if bracket_text.isdigit():
                    data['bracket'] = int(bracket_text)
                umaban_text = tds[2].get_text(strip=True)
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
            print(f"  レースデータ取得エラー: {e}")
            return None

    def get_horse_history(self, horse_id: str):
        if horse_id in self.horse_cache:
            return self.horse_cache[horse_id]

        url = f"{self.DB_URL}/horse/ajax_horse_results.html?id={horse_id}"
        try:
            time.sleep(self.delay)
            r = self.session.get(url, timeout=30)
            r.encoding = 'EUC-JP'
            soup = BeautifulSoup(r.text, 'lxml')

            results = []
            race_dates = []  # 前走日を取得
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 6:
                    continue

                # 日付を取得（最初の列）
                date_text = tds[0].get_text(strip=True)
                if re.match(r'\d{4}[./]\d{1,2}[./]\d{1,2}', date_text):
                    race_dates.append(date_text.replace('.', '/'))

                for td in tds[3:7]:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 20:
                        results.append(int(t))
                        break
                if len(results) >= 20:
                    break

            stats = self._calc_stats(results)
            # 最新のレース日を追加
            if race_dates:
                stats['last_race_date'] = race_dates[0]
            else:
                stats['last_race_date'] = None
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
            stats = {'jockey_win_rate': 0, 'jockey_place_rate': 0, 'jockey_show_rate': 0}

            # テーブルから成績を取得（累計行を探す）
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                # ヘッダー行で「勝率」列を探す
                header_row = rows[0] if rows else None
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                    # 勝率、連対率、複勝率の列インデックスを探す
                    # 注意: 「複勝率」には「勝率」が含まれるので、先に複勝率をチェック
                    win_idx = place_idx = show_idx = -1
                    for i, h in enumerate(headers):
                        if '複勝率' in h:
                            show_idx = i
                        elif '連対率' in h:
                            place_idx = i
                        elif '勝率' in h:  # 複勝率でない勝率
                            win_idx = i

                    if win_idx >= 0:
                        # 累計行（2行目）からデータを取得
                        for row in rows[1:3]:
                            cells = row.find_all(['th', 'td'])
                            cell_texts = [c.get_text(strip=True) for c in cells]
                            # 累計行かどうか確認（最初のセルが「累計」「通算」など、または年度が数字でない）
                            if len(cell_texts) > max(win_idx, place_idx, show_idx):
                                # 全角・半角両方のパーセント記号に対応
                                def parse_rate(text):
                                    # 「2.3％」「2.3%」→ 0.023
                                    m = re.search(r'(\d+\.?\d*)[％%]', text)
                                    return float(m.group(1)) / 100 if m else 0

                                if win_idx >= 0 and win_idx < len(cell_texts):
                                    stats['jockey_win_rate'] = parse_rate(cell_texts[win_idx])
                                if place_idx >= 0 and place_idx < len(cell_texts):
                                    stats['jockey_place_rate'] = parse_rate(cell_texts[place_idx])
                                if show_idx >= 0 and show_idx < len(cell_texts):
                                    stats['jockey_show_rate'] = parse_rate(cell_texts[show_idx])

                                if stats['jockey_win_rate'] > 0:
                                    break
                        if stats['jockey_win_rate'] > 0:
                            break

            self.jockey_cache[jockey_id] = stats
            return stats
        except Exception as e:
            print(f"  騎手成績取得エラー ({jockey_id}): {e}")
            return {'jockey_win_rate': 0, 'jockey_place_rate': 0, 'jockey_show_rate': 0}

    def get_pedigree_stats(self, horse_id: str):
        """馬の血統（父馬・母父馬）の成績を取得"""
        if horse_id in self.pedigree_cache:
            return self.pedigree_cache[horse_id]

        url = f"{self.DB_URL}/horse/{horse_id}/"
        try:
            soup = self._fetch(url)
            pedigree = {
                'father_win_rate': 0, 'father_show_rate': 0,
                'bms_win_rate': 0, 'bms_show_rate': 0
            }

            # 血統テーブルを探す
            pedigree_table = soup.find('table', class_='blood_table')
            if not pedigree_table:
                # 別のクラス名を試す
                for table in soup.find_all('table'):
                    if '血統' in str(table):
                        pedigree_table = table
                        break

            if pedigree_table:
                # 父馬（最初のリンク）
                links = pedigree_table.find_all('a', href=re.compile(r'/horse/ped/'))
                if links:
                    # 父馬の産駒成績ページから勝率を取得
                    father_link = links[0]
                    father_id_match = re.search(r'/horse/ped/(\w+)', father_link['href'])
                    if father_id_match:
                        father_stats = self._get_sire_stats(father_id_match.group(1))
                        pedigree['father_win_rate'] = father_stats.get('win_rate', 0)
                        pedigree['father_show_rate'] = father_stats.get('show_rate', 0)

                    # 母父馬（4番目のリンクが通常母父）
                    if len(links) > 3:
                        bms_link = links[3]
                        bms_id_match = re.search(r'/horse/ped/(\w+)', bms_link['href'])
                        if bms_id_match:
                            bms_stats = self._get_sire_stats(bms_id_match.group(1))
                            pedigree['bms_win_rate'] = bms_stats.get('win_rate', 0)
                            pedigree['bms_show_rate'] = bms_stats.get('show_rate', 0)

            self.pedigree_cache[horse_id] = pedigree
            return pedigree
        except Exception as e:
            return {'father_win_rate': 0, 'father_show_rate': 0, 'bms_win_rate': 0, 'bms_show_rate': 0}

    def _get_sire_stats(self, sire_id: str):
        """種牡馬の産駒成績を取得"""
        url = f"{self.DB_URL}/horse/sire/{sire_id}/"
        try:
            soup = self._fetch(url)
            stats = {'win_rate': 0, 'show_rate': 0}

            # 産駒成績テーブルを探す
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                if not rows:
                    continue

                headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]

                # 勝率、複勝率の列を探す
                win_idx = show_idx = -1
                for i, h in enumerate(headers):
                    if '複勝率' in h:
                        show_idx = i
                    elif '勝率' in h:
                        win_idx = i

                if win_idx >= 0:
                    for row in rows[1:3]:
                        cells = [c.get_text(strip=True) for c in row.find_all(['th', 'td'])]
                        if len(cells) > max(win_idx, show_idx if show_idx >= 0 else win_idx):
                            def parse_rate(text):
                                m = re.search(r'(\d+\.?\d*)[％%]', text)
                                return float(m.group(1)) / 100 if m else 0

                            if win_idx < len(cells):
                                stats['win_rate'] = parse_rate(cells[win_idx])
                            if show_idx >= 0 and show_idx < len(cells):
                                stats['show_rate'] = parse_rate(cells[show_idx])

                            if stats['win_rate'] > 0:
                                break
                    if stats['win_rate'] > 0:
                        break

            return stats
        except:
            return {'win_rate': 0, 'show_rate': 0}

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
            'horse_recent_avg_rank': 10, 'last_rank': 10,
            'last_race_date': None
        }

    def enrich_with_history(self, df, include_pedigree=False):
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

            # 血統データを取得
            if include_pedigree:
                pedigree_data = []
                for hid in df['horse_id'].dropna().unique():
                    pstats = self.get_pedigree_stats(str(hid))
                    pstats['horse_id'] = hid
                    pedigree_data.append(pstats)
                if pedigree_data:
                    pdf = pd.DataFrame(pedigree_data)
                    pdf['horse_id'] = pdf['horse_id'].astype(str)
                    df = df.merge(pdf, on='horse_id', how='left')

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
            # 環境特徴量
            'track_condition_encoded', 'weather_encoded',
            'trainer_encoded', 'horse_weight', 'horse_weight_change',
            # 計算特徴量
            'horse_number_ratio', 'last_rank_diff', 'win_rate_rank',
            # 相対特徴量（レース内での相対的な強さ）
            'horse_win_rate_vs_field', 'jockey_win_rate_vs_field',
            'horse_avg_rank_vs_field',
            # 休養・調子
            'days_since_last_race', 'rank_trend',
            # === 新規追加: 交互作用特徴量 ===
            'jockey_track_interaction',    # 騎手×競馬場の相性
            'trainer_distance_interaction', # 調教師×距離の相性
            'jockey_distance_interaction',  # 騎手×距離の相性
            # === 新規追加: 時系列強化 ===
            'win_streak',                  # 連勝数
            'show_streak',                 # 複勝連続数
            'recent_3_avg_rank',           # 直近3走平均着順
            'recent_10_avg_rank',          # 直近10走平均着順
            'rank_improvement',            # 着順改善トレンド
            # === 新規追加: 血統特徴量 ===
            'father_win_rate',             # 父馬の勝率
            'father_show_rate',            # 父馬の複勝率
            'bms_win_rate',                # 母父馬の勝率
            'bms_show_rate',               # 母父馬の複勝率
        ]

    def process(self, df):
        df = df.copy()
        if 'rank' in df.columns:
            df = df[df['rank'].notna() & (df['rank'] > 0)]

        num_cols = ['rank','bracket','horse_number','age','weight_carried','distance',
                    'field_size','horse_runs','horse_win_rate','horse_place_rate',
                    'horse_show_rate','horse_avg_rank','horse_recent_win_rate',
                    'horse_recent_show_rate','horse_recent_avg_rank','last_rank',
                    'jockey_win_rate','jockey_place_rate','jockey_show_rate',
                    'horse_weight', 'weight_change']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'sex' in df.columns:
            df['sex_encoded'] = df['sex'].map({'牡':0,'牝':1,'セ':2}).fillna(0)
        else:
            df['sex_encoded'] = 0

        df['track_encoded'] = 0

        if 'weight_carried' in df.columns and 'race_id' in df.columns:
            df['weight_diff'] = df.groupby('race_id')['weight_carried'].transform(lambda x: x - x.mean())
        else:
            df['weight_diff'] = 0

        if 'field_size' not in df.columns:
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

        # === 計算特徴量 ===
        # 馬番比率（馬番/出走頭数）
        if 'horse_number' in df.columns and 'field_size' in df.columns:
            df['horse_number_ratio'] = df['horse_number'] / df['field_size']
            df['horse_number_ratio'] = df['horse_number_ratio'].fillna(0.5)

        # 距離カテゴリ（短距離/中距離/長距離）
        if 'distance' in df.columns:
            df['distance_category'] = df['distance'].apply(
                lambda d: 0 if pd.notna(d) and d < 1400 else (2 if pd.notna(d) and d >= 1800 else 1)
            )
        else:
            df['distance_category'] = 1

        # 前走着順差（前走着順 - 平均着順）
        if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['last_rank_diff'] = df['last_rank'] - df['horse_avg_rank']
            df['last_rank_diff'] = df['last_rank_diff'].fillna(0)
        else:
            df['last_rank_diff'] = 0

        # レース内の勝率ランク
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False, method='min')
            df['win_rate_rank'] = df['win_rate_rank'].fillna(df['field_size'] / 2)
        else:
            df['win_rate_rank'] = 6

        # === 相対特徴量（レース内での相対的な強さ）===
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['field_avg_win_rate'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
            df['horse_win_rate_vs_field'] = df['horse_win_rate'] - df['field_avg_win_rate']
            df['horse_win_rate_vs_field'] = df['horse_win_rate_vs_field'].fillna(0)
        else:
            df['horse_win_rate_vs_field'] = 0

        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['field_avg_jockey_win_rate'] = df.groupby('race_id')['jockey_win_rate'].transform('mean')
            df['jockey_win_rate_vs_field'] = df['jockey_win_rate'] - df['field_avg_jockey_win_rate']
            df['jockey_win_rate_vs_field'] = df['jockey_win_rate_vs_field'].fillna(0)
        else:
            df['jockey_win_rate_vs_field'] = 0

        if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
            df['field_avg_rank'] = df.groupby('race_id')['horse_avg_rank'].transform('mean')
            df['horse_avg_rank_vs_field'] = df['field_avg_rank'] - df['horse_avg_rank']  # 低い方が良いので逆
            df['horse_avg_rank_vs_field'] = df['horse_avg_rank_vs_field'].fillna(0)
        else:
            df['horse_avg_rank_vs_field'] = 0

        # === 休養日数 ===
        # CSVにdays_since_last_raceがあればそれを使用
        if 'days_since_last_race' in df.columns:
            df['days_since_last_race'] = df['days_since_last_race'].fillna(30).clip(0, 365)
        elif 'last_race_date' in df.columns and 'race_date' in df.columns:
            df['days_since_last_race'] = (pd.to_datetime(df['race_date']) - pd.to_datetime(df['last_race_date'])).dt.days
            df['days_since_last_race'] = df['days_since_last_race'].fillna(30).clip(0, 365)
        else:
            df['days_since_last_race'] = 30  # デフォルト30日

        # === 着順トレンド（上り調子か下り調子か）===
        # last_rank と horse_avg_rank の差で代用
        if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
            # 前走が平均より良ければプラス（上り調子）
            df['rank_trend'] = df['horse_avg_rank'] - df['last_rank']
            df['rank_trend'] = df['rank_trend'].fillna(0)
        else:
            df['rank_trend'] = 0

        # === 交互作用特徴量 ===
        # 騎手×競馬場の相性（ハッシュベース）
        if 'jockey_id' in df.columns and 'race_id' in df.columns:
            # 競馬場コード（race_idの5-6桁目）
            df['track_code'] = df['race_id'].astype(str).str[4:6]
            df['jockey_track_interaction'] = df.apply(
                lambda x: hash(str(x.get('jockey_id', '')) + str(x.get('track_code', ''))) % 10000, axis=1
            )
        else:
            df['jockey_track_interaction'] = 0

        # 調教師×距離の相性
        if 'trainer_id' in df.columns and 'distance' in df.columns:
            # 距離カテゴリ（短/中/長）
            df['distance_cat'] = df['distance'].apply(
                lambda d: 'short' if pd.notna(d) and d < 1400 else ('long' if pd.notna(d) and d >= 1800 else 'mid')
            )
            df['trainer_distance_interaction'] = df.apply(
                lambda x: hash(str(x.get('trainer_id', '')) + str(x.get('distance_cat', ''))) % 10000, axis=1
            )
        else:
            df['trainer_distance_interaction'] = 0

        # 騎手×距離の相性
        if 'jockey_id' in df.columns and 'distance' in df.columns:
            df['jockey_distance_interaction'] = df.apply(
                lambda x: hash(str(x.get('jockey_id', '')) + str(x.get('distance_cat', ''))) % 10000, axis=1
            )
        else:
            df['jockey_distance_interaction'] = 0

        # === 時系列強化特徴量 ===
        # 連勝数（CSVにあれば使用、なければ0）
        if 'win_streak' not in df.columns:
            df['win_streak'] = 0
        if 'show_streak' not in df.columns:
            df['show_streak'] = 0

        # 直近3走・10走平均着順（CSVにあれば使用）
        if 'recent_3_avg_rank' not in df.columns:
            if 'horse_recent_avg_rank' in df.columns:
                df['recent_3_avg_rank'] = df['horse_recent_avg_rank']  # 5走平均で代用
            else:
                df['recent_3_avg_rank'] = 10
        if 'recent_10_avg_rank' not in df.columns:
            if 'horse_avg_rank' in df.columns:
                df['recent_10_avg_rank'] = df['horse_avg_rank']  # 全走平均で代用
            else:
                df['recent_10_avg_rank'] = 10

        # 着順改善トレンド（直近3走と全体平均の差）
        if 'recent_3_avg_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['rank_improvement'] = df['horse_avg_rank'] - df['recent_3_avg_rank']
            df['rank_improvement'] = df['rank_improvement'].fillna(0)
        else:
            df['rank_improvement'] = 0

        # === 血統特徴量 ===
        # CSVにあれば使用、なければ0
        for col in ['father_win_rate', 'father_show_rate', 'bms_win_rate', 'bms_show_rate']:
            if col not in df.columns:
                df[col] = 0

        if 'rank' in df.columns:
            df['target'] = (df['rank'] <= 3).astype(int)

        for f in self.features:
            if f not in df.columns:
                df[f] = 0

        return df


# ========== 学習 ==========
def train_model(df, features, use_smote=True, use_ensemble=True):
    X, y = df[features].fillna(-1), df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # === SMOTEでオーバーサンプリング ===
    if use_smote:
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr, y_tr)
            print(f"  SMOTE適用: {len(y_tr)} -> {len(y_tr_resampled)}件")
        except Exception as e:
            print(f"  SMOTE失敗（スキップ）: {e}")
            X_tr_resampled, y_tr_resampled = X_tr, y_tr
    else:
        X_tr_resampled, y_tr_resampled = X_tr, y_tr

    # === LightGBM ===
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'num_leaves': 112,
        'learning_rate': 0.067,
        'min_child_samples': 65,
        'reg_alpha': 6.8e-07,
        'reg_lambda': 0.025,
        'feature_fraction': 0.78,
        'bagging_fraction': 0.78,
        'bagging_freq': 3
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb.Dataset(X_tr_resampled, y_tr_resampled), 500, [lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    lgb_pred = lgb_model.predict(X_te)
    lgb_auc = roc_auc_score(y_te, lgb_pred)
    print(f"  LightGBM AUC: {lgb_auc:.4f}")

    # === XGBoost ===
    if use_ensemble:
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }
        xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=500, early_stopping_rounds=50)
        xgb_model.fit(
            X_tr_resampled, y_tr_resampled,
            eval_set=[(X_te, y_te)],
            verbose=False
        )
        xgb_pred = xgb_model.predict_proba(X_te)[:, 1]
        xgb_auc = roc_auc_score(y_te, xgb_pred)
        print(f"  XGBoost AUC: {xgb_auc:.4f}")

        # === アンサンブル（平均） ===
        ensemble_pred = (lgb_pred + xgb_pred) / 2
        ensemble_auc = roc_auc_score(y_te, ensemble_pred)
        print(f"  Ensemble AUC: {ensemble_auc:.4f}")

        # 最もAUCが高いモデルを返す
        if ensemble_auc >= max(lgb_auc, xgb_auc):
            print("  -> Ensemble採用")
            return {'lgb': lgb_model, 'xgb': xgb_model, 'type': 'ensemble'}, features, ensemble_auc
        elif xgb_auc > lgb_auc:
            print("  -> XGBoost採用")
            return {'xgb': xgb_model, 'type': 'xgb'}, features, xgb_auc
        else:
            print("  -> LightGBM採用")
            return lgb_model, features, lgb_auc
    else:
        return lgb_model, features, lgb_auc


def save_model(model, features, path, metadata=None):
    """モデルとメタデータを保存"""
    data = {
        'model': model,
        'features': features,
        'metadata': metadata or {}
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)

    # メタデータをJSONでも保存（API用）
    if metadata:
        import json
        meta_path = path.replace('.pkl', '_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


# ========== データ収集 ==========
def collect_data(scraper, start_date, end_date, existing_race_ids=None):
    """指定期間のデータを収集"""
    if existing_race_ids is None:
        existing_race_ids = set()

    all_data = []
    current = start_date
    total_days = (end_date - start_date).days
    processed_days = 0

    while current <= end_date:
        date_str = current.strftime('%Y%m%d')
        race_ids = scraper.get_race_list_by_date(date_str)

        # 新しいレースのみ処理
        new_race_ids = [rid for rid in race_ids if rid not in existing_race_ids]

        for rid in new_race_ids:
            df = scraper.get_race_data(rid)
            if df is not None and len(df) > 0:
                try:
                    df = scraper.enrich_with_history(df)
                except:
                    pass
                all_data.append(df)

        processed_days += 1
        if processed_days % 30 == 0:
            print(f"  {processed_days}/{total_days}日完了 ({len(all_data)}レース)")

        current += timedelta(days=1)

    return all_data


def get_latest_date_from_csv(csv_path):
    """CSVから最新の日付を取得"""
    if not Path(csv_path).exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        if 'race_date' in df.columns and len(df) > 0:
            latest = df['race_date'].max()
            return datetime.strptime(str(int(latest)), '%Y%m%d')
    except Exception as e:
        print(f"  CSV読み込みエラー: {e}")
    return None


def get_existing_race_ids(csv_path):
    """CSVから既存のレースIDを取得"""
    if not Path(csv_path).exists():
        return set()

    try:
        df = pd.read_csv(csv_path)
        if 'race_id' in df.columns:
            return set(df['race_id'].unique())
    except:
        pass
    return set()


# ========== メイン処理 ==========
def train_track(track_name, track_info, mode='init'):
    """
    競馬場のモデルを学習

    mode:
        'init'   - 初回モデル作成（3年分取得）
        'update' - モデル再学習（差分のみ取得）
    """
    print(f"\n{'='*50}")
    print(f"[{track_name}競馬場] {'初回モデル作成' if mode == 'init' else 'モデル再学習'}")
    print(f"{'='*50}")

    scraper = NARScraper(track_info['code'], delay=DELAY)
    processor = Processor()

    csv_path = track_info['data']
    model_path = track_info['model']

    today = datetime.now()
    yesterday = today - timedelta(days=1)  # 昨日まで（今日は結果未確定）

    if mode == 'init':
        # 初回: 365日分取得
        start_date = today - timedelta(days=INIT_DAYS)
        existing_race_ids = set()
        print(f"データ収集中（過去{INIT_DAYS}日）...")
    else:
        # 再学習: 差分のみ取得
        latest_date = get_latest_date_from_csv(csv_path)
        if latest_date is None:
            print(f"[WARN] CSVが存在しません。initモードで実行してください。")
            return False

        start_date = latest_date + timedelta(days=1)
        if start_date > yesterday:
            print(f"[OK] データは最新です（最終: {latest_date.strftime('%Y-%m-%d')}）")
            # データは最新だが、モデルは再学習する
            start_date = None

        existing_race_ids = get_existing_race_ids(csv_path)
        if start_date:
            print(f"差分データ収集中（{start_date.strftime('%Y-%m-%d')} 〜 {yesterday.strftime('%Y-%m-%d')}）...")

    # データ収集
    if mode == 'init' or (mode == 'update' and start_date):
        new_data = collect_data(scraper, start_date, yesterday, existing_race_ids)

        if new_data:
            df_new = pd.concat(new_data, ignore_index=True)
            print(f"新規データ: {len(df_new)}件")

            # CSVに保存/追記
            if mode == 'init' or not Path(csv_path).exists():
                df_new.to_csv(csv_path, index=False)
                print(f"[OK] CSV保存: {csv_path}")
            else:
                df_new.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"[OK] CSV追記: {csv_path}")
        else:
            if mode == 'init':
                print(f"[WARN] {track_name}: データが見つかりません")
                return False
            print(f"新規データなし")

    # CSVからデータ読み込み
    if not Path(csv_path).exists():
        print(f"⚠️ CSVが存在しません")
        return False

    df_all = pd.read_csv(csv_path)
    print(f"総データ: {len(df_all)}件")

    # 前処理
    df_processed = processor.process(df_all)
    print(f"処理後: {len(df_processed)}件")

    if len(df_processed) < 100:
        print(f"⚠️ {track_name}: データ不足（{len(df_processed)}件）")
        return False

    # 学習
    print("モデル学習中...")
    model, features, auc = train_model(df_processed, processor.features)
    print(f"AUC: {auc:.4f}")

    # メタデータ作成
    race_dates = df_all['race_date'].dropna().astype(int).astype(str)
    min_date = race_dates.min() if len(race_dates) > 0 else ""
    max_date = race_dates.max() if len(race_dates) > 0 else ""

    metadata = {
        "track_name": track_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_count": len(df_all),
        "race_count": df_all['race_id'].nunique() if 'race_id' in df_all.columns else 0,
        "date_range": {
            "from": f"{min_date[:4]}-{min_date[4:6]}-{min_date[6:8]}" if len(min_date) == 8 else "",
            "to": f"{max_date[:4]}-{max_date[4:6]}-{max_date[6:8]}" if len(max_date) == 8 else ""
        },
        "auc": round(auc, 4),
        "features": features
    }

    # 保存
    save_model(model, features, model_path, metadata)
    print(f"[OK] モデル保存: {model_path}")

    return True


def main():
    # コマンドライン引数
    if len(sys.argv) < 2:
        print("使い方: python train.py <競馬場名> [モード]")
        print("モード: init (初回) / update (再学習)")
        print("例: python train.py 大井 init")
        sys.exit(1)

    track_name = sys.argv[1].strip()
    mode = sys.argv[2].strip() if len(sys.argv) > 2 else 'update'

    if track_name not in TRACKS:
        print(f"[ERROR] 不明な競馬場: {track_name}")
        print(f"利用可能: {', '.join(TRACKS.keys())}")
        sys.exit(1)

    if mode not in ['init', 'update']:
        print(f"[ERROR] 不明なモード: {mode}")
        print("利用可能: init / update")
        sys.exit(1)

    print(f"[START] 学習開始: {track_name} ({mode}モード)")

    try:
        success = train_track(track_name, TRACKS[track_name], mode)
        if success:
            print(f"\n[OK] 完了: {track_name}")
        else:
            print(f"\n[WARN] 失敗: {track_name}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
