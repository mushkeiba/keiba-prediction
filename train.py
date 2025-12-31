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
        self.horse_cache = {}
        self.jockey_cache = {}

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

    def enrich_with_history(self, df):
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
            # 環境特徴量
            'track_condition_encoded', 'weather_encoded',
            'trainer_encoded', 'horse_weight', 'horse_weight_change',
            # 計算特徴量
            'horse_number_ratio', 'distance_category', 'last_rank_diff',
            'win_rate_rank', 'horse_position'
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

        # 馬番位置（内/中/外）
        if 'horse_number' in df.columns and 'field_size' in df.columns:
            df['horse_position'] = df.apply(
                lambda row: 0 if pd.notna(row.get('horse_number')) and pd.notna(row.get('field_size')) and row['horse_number'] / row['field_size'] <= 0.33
                else (2 if pd.notna(row.get('horse_number')) and pd.notna(row.get('field_size')) and row['horse_number'] / row['field_size'] > 0.66 else 1),
                axis=1
            )
        else:
            df['horse_position'] = 1

        if 'rank' in df.columns:
            df['target'] = (df['rank'] <= 3).astype(int)

        for f in self.features:
            if f not in df.columns:
                df[f] = 0

        return df


# ========== 学習 ==========
def train_model(df, features):
    X, y = df[features].fillna(-1), df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = lgb.train(
        {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.05,'verbose':-1},
        lgb.Dataset(X_tr, y_tr), 500, [lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    auc = roc_auc_score(y_te, model.predict(X_te))
    return model, features, auc


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
