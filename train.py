"""
地方競馬モデル学習スクリプト
GitHub Actionsから自動実行される
"""

import os
import sys
import time
import re
import pickle
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ========== 設定 ==========
TRACKS = {
    "大井": {"code": "44", "model": "model_ohi.pkl"},
    "川崎": {"code": "45", "model": "model_kawasaki.pkl"},
    "船橋": {"code": "43", "model": "model_funabashi.pkl"},
    "浦和": {"code": "42", "model": "model_urawa.pkl"},
    "門別": {"code": "30", "model": "model_monbetsu.pkl"},
    "盛岡": {"code": "35", "model": "model_morioka.pkl"},
    "水沢": {"code": "36", "model": "model_mizusawa.pkl"},
    "金沢": {"code": "46", "model": "model_kanazawa.pkl"},
    "笠松": {"code": "47", "model": "model_kasamatsu.pkl"},
    "名古屋": {"code": "48", "model": "model_nagoya.pkl"},
    "園田": {"code": "50", "model": "model_sonoda.pkl"},
    "姫路": {"code": "51", "model": "model_himeji.pkl"},
    "高知": {"code": "54", "model": "model_kochi.pkl"},
    "佐賀": {"code": "55", "model": "model_saga.pkl"},
}

DAYS = 180  # 取得日数
DELAY = 0.5  # リクエスト間隔


# ========== スクレイパー ==========
class NARScraper:
    BASE_URL = "https://nar.netkeiba.com"
    DB_URL = "https://db.netkeiba.com"

    def __init__(self, track_code, delay=1.0):
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

            nm = soup.find('h1', class_='RaceName')
            if nm:
                info['race_name'] = nm.get_text(strip=True)

            rd = soup.find('div', class_='RaceData01')
            if rd:
                dm = re.search(r'(\d{3,4})m', rd.get_text())
                if dm:
                    info['distance'] = int(dm.group(1))

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
            'sex_encoded', 'track_encoded', 'field_size', 'weight_diff'
        ]

    def process(self, df):
        df = df.copy()
        if 'rank' in df.columns:
            df = df[df['rank'].notna() & (df['rank'] > 0)]

        num_cols = ['rank','bracket','horse_number','age','weight_carried','distance',
                    'field_size','horse_runs','horse_win_rate','horse_place_rate',
                    'horse_show_rate','horse_avg_rank','horse_recent_win_rate',
                    'horse_recent_show_rate','horse_recent_avg_rank','last_rank',
                    'jockey_win_rate','jockey_place_rate','jockey_show_rate']
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


def save_model(model, features, path):
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'features': features}, f)


# ========== メイン処理 ==========
def train_track(track_name, track_info):
    print(f"\n{'='*50}")
    print(f"🏇 {track_name}競馬場")
    print(f"{'='*50}")

    scraper = NARScraper(track_info['code'], delay=DELAY)
    processor = Processor()

    # データ収集
    print(f"データ収集中（過去{DAYS}日）...")
    all_data = []
    today = datetime.now()

    for i in range(1, DAYS + 1):
        d = (today - timedelta(days=i)).strftime('%Y%m%d')
        race_ids = scraper.get_race_list_by_date(d)
        if not race_ids:
            continue

        for rid in race_ids:
            df = scraper.get_race_data(rid)
            if df is not None and len(df) > 0:
                try:
                    df = scraper.enrich_with_history(df)
                except:
                    pass
                all_data.append(df)

        if i % 30 == 0:
            print(f"  {i}/{DAYS}日完了 ({len(all_data)}レース)")

    if not all_data:
        print(f"⚠️ {track_name}: データが見つかりません")
        return False

    df_all = pd.concat(all_data, ignore_index=True)
    print(f"収集完了: {len(df_all)}件")

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

    # 保存
    save_model(model, features, track_info['model'])
    print(f"✅ 保存: {track_info['model']}")

    return True


def main():
    # コマンドライン引数から対象競馬場を取得
    target_tracks = []
    if len(sys.argv) > 1 and sys.argv[1]:
        target_tracks = [t.strip() for t in sys.argv[1].split(',')]
        target_tracks = [t for t in target_tracks if t in TRACKS]

    if not target_tracks:
        target_tracks = list(TRACKS.keys())

    print(f"🚀 学習開始: {', '.join(target_tracks)}")
    print(f"取得日数: {DAYS}日")

    results = {}
    for track_name in target_tracks:
        try:
            success = train_track(track_name, TRACKS[track_name])
            results[track_name] = "✅" if success else "⚠️"
        except Exception as e:
            print(f"❌ {track_name}: エラー - {e}")
            results[track_name] = "❌"

    # 結果サマリー
    print(f"\n{'='*50}")
    print("📊 結果サマリー")
    print(f"{'='*50}")
    for track, status in results.items():
        print(f"  {status} {track}")


if __name__ == "__main__":
    main()
