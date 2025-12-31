#!/usr/bin/env python3
"""
既存CSVの特徴量を更新するスクリプト
- 騎手成績（勝率、連対率、複勝率）を取得
- 追加特徴量を計算

使い方:
  python update_csv_features.py 大井
  python update_csv_features.py --all  # 全競馬場
"""
import sys
import time
import re
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# 競馬場設定
TRACKS = {
    "大井": {"code": "44", "data": "data/races_ohi.csv"},
    "川崎": {"code": "45", "data": "data/races_kawasaki.csv"},
    "船橋": {"code": "43", "data": "data/races_funabashi.csv"},
    "浦和": {"code": "42", "data": "data/races_urawa.csv"},
    "門別": {"code": "30", "data": "data/races_monbetsu.csv"},
    "盛岡": {"code": "35", "data": "data/races_morioka.csv"},
    "水沢": {"code": "36", "data": "data/races_mizusawa.csv"},
    "金沢": {"code": "46", "data": "data/races_kanazawa.csv"},
    "笠松": {"code": "47", "data": "data/races_kasamatsu.csv"},
    "名古屋": {"code": "48", "data": "data/races_nagoya.csv"},
    "園田": {"code": "50", "data": "data/races_sonoda.csv"},
    "姫路": {"code": "51", "data": "data/races_himeji.csv"},
    "高知": {"code": "54", "data": "data/races_kochi.csv"},
    "佐賀": {"code": "55", "data": "data/races_saga.csv"},
}


class JockeyScraper:
    """騎手成績を取得"""
    DB_URL = "https://db.netkeiba.com"

    def __init__(self, delay=0.3):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.session.verify = False  # SSL証明書検証をスキップ
        self.cache = {}
        # SSL警告を抑制
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def get_stats(self, jockey_id: str) -> dict:
        if jockey_id in self.cache:
            return self.cache[jockey_id]

        if pd.isna(jockey_id) or jockey_id == '':
            return {'win_rate': 0, 'place_rate': 0, 'show_rate': 0}

        url = f"{self.DB_URL}/jockey/{jockey_id}/"
        try:
            time.sleep(self.delay)
            r = self.session.get(url, timeout=30)
            r.encoding = 'EUC-JP'
            soup = BeautifulSoup(r.text, 'lxml')

            stats = {'win_rate': 0, 'place_rate': 0, 'show_rate': 0}

            # テーブルから成績を取得
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                if not rows:
                    continue

                headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]

                # 複勝率→連対率→勝率の順でチェック（誤マッチ防止）
                win_idx = place_idx = show_idx = -1
                for i, h in enumerate(headers):
                    if '複勝率' in h:
                        show_idx = i
                    elif '連対率' in h:
                        place_idx = i
                    elif '勝率' in h:
                        win_idx = i

                if win_idx >= 0:
                    for row in rows[1:3]:
                        cells = [c.get_text(strip=True) for c in row.find_all(['th', 'td'])]
                        if len(cells) > max(win_idx, place_idx, show_idx):
                            def parse_rate(text):
                                m = re.search(r'(\d+\.?\d*)[％%]', text)
                                return float(m.group(1)) / 100 if m else 0

                            if win_idx < len(cells):
                                stats['win_rate'] = parse_rate(cells[win_idx])
                            if place_idx < len(cells):
                                stats['place_rate'] = parse_rate(cells[place_idx])
                            if show_idx < len(cells):
                                stats['show_rate'] = parse_rate(cells[show_idx])

                            if stats['win_rate'] > 0:
                                break
                    if stats['win_rate'] > 0:
                        break

            self.cache[jockey_id] = stats
            return stats
        except Exception as e:
            print(f"    騎手 {jockey_id} 取得エラー: {e}")
            return {'win_rate': 0, 'place_rate': 0, 'show_rate': 0}


def add_calculated_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算で追加できる特徴量を追加"""
    df = df.copy()

    # 1. 馬番比率（馬番/出走頭数）- 内外の有利不利
    if 'horse_number' in df.columns and 'field_size' in df.columns:
        df['horse_number_ratio'] = df['horse_number'] / df['field_size']

    # 2. 距離カテゴリ（短距離/中距離/長距離）
    if 'distance' in df.columns:
        def categorize_distance(d):
            if pd.isna(d):
                return 1
            if d < 1400:
                return 0  # 短距離
            elif d < 1800:
                return 1  # 中距離
            else:
                return 2  # 長距離
        df['distance_category'] = df['distance'].apply(categorize_distance)

    # 3. 前走着順差（前走着順 - 平均着順）
    if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
        df['last_rank_diff'] = df['last_rank'] - df['horse_avg_rank']
        df['last_rank_diff'] = df['last_rank_diff'].fillna(0)

    # 4. レース内の勝率ランク
    if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
        df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False, method='min')
        df['win_rate_rank'] = df['win_rate_rank'].fillna(df['field_size'] / 2)

    # 5. 馬番位置（内/中/外）
    if 'horse_number' in df.columns and 'field_size' in df.columns:
        def get_position(row):
            if pd.isna(row['horse_number']) or pd.isna(row['field_size']):
                return 1
            ratio = row['horse_number'] / row['field_size']
            if ratio <= 0.33:
                return 0  # 内
            elif ratio <= 0.66:
                return 1  # 中
            else:
                return 2  # 外
        df['horse_position'] = df.apply(get_position, axis=1)

    return df


def update_csv(track_name: str, track_info: dict):
    """CSVを更新"""
    csv_path = track_info['data']

    if not Path(csv_path).exists():
        print(f"[ERROR] CSVが存在しません: {csv_path}")
        return False

    print(f"\n{'='*50}")
    print(f"[{track_name}競馬場] 特徴量更新")
    print(f"{'='*50}")

    # CSV読み込み
    df = pd.read_csv(csv_path)
    print(f"レコード数: {len(df)}")

    # バックアップ作成
    backup_path = csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(backup_path, index=False)
    print(f"バックアップ: {backup_path}")

    # === 騎手成績を更新 ===
    if 'jockey_id' in df.columns:
        print("\n騎手成績を取得中...")
        scraper = JockeyScraper(delay=0.2)

        # ユニークな騎手IDを取得
        unique_jockeys = df['jockey_id'].dropna().unique()
        print(f"騎手数: {len(unique_jockeys)}")

        # 騎手成績を取得
        jockey_stats = {}
        for i, jid in enumerate(unique_jockeys):
            if i % 50 == 0:
                print(f"  {i}/{len(unique_jockeys)} 完了...")
            stats = scraper.get_stats(str(jid))
            jockey_stats[str(jid)] = stats

        # CSVに反映
        df['jockey_id'] = df['jockey_id'].astype(str)
        df['jockey_win_rate'] = df['jockey_id'].map(lambda x: jockey_stats.get(x, {}).get('win_rate', 0))
        df['jockey_place_rate'] = df['jockey_id'].map(lambda x: jockey_stats.get(x, {}).get('place_rate', 0))
        df['jockey_show_rate'] = df['jockey_id'].map(lambda x: jockey_stats.get(x, {}).get('show_rate', 0))

        # 確認
        non_zero = (df['jockey_win_rate'] > 0).sum()
        print(f"騎手成績取得完了: {non_zero}/{len(df)} ({non_zero/len(df)*100:.1f}%) に値あり")

    # === 計算特徴量を追加 ===
    print("\n計算特徴量を追加中...")
    df = add_calculated_features(df)

    new_cols = ['horse_number_ratio', 'distance_category', 'last_rank_diff', 'win_rate_rank', 'horse_position']
    added = [c for c in new_cols if c in df.columns]
    print(f"追加した特徴量: {added}")

    # === 保存 ===
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] 保存完了: {csv_path}")

    return True


def main():
    if len(sys.argv) < 2:
        print("使い方: python update_csv_features.py <競馬場名>")
        print("        python update_csv_features.py --all")
        print(f"利用可能: {', '.join(TRACKS.keys())}")
        sys.exit(1)

    arg = sys.argv[1].strip()

    if arg == '--all':
        # 全競馬場を更新
        for track_name, track_info in TRACKS.items():
            if Path(track_info['data']).exists():
                update_csv(track_name, track_info)
    elif arg in TRACKS:
        update_csv(arg, TRACKS[arg])
    else:
        print(f"[ERROR] 不明な競馬場: {arg}")
        print(f"利用可能: {', '.join(TRACKS.keys())}")
        sys.exit(1)

    print("\n" + "="*50)
    print("[OK] 全ての更新が完了しました")
    print("次のステップ: python train.py <競馬場名> update")
    print("="*50)


if __name__ == "__main__":
    main()
