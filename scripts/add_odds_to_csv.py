#!/usr/bin/env python3
"""
過去レースのオッズをCSVに追加

使い方:
  python scripts/add_odds_to_csv.py --track 44 --start 2025-12-20 --end 2025-12-30
"""

import argparse
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent

TRACKS = {
    "44": {"name": "大井", "data": "data/races_ohi.csv"},
    "45": {"name": "川崎", "data": "data/races_kawasaki.csv"},
}

# User-Agent ローテーション
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]


def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    return session


def get_race_odds(session, race_id: str) -> dict:
    """レースの確定オッズを取得"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"

    try:
        time.sleep(random.uniform(0.5, 1.5))  # ランダム遅延
        response = session.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # 結果テーブルを探す
        result_table = soup.find('table', class_='RaceTable01')
        if not result_table:
            return {}

        odds_dict = {}
        rows = result_table.find_all('tr')

        for row in rows[1:]:  # ヘッダーをスキップ
            cols = row.find_all('td')
            if len(cols) < 10:
                continue

            try:
                # 馬番
                horse_num_elem = row.find('td', class_='Num')
                if not horse_num_elem:
                    continue
                horse_num = int(horse_num_elem.text.strip())

                # 単勝オッズ（通常は最後から2番目のカラム）
                # クラスで探す
                win_odds_elem = row.find('td', class_='Odds')
                if win_odds_elem:
                    win_odds_text = win_odds_elem.text.strip()
                    win_odds = float(win_odds_text) if win_odds_text and win_odds_text != '---' else 0
                else:
                    win_odds = 0

                odds_dict[horse_num] = {
                    'win_odds': win_odds,
                    'place_odds': win_odds / 3 if win_odds > 0 else 0  # 複勝は推定
                }

            except (ValueError, AttributeError):
                continue

        return odds_dict

    except Exception as e:
        print(f"  [WARN] {race_id}: {e}")
        return {}


def add_odds_to_csv(track_code: str, start_date: str, end_date: str):
    """CSVにオッズを追加"""
    track_info = TRACKS[track_code]
    csv_path = BASE_DIR / track_info["data"]

    print(f"\n{'='*60}")
    print(f"オッズ追加: {track_info['name']}競馬場")
    print(f"期間: {start_date} 〜 {end_date}")
    print(f"{'='*60}\n")

    # CSV読み込み
    df = pd.read_csv(csv_path)
    print(f"総レコード: {len(df)}件")

    # 日付カラム作成
    race_id_str = df["race_id"].astype(str)
    df["date"] = pd.to_datetime(race_id_str.str[:4] + race_id_str.str[6:10], format="%Y%m%d")

    # 対象期間のレースを抽出
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    target_races = df[mask]["race_id"].unique()
    print(f"対象レース: {len(target_races)}件")

    # オッズカラムがなければ追加
    if "win_odds" not in df.columns:
        df["win_odds"] = 0.0
    if "place_odds" not in df.columns:
        df["place_odds"] = 0.0

    # セッション作成
    session = create_session()

    # オッズ取得
    success_count = 0
    for i, race_id in enumerate(target_races):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(target_races)} 完了...")
            # セッション更新（User-Agent変更）
            session = create_session()

        odds = get_race_odds(session, str(race_id))

        if odds:
            for horse_num, odds_data in odds.items():
                mask = (df["race_id"] == race_id) & (df["horse_number"] == horse_num)
                df.loc[mask, "win_odds"] = odds_data["win_odds"]
                df.loc[mask, "place_odds"] = odds_data["place_odds"]
            success_count += 1

    print(f"\n取得成功: {success_count}/{len(target_races)} レース")

    # 一時カラム削除
    df = df.drop(columns=["date"])

    # 保存
    df.to_csv(csv_path, index=False)
    print(f"[OK] 保存: {csv_path}")

    # 確認
    target_df = df[df["race_id"].isin(target_races)]
    has_odds = target_df[target_df["win_odds"] > 0]
    print(f"オッズあり: {len(has_odds)}/{len(target_df)} レコード")


def main():
    parser = argparse.ArgumentParser(description="過去レースのオッズをCSVに追加")
    parser.add_argument("--track", required=True, help="競馬場コード (44=大井, 45=川崎)")
    parser.add_argument("--start", required=True, help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="終了日 (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.track not in TRACKS:
        print(f"[ERROR] 不明な競馬場コード: {args.track}")
        return

    add_odds_to_csv(args.track, args.start, args.end)


if __name__ == "__main__":
    main()
