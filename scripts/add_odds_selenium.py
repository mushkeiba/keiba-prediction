#!/usr/bin/env python3
"""
Seleniumで過去レースのオッズをCSVに追加

使い方:
  python scripts/add_odds_selenium.py --track 44 --start 2025-12-26 --end 2025-12-30
"""

import argparse
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

BASE_DIR = Path(__file__).parent.parent

TRACKS = {
    "44": {"name": "大井", "data": "data/races_ohi.csv"},
    "45": {"name": "川崎", "data": "data/races_kawasaki.csv"},
}


def create_driver():
    """ヘッドレスChromeを作成"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def get_race_odds(driver, race_id: str) -> dict:
    """レースの確定オッズを取得"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"

    try:
        driver.get(url)
        time.sleep(random.uniform(1, 2))  # ランダム遅延

        # ページタイトルでエラーチェック
        if "Error" in driver.title or "404" in driver.title:
            return {}

        odds_dict = {}

        # 結果テーブルを探す
        try:
            table = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "RaceTable01"))
            )
            rows = table.find_elements(By.TAG_NAME, "tr")

            for row in rows[1:]:  # ヘッダーをスキップ
                try:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) < 8:
                        continue

                    # 馬番（Numクラス）
                    try:
                        horse_num = int(row.find_element(By.CLASS_NAME, "Num").text.strip())
                    except:
                        continue

                    # 単勝オッズ（Oddsクラス）
                    try:
                        odds_elem = row.find_element(By.CLASS_NAME, "Odds")
                        win_odds_text = odds_elem.text.strip()
                        win_odds = float(win_odds_text) if win_odds_text and win_odds_text != '---' else 0
                    except:
                        win_odds = 0

                    if horse_num > 0:
                        odds_dict[horse_num] = {
                            'win_odds': win_odds,
                            'place_odds': win_odds / 3 if win_odds > 0 else 0
                        }

                except Exception as e:
                    continue

        except Exception as e:
            print(f"  [WARN] {race_id}: テーブル取得失敗")
            return {}

        return odds_dict

    except Exception as e:
        print(f"  [ERROR] {race_id}: {e}")
        return {}


def add_odds_to_csv(track_code: str, start_date: str, end_date: str):
    """CSVにオッズを追加"""
    track_info = TRACKS[track_code]
    csv_path = BASE_DIR / track_info["data"]

    print(f"\n{'='*60}")
    print(f"オッズ追加 (Selenium): {track_info['name']}競馬場")
    print(f"期間: {start_date} - {end_date}")
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

    if len(target_races) == 0:
        print("[ERROR] 対象レースがありません")
        return

    # オッズカラムがなければ追加
    if "win_odds" not in df.columns:
        df["win_odds"] = 0.0
    if "place_odds" not in df.columns:
        df["place_odds"] = 0.0

    # ドライバー作成
    print("Chrome起動中...")
    driver = create_driver()

    try:
        success_count = 0
        for i, race_id in enumerate(target_races):
            print(f"  [{i+1}/{len(target_races)}] {race_id}...", end=" ")

            odds = get_race_odds(driver, str(race_id))

            if odds:
                for horse_num, odds_data in odds.items():
                    mask = (df["race_id"] == race_id) & (df["horse_number"] == horse_num)
                    df.loc[mask, "win_odds"] = odds_data["win_odds"]
                    df.loc[mask, "place_odds"] = odds_data["place_odds"]
                success_count += 1
                print(f"OK ({len(odds)}頭)")
            else:
                print("SKIP")

            # 10件ごとに少し長めの休憩
            if (i + 1) % 10 == 0:
                time.sleep(random.uniform(2, 4))

        print(f"\n取得成功: {success_count}/{len(target_races)} レース")

    finally:
        driver.quit()

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
    parser = argparse.ArgumentParser(description="Seleniumで過去レースのオッズをCSVに追加")
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
