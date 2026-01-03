"""テストデータ用オッズ取得スクリプト"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import sys
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).parent

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}


def fetch_race_odds(race_id):
    """レース結果ページから確定オッズを取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        if not tables:
            return None

        result_table = tables[0]
        rows = result_table.find_all('tr')
        if len(rows) < 2:
            return None

        results = []
        for row in rows[1:]:
            cols = row.find_all(['td', 'th'])
            if len(cols) < 13:
                continue

            try:
                # 馬番
                horse_number_text = cols[2].get_text(strip=True)
                horse_number = int(horse_number_text) if horse_number_text.isdigit() else None

                # 単勝オッズ（12列目あたり）
                win_odds = None
                for i in range(10, min(15, len(cols))):
                    text = cols[i].get_text(strip=True)
                    if re.match(r'^\d+\.?\d*$', text):
                        odds_val = float(text)
                        if 1.0 <= odds_val <= 1000:  # オッズっぽい範囲
                            win_odds = odds_val
                            break

                if horse_number:
                    results.append({
                        'race_id': race_id,
                        'horse_number': horse_number,
                        'win_odds': win_odds,
                    })
            except:
                continue

        return results
    except:
        return None


def fetch_place_odds(race_id):
    """複勝オッズを払戻金から取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {}

        soup = BeautifulSoup(r.text, 'html.parser')

        # 払戻金テーブルを探す
        place_odds = {}

        # 複勝の払戻金を探す
        pay_tables = soup.find_all('table', class_='Payout_Detail_Table')
        if not pay_tables:
            pay_tables = soup.find_all('table')

        for table in pay_tables:
            text = table.get_text()
            if '複勝' in text:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    for i, cell in enumerate(cells):
                        cell_text = cell.get_text(strip=True)
                        # 馬番と払戻金のパターンを探す
                        # 例: "3" "150円" または "3 150"
                        nums = re.findall(r'\d+', cell_text)
                        if len(nums) >= 2:
                            horse_num = int(nums[0])
                            payout = int(nums[1])
                            if 1 <= horse_num <= 18 and 100 <= payout <= 100000:
                                # 払戻金からオッズを計算（100円あたり）
                                odds = payout / 100
                                place_odds[horse_num] = odds

        return place_odds
    except:
        return {}


def process_batch(race_ids):
    """バッチ処理"""
    results_all = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_race = {executor.submit(fetch_race_odds, rid): rid for rid in race_ids}

        for future in as_completed(future_to_race):
            try:
                results = future.result()
                if results:
                    results_all.extend(results)
            except:
                pass

    return results_all


def main(track_name):
    csv_path = BASE_DIR / f'data/races_{track_name}.csv'
    df = pd.read_csv(csv_path)

    # テストデータのレースIDを取得（後ろ20%）
    df = df.sort_values('race_id')
    split_idx = int(len(df) * 0.8)
    test_race_ids = df.iloc[split_idx:]['race_id'].unique().tolist()

    print(f'=== {track_name.upper()} オッズ取得 ===')
    print(f'テストレース数: {len(test_race_ids)}')

    # オッズを格納するdict
    odds_data = {}

    batch_size = 50
    total = len(test_race_ids)
    done = 0

    for i in range(0, total, batch_size):
        batch = test_race_ids[i:i+batch_size]
        results = process_batch(batch)

        for r in results:
            key = (r['race_id'], r['horse_number'])
            odds_data[key] = r['win_odds']

        done += len(batch)
        pct = done / total * 100
        print(f'[{done}/{total}] {pct:.1f}% 完了')

        time.sleep(0.3)

    # CSVを更新
    updated = 0
    for idx, row in df.iterrows():
        key = (row['race_id'], row['horse_number'])
        if key in odds_data and odds_data[key]:
            df.at[idx, 'win_odds'] = odds_data[key]
            # 複勝オッズは単勝の約1/3と推定（ざっくり）
            df.at[idx, 'place_odds'] = odds_data[key] / 3 if odds_data[key] else 0
            updated += 1

    df.to_csv(csv_path, index=False)

    print(f'\n=== 完了 ===')
    print(f'更新: {updated}件')
    print(f'保存: {csv_path}')

    # 確認
    test_df = df.iloc[split_idx:]
    valid_odds = test_df[test_df['win_odds'] > 0]
    print(f'オッズ有効: {len(valid_odds):,} / {len(test_df):,} ({len(valid_odds)/len(test_df)*100:.1f}%)')


if __name__ == '__main__':
    track = sys.argv[1] if len(sys.argv) > 1 else 'ohi'
    main(track)
