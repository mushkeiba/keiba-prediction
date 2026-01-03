"""複勝オッズ取得スクリプト v2 - netkeibaから正確に取得"""
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


def fetch_place_odds(race_id):
    """結果ページから複勝オッズを取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        tables = soup.find_all('table')
        if len(tables) < 2:
            return None

        # Table 1に払戻金がある
        payout_table = tables[1]
        rows = payout_table.find_all('tr')

        place_odds = {}

        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 3:
                continue

            row_text = cells[0].get_text(strip=True)

            # 複勝の行を探す（文字化けしてるので「複」で検索）
            if '複' in row_text or 'fukusho' in row_text.lower():
                # 馬番: cells[1] = "1394" → 13, 9, 4
                # 払戻: cells[2] = "120円190円110円"
                horse_nums_text = cells[1].get_text(strip=True)
                payouts_text = cells[2].get_text(strip=True)

                # 馬番をパース（2桁ずつ or 1桁ずつ）
                # "1394" → [13, 9, 4] or [1, 3, 9, 4]
                # 払戻金の数で判断
                payouts = re.findall(r'(\d+)円', payouts_text)

                if len(payouts) == 3:
                    # 3頭の複勝
                    # 馬番を分解
                    nums_str = horse_nums_text
                    horse_nums = []

                    # 2桁+1桁+1桁 or 1桁+2桁+1桁 など試す
                    # 払戻金額から人気順を推測して分解
                    if len(nums_str) == 4:
                        # パターン: 13,9,4 or 1,3,94 など
                        patterns = [
                            [nums_str[:2], nums_str[2], nums_str[3]],
                            [nums_str[0], nums_str[1:3], nums_str[3]],
                            [nums_str[0], nums_str[1], nums_str[2:4]],
                            [nums_str[0], nums_str[1], nums_str[2], nums_str[3]],
                        ]
                        for pat in patterns:
                            try:
                                nums = [int(x) for x in pat if x]
                                if len(nums) == 3 and all(1 <= n <= 18 for n in nums):
                                    horse_nums = nums
                                    break
                            except:
                                continue
                    elif len(nums_str) == 3:
                        # 1桁ずつ
                        horse_nums = [int(nums_str[0]), int(nums_str[1]), int(nums_str[2])]
                    elif len(nums_str) == 5:
                        # 2桁+2桁+1桁 or 2桁+1桁+2桁 など
                        patterns = [
                            [nums_str[:2], nums_str[2:4], nums_str[4]],
                            [nums_str[:2], nums_str[2], nums_str[3:5]],
                            [nums_str[0], nums_str[1:3], nums_str[3:5]],
                        ]
                        for pat in patterns:
                            try:
                                nums = [int(x) for x in pat if x]
                                if len(nums) == 3 and all(1 <= n <= 18 for n in nums):
                                    horse_nums = nums
                                    break
                            except:
                                continue

                    # 馬番と払戻金をマッピング
                    if len(horse_nums) == 3 and len(payouts) == 3:
                        for hn, payout in zip(horse_nums, payouts):
                            odds = int(payout) / 100
                            place_odds[hn] = odds

                elif len(payouts) == 2:
                    # 2頭の複勝（少頭数レース）
                    if len(horse_nums_text) == 2:
                        horse_nums = [int(horse_nums_text[0]), int(horse_nums_text[1])]
                    elif len(horse_nums_text) == 3:
                        patterns = [
                            [horse_nums_text[:2], horse_nums_text[2]],
                            [horse_nums_text[0], horse_nums_text[1:3]],
                        ]
                        for pat in patterns:
                            try:
                                nums = [int(x) for x in pat]
                                if all(1 <= n <= 18 for n in nums):
                                    horse_nums = nums
                                    break
                            except:
                                continue

                    if len(horse_nums) == 2 and len(payouts) == 2:
                        for hn, payout in zip(horse_nums, payouts):
                            odds = int(payout) / 100
                            place_odds[hn] = odds

        if place_odds:
            return {'race_id': race_id, 'place_odds': place_odds}
        return None

    except Exception as e:
        return None


def process_batch(race_ids):
    """バッチ処理"""
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_race = {executor.submit(fetch_place_odds, rid): rid for rid in race_ids}
        for future in as_completed(future_to_race):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except:
                pass
    return results


def main(track_name):
    csv_path = BASE_DIR / f'data/races_{track_name}.csv'
    df = pd.read_csv(csv_path)

    # テストデータのレースID
    df = df.sort_values('race_id')
    split_idx = int(len(df) * 0.8)
    test_race_ids = df.iloc[split_idx:]['race_id'].unique().tolist()

    print(f'=== {track_name.upper()} 複勝オッズ取得 ===')
    print(f'テストレース数: {len(test_race_ids)}')

    place_odds_data = {}
    batch_size = 50
    total = len(test_race_ids)
    done = 0
    success = 0

    for i in range(0, total, batch_size):
        batch = test_race_ids[i:i+batch_size]
        results = process_batch(batch)

        for r in results:
            place_odds_data[r['race_id']] = r['place_odds']
            success += 1

        done += len(batch)
        pct = done / total * 100
        print(f'[{done}/{total}] {pct:.1f}% - 成功: {success}')

        time.sleep(0.3)

    # CSVを更新
    updated = 0
    for idx, row in df.iterrows():
        race_id = row['race_id']
        horse_num = row['horse_number']

        if race_id in place_odds_data:
            odds_dict = place_odds_data[race_id]
            if horse_num in odds_dict:
                df.at[idx, 'place_odds'] = odds_dict[horse_num]
                updated += 1

    df.to_csv(csv_path, index=False)

    print(f'\n=== 完了 ===')
    print(f'更新: {updated}件')

    # 確認
    test_df = df.iloc[split_idx:]
    valid = test_df[(test_df['place_odds'] > 0) & (test_df['place_odds'] < 100)]
    print(f'複勝オッズ有効: {len(valid):,} / {len(test_df):,} ({len(valid)/len(test_df)*100:.1f}%)')
    if len(valid) > 0:
        print(f'複勝オッズ平均: {valid["place_odds"].mean():.2f}倍')


if __name__ == '__main__':
    track = sys.argv[1] if len(sys.argv) > 1 else 'ohi'
    main(track)
