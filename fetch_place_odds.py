"""複勝オッズ取得スクリプト - 払戻金から計算"""
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
    """払戻金ページから複勝オッズを取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, 'html.parser')

        # 払戻金テーブルを探す
        results = {}

        # 全テーブルをチェック
        for table in soup.find_all('table'):
            text = table.get_text()

            # 複勝を含むテーブルを探す
            if '複勝' in text:
                rows = table.find_all('tr')
                for row in rows:
                    row_text = row.get_text()
                    if '複勝' in row_text:
                        # 複勝の行から馬番と払戻金を抽出
                        cells = row.find_all(['td', 'th'])
                        for cell in cells:
                            cell_text = cell.get_text(strip=True)
                            # パターン: "1" "230" または "1 230"
                            # 馬番（1-18）と払戻金（100-99999）を探す
                            matches = re.findall(r'(\d+)\s*(\d{3,5})', cell_text)
                            for match in matches:
                                horse_num = int(match[0])
                                payout = int(match[1])
                                if 1 <= horse_num <= 18 and 100 <= payout <= 99999:
                                    odds = payout / 100
                                    results[horse_num] = odds

        # 別のパターンも試す（Payout_Detail_Table）
        if not results:
            payout_tables = soup.find_all('table', class_='Payout_Detail_Table')
            for table in payout_tables:
                rows = table.find_all('tr')
                for row in rows:
                    if '複勝' in row.get_text():
                        # 次の行に馬番と払戻金がある場合
                        tds = row.find_all('td')
                        for td in tds:
                            spans = td.find_all('span')
                            for i in range(0, len(spans)-1, 2):
                                try:
                                    horse_num = int(spans[i].get_text(strip=True))
                                    payout_text = spans[i+1].get_text(strip=True).replace(',', '').replace('円', '')
                                    payout = int(payout_text)
                                    if 1 <= horse_num <= 18 and 100 <= payout <= 99999:
                                        results[horse_num] = payout / 100
                                except:
                                    pass

        # さらに別パターン（ResultPayback）
        if not results:
            payback = soup.find('div', class_='ResultPayback')
            if payback:
                rows = payback.find_all('tr')
                in_place = False
                for row in rows:
                    text = row.get_text()
                    if '複勝' in text:
                        in_place = True
                    elif in_place and ('単勝' in text or '枠連' in text or '馬連' in text):
                        in_place = False

                    if in_place:
                        # 馬番と払戻金を探す
                        nums = re.findall(r'\d+', text)
                        if len(nums) >= 2:
                            for i in range(0, len(nums)-1, 2):
                                try:
                                    horse_num = int(nums[i])
                                    payout = int(nums[i+1])
                                    if 1 <= horse_num <= 18 and 100 <= payout <= 99999:
                                        results[horse_num] = payout / 100
                                except:
                                    pass

        if results:
            return {'race_id': race_id, 'place_odds': results}
        return None

    except Exception as e:
        return None


def process_batch(race_ids):
    """バッチ処理"""
    results_all = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_race = {executor.submit(fetch_place_odds, rid): rid for rid in race_ids}

        for future in as_completed(future_to_race):
            try:
                result = future.result()
                if result:
                    results_all.append(result)
            except:
                pass

    return results_all


def main(track_name):
    csv_path = BASE_DIR / f'data/races_{track_name}.csv'
    df = pd.read_csv(csv_path)

    # テストデータのレースID
    df = df.sort_values('race_id')
    split_idx = int(len(df) * 0.8)
    test_race_ids = df.iloc[split_idx:]['race_id'].unique().tolist()

    print(f'=== {track_name.upper()} 複勝オッズ取得 ===')
    print(f'テストレース数: {len(test_race_ids)}')

    # 複勝オッズを格納
    place_odds_data = {}

    batch_size = 50
    total = len(test_race_ids)
    done = 0

    for i in range(0, total, batch_size):
        batch = test_race_ids[i:i+batch_size]
        results = process_batch(batch)

        for r in results:
            place_odds_data[r['race_id']] = r['place_odds']

        done += len(batch)
        pct = done / total * 100
        got = sum(1 for r in results if r)
        print(f'[{done}/{total}] {pct:.1f}% - 取得: {got}/{len(batch)}')

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
    print(f'保存: {csv_path}')

    # 確認
    test_df = df.iloc[split_idx:]
    valid = test_df[test_df['place_odds'] > 0]
    print(f'複勝オッズ有効: {len(valid):,} / {len(test_df):,} ({len(valid)/len(test_df)*100:.1f}%)')


if __name__ == '__main__':
    track = sys.argv[1] if len(sys.argv) > 1 else 'ohi'
    main(track)
