"""
並列版：追加特徴量取得スクリプト
5並列でデータを爆速取得
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import sys
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).parent

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}


def fetch_race_result(race_id):
    """レース結果ページから追加特徴量を取得"""
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
            if len(cols) < 10:
                continue

            try:
                rank_text = cols[0].get_text(strip=True)
                rank = int(re.search(r'\d+', rank_text).group()) if re.search(r'\d+', rank_text) else None

                horse_number_text = cols[2].get_text(strip=True)
                horse_number = int(horse_number_text) if horse_number_text.isdigit() else None

                time_text = cols[7].get_text(strip=True) if len(cols) > 7 else ''
                race_time = parse_time(time_text)

                last_3f_text = cols[11].get_text(strip=True) if len(cols) > 11 else ''
                last_3f = float(last_3f_text) if re.match(r'^\d+\.?\d*$', last_3f_text) else None

                results.append({
                    'race_id': race_id,
                    'horse_number': horse_number,
                    'race_time': race_time,
                    'last_3f': last_3f,
                })
            except:
                continue

        return results
    except:
        return None


def parse_time(time_str):
    if not time_str:
        return None
    match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2)) + int(match.group(3)) / 10
    match = re.match(r'(\d+)\.(\d+)', time_str)
    if match:
        return int(match.group(1)) + int(match.group(2)) / 10
    return None


def process_batch(race_ids, df, csv_path):
    """バッチ処理"""
    results_all = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_race = {executor.submit(fetch_race_result, rid): rid for rid in race_ids}

        for future in as_completed(future_to_race):
            race_id = future_to_race[future]
            try:
                results = future.result()
                if results:
                    results_all.extend(results)
            except:
                pass

    # DataFrameを更新
    for r in results_all:
        mask = (df['race_id'] == r['race_id']) & (df['horse_number'] == r['horse_number'])
        if mask.any():
            df.loc[mask, 'race_time_seconds'] = r['race_time']
            df.loc[mask, 'last_3f'] = r['last_3f']

    return len([r for r in results_all if r])


def main(track_name):
    csv_path = BASE_DIR / f'data/races_{track_name}.csv'
    df = pd.read_csv(csv_path)

    # 新カラム追加
    if 'race_time_seconds' not in df.columns:
        df['race_time_seconds'] = None
    if 'last_3f' not in df.columns:
        df['last_3f'] = None

    # 未取得レース
    incomplete = df[df['last_3f'].isna()]['race_id'].unique().tolist()
    total = len(incomplete)

    print(f'=== {track_name.upper()} 並列取得 ===')
    print(f'残りレース: {total}')
    print(f'並列数: 5')
    print()

    batch_size = 50
    done = 0

    for i in range(0, total, batch_size):
        batch = incomplete[i:i+batch_size]
        count = process_batch(batch, df, csv_path)
        done += len(batch)

        # 保存
        df.to_csv(csv_path, index=False)

        pct = done / total * 100
        print(f'[{done}/{total}] {pct:.1f}% 完了')

        time.sleep(0.5)  # バッチ間の休憩

    print(f'\n=== 完了！ ===')
    print(f'保存先: {csv_path}')


if __name__ == '__main__':
    import sys
    track = sys.argv[1] if len(sys.argv) > 1 else 'ohi'
    main(track)
