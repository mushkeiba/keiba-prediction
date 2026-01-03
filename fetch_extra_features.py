"""
追加特徴量（上がり3F・脚質・タイム）取得スクリプト
netkeibaからデータをスクレイピングしてCSVに追加
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import sys
import io
from pathlib import Path

# 文字化け対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).parent

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
}


def fetch_race_result(race_id):
    """レース結果ページから追加特徴量を取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, 'html.parser')

        # 結果テーブルを探す
        result_table = soup.find('table', class_='RaceTable01')
        if not result_table:
            # クラス名が違う場合、最初のテーブルを試す
            tables = soup.find_all('table')
            if tables:
                result_table = tables[0]
            else:
                return None

        rows = result_table.find_all('tr')
        if len(rows) < 2:
            return None

        results = []

        for row in rows[1:]:  # ヘッダーをスキップ
            cols = row.find_all(['td', 'th'])
            if len(cols) < 10:
                continue

            try:
                # 各カラムからデータ抽出
                rank_text = cols[0].get_text(strip=True)
                rank = int(re.search(r'\d+', rank_text).group()) if re.search(r'\d+', rank_text) else None

                horse_number_text = cols[2].get_text(strip=True)
                horse_number = int(horse_number_text) if horse_number_text.isdigit() else None

                # タイム（分:秒.ミリ秒形式）
                time_text = cols[7].get_text(strip=True) if len(cols) > 7 else ''
                race_time = parse_time(time_text)

                # 上がり3F
                last_3f_text = cols[11].get_text(strip=True) if len(cols) > 11 else ''
                last_3f = float(last_3f_text) if re.match(r'^\d+\.?\d*$', last_3f_text) else None

                # コーナー通過順（脚質計算用）
                corner_text = ''
                corner_span = row.find('span', class_='PassageNum')
                if corner_span:
                    corner_text = corner_span.get_text(strip=True)

                # 脚質を計算
                running_style = calc_running_style(corner_text, rank)

                results.append({
                    'race_id': race_id,
                    'horse_number': horse_number,
                    'rank': rank,
                    'race_time': race_time,
                    'last_3f': last_3f,
                    'running_style': running_style,
                    'corner_positions': corner_text,
                })

            except Exception as e:
                continue

        return results

    except Exception as e:
        print(f'Error fetching {race_id}: {e}')
        return None


def parse_time(time_str):
    """タイム文字列を秒に変換 (1:23.4 -> 83.4)"""
    if not time_str:
        return None

    match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        decimals = int(match.group(3))
        return minutes * 60 + seconds + decimals / 10

    match = re.match(r'(\d+)\.(\d+)', time_str)
    if match:
        seconds = int(match.group(1))
        decimals = int(match.group(2))
        return seconds + decimals / 10

    return None


def calc_running_style(corner_text, final_rank):
    """
    コーナー通過順から脚質を計算
    0: 逃げ（先頭）
    1: 先行（2-4番手）
    2: 差し（5-8番手）
    3: 追込（9番手以降）
    """
    if not corner_text:
        return None

    # 最初のコーナー通過順を取得
    positions = re.findall(r'\d+', corner_text)
    if not positions:
        return None

    first_corner = int(positions[0])

    if first_corner == 1:
        return 0  # 逃げ
    elif first_corner <= 4:
        return 1  # 先行
    elif first_corner <= 8:
        return 2  # 差し
    else:
        return 3  # 追込


def update_csv_with_features(track_name, max_races=None, delay=0.5):
    """CSVに追加特徴量を追加"""
    csv_path = BASE_DIR / f'data/races_{track_name}.csv'

    if not csv_path.exists():
        print(f'CSV not found: {csv_path}')
        return

    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} rows from {csv_path}')

    # 既存のカラムを確認
    new_cols = ['race_time_seconds', 'last_3f', 'running_style']
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    # 未取得のレースIDを抽出
    race_ids = df['race_id'].unique()

    # 既に取得済みのレースを除外
    incomplete = df[df['last_3f'].isna()]['race_id'].unique()

    if len(incomplete) == 0:
        print('All races already have extra features!')
        return

    print(f'Races to fetch: {len(incomplete)}')

    if max_races:
        incomplete = incomplete[:max_races]
        print(f'Limiting to {max_races} races')

    success = 0
    failed = 0

    for i, race_id in enumerate(incomplete):
        print(f'[{i+1}/{len(incomplete)}] Fetching {race_id}...', end=' ')

        results = fetch_race_result(race_id)

        if results:
            # DataFrameを更新
            for r in results:
                mask = (df['race_id'] == race_id) & (df['horse_number'] == r['horse_number'])
                if mask.any():
                    df.loc[mask, 'race_time_seconds'] = r['race_time']
                    df.loc[mask, 'last_3f'] = r['last_3f']
                    df.loc[mask, 'running_style'] = r['running_style']

            print(f'OK ({len(results)} horses)')
            success += 1
        else:
            print('FAILED')
            failed += 1

        # 10件ごとに保存
        if (i + 1) % 10 == 0:
            df.to_csv(csv_path, index=False)
            print(f'  Saved! ({success} success, {failed} failed)')

        time.sleep(delay)

    # 最終保存
    df.to_csv(csv_path, index=False)
    print(f'\n=== Complete ===')
    print(f'Success: {success}')
    print(f'Failed: {failed}')
    print(f'Saved to {csv_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='追加特徴量取得')
    parser.add_argument('track', choices=['ohi', 'kawasaki'], help='競馬場')
    parser.add_argument('--max', type=int, default=None, help='最大レース数')
    parser.add_argument('--delay', type=float, default=0.5, help='リクエスト間隔(秒)')

    args = parser.parse_args()

    print('='*50)
    print('追加特徴量取得スクリプト')
    print('='*50)
    print(f'Track: {args.track}')
    print(f'Max races: {args.max or "all"}')
    print(f'Delay: {args.delay}s')
    print('='*50)

    update_csv_with_features(args.track, max_races=args.max, delay=args.delay)
