#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
馬場状態・天気を本物のデータで更新するスクリプト
"""
import sys
import time
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

# 出力エンコーディング設定
sys.stdout.reconfigure(encoding='utf-8')

CSV_PATH = 'data/races_ohi.csv'
DELAY = 0.3  # リクエスト間隔

def fetch_race_conditions(race_id: str, session: requests.Session) -> dict:
    """レースページから馬場状態・天気を取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    try:
        r = session.get(url, timeout=30)
        r.encoding = 'EUC-JP'
        soup = BeautifulSoup(r.text, 'lxml')

        rd = soup.find('div', class_='RaceData01')
        if not rd:
            return {'track_condition': None, 'weather': None}

        rd_text = rd.get_text()

        # 馬場状態
        track_match = re.search(r'馬場[:：]\s*(良|稍重|重|不良)', rd_text)
        track_cond = track_match.group(1) if track_match else None

        # 天気
        weather_match = re.search(r'天候[:：]\s*(晴|曇|雨|小雨|雪)', rd_text)
        weather = weather_match.group(1) if weather_match else None

        return {'track_condition': track_cond, 'weather': weather}

    except Exception as e:
        return {'track_condition': None, 'weather': None}


def main():
    print('=' * 60, flush=True)
    print('[大井競馬場] 馬場状態・天気を本物のデータで更新', flush=True)
    print('=' * 60, flush=True)

    # CSV読み込み
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f'レコード数: {len(df)}')

    # バックアップ
    backup_path = CSV_PATH.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(backup_path, index=False)
    print(f'バックアップ: {backup_path}')

    # ユニークレースID
    unique_races = df['race_id'].unique()
    print(f'ユニークレース数: {len(unique_races)}')
    print(f'推定時間: {len(unique_races) * DELAY / 60:.1f}分')
    print()

    # セッション作成
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # 各レースの条件を取得
    race_conditions = {}
    success_count = 0

    for i, race_id in enumerate(unique_races):
        if i % 50 == 0:
            print(f'{i}/{len(unique_races)} 完了... (成功: {success_count})', flush=True)

        conditions = fetch_race_conditions(str(race_id), session)
        race_conditions[race_id] = conditions

        if conditions['track_condition'] and conditions['weather']:
            success_count += 1

        time.sleep(DELAY)

    print(f'\n取得完了: {success_count}/{len(unique_races)} レースで成功')

    # DataFrameに反映
    df['track_condition_new'] = df['race_id'].map(
        lambda x: race_conditions.get(x, {}).get('track_condition')
    )
    df['weather_new'] = df['race_id'].map(
        lambda x: race_conditions.get(x, {}).get('weather')
    )

    # 新しい値があれば更新、なければ元の値を維持
    updated_track = df['track_condition_new'].notna().sum()
    updated_weather = df['weather_new'].notna().sum()

    df['track_condition'] = df['track_condition_new'].fillna(df['track_condition'])
    df['weather'] = df['weather_new'].fillna(df['weather'])

    # 一時列を削除
    df = df.drop(columns=['track_condition_new', 'weather_new'])

    # 保存
    df.to_csv(CSV_PATH, index=False)

    print()
    print('=== 結果 ===')
    print(f'track_condition更新: {updated_track}/{len(df)} レコード')
    print(f'weather更新: {updated_weather}/{len(df)} レコード')
    print()
    print('track_condition分布:')
    print(df['track_condition'].value_counts())
    print()
    print('weather分布:')
    print(df['weather'].value_counts())
    print()
    print(f'[OK] 保存完了: {CSV_PATH}')


if __name__ == '__main__':
    main()
