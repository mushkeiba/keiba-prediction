# -*- coding: utf-8 -*-
"""
全地方競馬場のデータを収集してCSV保存
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import os
import csv
from datetime import datetime

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

TRACKS = {
    '30': '門別', '35': '盛岡', '36': '水沢',
    '42': '浦和', '43': '船橋', '44': '大井', '45': '川崎',
    '46': '金沢', '47': '笠松', '48': '名古屋',
    '50': '園田', '51': '姫路', '54': '高知', '55': '佐賀',
}


def fetch_shutuba_odds(race_id):
    """出馬表から複勝オッズを取得"""
    url = f'https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')
        odds_dict = {}
        for row in soup.find_all('tr'):
            cols = row.find_all(['td', 'th'])
            if len(cols) < 3:
                continue
            horse_num = None
            odds_val = None
            for col in cols:
                text = col.get_text(strip=True)
                if horse_num is None and re.match(r'^[1-9]$|^1[0-8]$', text):
                    horse_num = int(text)
                if re.match(r'^\d+\.\d+$', text):
                    val = float(text)
                    if 1.0 <= val <= 500:
                        odds_val = val
            if horse_num and odds_val:
                odds_dict[horse_num] = odds_val
        return odds_dict
    except:
        return {}


def fetch_race_result(race_id):
    """レース結果を取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'euc-jp'
        if len(r.text) < 5000:
            return None
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        if not tables:
            return None
        horses = []
        for row in tables[0].find_all('tr')[1:]:
            cols = row.find_all(['td', 'th'])
            if len(cols) < 4:
                continue
            try:
                rank_text = cols[0].get_text(strip=True)
                rank_match = re.search(r'\d+', rank_text)
                if not rank_match:
                    continue
                rank = int(rank_match.group())
                horse_num = int(cols[2].get_text(strip=True))
                horse_name = cols[3].get_text(strip=True)
                horses.append({
                    'num': horse_num,
                    'name': horse_name,
                    'rank': rank
                })
            except:
                continue
        return horses
    except:
        return None


def collect_track_data(track_code, track_name, max_races=150):
    """1競馬場のデータを収集"""
    print(f'\n{track_name}競馬のデータ収集中...')

    all_data = []
    race_count = 0

    for year in [2026, 2025]:
        for kai in range(12, 0, -1):
            found_any = False

            for day in range(1, 9):
                test_id = f"{year}{track_code}{kai:02d}{day:02d}01"
                result = fetch_race_result(test_id)
                time.sleep(0.2)

                if not result:
                    if found_any:
                        break
                    continue

                found_any = True
                print(f'  {year}年{kai}回{day}日目...', end='', flush=True)
                day_races = 0

                for race_num in range(1, 13):
                    race_id = f"{year}{track_code}{kai:02d}{day:02d}{race_num:02d}"

                    if race_num == 1:
                        horses = result
                    else:
                        horses = fetch_race_result(race_id)
                        time.sleep(0.2)

                    if not horses:
                        continue

                    odds = fetch_shutuba_odds(race_id)
                    time.sleep(0.2)

                    if not odds:
                        continue

                    race_count += 1
                    day_races += 1

                    # 全馬のデータを保存
                    for h in horses:
                        pre_odds = odds.get(h['num'])
                        if pre_odds:
                            all_data.append({
                                'race_id': race_id,
                                'track_code': track_code,
                                'track_name': track_name,
                                'year': year,
                                'kai': kai,
                                'day': day,
                                'race_num': race_num,
                                'horse_num': h['num'],
                                'horse_name': h['name'],
                                'rank': h['rank'],
                                'pre_odds': pre_odds,
                                'is_target': 3.0 <= pre_odds <= 5.0,
                                'is_hit': h['rank'] <= 3
                            })

                print(f' {day_races}R', flush=True)

                if race_count >= max_races:
                    break

            if race_count >= max_races:
                break

        if race_count >= max_races:
            break

    return all_data


def main():
    # 保存先ディレクトリ
    data_dir = 'scraped_data'
    os.makedirs(data_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_tracks_data = []
    summary = []

    for track_code, track_name in TRACKS.items():
        data = collect_track_data(track_code, track_name, max_races=150)
        all_tracks_data.extend(data)

        # 競馬場別サマリー
        targets = [d for d in data if d['is_target']]
        hits = [d for d in targets if d['is_hit']]

        if targets:
            bets = len(targets)
            hit_count = len(hits)
            payout = sum(int(d['pre_odds'] * 1000) for d in hits)
            roi = payout / (bets * 1000) * 100

            summary.append({
                'track_name': track_name,
                'races': len(set(d['race_id'] for d in data)),
                'bets': bets,
                'hits': hit_count,
                'hit_rate': hit_count / bets * 100,
                'roi': roi
            })

            print(f'  -> {bets}頭中{hit_count}的中, ROI {roi:.1f}%')

        # 競馬場別CSV保存
        track_file = os.path.join(data_dir, f'{track_name}_{timestamp}.csv')
        with open(track_file, 'w', encoding='utf-8', newline='') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        print(f'  保存: {track_file}')

    # 全競馬場まとめCSV
    all_file = os.path.join(data_dir, f'all_tracks_{timestamp}.csv')
    with open(all_file, 'w', encoding='utf-8', newline='') as f:
        if all_tracks_data:
            writer = csv.DictWriter(f, fieldnames=all_tracks_data[0].keys())
            writer.writeheader()
            writer.writerows(all_tracks_data)
    print(f'\n全データ保存: {all_file}')

    # サマリー表示
    print('\n' + '='*60)
    print('【全競馬場サマリー】')
    print('='*60)
    print(f'{"競馬場":6} {"レース":>6} {"対象":>6} {"的中":>6} {"的中率":>8} {"ROI":>8}')
    print('-'*50)

    for s in sorted(summary, key=lambda x: x['roi'], reverse=True):
        print(f'{s["track_name"]:6} {s["races"]:>6} {s["bets"]:>6} {s["hits"]:>6} {s["hit_rate"]:>7.1f}% {s["roi"]:>7.1f}%')

    # サマリーCSV
    summary_file = os.path.join(data_dir, f'summary_{timestamp}.csv')
    with open(summary_file, 'w', encoding='utf-8', newline='') as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    print(f'\nサマリー保存: {summary_file}')


if __name__ == '__main__':
    main()
