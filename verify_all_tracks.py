# -*- coding: utf-8 -*-
"""
全地方競馬場で複勝3-5倍戦略を検証
使い方: python verify_all_tracks.py [競馬場コード]
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import sys

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# 全地方競馬場
TRACKS = {
    '30': '門別',
    '35': '盛岡',
    '36': '水沢',
    '42': '浦和',
    '43': '船橋',
    '44': '大井',
    '45': '川崎',
    '46': '金沢',
    '47': '笠松',
    '48': '名古屋',
    '50': '園田',
    '51': '姫路',
    '54': '高知',
    '55': '佐賀',
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


def verify_track(track_code, track_name, max_races=150):
    """1競馬場を検証"""
    print(f'\n{"="*60}')
    print(f'{track_name}競馬（コード: {track_code}）')
    print(f'{"="*60}')

    stats = {'races': 0, 'bets': 0, 'hits': 0, 'payout': 0}
    details = []

    # 2026年と2025年を探索
    for year in [2026, 2025]:
        for kai in range(12, 0, -1):
            found_any = False

            for day in range(1, 9):
                # まず1Rを確認
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

                    stats['races'] += 1
                    day_races += 1

                    for h in horses:
                        o = odds.get(h['num'])
                        if o and 3.0 <= o <= 5.0:
                            stats['bets'] += 1
                            hit = h['rank'] <= 3

                            details.append({
                                'race_id': race_id,
                                'horse': h['name'],
                                'odds': o,
                                'rank': h['rank'],
                                'hit': hit
                            })

                            if hit:
                                stats['hits'] += 1
                                stats['payout'] += int(o * 1000)

                print(f' {day_races}R', flush=True)

                if stats['races'] >= max_races:
                    break

            if stats['races'] >= max_races:
                break

        if stats['races'] >= max_races:
            break

    # 結果表示
    print()
    if stats['bets'] == 0:
        print('対象馬なし')
        return stats, details

    total_bet = stats['bets'] * 1000
    hit_rate = stats['hits'] / stats['bets'] * 100
    roi = stats['payout'] / total_bet * 100
    profit = stats['payout'] - total_bet

    print(f'レース数: {stats["races"]}')
    print(f'対象馬数: {stats["bets"]}頭')
    print(f'的中数:   {stats["hits"]}頭')
    print(f'的中率:   {hit_rate:.1f}%')
    print(f'投資額:   {total_bet:,}円')
    print(f'払戻額:   {stats["payout"]:,}円')
    print(f'収支:     {profit:+,}円')
    print(f'ROI:      {roi:.1f}%')

    return stats, details


def main():
    if len(sys.argv) > 1:
        # 指定競馬場のみ
        code = sys.argv[1]
        if code in TRACKS:
            verify_track(code, TRACKS[code])
        else:
            print(f'Unknown track code: {code}')
            print('Available:', list(TRACKS.keys()))
    else:
        # 全競馬場
        all_stats = {}
        for code, name in TRACKS.items():
            stats, _ = verify_track(code, name, max_races=100)
            all_stats[name] = stats

        # サマリー
        print('\n' + '='*60)
        print('【全競馬場サマリー】')
        print('='*60)
        print(f'{"競馬場":6} {"レース":>6} {"対象":>6} {"的中":>6} {"的中率":>8} {"ROI":>8}')
        print('-'*50)

        for name, s in all_stats.items():
            if s['bets'] > 0:
                hr = s['hits'] / s['bets'] * 100
                roi = s['payout'] / (s['bets'] * 1000) * 100
                print(f'{name:6} {s["races"]:>6} {s["bets"]:>6} {s["hits"]:>6} {hr:>7.1f}% {roi:>7.1f}%')
            else:
                print(f'{name:6} {s["races"]:>6} {"N/A":>6}')


if __name__ == '__main__':
    main()
