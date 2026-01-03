# -*- coding: utf-8 -*-
"""特定日のレース結果を取得"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

def fetch_race_list(date_str, track_code):
    """指定日のレース一覧を取得"""
    # track_code: 44=大井, 45=川崎
    url = f'https://nar.netkeiba.com/top/race_list.html?kaisai_date={date_str}'

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = 'euc-jp'
    soup = BeautifulSoup(r.text, 'html.parser')

    race_ids = []
    links = soup.find_all('a', href=re.compile(r'race_id=\d+'))

    for link in links:
        href = link.get('href', '')
        match = re.search(r'race_id=(\d+)', href)
        if match:
            race_id = match.group(1)
            # track_codeでフィルタ
            if race_id[4:6] == track_code:
                race_ids.append(race_id)

    return list(set(race_ids))

def fetch_race_result(race_id):
    """レース結果を取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = 'euc-jp'
    soup = BeautifulSoup(r.text, 'html.parser')

    results = []

    # 結果テーブル
    tables = soup.find_all('table')
    if not tables:
        return None, None

    result_table = tables[0]
    rows = result_table.find_all('tr')

    for row in rows[1:]:
        cols = row.find_all(['td', 'th'])
        if len(cols) < 10:
            continue

        try:
            rank_text = cols[0].get_text(strip=True)
            rank = int(re.search(r'\d+', rank_text).group()) if re.search(r'\d+', rank_text) else None

            horse_number = int(cols[2].get_text(strip=True))
            horse_name = cols[3].get_text(strip=True)

            results.append({
                'race_id': int(race_id),
                'horse_number': horse_number,
                'horse_name': horse_name,
                'rank': rank,
            })
        except:
            continue

    # 複勝オッズ取得
    place_odds = {}
    if len(tables) > 1:
        payout_table = tables[1]
        for row in payout_table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                row_text = cells[0].get_text(strip=True)
                if '複' in row_text:
                    payouts_text = cells[2].get_text(strip=True)
                    horse_nums_text = cells[1].get_text(strip=True)
                    payouts = re.findall(r'(\d+)円', payouts_text)

                    # 簡易パース
                    if len(payouts) == 3 and len(horse_nums_text) >= 3:
                        try:
                            if len(horse_nums_text) == 3:
                                nums = [int(horse_nums_text[0]), int(horse_nums_text[1]), int(horse_nums_text[2])]
                            elif len(horse_nums_text) == 4:
                                # 2桁が1つある
                                nums = [int(horse_nums_text[:2]), int(horse_nums_text[2]), int(horse_nums_text[3])]
                                if nums[0] > 18:
                                    nums = [int(horse_nums_text[0]), int(horse_nums_text[1:3]), int(horse_nums_text[3])]
                            else:
                                nums = []

                            for hn, payout in zip(nums, payouts):
                                place_odds[hn] = int(payout) / 100
                        except:
                            pass

    # 複勝オッズを結果に追加
    for r in results:
        r['place_odds'] = place_odds.get(r['horse_number'], None)

    return results

def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else '20260101'
    track = sys.argv[2] if len(sys.argv) > 2 else 'kawasaki'

    track_code = '45' if track == 'kawasaki' else '44'

    print(f'=== {date_str} {track.upper()} レース取得 ===')

    # レース一覧取得
    race_ids = fetch_race_list(date_str, track_code)
    print(f'レース数: {len(race_ids)}')

    if not race_ids:
        print('レースが見つかりません')
        return

    all_results = []

    for race_id in sorted(race_ids):
        print(f'取得中: {race_id}')
        results = fetch_race_result(race_id)
        if results:
            all_results.extend(results)
        time.sleep(0.5)

    if all_results:
        df = pd.DataFrame(all_results)
        print()
        print(df.to_string())

        # 複勝3-5倍をフィルタ
        target = df[(df['place_odds'] >= 3.0) & (df['place_odds'] <= 5.0)]
        print()
        print('=== 複勝3-5倍の馬 ===')
        if len(target) > 0:
            for _, row in target.iterrows():
                hit = '◎' if row['rank'] <= 3 else '×'
                print(f"{row['horse_name']}: 複勝{row['place_odds']}倍 → {row['rank']}着 {hit}")

            # 収支計算
            bets = len(target) * 1000
            payout = target[target['rank'] <= 3]['place_odds'].sum() * 1000
            profit = payout - bets
            print()
            print(f'賭け: {len(target)}頭 × 1000円 = {bets:,}円')
            print(f'払戻: {payout:,.0f}円')
            print(f'収支: {profit:+,.0f}円')
        else:
            print('該当馬なし')

if __name__ == '__main__':
    main()
