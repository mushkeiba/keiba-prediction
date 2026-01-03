# -*- coding: utf-8 -*-
"""
複勝3-5倍戦略の検証スクリプト
川崎・大井競馬 過去60日分
レースIDを総当たりで検索
"""

import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# 競馬場コード
TRACKS = {
    '44': '大井',
    '45': '川崎'
}


def fetch_shutuba_odds(race_id):
    """出馬表から全馬の複勝オッズを取得"""
    url = f'https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}'

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        odds_dict = {}

        # ShutubaTableを探す
        table = soup.find('table', class_='ShutubaTable')
        if not table:
            # 別のテーブルを試す
            table = soup.find('table')

        if not table:
            return {}

        for row in table.find_all('tr'):
            cols = row.find_all(['td', 'th'])
            if len(cols) < 3:
                continue

            try:
                # 馬番を探す（通常2列目か3列目）
                horse_num = None
                odds_val = None

                for i, col in enumerate(cols):
                    text = col.get_text(strip=True)

                    # 馬番（1-18の数字）
                    if horse_num is None and re.match(r'^[1-9]$|^1[0-8]$', text):
                        horse_num = int(text)

                    # 複勝オッズ（小数点を含む数値）
                    if re.match(r'^\d+\.\d+$', text):
                        val = float(text)
                        # オッズらしい範囲（1.0〜500倍程度）
                        if 1.0 <= val <= 500:
                            odds_val = val

                if horse_num and odds_val:
                    odds_dict[horse_num] = odds_val

            except:
                continue

        return odds_dict

    except Exception as e:
        return {}


def fetch_race_result(race_id):
    """1レースの結果と着順を取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        tables = soup.find_all('table')
        if len(tables) < 2:
            return None

        # Table 0: 着順結果
        result_table = tables[0]
        horses = []
        for row in result_table.find_all('tr')[1:]:
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

        # Table 1: 払戻金テーブルから複勝払戻を取得（確定オッズ）
        payout_table = tables[1]
        payout_text = payout_table.get_text()

        place_odds = {}
        place_match = re.search(r'複勝\s*([\d\s]+?)\s*([\d,円\s]+円)', payout_text)
        if place_match:
            nums_part = place_match.group(1).strip()
            odds_part = place_match.group(2)

            horse_nums = [int(x) for x in re.findall(r'\d+', nums_part)]
            payouts = re.findall(r'([\d,]+)円', odds_part)
            payouts = [int(p.replace(',', '')) for p in payouts]

            for hn, payout in zip(horse_nums, payouts):
                place_odds[hn] = payout / 100

        for h in horses:
            h['place_odds'] = place_odds.get(h['num'])

        return horses

    except Exception as e:
        return None


def fetch_race_with_pre_odds(race_id):
    """レース結果と事前オッズを両方取得"""
    url = f'https://nar.netkeiba.com/race/odds_place.html?race_id={race_id}'

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        # 複勝オッズテーブルを探す
        pre_odds = {}

        for row in soup.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 3:
                try:
                    # 馬番を探す
                    num_text = cols[0].get_text(strip=True)
                    num_match = re.search(r'^\d+$', num_text)
                    if not num_match:
                        continue
                    horse_num = int(num_text)

                    # オッズを探す（範囲形式: "2.3 - 4.5"）
                    for col in cols[1:]:
                        odds_text = col.get_text(strip=True)
                        range_match = re.search(r'([\d.]+)\s*[-~]\s*([\d.]+)', odds_text)
                        if range_match:
                            low = float(range_match.group(1))
                            high = float(range_match.group(2))
                            pre_odds[horse_num] = (low + high) / 2
                            break
                except:
                    continue

        return pre_odds

    except:
        return {}


def fetch_race_with_odds(race_id):
    """レース結果と出馬表オッズを両方取得"""
    # 結果を取得（着順）
    result = fetch_race_result(race_id)
    if not result:
        return None

    # 出馬表からオッズを取得
    odds = fetch_shutuba_odds(race_id)

    # オッズを結果に紐付け
    for h in result:
        h['pre_odds'] = odds.get(h['num'])

    return result


def find_recent_races(track_code, max_kaisai=15, max_race=12):
    """
    最近の開催を効率的に探索
    レースID形式: {年4桁}{競馬場2桁}{回2桁}{日2桁}{レース番号2桁}
    """
    track_name = TRACKS.get(track_code, track_code)
    found_races = []

    print(f"\n{track_name}競馬の最近開催を探索中...")

    # 最新年から逆順で探索
    for year in [2026, 2025]:
        # 回を逆順で（最新の開催から）
        for kai in range(12, 0, -1):
            found_any_day = False

            for day in range(1, 9):  # 通常1回の開催は1-8日程度
                test_id = f"{year}{track_code}{kai:02d}{day:02d}01"
                horses = fetch_race_with_odds(test_id)
                time.sleep(0.3)

                if not horses:
                    if found_any_day:  # 既に日があったなら次の回へ
                        break
                    continue

                found_any_day = True
                print(f"  {year}年{kai}回{day}日目", end='', flush=True)
                day_races = 0

                for race_num in range(1, max_race + 1):
                    race_id = f"{year}{track_code}{kai:02d}{day:02d}{race_num:02d}"

                    # 1Rは既に取得済み
                    if race_num == 1:
                        race_horses = horses
                    else:
                        race_horses = fetch_race_with_odds(race_id)
                        time.sleep(0.3)

                    if race_horses:
                        found_races.append({
                            'race_id': race_id,
                            'track': track_name,
                            'year': year,
                            'kai': kai,
                            'day': day,
                            'race_num': race_num,
                            'horses': race_horses
                        })
                        day_races += 1

                print(f" ({day_races}R)")

            # 十分なデータが集まったら終了
            if len(found_races) >= max_kaisai * 10:
                return found_races

    return found_races


def main():
    """メイン処理: 最近の開催を検証"""
    print("=" * 60)
    print("複勝3-5倍戦略 検証 (川崎・大井)")
    print("=" * 60)

    # 結果を競馬場別に集計
    stats = {
        '川崎': {'bets': 0, 'hits': 0, 'payout': 0, 'races': 0},
        '大井': {'bets': 0, 'hits': 0, 'payout': 0, 'races': 0}
    }

    all_races = []
    details = []  # 対象馬の詳細

    # 川崎・大井の最近開催を取得
    for track_code in ['45', '44']:  # 川崎, 大井
        races = find_recent_races(track_code, max_kaisai=15)
        all_races.extend(races)

    print(f"\n合計 {len(all_races)} レースを取得")

    # 戦略を適用（出馬表の事前オッズを使用）
    for race in all_races:
        track_name = race['track']
        stats[track_name]['races'] += 1

        for h in race['horses']:
            # 事前オッズ（出馬表から取得）を使用
            odds = h.get('pre_odds')
            if odds and 3.0 <= odds <= 5.0:
                stats[track_name]['bets'] += 1
                hit = h['rank'] <= 3

                # 払戻は確定オッズを使用（あれば）
                payout_odds = h.get('place_odds') if hit else None

                detail = {
                    'race_id': race['race_id'],
                    'track': track_name,
                    'kai': race['kai'],
                    'day': race['day'],
                    'race_num': race['race_num'],
                    'horse_name': h['name'],
                    'horse_num': h['num'],
                    'pre_odds': odds,  # 事前オッズ
                    'payout_odds': payout_odds,  # 確定オッズ
                    'rank': h['rank'],
                    'hit': hit
                }
                details.append(detail)

                if hit:
                    stats[track_name]['hits'] += 1
                    # 確定オッズがあればそれを使用、なければ事前オッズで計算
                    if payout_odds:
                        stats[track_name]['payout'] += int(payout_odds * 1000)
                    else:
                        stats[track_name]['payout'] += int(odds * 1000)

    # 結果表示
    print()
    print("=" * 60)
    print("【検証結果】複勝オッズ 3.0〜5.0倍 戦略")
    print("=" * 60)

    for track in ['川崎', '大井']:
        s = stats[track]
        if s['bets'] == 0:
            print(f"\n{track}: データなし")
            continue

        total_bet = s['bets'] * 1000
        hit_rate = s['hits'] / s['bets'] * 100
        roi = s['payout'] / total_bet * 100
        profit = s['payout'] - total_bet

        print(f"\n【{track}競馬】")
        print(f"  レース数: {s['races']}")
        print(f"  対象馬数: {s['bets']}頭")
        print(f"  的中数:   {s['hits']}頭")
        print(f"  的中率:   {hit_rate:.1f}%")
        print(f"  投資額:   {total_bet:,}円")
        print(f"  払戻額:   {s['payout']:,}円")
        print(f"  収支:     {profit:+,}円")
        print(f"  ROI:      {roi:.1f}%")

    # 合計
    total_bets = stats['川崎']['bets'] + stats['大井']['bets']
    total_hits = stats['川崎']['hits'] + stats['大井']['hits']
    total_payout = stats['川崎']['payout'] + stats['大井']['payout']

    if total_bets > 0:
        total_bet_yen = total_bets * 1000
        print(f"\n【合計】")
        print(f"  対象馬数: {total_bets}頭")
        print(f"  的中率:   {total_hits/total_bets*100:.1f}%")
        print(f"  投資額:   {total_bet_yen:,}円")
        print(f"  払戻額:   {total_payout:,}円")
        print(f"  収支:     {total_payout - total_bet_yen:+,}円")
        print(f"  ROI:      {total_payout/total_bet_yen*100:.1f}%")

    print()
    print("=" * 60)

    # 詳細をファイルに保存
    if details:
        with open('verify_result.csv', 'w', encoding='utf-8') as f:
            f.write('race_id,track,kai,day,race_num,horse_num,horse_name,pre_odds,payout_odds,rank,hit\n')
            for d in details:
                hit_mark = 'O' if d['hit'] else 'X'
                payout = d.get('payout_odds') or ''
                f.write(f"{d['race_id']},{d['track']},{d['kai']},{d['day']},{d['race_num']},"
                        f"{d['horse_num']},{d['horse_name']},{d['pre_odds']},{payout},{d['rank']},{hit_mark}\n")
        print(f"\n詳細を verify_result.csv に保存しました")

    return stats, details


if __name__ == '__main__':
    stats, details = main()
