# スクレイピング方法メモ

## 戦略

**複勝オッズ3〜5倍の馬を全部買う**

- 川崎・大井両方で有効
- 過去8ヶ月で ROI 124-129%
- MLモデル不要、単純ルール

---

## レースID形式

```
{年4桁}{競馬場2桁}{回2桁}{日2桁}{レース番号2桁}
```

例: `202645010103`
- 2026 = 年
- 45 = 川崎 (44=大井)
- 01 = 第1回
- 01 = 1日目
- 03 = 3R

---

## スクレイピングコード

```python
import requests
from bs4 import BeautifulSoup
import re
import time

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def fetch_race(race_id):
    """1レースの結果と複勝オッズを取得"""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = 'euc-jp'  # 重要！
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
            rank = int(re.search(r'\d+', cols[0].get_text()).group())
            horse_num = int(cols[2].get_text(strip=True))
            horse_name = cols[3].get_text(strip=True)
            horses.append({
                'num': horse_num,
                'name': horse_name,
                'rank': rank
            })
        except:
            continue

    # Table 1: 払戻金
    # 形式: 「複勝  3  8  5    1,200円300円110円」
    payout_table = tables[1]
    payout_text = payout_table.get_text()

    place_match = re.search(r'複勝\s*([\d\s]+)\s*([\d,円]+)', payout_text)
    place_odds = {}

    if place_match:
        nums_part = place_match.group(1).strip()
        odds_part = place_match.group(2)

        horse_nums = [int(x) for x in re.findall(r'\d+', nums_part)]
        payouts = re.findall(r'([\d,]+)円', odds_part)
        payouts = [int(p.replace(',', '')) for p in payouts]

        for hn, payout in zip(horse_nums, payouts):
            place_odds[hn] = payout / 100  # 円→倍率

    # 紐付け
    for h in horses:
        h['place_odds'] = place_odds.get(h['num'])

    return horses


def fetch_day(date_str, track='kawasaki'):
    """1日分の全レースを取得"""
    track_code = '45' if track == 'kawasaki' else '44'

    all_results = []

    for race_num in range(1, 13):
        race_id = f'{date_str[:4]}{track_code}0101{race_num:02d}'
        horses = fetch_race(race_id)

        if horses:
            for h in horses:
                h['race'] = race_num
                all_results.append(h)

        time.sleep(0.3)

    return all_results


def apply_strategy(results):
    """複勝3-5倍戦略を適用"""
    targets = [r for r in results
               if r.get('place_odds') and 3.0 <= r['place_odds'] <= 5.0]

    if not targets:
        print('対象馬なし')
        return

    total_bet = len(targets) * 1000
    total_payout = 0

    for t in targets:
        hit = t['rank'] <= 3
        payout = t['place_odds'] * 1000 if hit else 0
        total_payout += payout
        mark = '◎' if hit else '×'
        print(f"{t['race']:2d}R {t['name'][:12]:12s} "
              f"複勝{t['place_odds']:.1f}倍 → {t['rank']}着 {mark}")

    print()
    print(f'賭け: {total_bet:,}円')
    print(f'払戻: {total_payout:,.0f}円')
    print(f'収支: {total_payout - total_bet:+,.0f}円')


# 使用例
if __name__ == '__main__':
    results = fetch_day('20260101', 'kawasaki')
    apply_strategy(results)
```

---

## 注意点

1. **エンコーディング**: `r.encoding = 'euc-jp'` 必須
2. **レート制限**: `time.sleep(0.3)` を入れる
3. **テーブル構造**:
   - Table 0 = 着順結果
   - Table 1 = 払戻金（単勝・複勝・枠連など）
4. **複勝の形式**: 「複勝  馬番1 馬番2 馬番3  払戻1円払戻2円払戻3円」

---

## 確認用URL

```
https://nar.netkeiba.com/race/result.html?race_id={race_id}
```

---

## 過去検証結果

川崎（2025/05-12）: 573件 | 的中33.9% | ROI 128.7% | +16,450円
大井（2025/05-12）: 837件 | 的中32.5% | ROI 124.2% | +20,283円

---

## 2026/01/01 川崎 実績

| R | 馬名 | 複勝 | 着順 |
|---|------|------|------|
| 3R | シシフンジン | 3.0倍 | 2着 ◎ |
| 4R | ヘウレシス | 4.5倍 | 1着 ◎ |
| 9R | ミトノキャット | 3.8倍 | 1着 ◎ |

**3頭全的中: +8,300円**
