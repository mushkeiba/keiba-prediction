#!/usr/bin/env python3
"""予測の安定性を調査するスクリプト"""
import pickle
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time

print('=== 大井 1R の予測を2回実行して比較 ===')
print()

# モデル読み込み
with open('models/model_ohi.pkl', 'rb') as f:
    d = pickle.load(f)
model = d['model']
model_features = d['features']

# 簡易スクレイパー
class SimpleScraper:
    BASE_URL = 'https://nar.netkeiba.com'
    DB_URL = 'https://db.netkeiba.com'

    def __init__(self, delay=0.5):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.horse_cache = {}

    def _fetch(self, url, encoding='EUC-JP'):
        time.sleep(self.delay)
        r = self.session.get(url)
        r.encoding = encoding
        return BeautifulSoup(r.text, 'lxml')

    def get_race_list(self, date):
        url = f'{self.BASE_URL}/top/race_list_sub.html?kaisai_date={date}'
        soup = self._fetch(url, 'UTF-8')
        ids = []
        for a in soup.find_all('a', href=True):
            m = re.search(r'race_id=(\d+)', a['href'])
            if m:
                rid = m.group(1)
                if len(rid) >= 6 and rid[4:6] == '44':
                    ids.append(rid)
        return list(set(ids))

    def get_race_data(self, race_id):
        url = f'{self.BASE_URL}/race/shutuba.html?race_id={race_id}'
        soup = self._fetch(url)
        info = {'race_id': race_id}

        table = soup.find('table', class_='ShutubaTable')
        if not table:
            return None

        rows = []
        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 4:
                continue

            data = info.copy()

            bracket_text = tds[0].get_text(strip=True)
            if bracket_text.isdigit():
                data['bracket'] = int(bracket_text)
            umaban_text = tds[1].get_text(strip=True)
            if umaban_text.isdigit():
                data['horse_number'] = int(umaban_text)

            horse_link = tr.find('a', href=re.compile(r'/horse/\d+'))
            if horse_link:
                data['horse_name'] = horse_link.get_text(strip=True)
                m = re.search(r'/horse/(\d+)', horse_link['href'])
                if m:
                    data['horse_id'] = m.group(1)

            # 馬体重
            for td in tds:
                wt = td.get_text(strip=True)
                wm = re.match(r'^(\d{3,4})(?:\(([+-]?\d+)\))?$', wt)
                if wm and 300 <= int(wm.group(1)) <= 600:
                    data['horse_weight'] = int(wm.group(1))
                    data['weight_change'] = int(wm.group(2)) if wm.group(2) else 0
                    break

            for td in tds:
                text = td.get_text(strip=True)
                if re.match(r'^[牡牝セ]\d$', text):
                    data['sex'] = text[0]
                    data['age'] = int(text[1])
                if re.match(r'^\d{2}(\.\d)?$', text):
                    w = float(text)
                    if 45 <= w <= 65 and 'weight_carried' not in data:
                        data['weight_carried'] = w

            if data.get('horse_name'):
                rows.append(data)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df['field_size'] = len(df)
        df['distance'] = 1600
        return df

    def get_horse_history(self, horse_id):
        if horse_id in self.horse_cache:
            return self.horse_cache[horse_id]

        url = f'{self.DB_URL}/horse/ajax_horse_results.html?id={horse_id}'
        try:
            time.sleep(self.delay)
            r = self.session.get(url)
            r.encoding = 'EUC-JP'
            soup = BeautifulSoup(r.text, 'lxml')

            results = []
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 6:
                    continue
                for td in tds[3:7]:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 20:
                        results.append(int(t))
                        break
                if len(results) >= 20:
                    break

            if not results:
                return self._empty_stats()

            total = len(results)
            recent = results[:5]
            stats = {
                'horse_runs': total,
                'horse_win_rate': sum(1 for r in results if r == 1) / total,
                'horse_place_rate': sum(1 for r in results if r <= 2) / total,
                'horse_show_rate': sum(1 for r in results if r <= 3) / total,
                'horse_avg_rank': np.mean(results),
                'horse_recent_win_rate': sum(1 for r in recent if r == 1) / len(recent) if recent else 0,
                'horse_recent_show_rate': sum(1 for r in recent if r <= 3) / len(recent) if recent else 0,
                'horse_recent_avg_rank': np.mean(recent) if recent else 10,
                'last_rank': results[0] if results else 10
            }
            self.horse_cache[horse_id] = stats
            return stats
        except Exception as e:
            print(f'  [WARN] 馬成績取得失敗: {horse_id} - {e}')
            return self._empty_stats()

    def _empty_stats(self):
        return {
            'horse_runs': 0, 'horse_win_rate': 0, 'horse_place_rate': 0,
            'horse_show_rate': 0, 'horse_avg_rank': 10,
            'horse_recent_win_rate': 0, 'horse_recent_show_rate': 0,
            'horse_recent_avg_rank': 10, 'last_rank': 10
        }

    def enrich_data(self, df):
        df = df.copy()
        if 'horse_id' in df.columns:
            horse_data = []
            for hid in df['horse_id'].dropna().unique():
                stats = self.get_horse_history(str(hid))
                stats['horse_id'] = hid
                horse_data.append(stats)
            if horse_data:
                hdf = pd.DataFrame(horse_data)
                df['horse_id'] = df['horse_id'].astype(str)
                hdf['horse_id'] = hdf['horse_id'].astype(str)
                df = df.merge(hdf, on='horse_id', how='left')
        return df


def process(df):
    df = df.copy()
    if 'sex' in df.columns:
        df['sex_encoded'] = df['sex'].map({'牡':0,'牝':1,'セ':2}).fillna(0)
    else:
        df['sex_encoded'] = 0
    df['track_encoded'] = 0
    df['track_condition_encoded'] = 0
    df['weather_encoded'] = 0
    df['trainer_encoded'] = 0
    if 'horse_weight' not in df.columns:
        df['horse_weight'] = 450
    else:
        df['horse_weight'] = df['horse_weight'].fillna(450)
    if 'weight_change' not in df.columns:
        df['horse_weight_change'] = 0
    else:
        df['horse_weight_change'] = df['weight_change'].fillna(0)
    if 'weight_carried' in df.columns:
        df['weight_diff'] = df['weight_carried'] - df['weight_carried'].mean()
    else:
        df['weight_diff'] = 0
    for f in model_features:
        if f not in df.columns:
            df[f] = 0
    return df


# 1回目の予測
print('【1回目の予測】')
scraper1 = SimpleScraper(delay=0.3)
race_ids = scraper1.get_race_list('20251231')
if not race_ids:
    print('レースが見つかりません')
    exit()

race_id = sorted(race_ids)[0]
print(f'レースID: {race_id}')

df1 = scraper1.get_race_data(race_id)
print(f'出走頭数: {len(df1)}')
df1 = scraper1.enrich_data(df1)
df1 = process(df1)

X1 = df1[model_features].fillna(-1)
df1['prob'] = model.predict(X1)
df1 = df1.sort_values('prob', ascending=False).reset_index(drop=True)

print('予測順位 (確率):')
for i in range(min(5, len(df1))):
    row = df1.iloc[i]
    print(f'  {i+1}位: {int(row["horse_number"]):2d}番 {row["horse_name"]:10s} ({row["prob"]:.4f})')

print()
print('【2回目の予測（5秒後・キャッシュなし）】')
time.sleep(5)

scraper2 = SimpleScraper(delay=0.3)  # 新しいインスタンス（キャッシュなし）
df2 = scraper2.get_race_data(race_id)
df2 = scraper2.enrich_data(df2)
df2 = process(df2)

X2 = df2[model_features].fillna(-1)
df2['prob'] = model.predict(X2)
df2 = df2.sort_values('prob', ascending=False).reset_index(drop=True)

print('予測順位 (確率):')
for i in range(min(5, len(df2))):
    row = df2.iloc[i]
    print(f'  {i+1}位: {int(row["horse_number"]):2d}番 {row["horse_name"]:10s} ({row["prob"]:.4f})')

# 比較
print()
print('='*50)
print('=== 結果比較 ===')
print('='*50)

order1 = df1['horse_number'].tolist()[:5]
order2 = df2['horse_number'].tolist()[:5]
if order1 == order2:
    print('上位5頭: ✅ 変化なし')
else:
    print('上位5頭: ⚠️ 変化あり!')
    print(f'  1回目: {order1}')
    print(f'  2回目: {order2}')

print()
print('=== 入力データの差分 ===')
df1_by_num = df1.set_index('horse_number')
df2_by_num = df2.set_index('horse_number')

important = ['horse_weight', 'horse_avg_rank', 'horse_recent_avg_rank', 'last_rank', 'horse_runs', 'horse_weight_change']
diff_found = False
for feat in important:
    if feat in df1_by_num.columns and feat in df2_by_num.columns:
        for num in df1_by_num.index:
            if num in df2_by_num.index:
                v1 = df1_by_num.loc[num, feat]
                v2 = df2_by_num.loc[num, feat]
                if pd.notna(v1) and pd.notna(v2) and abs(float(v1) - float(v2)) > 0.001:
                    diff_found = True
                    print(f'  {feat} 馬番{num}: {v1:.2f} → {v2:.2f}')

if not diff_found:
    print('  差異なし（同一データ）')

print()
print('=== 予測確率の分布 ===')
probs = df1['prob'].values
print(f'最大: {probs.max():.4f}')
print(f'最小: {probs.min():.4f}')
print(f'平均: {probs.mean():.4f}')
print(f'範囲: {probs.max() - probs.min():.4f}')
print()
print('順位間の確率差:')
sorted_probs = sorted(probs, reverse=True)
for i in range(min(4, len(sorted_probs)-1)):
    diff = sorted_probs[i] - sorted_probs[i+1]
    warning = '⚠️ 僅差!' if diff < 0.01 else ''
    print(f'  {i+1}位-{i+2}位: {diff:.4f} {warning}')
