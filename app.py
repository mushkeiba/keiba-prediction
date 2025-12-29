# 大井競馬 予測アプリ
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta
import os

# ページ設定
st.set_page_config(
    page_title="大井競馬 予測",
    page_icon="🏇",
    layout="wide"
)

# ========== スクレイパー ==========
class OhiScraper:
    """大井競馬場スクレイパー（簡易版）"""

    BASE_URL = "https://nar.netkeiba.com"
    DB_URL = "https://db.netkeiba.com"
    OHI_CODE = "44"

    def __init__(self, delay=1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.horse_cache = {}
        self.jockey_cache = {}

    def _fetch(self, url, encoding='EUC-JP'):
        time.sleep(self.delay)
        r = self.session.get(url)
        r.encoding = encoding
        return BeautifulSoup(r.text, 'lxml')

    def get_race_list_by_date(self, date: str) -> list:
        """指定日のレース一覧を取得"""
        url = f"{self.BASE_URL}/top/race_list_sub.html?kaisai_date={date}"
        try:
            soup = self._fetch(url, encoding='UTF-8')
            ids = []
            for a in soup.find_all('a', href=True):
                m = re.search(r'race_id=(\d+)', a['href'])
                if m and self.OHI_CODE in m.group(1)[4:6]:
                    ids.append(m.group(1))
            return list(set(ids))
        except:
            return []

    def get_race_data(self, race_id: str):
        """出馬表データを取得"""
        url = f"{self.BASE_URL}/race/shutuba.html?race_id={race_id}"

        try:
            soup = self._fetch(url)
            info = {'race_id': race_id}

            # レース名
            nm = soup.find('h1', class_='RaceName')
            if nm:
                info['race_name'] = nm.get_text(strip=True)

            # 距離
            rd = soup.find('div', class_='RaceData01')
            if rd:
                dm = re.search(r'(\d{3,4})m', rd.get_text())
                if dm:
                    info['distance'] = int(dm.group(1))

            # テーブル取得
            table = soup.find('table', class_='ShutubaTable')
            if not table:
                table = soup.find('table', class_='RaceTable01')
            if not table:
                for t in soup.find_all('table'):
                    if t.find('a', href=re.compile(r'/horse/')):
                        table = t
                        break
            if not table:
                return None

            rows = []
            for tr in table.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 4:
                    continue

                data = info.copy()

                # 枠番・馬番
                bracket_text = tds[0].get_text(strip=True)
                if bracket_text.isdigit():
                    data['bracket'] = int(bracket_text)
                umaban_text = tds[1].get_text(strip=True)
                if umaban_text.isdigit():
                    data['horse_number'] = int(umaban_text)

                # 馬名・馬ID
                horse_link = tr.find('a', href=re.compile(r'/horse/\d+'))
                if horse_link:
                    data['horse_name'] = horse_link.get_text(strip=True)
                    m = re.search(r'/horse/(\d+)', horse_link['href'])
                    if m:
                        data['horse_id'] = m.group(1)

                # 騎手
                jockey_link = tr.find('a', href=re.compile(r'/jockey/'))
                if jockey_link:
                    data['jockey_name'] = jockey_link.get_text(strip=True)
                    m = re.search(r'/jockey/(?:result/recent/)?([a-zA-Z0-9]+)', jockey_link['href'])
                    if m:
                        data['jockey_id'] = m.group(1)

                # 性齢・斤量
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
            return df

        except Exception as e:
            st.error(f'エラー: {e}')
            return None

    def get_horse_history(self, horse_id: str):
        """馬の過去成績を取得"""
        if horse_id in self.horse_cache:
            return self.horse_cache[horse_id]

        url = f"{self.DB_URL}/horse/ajax_horse_results.html?id={horse_id}"
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

            stats = self._calc_stats(results)
            self.horse_cache[horse_id] = stats
            return stats
        except:
            return self._empty_stats()

    def get_jockey_stats(self, jockey_id: str):
        """騎手成績を取得"""
        if jockey_id in self.jockey_cache:
            return self.jockey_cache[jockey_id]

        url = f"{self.DB_URL}/jockey/{jockey_id}/"
        try:
            soup = self._fetch(url)
            text = soup.get_text()
            stats = {'jockey_win_rate': 0, 'jockey_place_rate': 0, 'jockey_show_rate': 0}

            m = re.search(r'勝率[：:\s]*(\d+\.?\d*)', text)
            if m:
                stats['jockey_win_rate'] = float(m.group(1)) / 100
            m = re.search(r'連対率[：:\s]*(\d+\.?\d*)', text)
            if m:
                stats['jockey_place_rate'] = float(m.group(1)) / 100
            m = re.search(r'複勝率[：:\s]*(\d+\.?\d*)', text)
            if m:
                stats['jockey_show_rate'] = float(m.group(1)) / 100

            self.jockey_cache[jockey_id] = stats
            return stats
        except:
            return {'jockey_win_rate': 0, 'jockey_place_rate': 0, 'jockey_show_rate': 0}

    def _calc_stats(self, ranks):
        if not ranks:
            return self._empty_stats()
        total = len(ranks)
        wins = sum(1 for r in ranks if r == 1)
        place = sum(1 for r in ranks if r <= 2)
        show = sum(1 for r in ranks if r <= 3)
        recent = ranks[:5]
        r_total = len(recent)
        return {
            'horse_runs': total,
            'horse_win_rate': wins / total,
            'horse_place_rate': place / total,
            'horse_show_rate': show / total,
            'horse_avg_rank': np.mean(ranks),
            'horse_recent_win_rate': sum(1 for r in recent if r == 1) / r_total if r_total else 0,
            'horse_recent_show_rate': sum(1 for r in recent if r <= 3) / r_total if r_total else 0,
            'horse_recent_avg_rank': np.mean(recent) if recent else 10,
            'last_rank': ranks[0] if ranks else 10
        }

    def _empty_stats(self):
        return {
            'horse_runs': 0, 'horse_win_rate': 0, 'horse_place_rate': 0,
            'horse_show_rate': 0, 'horse_avg_rank': 10,
            'horse_recent_win_rate': 0, 'horse_recent_show_rate': 0,
            'horse_recent_avg_rank': 10, 'last_rank': 10
        }

    def enrich_data(self, df, progress_callback=None):
        """馬・騎手情報を付与"""
        df = df.copy()

        # 馬の過去成績
        if 'horse_id' in df.columns:
            horse_data = []
            unique_horses = df['horse_id'].dropna().unique()
            for i, hid in enumerate(unique_horses):
                if progress_callback:
                    progress_callback(i / len(unique_horses), f'馬情報取得中... {i+1}/{len(unique_horses)}')
                stats = self.get_horse_history(str(hid))
                stats['horse_id'] = hid
                horse_data.append(stats)
            if horse_data:
                hdf = pd.DataFrame(horse_data)
                df['horse_id'] = df['horse_id'].astype(str)
                hdf['horse_id'] = hdf['horse_id'].astype(str)
                df = df.merge(hdf, on='horse_id', how='left')

        # 騎手成績
        if 'jockey_id' in df.columns:
            jockey_data = []
            unique_jockeys = df['jockey_id'].dropna().unique()
            for i, jid in enumerate(unique_jockeys):
                if progress_callback:
                    progress_callback(0.5 + i / len(unique_jockeys) * 0.5, f'騎手情報取得中... {i+1}/{len(unique_jockeys)}')
                stats = self.get_jockey_stats(str(jid))
                stats['jockey_id'] = jid
                jockey_data.append(stats)
            if jockey_data:
                jdf = pd.DataFrame(jockey_data)
                df['jockey_id'] = df['jockey_id'].astype(str)
                jdf['jockey_id'] = jdf['jockey_id'].astype(str)
                df = df.merge(jdf, on='jockey_id', how='left')

        return df


# ========== 前処理 ==========
class Processor:
    def __init__(self):
        self.features = [
            'horse_runs', 'horse_win_rate', 'horse_place_rate', 'horse_show_rate',
            'horse_avg_rank', 'horse_recent_win_rate', 'horse_recent_show_rate',
            'horse_recent_avg_rank', 'last_rank',
            'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
            'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
            'sex_encoded', 'track_encoded', 'field_size', 'weight_diff'
        ]

    def process(self, df):
        df = df.copy()

        num_cols = ['bracket','horse_number','age','weight_carried','distance',
                    'field_size','horse_runs','horse_win_rate','horse_place_rate',
                    'horse_show_rate','horse_avg_rank','horse_recent_win_rate',
                    'horse_recent_show_rate','horse_recent_avg_rank','last_rank',
                    'jockey_win_rate','jockey_place_rate','jockey_show_rate']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'sex' in df.columns:
            df['sex_encoded'] = df['sex'].map({'牡':0,'牝':1,'セ':2}).fillna(0)
        else:
            df['sex_encoded'] = 0

        df['track_encoded'] = 0  # 出馬表では馬場不明

        if 'weight_carried' in df.columns and 'race_id' in df.columns:
            df['weight_diff'] = df.groupby('race_id')['weight_carried'].transform(lambda x: x - x.mean())
        else:
            df['weight_diff'] = 0

        if 'field_size' not in df.columns:
            if 'race_id' in df.columns:
                df['field_size'] = df.groupby('race_id')['race_id'].transform('count')
            else:
                df['field_size'] = 12

        for f in self.features:
            if f not in df.columns:
                df[f] = 0

        return df

    def get_features(self):
        return self.features


# ========== モデル読み込み ==========
@st.cache_resource
def load_model():
    """学習済みモデルを読み込み"""
    model_path = 'model_v2.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            d = pickle.load(f)
        return d['model'], d['features']
    return None, None


# ========== メイン画面 ==========
st.title('🏇 大井競馬 予測')
st.markdown('---')

# モデル読み込み
model, model_features = load_model()

if model is None:
    st.error('⚠️ モデルファイル（model_v2.pkl）が見つかりません')
    st.info('Google Colabで学習したモデルをアップロードしてください')
    st.stop()

# サイドバー
st.sidebar.header('設定')
target_date = st.sidebar.date_input(
    '予測日',
    value=datetime.now()
)
date_str = target_date.strftime('%Y%m%d')

# 予測実行ボタン
if st.sidebar.button('🔍 予測実行', type='primary'):
    scraper = OhiScraper(delay=0.5)
    processor = Processor()

    with st.spinner(f'{target_date.strftime("%Y/%m/%d")} のレースを取得中...'):
        race_ids = scraper.get_race_list_by_date(date_str)

    if not race_ids:
        st.warning('レースが見つかりません。日付を確認してください。')
    else:
        st.success(f'{len(race_ids)}レース発見！')

        # 全レース処理
        for rid in sorted(race_ids):
            with st.spinner(f'レース {rid} を処理中...'):
                df = scraper.get_race_data(rid)
                if df is None:
                    st.warning(f'{rid}: データ取得失敗')
                    continue

                # 進捗表示用
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                df = scraper.enrich_data(df, update_progress)
                progress_bar.empty()
                status_text.empty()

                df = processor.process(df)

                # 予測
                X = df[model_features].fillna(-1)
                df['prob'] = model.predict(X)
                df['pred_rank'] = df['prob'].rank(ascending=False, method='min').astype(int)
                df = df.sort_values('prob', ascending=False)

                # レース名表示
                race_name = df['race_name'].iloc[0] if 'race_name' in df.columns else rid
                st.subheader(f'🏁 {race_name}')

                # 結果表示
                cols = st.columns(3)
                for i, (_, row) in enumerate(df.head(3).iterrows()):
                    with cols[i]:
                        medal = ['🥇', '🥈', '🥉'][i]
                        num = int(row['horse_number']) if pd.notna(row.get('horse_number')) else '-'
                        st.metric(
                            label=f"{medal} {i+1}位予測",
                            value=f"{num}番 {row.get('horse_name', '?')}",
                            delta=f"確率: {row['prob']:.1%}"
                        )
                        st.caption(f"勝率: {row.get('horse_win_rate', 0)*100:.0f}% / 複勝率: {row.get('horse_show_rate', 0)*100:.0f}%")

                # 全馬一覧（折りたたみ）
                with st.expander('全馬一覧'):
                    display_df = df[['pred_rank', 'horse_number', 'horse_name', 'jockey_name', 'prob']].copy()
                    display_df.columns = ['予測順位', '馬番', '馬名', '騎手', '確率']
                    display_df['確率'] = display_df['確率'].apply(lambda x: f'{x:.1%}')
                    st.dataframe(display_df, hide_index=True)

                st.markdown('---')

# フッター
st.sidebar.markdown('---')
st.sidebar.caption('大井競馬予測システム v2')
st.sidebar.caption('データ: netkeiba.com')
