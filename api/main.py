# 地方競馬 予測API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from datetime import datetime
import os
import json
from pathlib import Path
from collections import defaultdict
import asyncio

# ========== モデル用カスタムクラス ==========
# pickleでモデルを読み込む際に必要
class TargetEncoderSafe:
    """リークしないTarget Encoder（学習データのみで統計作成）"""

    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.global_mean = None
        self.mappings = {}

    def fit(self, train_df, cols, target):
        """学習データのみで統計を作成"""
        self.global_mean = train_df[target].mean()

        for col in cols:
            stats = train_df.groupby(col)[target].agg(['mean', 'count'])
            smooth_mean = (stats['mean'] * stats['count'] + self.global_mean * self.smoothing) / \
                         (stats['count'] + self.smoothing)
            self.mappings[col] = smooth_mean.to_dict()

        return self

    def transform(self, df, cols):
        """学習データの統計でテストデータを変換"""
        df = df.copy()
        for col in cols:
            te_col = f'{col}_te'
            df[te_col] = df[col].map(self.mappings.get(col, {})).fillna(self.global_mean)
        return df


def create_features_v3(df):
    """
    v3特徴量作成（人気ベース - 的中率77%達成）
    市場の知恵（オッズ）を活用したアプローチ
    """
    df = df.copy()

    # 数値変換
    num_cols = ['horse_win_rate', 'horse_show_rate', 'last_rank',
                'jockey_win_rate', 'field_size', 'win_odds', 'last_3f']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 人気（最重要特徴量）
    if 'win_odds' in df.columns:
        df['popularity'] = df.groupby('race_id')['win_odds'].rank(ascending=True)
        df['odds_implied_prob'] = 1 / df['win_odds'].clip(lower=1)
    else:
        df['popularity'] = 5
        df['odds_implied_prob'] = 0.1

    # 人気に対する実力の乖離
    if 'horse_show_rate' in df.columns:
        df['show_rate_rank'] = df.groupby('race_id')['horse_show_rate'].rank(ascending=False)
        df['value_gap'] = df['show_rate_rank'] - df['popularity']
    else:
        df['value_gap'] = 0

    # 上がり3F順位
    if 'last_3f' in df.columns:
        df['last_3f_rank'] = df.groupby('race_id')['last_3f'].rank(ascending=True)
    else:
        df['last_3f_rank'] = 5

    # 特徴量リスト（v3モデルと同じ順序）
    features = [
        'popularity', 'odds_implied_prob', 'value_gap',
        'horse_show_rate', 'jockey_win_rate', 'last_rank',
        'last_3f_rank', 'field_size'
    ]

    # 欠損埋め
    defaults = {
        'popularity': 5, 'odds_implied_prob': 0.1, 'value_gap': 0,
        'horse_show_rate': 0.27, 'jockey_win_rate': 0.1, 'last_rank': 5,
        'last_3f_rank': 5, 'field_size': 11
    }
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(defaults.get(f, 0))
        else:
            df[f] = defaults.get(f, 0)

    return df, features


def create_features_v8(df):
    """
    v8特徴量作成（オッズ除外 + 過去データのみ）
    - 回収率108-110%達成
    - データリークなし
    """
    df = df.copy()

    # 数値変換
    num_cols = [
        'horse_runs', 'horse_win_rate', 'horse_show_rate', 'horse_avg_rank',
        'horse_recent_win_rate', 'horse_recent_show_rate', 'horse_recent_avg_rank',
        'last_rank', 'jockey_win_rate', 'jockey_show_rate',
        'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
        'field_size', 'horse_weight', 'weight_change'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- 過去スピード指数（推論時は過去データから計算済みの値を使用） ---
    # 推論時はスクレイピングした過去成績から計算
    if 'past_speed_index' not in df.columns:
        df['past_speed_index'] = 50  # デフォルト
    if 'past_3_speed_index' not in df.columns:
        df['past_3_speed_index'] = 50

    # --- 過去の上がり3F ---
    if 'past_last_3f' not in df.columns:
        df['past_last_3f'] = 40  # デフォルト
    if 'past_3_last_3f' not in df.columns:
        df['past_3_last_3f'] = 40

    # --- 前走経過日数 ---
    if 'days_since_last' not in df.columns:
        df['days_since_last'] = 30  # デフォルト

    # --- レース内での相対順位 ---
    df['show_rate_rank'] = df.groupby('race_id')['horse_show_rate'].rank(ascending=False)
    df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False)
    df['jockey_rank'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)
    df['avg_rank_rank'] = df.groupby('race_id')['horse_avg_rank'].rank(ascending=True)
    df['past_speed_rank'] = df.groupby('race_id')['past_speed_index'].rank(ascending=False)
    df['past_3f_rank'] = df.groupby('race_id')['past_last_3f'].rank(ascending=True)

    # --- レース内での相対値 ---
    df['show_rate_vs_field'] = df['horse_show_rate'] - df.groupby('race_id')['horse_show_rate'].transform('mean')
    df['win_rate_vs_field'] = df['horse_win_rate'] - df.groupby('race_id')['horse_win_rate'].transform('mean')
    df['jockey_vs_field'] = df['jockey_win_rate'] - df.groupby('race_id')['jockey_win_rate'].transform('mean')
    df['past_speed_vs_field'] = df['past_speed_index'] - df.groupby('race_id')['past_speed_index'].transform('mean')

    # --- 経験値スコア ---
    df['experience_score'] = np.log1p(df['horse_runs']) * df['horse_show_rate']

    # --- 調子スコア ---
    df['form_score'] = df['horse_recent_show_rate'].fillna(df['horse_show_rate'])
    df['form_trend'] = df['form_score'] - df['horse_show_rate']

    # --- 前走の成績 ---
    df['last_rank_score'] = np.where(df['last_rank'] <= 3, 1, 0)
    df['last_rank_normalized'] = df['last_rank'] / df['field_size'].clip(lower=1)

    # --- 馬場 ---
    condition_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    if 'track_condition' in df.columns:
        df['track_condition_code'] = df['track_condition'].map(condition_map).fillna(0)
    else:
        df['track_condition_code'] = 0

    # --- 休み明け効果 ---
    df['is_fresh'] = (df['days_since_last'] >= 30).astype(int)
    df['is_long_rest'] = (df['days_since_last'] >= 60).astype(int)

    # --- 特徴量リスト ---
    features = [
        # 相対順位
        'show_rate_rank', 'win_rate_rank', 'jockey_rank', 'avg_rank_rank',
        'past_speed_rank', 'past_3f_rank',
        # 相対値
        'show_rate_vs_field', 'win_rate_vs_field', 'jockey_vs_field',
        'past_speed_vs_field',
        # 実績
        'horse_show_rate', 'horse_win_rate', 'horse_avg_rank',
        'jockey_win_rate', 'jockey_show_rate',
        # 経験・調子
        'experience_score', 'form_score', 'form_trend', 'horse_runs',
        # 前走
        'last_rank', 'last_rank_score', 'last_rank_normalized',
        # 過去のスピード・タイム
        'past_speed_index', 'past_3_speed_index',
        'past_last_3f', 'past_3_last_3f',
        # 経過日数
        'days_since_last', 'is_fresh', 'is_long_rest',
        # その他
        'field_size', 'age', 'horse_number', 'track_condition_code',
        'weight_carried', 'horse_weight'
    ]

    # デフォルト値
    defaults = {
        'show_rate_rank': 5, 'win_rate_rank': 5, 'jockey_rank': 5, 'avg_rank_rank': 5,
        'past_speed_rank': 5, 'past_3f_rank': 5,
        'show_rate_vs_field': 0, 'win_rate_vs_field': 0, 'jockey_vs_field': 0,
        'past_speed_vs_field': 0,
        'horse_show_rate': 0.27, 'horse_win_rate': 0.1, 'horse_avg_rank': 5,
        'jockey_win_rate': 0.1, 'jockey_show_rate': 0.27,
        'experience_score': 0.5, 'form_score': 0.27, 'form_trend': 0, 'horse_runs': 10,
        'last_rank': 5, 'last_rank_score': 0, 'last_rank_normalized': 0.5,
        'past_speed_index': 50, 'past_3_speed_index': 50,
        'past_last_3f': 40, 'past_3_last_3f': 40,
        'days_since_last': 30, 'is_fresh': 0, 'is_long_rest': 0,
        'field_size': 11, 'age': 4, 'horse_number': 5, 'track_condition_code': 0,
        'weight_carried': 55, 'horse_weight': 470
    }

    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(defaults.get(f, 0))
        else:
            df[f] = defaults.get(f, 0)

    return df, features


# プロジェクトのルートディレクトリを取得
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="地方競馬予測API",
    description="AIが予測する地方競馬の3着以内予測",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 競馬場設定 ==========
# v8モデル（オッズ除外・閾値フィルタリング対応）を優先使用
TRACKS = {
    "44": {"name": "大井", "model": "models/model_ohi_v8.pkl", "emoji": "🏟️"},
    "45": {"name": "川崎", "model": "models/model_kawasaki_v10.pkl", "emoji": "🌊"},
    "43": {"name": "船橋", "model": "models/model_funabashi.pkl", "emoji": "⚓"},
    "42": {"name": "浦和", "model": "models/model_urawa.pkl", "emoji": "🌸"},
    "30": {"name": "門別", "model": "models/model_monbetsu.pkl", "emoji": "🐴"},
    "35": {"name": "盛岡", "model": "models/model_morioka.pkl", "emoji": "⛰️"},
    "36": {"name": "水沢", "model": "models/model_mizusawa.pkl", "emoji": "💧"},
    "46": {"name": "金沢", "model": "models/model_kanazawa.pkl", "emoji": "✨"},
    "47": {"name": "笠松", "model": "models/model_kasamatsu.pkl", "emoji": "🎋"},
    "48": {"name": "名古屋", "model": "models/model_nagoya.pkl", "emoji": "🏯"},
    "50": {"name": "園田", "model": "models/model_sonoda.pkl", "emoji": "🌳"},
    "51": {"name": "姫路", "model": "models/model_himeji.pkl", "emoji": "🏰"},
    "54": {"name": "高知", "model": "models/model_kochi.pkl", "emoji": "🐋"},
    "55": {"name": "佐賀", "model": "models/model_saga.pkl", "emoji": "🎋"},
}

# モデルキャッシュ
model_cache = {}

# ========== v6選択的ベッティング設定 ==========
# バックテスト結果: prob_diff >= 20% で100%超ROI達成
# prob_diff = 予測1位の確率 - 予測2位の確率

# 競馬場別の推奨フィルター設定
SELECTIVE_BETTING_CONFIG = {
    "44": {  # 大井
        "min_prob_diff": 0.20,  # 確率差20%以上
        "expected_roi": 1.057,  # 期待ROI 105.7%
        "hit_rate": 0.596,      # 的中率 59.6%
    },
    "45": {  # 川崎
        "min_prob_diff": 0.20,  # 確率差20%以上
        "expected_roi": 1.147,  # 期待ROI 114.7%
        "hit_rate": 0.649,      # 的中率 64.9%
    },
    # その他は保守的設定
    "default": {
        "min_prob_diff": 0.15,
        "expected_roi": 1.0,
        "hit_rate": 0.55,
    }
}

def get_betting_config(track_code: str) -> dict:
    """競馬場の選択的ベッティング設定を取得"""
    return SELECTIVE_BETTING_CONFIG.get(track_code, SELECTIVE_BETTING_CONFIG["default"])

# 旧設定との互換性（他の箇所で参照されている場合用）
MIN_PLACE_ODDS_FOR_ROI = {
    "44": 1.5, "45": 1.8, "43": 2.0, "42": 2.0, "30": 2.0,
    "35": 2.0, "36": 2.0, "46": 2.0, "47": 2.0, "48": 2.0,
    "50": 2.0, "51": 2.0, "54": 2.0, "55": 2.0,
}

# 賭け金計算（期待値に応じた可変金額）
def calculate_bet_amount(expected_value: float, base_bet: int = 100) -> int:
    """期待値に応じた賭け金を計算"""
    if expected_value <= 1.0:
        return 0  # 期待値1.0以下は買わない
    elif expected_value <= 1.2:
        return base_bet  # 100円
    elif expected_value <= 1.5:
        return base_bet * 2  # 200円
    elif expected_value <= 2.0:
        return base_bet * 3  # 300円
    else:
        return base_bet * 5  # 500円（期待値2.0超）

# ========== 予測ログ保存 ==========
def save_prediction_log(race_id: str, track_code: str, predictions: list, metadata: dict = None):
    """予測結果をJSONに保存（後で結果と照合するため）"""
    try:
        # 日付を抽出（race_idから）
        date_str = race_id[:4] + "-" + race_id[6:8] + "-" + race_id[8:10]

        # ディレクトリ作成
        log_dir = BASE_DIR / "prediction_logs" / date_str
        log_dir.mkdir(parents=True, exist_ok=True)

        # ログデータ作成
        log_data = {
            "race_id": race_id,
            "track_code": track_code,
            "track_name": TRACKS.get(track_code, {}).get("name", "不明"),
            "predicted_at": datetime.now().isoformat(),
            "predictions": predictions,
            "metadata": metadata or {}
        }

        # レースごとにファイル保存
        log_file = log_dir / f"{race_id}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"Prediction log saved: {log_file}")
    except Exception as e:
        print(f"Failed to save prediction log: {e}")

# 旧モデル名との互換性（v8モデルがなければ旧モデルにフォールバック）
MODEL_ALIASES = {
    "models/model_ohi_v8.pkl": ["models/model_ohi_v8.pkl", "models/model_ohi.pkl"],
    "models/model_kawasaki_v8.pkl": ["models/model_kawasaki_v8.pkl", "models/model_kawasaki.pkl"],
    "models/model_ohi.pkl": ["models/model_ohi.pkl", "model_v2.pkl"],
}


# ========== スクレイピング対策ヘルパー ==========
# User-Agentリスト（実際のブラウザから取得）
SCRAPER_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]

def create_scraper_session():
    """スクレイピング対策済みセッションを作成"""
    session = requests.Session()
    ua = random.choice(SCRAPER_USER_AGENTS)
    session.headers.update({
        'User-Agent': ua,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    })
    return session

def fetch_with_retry(url, encoding='EUC-JP', retries=3, delay=0.3):
    """リトライ機能付きフェッチ（スタンドアロン関数用）"""
    time.sleep(delay + random.uniform(0, 0.3))

    for attempt in range(retries):
        try:
            session = create_scraper_session()
            r = session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = encoding
            return BeautifulSoup(r.text, 'lxml')
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                return None
    return None


# ========== スクレイパー ==========
class NARScraper:
    BASE_URL = "https://nar.netkeiba.com"
    DB_URL = "https://db.netkeiba.com"

    # User-Agentローテーション用リスト（実際のブラウザから取得）
    USER_AGENTS = [
        # Chrome (Windows)
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        # Chrome (Mac)
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Firefox (Windows)
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        # Safari (Mac)
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        # Edge (Windows)
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    ]

    def __init__(self, track_code, delay=0.5):
        self.track_code = track_code
        self.delay = delay
        self.session = requests.Session()
        self.session.verify = False  # SSL証明書検証をスキップ
        self.horse_cache = {}
        self.jockey_cache = {}
        self._request_count = 0
        # SSL警告を抑制
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # 初期ヘッダー設定
        self._update_headers()

    def _update_headers(self, referer=None):
        """リクエストごとにヘッダーを更新（ブラウザを模倣）"""
        ua = random.choice(self.USER_AGENTS)
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin' if referer else 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        if referer:
            headers['Referer'] = referer
        self.session.headers.update(headers)

    def _random_delay(self):
        """ランダムな遅延（人間らしいアクセス間隔を模倣）"""
        # 基本遅延 ± 30%のランダム + 時々長めの休憩
        jitter = self.delay * random.uniform(0.7, 1.3)
        # 10リクエストごとに少し長めの休憩
        if self._request_count > 0 and self._request_count % 10 == 0:
            jitter += random.uniform(1.0, 2.0)
        time.sleep(jitter)
        self._request_count += 1

    def _fetch(self, url, encoding='EUC-JP', retries=3):
        """リトライ機能付きフェッチ"""
        referer = f"{self.BASE_URL}/" if self.BASE_URL in url else None
        self._update_headers(referer)
        self._random_delay()

        for attempt in range(retries):
            try:
                r = self.session.get(url, timeout=30)
                r.raise_for_status()
                r.encoding = encoding
                return BeautifulSoup(r.text, 'lxml')
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    # 指数バックオフ
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    self._update_headers(referer)  # ヘッダー更新
                else:
                    raise e
        return None

    def get_race_list_by_date(self, date: str) -> list:
        url = f"{self.BASE_URL}/top/race_list_sub.html?kaisai_date={date}"
        try:
            soup = self._fetch(url, encoding='UTF-8')
            ids = []
            for a in soup.find_all('a', href=True):
                m = re.search(r'race_id=(\d+)', a['href'])
                if m:
                    race_id = m.group(1)
                    if len(race_id) >= 6 and race_id[4:6] == self.track_code:
                        ids.append(race_id)
            return list(set(ids))
        except:
            return []

    def get_race_data(self, race_id: str):
        url = f"{self.BASE_URL}/race/shutuba.html?race_id={race_id}"
        try:
            soup = self._fetch(url)
            info = {'race_id': race_id}

            # レース名
            nm = soup.find('h1', class_='RaceName')
            if nm:
                info['race_name'] = nm.get_text(strip=True)

            # 発走時刻
            rd = soup.find('div', class_='RaceData01')
            if rd:
                rd_text = rd.get_text()
                tm = re.search(r'(\d{1,2}):(\d{2})', rd_text)
                if tm:
                    info['start_time'] = f"{tm.group(1)}:{tm.group(2)}"
                dm = re.search(r'(\d{3,4})m', rd_text)
                if dm:
                    info['distance'] = int(dm.group(1))

                # 馬場状態を抽出（良/稍重/重/不良）
                track_cond_match = re.search(r'[ダ芝].*?[:：]\s*(良|稍重|重|不良)', rd_text)
                if track_cond_match:
                    info['track_condition'] = track_cond_match.group(1)
                else:
                    info['track_condition'] = '良'

                # 天気を抽出（晴/曇/雨/小雨/雪）
                weather_match = re.search(r'天気[:：]\s*(晴|曇|雨|小雨|雪)', rd_text)
                if weather_match:
                    info['weather'] = weather_match.group(1)
                else:
                    info['weather'] = '晴'

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

                jockey_link = tr.find('a', href=re.compile(r'/jockey/'))
                if jockey_link:
                    data['jockey_name'] = jockey_link.get_text(strip=True)
                    m = re.search(r'/jockey/(?:result/recent/)?([a-zA-Z0-9]+)', jockey_link['href'])
                    if m:
                        data['jockey_id'] = m.group(1)

                # 調教師を抽出
                trainer_link = tr.find('a', href=re.compile(r'/trainer/'))
                if trainer_link:
                    data['trainer_name'] = trainer_link.get_text(strip=True)
                    m = re.search(r'/trainer/(?:result/recent/)?([a-zA-Z0-9]+)', trainer_link['href'])
                    if m:
                        data['trainer_id'] = m.group(1)

                # 馬体重を抽出（例: 450(+4), 448(-2), 452）
                for td in tds:
                    weight_text = td.get_text(strip=True)
                    weight_match = re.match(r'^(\d{3,4})(?:\(([+-]?\d+)\))?$', weight_text)
                    if weight_match and 300 <= int(weight_match.group(1)) <= 600:
                        data['horse_weight'] = int(weight_match.group(1))
                        if weight_match.group(2):
                            data['weight_change'] = int(weight_match.group(2))
                        else:
                            data['weight_change'] = 0
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
            return df
        except Exception as e:
            print(f'Error: {e}')

    def get_all_odds(self, race_id: str) -> dict:
        """単勝・複勝オッズを一括取得（API呼び出し最小化）"""
        result = {'win': {}, 'place': {}}

        # 1. 出馬表ページから単勝オッズを取得（1リクエスト目）
        shutuba_url = f"{self.BASE_URL}/race/shutuba.html?race_id={race_id}"
        try:
            soup = self._fetch(shutuba_url)
            table = soup.find('table', class_='ShutubaTable')
            if not table:
                table = soup.find('table', class_='RaceTable01')

            if table:
                for tr in table.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) >= 2:
                        umaban = None
                        odds_val = None

                        for i, td in enumerate(tds[:3]):
                            td_class = ' '.join(td.get('class', []))
                            text = td.get_text(strip=True)
                            if 'Umaban' in td_class or (i == 1 and text.isdigit()):
                                if text.isdigit() and 1 <= int(text) <= 18:
                                    umaban = int(text)
                                    break

                        for td in tds:
                            td_class = ' '.join(td.get('class', []))
                            if 'Popular' in td_class or 'Odds' in td_class or 'odds' in td_class.lower():
                                text = td.get_text(strip=True)
                                odds_match = re.search(r'(\d+\.?\d*)', text)
                                if odds_match:
                                    val = float(odds_match.group(1))
                                    if 1.0 <= val <= 999.9:
                                        odds_val = val
                                        break

                        if umaban and odds_val:
                            result['win'][umaban] = odds_val
        except Exception as e:
            print(f'Win odds error: {e}')

        # 2. 複勝オッズページから取得（2リクエスト目）
        place_url = f"{self.BASE_URL}/odds/odds_get_form.html?type=b2&race_id={race_id}"
        try:
            soup = self._fetch(place_url)
            tables = soup.find_all('table')
            if len(tables) >= 2:
                table = tables[1]
                for tr in table.find_all('tr'):
                    tds = tr.find_all('td')
                    # td構造: [枠番, 馬番, 空, 馬名, オッズ]
                    if len(tds) >= 5:
                        umaban_text = tds[1].get_text(strip=True)  # td[1]が馬番
                        if umaban_text.isdigit():
                            umaban = int(umaban_text)
                            # オッズは最後のtd
                            odds_text = tds[-1].get_text(strip=True)
                            # 「1.4 - 2.6」形式をパース
                            odds_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', odds_text)
                            if odds_match:
                                min_odds = float(odds_match.group(1))
                                max_odds = float(odds_match.group(2))
                                result['place'][umaban] = {
                                    'min': min_odds,
                                    'max': max_odds,
                                    'avg': round((min_odds + max_odds) / 2, 2)
                                }
                            else:
                                # 単一の数値の場合
                                single_match = re.search(r'(\d+\.?\d*)', odds_text)
                                if single_match:
                                    odds_val = float(single_match.group(1))
                                    result['place'][umaban] = {
                                        'min': odds_val,
                                        'max': odds_val,
                                        'avg': odds_val
                                    }
        except Exception as e:
            print(f'Place odds error: {e}')

        return result

    def get_odds(self, race_id: str, horse_names: list = None) -> dict:
        """単勝オッズを取得（出馬表の予想オッズ列から）"""
        odds_dict = {}

        # 1. 出馬表ページから予想オッズを取得（最も確実）
        shutuba_url = f"{self.BASE_URL}/race/shutuba.html?race_id={race_id}"
        try:
            soup = self._fetch(shutuba_url)
            table = soup.find('table', class_='ShutubaTable')
            if not table:
                table = soup.find('table', class_='RaceTable01')

            if table:
                for tr in table.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) >= 2:
                        # 馬番は通常2番目のtd（1番目は枠番）
                        umaban = None
                        odds_val = None

                        # 馬番を取得（Umabanクラスまたは2番目のtd）
                        for i, td in enumerate(tds[:3]):
                            td_class = ' '.join(td.get('class', []))
                            text = td.get_text(strip=True)
                            if 'Umaban' in td_class or (i == 1 and text.isdigit()):
                                if text.isdigit() and 1 <= int(text) <= 18:
                                    umaban = int(text)
                                    break

                        # 予想オッズを取得（Popular列、通常は後ろの方のtd）
                        for td in tds:
                            td_class = ' '.join(td.get('class', []))
                            # Popularクラスまたはodds関連のクラスを持つセル
                            if 'Popular' in td_class or 'Odds' in td_class or 'odds' in td_class.lower():
                                text = td.get_text(strip=True)
                                odds_match = re.search(r'(\d+\.?\d*)', text)
                                if odds_match:
                                    val = float(odds_match.group(1))
                                    if 1.0 <= val <= 999.9:
                                        odds_val = val
                                        break

                        if umaban and odds_val:
                            odds_dict[umaban] = odds_val

            if odds_dict:
                print(f"DEBUG shutuba odds: {odds_dict}")
                return odds_dict
        except Exception as e:
            print(f'Shutuba odds error: {e}')

        # オッズページから取得できなかった場合、スマホ版結果ページから全馬のオッズを取得
        # スマホ版は結果テーブルに全馬の単勝オッズが含まれている
        sp_result_url = f"https://nar.sp.netkeiba.com/race/race_result.html?race_id={race_id}"
        try:
            soup = self._fetch(sp_result_url, encoding='UTF-8')

            # スマホ版の結果テーブルからオッズを取得
            # テーブル内の各行から馬番とオッズを抽出
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 8:
                    try:
                        # 馬番を探す（通常は最初の方のtd）
                        umaban = None
                        odds_val = None

                        for i, td in enumerate(tds):
                            text = td.get_text(strip=True)
                            # 馬番（1-18の数字、通常2桁以下）
                            if text.isdigit() and 1 <= int(text) <= 18 and umaban is None:
                                # 着順ではなく馬番かを確認（着順は1から始まる小さい数字）
                                # class属性やdata属性で判別できる場合もある
                                td_class = td.get('class', [])
                                if 'Umaban' in str(td_class) or i >= 1:
                                    umaban = int(text)

                        # オッズを探す（小数点を含む数字）
                        for td in tds:
                            text = td.get_text(strip=True)
                            # オッズパターン: "1.5" or "29.8" など
                            odds_match = re.match(r'^(\d+\.\d+)$', text)
                            if odds_match:
                                val = float(odds_match.group(1))
                                if 1.0 <= val <= 999.9:
                                    odds_val = val
                                    break

                        if umaban and odds_val:
                            odds_dict[umaban] = odds_val

                    except (ValueError, IndexError):
                        continue

            if odds_dict:
                return odds_dict

        except Exception as e:
            print(f'SP result page error: {e}')

        # 最後の手段: PC版結果ページの払戻金から勝ち馬のみ取得
        result_url = f"{self.BASE_URL}/race/result.html?race_id={race_id}"
        try:
            soup = self._fetch(result_url)
            payout_table = soup.find('table', class_='Payout_Detail_Table')
            if payout_table:
                for tr in payout_table.find_all('tr'):
                    th = tr.find('th')
                    if th and '単勝' in th.get_text():
                        tds = tr.find_all('td')
                        if len(tds) >= 2:
                            umaban_text = tds[0].get_text(strip=True)
                            payout_text = tds[1].get_text(strip=True)
                            if umaban_text.isdigit():
                                umaban = int(umaban_text)
                                payout_match = re.search(r'([\d,]+)', payout_text)
                                if payout_match:
                                    payout = int(payout_match.group(1).replace(',', ''))
                                    odds_dict[umaban] = payout / 100
            return odds_dict
        except Exception as e:
            print(f'Result page error: {e}')
            return {}

    def get_horse_history(self, horse_id: str):
        if horse_id in self.horse_cache:
            return self.horse_cache[horse_id]

        url = f"{self.DB_URL}/horse/ajax_horse_results.html?id={horse_id}"
        try:
            time.sleep(self.delay)
            r = self.session.get(url)
            r.encoding = 'EUC-JP'
            soup = BeautifulSoup(r.text, 'lxml')

            results = []
            last_3f_times = []  # 上がり3Fタイム
            race_dates = []  # レース日付

            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 6:
                    continue

                # 着順を取得（tds[3:7]のどこかに着順がある）
                rank = None
                for td in tds[3:7]:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 20:
                        rank = int(t)
                        break

                if rank is None:
                    continue

                results.append(rank)

                # 日付を取得（最初のtd、YYYY/MM/DD形式）
                if len(tds) > 0:
                    date_text = tds[0].get_text(strip=True)
                    date_match = re.search(r'(\d{4})/(\d{2})/(\d{2})', date_text)
                    if date_match:
                        race_dates.append(f"{date_match.group(1)}{date_match.group(2)}{date_match.group(3)}")

                # 上がり3Fを取得（通常index 9-11あたり）
                # テーブル構造: 日付,開催,R,レース名,映像,頭数,枠番,馬番,オッズ,人気,着順,着差,タイム,上がり...
                for idx in [13, 12, 11, 10, 9]:  # 可能性のあるインデックスを試す
                    if len(tds) > idx:
                        l3f_text = tds[idx].get_text(strip=True)
                        # 上がり3Fは30-50秒台（例: 38.5, 41.2）
                        if re.match(r'^3[0-9]\.\d$|^4[0-9]\.\d$|^5[0-2]\.\d$', l3f_text):
                            last_3f_times.append(float(l3f_text))
                            break
                else:
                    last_3f_times.append(None)

                if len(results) >= 20:
                    break

            stats = self._calc_stats(results, last_3f_times, race_dates)
            self.horse_cache[horse_id] = stats
            return stats
        except Exception as e:
            return self._empty_stats()

    def get_jockey_stats(self, jockey_id: str):
        if jockey_id in self.jockey_cache:
            return self.jockey_cache[jockey_id]

        url = f"{self.DB_URL}/jockey/{jockey_id}/"
        try:
            soup = self._fetch(url)
            stats = {'jockey_win_rate': 0, 'jockey_place_rate': 0, 'jockey_show_rate': 0}

            # テーブルから成績を取得（累計行を探す）
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                # ヘッダー行で「勝率」列を探す
                header_row = rows[0] if rows else None
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                    # 勝率、連対率、複勝率の列インデックスを探す
                    # 注意: 「複勝率」には「勝率」が含まれるので、先に複勝率をチェック
                    win_idx = place_idx = show_idx = -1
                    for i, h in enumerate(headers):
                        if '複勝率' in h:
                            show_idx = i
                        elif '連対率' in h:
                            place_idx = i
                        elif '勝率' in h:  # 複勝率でない勝率
                            win_idx = i

                    if win_idx >= 0:
                        # 累計行（2行目）からデータを取得
                        for row in rows[1:3]:
                            cells = row.find_all(['th', 'td'])
                            cell_texts = [c.get_text(strip=True) for c in cells]
                            if len(cell_texts) > max(win_idx, place_idx, show_idx):
                                # 全角・半角両方のパーセント記号に対応
                                def parse_rate(text):
                                    m = re.search(r'(\d+\.?\d*)[％%]', text)
                                    return float(m.group(1)) / 100 if m else 0

                                if win_idx >= 0 and win_idx < len(cell_texts):
                                    stats['jockey_win_rate'] = parse_rate(cell_texts[win_idx])
                                if place_idx >= 0 and place_idx < len(cell_texts):
                                    stats['jockey_place_rate'] = parse_rate(cell_texts[place_idx])
                                if show_idx >= 0 and show_idx < len(cell_texts):
                                    stats['jockey_show_rate'] = parse_rate(cell_texts[show_idx])

                                if stats['jockey_win_rate'] > 0:
                                    break
                        if stats['jockey_win_rate'] > 0:
                            break

            self.jockey_cache[jockey_id] = stats
            return stats
        except:
            return {'jockey_win_rate': 0, 'jockey_place_rate': 0, 'jockey_show_rate': 0}

    def _calc_stats(self, ranks, last_3f_times=None, race_dates=None):
        if not ranks:
            return self._empty_stats()
        total = len(ranks)
        wins = sum(1 for r in ranks if r == 1)
        place = sum(1 for r in ranks if r <= 2)
        show = sum(1 for r in ranks if r <= 3)
        recent = ranks[:5]
        r_total = len(recent)

        # 連勝数/連複勝数を計算（直近から数える）
        win_streak = 0
        for r in ranks:
            if r == 1:
                win_streak += 1
            else:
                break

        show_streak = 0
        for r in ranks:
            if r <= 3:
                show_streak += 1
            else:
                break

        # 着順の標準偏差（安定性指標）
        past_rank_std = np.std(ranks) if len(ranks) >= 2 else 3.0

        # 上がり3F関連（デフォルト値: 学習データ平均）
        prev_last_3f = 41.2
        avg_last_3f_3races = 41.2
        avg_last_3f_5races = 41.2

        if last_3f_times:
            # 有効な上がり3Fのみ抽出
            valid_3f = [t for t in last_3f_times if t is not None]
            if valid_3f:
                prev_last_3f = valid_3f[0]  # 直近の上がり3F
                avg_last_3f_3races = np.mean(valid_3f[:3]) if len(valid_3f) >= 1 else 41.2
                avg_last_3f_5races = np.mean(valid_3f[:5]) if len(valid_3f) >= 1 else 41.2

        # 前走からの日数計算
        days_since_last_race = 30  # デフォルト値
        if race_dates and len(race_dates) >= 1:
            try:
                from datetime import datetime
                last_race_date = datetime.strptime(race_dates[0], '%Y%m%d')
                today = datetime.now()
                days_since_last_race = (today - last_race_date).days
            except:
                pass

        return {
            'horse_runs': total,
            'horse_win_rate': wins / total,
            'horse_place_rate': place / total,
            'horse_show_rate': show / total,
            'horse_avg_rank': np.mean(ranks),
            'horse_recent_win_rate': sum(1 for r in recent if r == 1) / r_total if r_total else 0,
            'horse_recent_show_rate': sum(1 for r in recent if r <= 3) / r_total if r_total else 0,
            'horse_recent_avg_rank': np.mean(recent) if recent else 10,
            'last_rank': ranks[0] if ranks else 10,
            'win_streak': win_streak,
            'show_streak': show_streak,
            'past_rank_std': past_rank_std,
            'prev_last_3f': prev_last_3f,
            'avg_last_3f_3races': avg_last_3f_3races,
            'avg_last_3f_5races': avg_last_3f_5races,
            'days_since_last_race': days_since_last_race
        }

    def _empty_stats(self):
        return {
            'horse_runs': 0, 'horse_win_rate': 0, 'horse_place_rate': 0,
            'horse_show_rate': 0, 'horse_avg_rank': 10,
            'horse_recent_win_rate': 0, 'horse_recent_show_rate': 0,
            'horse_recent_avg_rank': 10, 'last_rank': 10,
            'win_streak': 0, 'show_streak': 0, 'past_rank_std': 3.0,
            'prev_last_3f': 41.2, 'avg_last_3f_3races': 41.2, 'avg_last_3f_5races': 41.2,
            'days_since_last_race': 30
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

        if 'jockey_id' in df.columns:
            jockey_data = []
            for jid in df['jockey_id'].dropna().unique():
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
    def __init__(self, te_encoder=None):
        self.te_encoder = te_encoder  # Target Encoder（モデルから取得）
        # 新モデル（optimize_v2.py）対応: 52特徴量
        self.features = [
            # 基本特徴量
            'horse_runs', 'horse_win_rate', 'horse_place_rate', 'horse_show_rate',
            'horse_avg_rank', 'horse_recent_win_rate', 'horse_recent_show_rate',
            'horse_recent_avg_rank', 'last_rank',
            'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
            'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
            'sex_encoded', 'field_size', 'weight_diff',
            # 環境特徴量
            'track_condition_encoded', 'weather_encoded',
            'horse_weight', 'horse_weight_change',
            # 計算特徴量
            'horse_number_ratio', 'last_rank_diff', 'win_rate_rank',
            # 相対特徴量（レース内での相対的な強さ）
            'horse_win_rate_vs_field', 'jockey_win_rate_vs_field',
            'horse_avg_rank_vs_field',
            # 休養・調子
            'days_since_last_race', 'rank_trend',
            # 時系列特徴量
            'win_streak', 'show_streak', 'recent_3_avg_rank', 'recent_10_avg_rank', 'rank_improvement',
            # Target Encoding（推論時はグローバル平均を使用）
            'jockey_id_te', 'trainer_id_te', 'horse_id_te',
            # 追加特徴量（AUC 0.8目標の最適化で追加）
            'horse_jockey_synergy', 'form_score', 'class_indicator',
            'horse_win_rate_std', 'field_strength', 'inner_outer',
            'avg_rank_percentile', 'jockey_rank_in_race', 'odds_implied_prob',
            'distance_fitness', 'weight_per_meter', 'experience_score',
            # v6追加特徴量（上がり3F関連）
            'prev_last_3f', 'avg_last_3f_3races', 'avg_last_3f_5races',
            'prev_last_3f_rank', 'prev_last_3f_vs_field',
            'past_rank_std', 'is_first_race'
        ]

    def process(self, df):
        df = df.copy()
        num_cols = ['bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
                    'horse_weight', 'weight_change']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'sex' in df.columns:
            df['sex_encoded'] = df['sex'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        else:
            df['sex_encoded'] = 0

        df['track_encoded'] = 0

        if 'weight_carried' in df.columns and 'race_id' in df.columns:
            df['weight_diff'] = df.groupby('race_id')['weight_carried'].transform(lambda x: x - x.mean())
        else:
            df['weight_diff'] = 0

        if 'field_size' not in df.columns:
            if 'race_id' in df.columns:
                df['field_size'] = df.groupby('race_id')['race_id'].transform('count')
            else:
                df['field_size'] = 12

        # 馬場状態エンコーディング（良=0, 稍重=1, 重=2, 不良=3）
        if 'track_condition' in df.columns:
            df['track_condition_encoded'] = df['track_condition'].map(
                {'良': 0, '稍重': 1, '重': 2, '不良': 3}
            ).fillna(0)
        else:
            df['track_condition_encoded'] = 0

        # 天気エンコーディング（晴=0, 曇=1, 小雨=2, 雨=3, 雪=4）
        if 'weather' in df.columns:
            df['weather_encoded'] = df['weather'].map(
                {'晴': 0, '曇': 1, '小雨': 2, '雨': 3, '雪': 4}
            ).fillna(0)
        else:
            df['weather_encoded'] = 0

        # 調教師エンコーディング（ハッシュベース）
        if 'trainer_id' in df.columns:
            df['trainer_encoded'] = df['trainer_id'].apply(
                lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0
            )
        else:
            df['trainer_encoded'] = 0

        # 馬体重（欠損は450kgで補完）
        if 'horse_weight' in df.columns:
            df['horse_weight'] = df['horse_weight'].fillna(450)
        else:
            df['horse_weight'] = 450

        # 馬体重増減
        if 'weight_change' in df.columns:
            df['horse_weight_change'] = df['weight_change'].fillna(0)
        else:
            df['horse_weight_change'] = 0

        # === 計算特徴量 ===
        # 馬番比率（馬番/出走頭数）
        if 'horse_number' in df.columns and 'field_size' in df.columns:
            df['horse_number_ratio'] = df['horse_number'] / df['field_size']
            df['horse_number_ratio'] = df['horse_number_ratio'].fillna(0.5)

        # 距離カテゴリ（短距離/中距離/長距離）
        if 'distance' in df.columns:
            df['distance_category'] = df['distance'].apply(
                lambda d: 0 if pd.notna(d) and d < 1400 else (2 if pd.notna(d) and d >= 1800 else 1)
            )
        else:
            df['distance_category'] = 1

        # 前走着順差（前走着順 - 平均着順）
        if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['last_rank_diff'] = df['last_rank'] - df['horse_avg_rank']
            df['last_rank_diff'] = df['last_rank_diff'].fillna(0)
        else:
            df['last_rank_diff'] = 0

        # レース内の勝率ランク
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False, method='min')
            df['win_rate_rank'] = df['win_rate_rank'].fillna(df['field_size'] / 2)
        else:
            df['win_rate_rank'] = 6

        # === 相対特徴量（レース内での相対的な強さ）===
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['field_avg_win_rate'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
            df['horse_win_rate_vs_field'] = df['horse_win_rate'] - df['field_avg_win_rate']
            df['horse_win_rate_vs_field'] = df['horse_win_rate_vs_field'].fillna(0)
        else:
            df['horse_win_rate_vs_field'] = 0

        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['field_avg_jockey_win_rate'] = df.groupby('race_id')['jockey_win_rate'].transform('mean')
            df['jockey_win_rate_vs_field'] = df['jockey_win_rate'] - df['field_avg_jockey_win_rate']
            df['jockey_win_rate_vs_field'] = df['jockey_win_rate_vs_field'].fillna(0)
        else:
            df['jockey_win_rate_vs_field'] = 0

        if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
            df['field_avg_rank'] = df.groupby('race_id')['horse_avg_rank'].transform('mean')
            df['horse_avg_rank_vs_field'] = df['field_avg_rank'] - df['horse_avg_rank']
            df['horse_avg_rank_vs_field'] = df['horse_avg_rank_vs_field'].fillna(0)
        else:
            df['horse_avg_rank_vs_field'] = 0

        # === 休養日数 ===
        # スクレイピングしたデータを使用（enrich_dataでマージ済み、欠損はv6セクションで埋める）
        if 'days_since_last_race' not in df.columns:
            df['days_since_last_race'] = 30

        # === 着順トレンド ===
        if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['rank_trend'] = df['horse_avg_rank'] - df['last_rank']
            df['rank_trend'] = df['rank_trend'].fillna(0)
        else:
            df['rank_trend'] = 0

        # === 交互作用特徴量（train.pyと同じロジック）===
        # 騎手×競馬場の相性（ハッシュベース）
        if 'jockey_id' in df.columns and 'race_id' in df.columns:
            df['track_code'] = df['race_id'].astype(str).str[4:6]
            df['jockey_track_interaction'] = df.apply(
                lambda x: hash(str(x.get('jockey_id', '')) + str(x.get('track_code', ''))) % 10000, axis=1
            )
        else:
            df['jockey_track_interaction'] = 0

        # 調教師×距離の相性
        if 'trainer_id' in df.columns and 'distance' in df.columns:
            df['distance_cat'] = df['distance'].apply(
                lambda d: 'short' if pd.notna(d) and d < 1400 else ('long' if pd.notna(d) and d >= 1800 else 'mid')
            )
            df['trainer_distance_interaction'] = df.apply(
                lambda x: hash(str(x.get('trainer_id', '')) + str(x.get('distance_cat', ''))) % 10000, axis=1
            )
        else:
            df['trainer_distance_interaction'] = 0

        # 騎手×距離の相性
        if 'jockey_id' in df.columns and 'distance' in df.columns:
            if 'distance_cat' not in df.columns:
                df['distance_cat'] = df['distance'].apply(
                    lambda d: 'short' if pd.notna(d) and d < 1400 else ('long' if pd.notna(d) and d >= 1800 else 'mid')
                )
            df['jockey_distance_interaction'] = df.apply(
                lambda x: hash(str(x.get('jockey_id', '')) + str(x.get('distance_cat', ''))) % 10000, axis=1
            )
        else:
            df['jockey_distance_interaction'] = 0

        # === 時系列強化特徴量 ===
        # 連勝数（CSVにあれば使用、なければ0）
        if 'win_streak' not in df.columns:
            df['win_streak'] = 0
        if 'show_streak' not in df.columns:
            df['show_streak'] = 0

        # 直近3走・10走平均着順
        if 'recent_3_avg_rank' not in df.columns:
            if 'horse_recent_avg_rank' in df.columns:
                df['recent_3_avg_rank'] = df['horse_recent_avg_rank']
            else:
                df['recent_3_avg_rank'] = 10
        if 'recent_10_avg_rank' not in df.columns:
            if 'horse_avg_rank' in df.columns:
                df['recent_10_avg_rank'] = df['horse_avg_rank']
            else:
                df['recent_10_avg_rank'] = 10

        # 着順改善トレンド
        if 'recent_3_avg_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['rank_improvement'] = df['horse_avg_rank'] - df['recent_3_avg_rank']
            df['rank_improvement'] = df['rank_improvement'].fillna(0)
        else:
            df['rank_improvement'] = 0

        # === Target Encoding ===
        if self.te_encoder is not None:
            # モデルのte_encoderを使用して実際のエンコーディングを適用
            te_cols = ['jockey_id', 'trainer_id', 'horse_id']
            for col in te_cols:
                te_col = f'{col}_te'
                if col in df.columns and col in self.te_encoder.mappings:
                    df[te_col] = df[col].map(self.te_encoder.mappings[col]).fillna(self.te_encoder.global_mean)
                else:
                    df[te_col] = self.te_encoder.global_mean
        else:
            # te_encoderがない場合はグローバル平均を使用
            global_te_default = 0.274  # 学習データの複勝率平均
            df['jockey_id_te'] = global_te_default
            df['trainer_id_te'] = global_te_default
            df['horse_id_te'] = global_te_default

        # === 追加特徴量（AUC 0.8目標の最適化で追加）===
        # 馬×騎手シナジー
        if 'horse_win_rate' in df.columns and 'jockey_win_rate' in df.columns:
            df['horse_jockey_synergy'] = df['horse_win_rate'] * df['jockey_win_rate']
        else:
            df['horse_jockey_synergy'] = 0

        # フォームスコア（調子指標）
        if all(c in df.columns for c in ['last_rank', 'field_size', 'horse_recent_avg_rank', 'horse_win_rate']):
            df['form_score'] = (
                0.5 * (1 - df['last_rank'] / df['field_size'].clip(lower=1)) +
                0.3 * (1 - df['horse_recent_avg_rank'] / df['field_size'].clip(lower=1)) +
                0.2 * df['horse_win_rate']
            ).fillna(0)
        else:
            df['form_score'] = 0

        # クラス指標（出走頭数/平均着順）
        if 'field_size' in df.columns and 'horse_avg_rank' in df.columns:
            df['class_indicator'] = df['field_size'] / (df['horse_avg_rank'] + 1)
            df['class_indicator'] = df['class_indicator'].fillna(1)
        else:
            df['class_indicator'] = 1

        # 勝率の標準偏差（推論時は計算不可、0をデフォルト）
        df['horse_win_rate_std'] = 0

        # フィールド強度（レース内の平均勝率）
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['field_strength'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
            df['field_strength'] = df['field_strength'].fillna(0.1)
        else:
            df['field_strength'] = 0.1

        # 内外（馬番による枠位置）: 0=内, 1=中, 2=外
        if 'horse_number' in df.columns:
            df['inner_outer'] = df['horse_number'].apply(
                lambda x: 0 if pd.notna(x) and x <= 4 else (2 if pd.notna(x) and x >= 10 else 1)
            )
        else:
            df['inner_outer'] = 1

        # 平均着順パーセンタイル（レース内での相対順位）
        if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
            df['avg_rank_percentile'] = df.groupby('race_id')['horse_avg_rank'].rank(pct=True)
            df['avg_rank_percentile'] = df['avg_rank_percentile'].fillna(0.5)
        else:
            df['avg_rank_percentile'] = 0.5

        # 騎手のレース内ランク
        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['jockey_rank_in_race'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)
            df['jockey_rank_in_race'] = df['jockey_rank_in_race'].fillna(6)
        else:
            df['jockey_rank_in_race'] = 6

        # オッズからの暗黙的勝率（オッズがない場合はデフォルト）
        if 'odds' in df.columns:
            df['odds_implied_prob'] = 1 / (df['odds'].clip(lower=1) + 1)
        else:
            df['odds_implied_prob'] = 0.1  # デフォルト（約10倍相当）

        # 距離適性（推論時はデフォルト）
        df['distance_fitness'] = 1.0

        # 斤量/距離（負担重量の効率）
        if 'weight_carried' in df.columns and 'distance' in df.columns:
            df['weight_per_meter'] = df['weight_carried'] / (df['distance'] / 1000).clip(lower=0.1)
            df['weight_per_meter'] = df['weight_per_meter'].fillna(50)
        else:
            df['weight_per_meter'] = 50

        # 経験スコア（出走数×複勝率）
        if 'horse_runs' in df.columns and 'horse_show_rate' in df.columns:
            df['experience_score'] = np.log1p(df['horse_runs']) * df['horse_show_rate']
            df['experience_score'] = df['experience_score'].fillna(0)
        else:
            df['experience_score'] = 0

        # === 血統特徴量 ===
        for col in ['father_win_rate', 'father_show_rate', 'bms_win_rate', 'bms_show_rate']:
            if col not in df.columns:
                df[col] = 0

        # === v6追加特徴量（上がり3F関連）===
        # スクレイピングしたデータを使用、欠損値はデフォルトで埋める
        # last_3f mean=41.2, std=2.0
        if 'prev_last_3f' not in df.columns:
            df['prev_last_3f'] = 41.2
        df['prev_last_3f'] = df['prev_last_3f'].fillna(41.2)

        # 過去3走・5走の上がり3F平均
        if 'avg_last_3f_3races' not in df.columns:
            df['avg_last_3f_3races'] = 41.2
        df['avg_last_3f_3races'] = df['avg_last_3f_3races'].fillna(41.2)

        if 'avg_last_3f_5races' not in df.columns:
            df['avg_last_3f_5races'] = 41.2
        df['avg_last_3f_5races'] = df['avg_last_3f_5races'].fillna(41.2)

        # 前走からの日数（スクレイピングしたデータを使用）
        if 'days_since_last_race' not in df.columns:
            df['days_since_last_race'] = 30
        df['days_since_last_race'] = df['days_since_last_race'].fillna(30)

        # 前走の上がり3F順位（フィールド内での相対順位）- 実データから計算
        if 'prev_last_3f' in df.columns:
            df['prev_last_3f_rank'] = df.groupby(level=0, group_keys=False).apply(
                lambda x: x['prev_last_3f'].rank(ascending=True)
            ).reset_index(drop=True) if len(df) > 1 else 5.5
        if 'prev_last_3f_rank' not in df.columns or df['prev_last_3f_rank'].isna().all():
            df['prev_last_3f_rank'] = 5.5
        df['prev_last_3f_rank'] = df['prev_last_3f_rank'].fillna(5.5)

        # 上がり3Fのフィールド平均との差 - 実データから計算
        if 'prev_last_3f' in df.columns and len(df) > 1:
            field_mean = df['prev_last_3f'].mean()
            df['prev_last_3f_vs_field'] = field_mean - df['prev_last_3f']  # 速いほど+
        else:
            df['prev_last_3f_vs_field'] = 0
        df['prev_last_3f_vs_field'] = df['prev_last_3f_vs_field'].fillna(0)

        # 過去着順の標準偏差（安定性指標）
        if 'past_rank_std' not in df.columns:
            df['past_rank_std'] = 2.66
        df['past_rank_std'] = df['past_rank_std'].fillna(2.66)

        # 初出走フラグ（出走回数が0または1なら初出走）
        if 'is_first_race' not in df.columns:
            if 'horse_runs' in df.columns:
                df['is_first_race'] = (df['horse_runs'] <= 1).astype(int)
            else:
                df['is_first_race'] = 0

        # === v10モデル用特徴量 ===
        # レース内相対順位
        if 'horse_show_rate' in df.columns and 'race_id' in df.columns:
            df['show_rate_rank'] = df.groupby('race_id')['horse_show_rate'].rank(ascending=False, method='min').fillna(6)
        else:
            df['show_rate_rank'] = 6

        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['jockey_rank'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False, method='min').fillna(6)
        else:
            df['jockey_rank'] = 6

        if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
            df['avg_rank_rank'] = df.groupby('race_id')['horse_avg_rank'].rank(ascending=True, method='min').fillna(6)
        else:
            df['avg_rank_rank'] = 6

        # 過去スピード指数（デフォルト50）
        df['past_speed_index'] = 50
        df['past_3_speed_index'] = 50
        df['past_speed_rank'] = 6

        # 過去上がり3F
        if 'prev_last_3f' in df.columns:
            df['past_last_3f'] = df['prev_last_3f'].fillna(40)
            df['past_3_last_3f'] = df['avg_last_3f_3races'].fillna(40) if 'avg_last_3f_3races' in df.columns else 40
            df['past_3f_rank'] = df.groupby('race_id')['past_last_3f'].rank(ascending=True, method='min').fillna(6) if 'race_id' in df.columns else 6
        else:
            df['past_last_3f'] = 40
            df['past_3_last_3f'] = 40
            df['past_3f_rank'] = 6

        # 相対値
        if 'horse_show_rate' in df.columns and 'race_id' in df.columns:
            df['show_rate_vs_field'] = df['horse_show_rate'] - df.groupby('race_id')['horse_show_rate'].transform('mean')
            df['show_rate_vs_field'] = df['show_rate_vs_field'].fillna(0)
        else:
            df['show_rate_vs_field'] = 0

        # win_rate_vs_fieldはhorse_win_rate_vs_fieldからコピー
        if 'horse_win_rate_vs_field' in df.columns:
            df['win_rate_vs_field'] = df['horse_win_rate_vs_field']
        else:
            df['win_rate_vs_field'] = 0

        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['jockey_vs_field'] = df['jockey_win_rate'] - df.groupby('race_id')['jockey_win_rate'].transform('mean')
            df['jockey_vs_field'] = df['jockey_vs_field'].fillna(0)
        else:
            df['jockey_vs_field'] = 0

        df['past_speed_vs_field'] = 0  # デフォルト

        # form_trend
        if 'form_score' in df.columns and 'horse_show_rate' in df.columns:
            df['form_trend'] = df['form_score'] - df['horse_show_rate']
            df['form_trend'] = df['form_trend'].fillna(0)
        else:
            df['form_trend'] = 0

        # 前走成績スコア
        if 'last_rank' in df.columns:
            df['last_rank_score'] = (df['last_rank'] <= 3).astype(int)
            df['last_rank_normalized'] = df['last_rank'] / df['field_size'].clip(lower=1) if 'field_size' in df.columns else 0.5
        else:
            df['last_rank_score'] = 0
            df['last_rank_normalized'] = 0.5

        # 経過日数関連
        if 'days_since_last_race' in df.columns:
            df['days_since_last'] = df['days_since_last_race'].fillna(30).clip(0, 180)
        else:
            df['days_since_last'] = 30
        df['is_fresh'] = (df['days_since_last'] >= 30).astype(int)
        df['is_long_rest'] = (df['days_since_last'] >= 60).astype(int)

        # track_condition_code (track_condition_encodedのエイリアス)
        if 'track_condition_encoded' in df.columns:
            df['track_condition_code'] = df['track_condition_encoded']
        else:
            df['track_condition_code'] = 0

        for f in self.features:
            if f not in df.columns:
                df[f] = 0
        return df

    def get_features(self):
        return self.features


# ========== モデル読み込み ==========
def load_model(track_code: str):
    """
    モデルを読み込む
    Returns: (model, features, te_encoder, version, best_threshold)
    """
    if track_code in model_cache:
        return model_cache[track_code]

    if track_code not in TRACKS:
        return None, None, None, None, None

    model_name = TRACKS[track_code]['model']

    # エイリアスをチェック（旧モデル名との互換性）
    paths_to_try = [model_name]
    if model_name in MODEL_ALIASES:
        paths_to_try = MODEL_ALIASES[model_name]

    # pickleでカスタムクラスを読み込めるよう__main__に登録
    import __main__
    __main__.TargetEncoderSafe = TargetEncoderSafe

    for model_name in paths_to_try:
        model_path = BASE_DIR / model_name
        if model_path.exists():
            with open(model_path, 'rb') as f:
                d = pickle.load(f)
            te_encoder = d.get('te_encoder')  # Target Encoder取得
            version = d.get('version', 'legacy')  # モデルバージョン取得
            best_threshold = d.get('best_threshold', 0.15)  # v8の閾値（デフォルト0.15）
            model_cache[track_code] = (d['model'], d['features'], te_encoder, version, best_threshold)
            return d['model'], d['features'], te_encoder, version, best_threshold
    return None, None, None, None, None


def predict_with_model(model, X):
    """モデルで予測（アンサンブル対応）- 確率を返す"""
    if isinstance(model, dict):
        # アンサンブルモデルの場合
        model_type = model.get('type', 'ensemble')
        if model_type == 'ensemble':
            # 分類器の場合はpredict_probaを使用（クラス1の確率）
            lgb_pred = model['lgb'].predict_proba(X)[:, 1]
            xgb_pred = model['xgb'].predict_proba(X)[:, 1]
            return (lgb_pred + xgb_pred) / 2
        elif model_type == 'xgb':
            return model['xgb'].predict_proba(X)[:, 1]
        elif model_type == 'lgb':
            return model['lgb'].predict_proba(X)[:, 1]
        else:
            # フォールバック: 最初に見つかったモデルを使用
            for key in ['lgb', 'xgb', 'model']:
                if key in model:
                    m = model[key]
                    if hasattr(m, 'predict_proba'):
                        return m.predict_proba(X)[:, 1]
                    return m.predict(X)
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        # 単一モデルの場合
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)


# ========== APIエンドポイント ==========

@app.get("/")
def root():
    return {"message": "地方競馬予測API", "version": "1.0.0"}


@app.get("/api/tracks")
def get_tracks():
    """利用可能な競馬場一覧を取得"""
    tracks = []
    for code, info in TRACKS.items():
        model_name = info['model']
        # エイリアスも含めてチェック
        paths_to_check = [model_name]
        if model_name in MODEL_ALIASES:
            paths_to_check = MODEL_ALIASES[model_name]
        model_exists = any((BASE_DIR / p).exists() for p in paths_to_check)
        tracks.append({
            "code": code,
            "name": info['name'],
            "emoji": info['emoji'],
            "model_available": model_exists
        })
    return {"tracks": tracks}


class PredictRequest(BaseModel):
    track_code: str
    date: str  # YYYY-MM-DD形式


class PredictionResult(BaseModel):
    rank: int
    number: int
    name: str
    jockey: str
    prob: float
    win_rate: float
    show_rate: float


class RaceResult(BaseModel):
    id: str
    name: str
    distance: int
    time: str
    predictions: list[PredictionResult]


@app.post("/api/predict")
def predict(request: PredictRequest):
    """予測を実行（事前生成JSONがあれば優先使用）"""
    track_code = request.track_code
    date_str = request.date.replace("-", "")

    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    # === 事前生成JSONをチェック ===
    predictions_file = BASE_DIR / "predictions" / request.date / f"{track_code}.json"
    if predictions_file.exists():
        # JSONが存在すれば読み込んで返す（一貫性のある予測）
        with open(predictions_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        # 事前生成データを使用していることを示すフラグを追加
        cached_data["from_cache"] = True
        return cached_data

    # === JSONがない場合はスクレイピング ===
    model, model_features, te_encoder, model_version, best_threshold = load_model(track_code)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"{TRACKS[track_code]['name']}のモデルがありません"
        )

    # モデルバージョンに応じた処理を選択
    use_v8 = model_version == 'v8_no_leak'
    use_v3 = model_version == 'auto_optimized'

    scraper = NARScraper(track_code, delay=0.3)
    processor = Processor(te_encoder=te_encoder) if not (use_v3 or use_v8) else None

    # レース一覧取得
    race_ids = scraper.get_race_list_by_date(date_str)
    if not race_ids:
        return {"races": [], "message": "レースが見つかりません"}

    results = []
    for rid in sorted(race_ids):
        df = scraper.get_race_data(rid)
        if df is None:
            continue

        df = scraper.enrich_data(df)

        # オッズ取得（単勝・複勝を一括取得）- v3では特徴量に必要
        all_odds = scraper.get_all_odds(rid)
        win_odds_dict = all_odds.get('win', {})
        place_odds_dict = all_odds.get('place', {})

        # オッズをDataFrameに追加（v3特徴量で必要）
        if 'horse_number' in df.columns:
            df['win_odds'] = df['horse_number'].apply(lambda x: win_odds_dict.get(int(x), 10) if pd.notna(x) else 10)

        # 特徴量作成
        if use_v8:
            # v8: オッズ除外アプローチ（回収率108-110%）
            df, features_to_use = create_features_v8(df)
        elif use_v3:
            # v3: 人気ベースアプローチ（的中率77%）
            df, features_to_use = create_features_v3(df)
        else:
            # 従来モデル
            df = processor.process(df)
            features_to_use = model_features

        # 予測
        X = df[features_to_use].fillna(-1)
        df['prob'] = predict_with_model(model, X)
        df['pred_rank'] = df['prob'].rank(ascending=False, method='min').astype(int)
        df = df.sort_values('prob', ascending=False)

        # レース番号抽出
        race_num = rid[-2:]
        race_name = df['race_name'].iloc[0] if 'race_name' in df.columns else f"{race_num}R"
        distance = int(df['distance'].iloc[0]) if 'distance' in df.columns else 0
        start_time = df['start_time'].iloc[0] if 'start_time' in df.columns else ""

        predictions = []
        for i, (_, row) in enumerate(df.iterrows()):  # 全馬を返す
            horse_num = int(row['horse_number']) if pd.notna(row.get('horse_number')) else 0
            win_odds = win_odds_dict.get(horse_num, 0)
            place_odds_data = place_odds_dict.get(horse_num, {})
            place_odds = place_odds_data.get('avg', 0) if place_odds_data else 0
            place_odds_min = place_odds_data.get('min', 0) if place_odds_data else 0
            place_odds_max = place_odds_data.get('max', 0) if place_odds_data else 0
            prob = float(row['prob'])

            # 勝率・複勝率を取得（0-1の範囲であるべき）
            raw_win_rate = float(row.get('horse_win_rate') or 0)
            raw_show_rate = float(row.get('horse_show_rate') or 0)

            win_rate = raw_win_rate * 100
            show_rate = raw_show_rate * 100

            # 期待値計算（複勝オッズ × AI確率）
            # 複勝オッズがあればそれを使用、なければ単勝/3で推定
            effective_place_odds = place_odds if place_odds > 0 else (win_odds / 3 if win_odds > 0 else 0)
            expected_value = prob * effective_place_odds if effective_place_odds > 0 else 0

            # 妙味判定: 期待値 > 1.0 なら黒字期待
            is_value = expected_value > 1.0

            # ========== 回収率ベース買い目判定 ==========
            # バックテスト結果: 予測1位のみを対象、オッズ条件で回収率100%+を狙う
            bet_layer = None
            recommended_bet = 0
            min_odds = MIN_PLACE_ODDS_FOR_ROI.get(track_code, 2.0)

            if i == 0:  # 予測1位のみ対象（バックテストと同じ条件）
                if effective_place_odds >= min_odds:
                    # 回収率100%以上が期待できる
                    bet_layer = "roi_buy"
                    recommended_bet = calculate_bet_amount(expected_value)
                elif effective_place_odds > 0:
                    # オッズ不足だが参考表示
                    bet_layer = "watch"
                    recommended_bet = 0

            predictions.append({
                "rank": i + 1,
                "number": horse_num,
                "name": row.get('horse_name', '不明'),
                "jockey": row.get('jockey_name', '不明'),
                "prob": round(prob, 3),
                "win_rate": round(win_rate, 1),
                "show_rate": round(show_rate, 1),
                "odds": win_odds,
                "place_odds": place_odds,
                "place_odds_min": place_odds_min,
                "place_odds_max": place_odds_max,
                "expected_value": round(expected_value, 2),
                "is_value": is_value,
                "bet_layer": bet_layer,
                "recommended_bet": recommended_bet
            })

        # 信頼度指標（prob_gap）を計算
        prob_gap = 0
        if len(df) >= 2:
            sorted_probs = df['prob'].sort_values(ascending=False).values
            prob_gap = float(sorted_probs[0] - sorted_probs[1])

        results.append({
            "id": race_num,
            "name": race_name,
            "distance": distance,
            "time": start_time,
            "field_size": len(df),  # 出走頭数を追加
            "prob_gap": round(prob_gap, 3),  # 1位と2位の確率差
            "predictions": predictions
        })

    # ========== 選択的ベッティング ==========
    # v8モデルの場合はbest_thresholdを使用、それ以外はv6設定を使用
    if use_v8:
        min_prob_diff = best_threshold
        expected_roi = 1.08  # v8バックテスト結果: 108%
        strategy_name = f"v8閾値フィルタ: 確率差{int(min_prob_diff*100)}%以上で購入（期待ROI {expected_roi:.0%}）"
    else:
        config = get_betting_config(track_code)
        min_prob_diff = config["min_prob_diff"]
        expected_roi = config["expected_roi"]
        strategy_name = f"確率差{int(min_prob_diff*100)}%以上で購入（期待ROI {expected_roi:.1%}）"

    betting_picks = {
        "roi_buy": [],     # 推奨買い（prob_diff条件クリア）※フロントエンド互換
        "v6_buy": [],      # 同じ内容（旧キー名、互換性）
        "v8_buy": [],      # v8用（新キー名）
        "watch": [],       # 様子見（prob_diff不足）
        "total_bet": 0,
        "expected_return": 0,
        "strategy": strategy_name,
        "model_version": model_version,
        "min_prob_diff": min_prob_diff,
        "expected_roi": expected_roi,
    }

    for race in results:
        preds = race["predictions"]
        if len(preds) < 2:
            continue

        # prob_diff計算（1位と2位の確率差）
        prob_diff = preds[0]["prob"] - preds[1]["prob"]
        top_pred = preds[0]

        # フィルター: prob_diff >= 閾値
        if prob_diff >= min_prob_diff:
            pick = {
                "race_id": race["id"],
                "race_name": race["name"],
                "race_time": race["time"],
                "number": top_pred["number"],
                "name": top_pred["name"],
                "prob": top_pred["prob"],
                "prob_diff": round(prob_diff, 3),
                "place_odds": top_pred["place_odds"],
                "odds": top_pred["odds"],
                "recommended_bet": 100,  # 固定100円
                "confidence": "高" if prob_diff >= 0.25 else ("中" if prob_diff >= 0.15 else "低"),
            }
            betting_picks["v6_buy"].append(pick)
            betting_picks["v8_buy"].append(pick)
            betting_picks["roi_buy"].append(pick)  # フロントエンド互換
            betting_picks["total_bet"] += 100
            # 期待リターン = 賭け金 × 期待ROI
            betting_picks["expected_return"] += 100 * expected_roi
        else:
            # prob_diff不足 → 様子見
            pick = {
                "race_id": race["id"],
                "race_name": race["name"],
                "race_time": race["time"],
                "number": top_pred["number"],
                "name": top_pred["name"],
                "prob": top_pred["prob"],
                "prob_diff": round(prob_diff, 3),
                "reason": f"確率差{prob_diff:.1%} < {min_prob_diff:.0%}",
            }
            betting_picks["watch"].append(pick)

    return {
        "track": {
            "code": track_code,
            "name": TRACKS[track_code]['name'],
            "emoji": TRACKS[track_code]['emoji']
        },
        "date": request.date,
        "races": results,
        "betting_picks": betting_picks
    }


class RaceListRequest(BaseModel):
    track_code: str
    date: str


@app.post("/api/races")
def get_race_list(request: RaceListRequest):
    """レース一覧を取得（軽量）"""
    track_code = request.track_code
    date_str = request.date.replace("-", "")

    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    scraper = NARScraper(track_code, delay=0.3)
    race_ids = scraper.get_race_list_by_date(date_str)

    return {
        "track": TRACKS[track_code],
        "race_ids": sorted(race_ids)
    }


class SingleRaceRequest(BaseModel):
    race_id: str
    track_code: str


@app.post("/api/predict/race")
def predict_single_race(request: SingleRaceRequest):
    """単一レースの予測"""
    race_id = request.race_id
    track_code = request.track_code

    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    model, model_features, te_encoder, model_version, best_threshold = load_model(track_code)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"{TRACKS[track_code]['name']}のモデルがありません"
        )

    # モデルバージョンに応じた処理を選択
    use_v8 = model_version == 'v8_no_leak'
    use_v3 = model_version == 'auto_optimized'

    scraper = NARScraper(track_code, delay=0.3)
    processor = Processor(te_encoder=te_encoder) if not (use_v3 or use_v8) else None

    df = scraper.get_race_data(race_id)
    if df is None:
        raise HTTPException(status_code=404, detail="レースデータが取得できません")

    df = scraper.enrich_data(df)

    # オッズ取得（v3で必要）
    all_odds = scraper.get_all_odds(race_id)
    win_odds_dict = all_odds.get('win', {})
    if 'horse_number' in df.columns:
        df['win_odds'] = df['horse_number'].apply(lambda x: win_odds_dict.get(int(x), 10) if pd.notna(x) else 10)

    # 特徴量作成
    if use_v8:
        df, features_to_use = create_features_v8(df)
    elif use_v3:
        df, features_to_use = create_features_v3(df)
    else:
        df = processor.process(df)
        features_to_use = model_features

    # 複勝オッズ取得
    place_odds_dict = all_odds.get('place', {})

    # 予測
    X = df[features_to_use].fillna(-1)
    df['prob'] = predict_with_model(model, X)
    df['pred_rank'] = df['prob'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values('prob', ascending=False)

    # レース情報
    race_num = race_id[-2:]
    race_name = df['race_name'].iloc[0] if 'race_name' in df.columns else f"{race_num}R"
    distance = int(df['distance'].iloc[0]) if 'distance' in df.columns else 0
    start_time = df['start_time'].iloc[0] if 'start_time' in df.columns else ""

    predictions = []
    for i, (_, row) in enumerate(df.iterrows()):  # 全馬を返す
        horse_num = int(row['horse_number']) if pd.notna(row.get('horse_number')) else 0
        win_odds = win_odds_dict.get(horse_num, 0)
        place_odds_data = place_odds_dict.get(horse_num, {})
        place_odds = place_odds_data.get('avg', 0) if place_odds_data else 0
        place_odds_min = place_odds_data.get('min', 0) if place_odds_data else 0
        place_odds_max = place_odds_data.get('max', 0) if place_odds_data else 0
        prob = float(row['prob'])

        # 勝率・複勝率を取得（0-1の範囲であるべき）
        raw_win_rate = float(row.get('horse_win_rate') or 0)
        raw_show_rate = float(row.get('horse_show_rate') or 0)

        win_rate = raw_win_rate * 100
        show_rate = raw_show_rate * 100

        # 期待値計算（複勝オッズ × AI確率）
        effective_place_odds = place_odds if place_odds > 0 else (win_odds / 3 if win_odds > 0 else 0)
        expected_value = prob * effective_place_odds if effective_place_odds > 0 else 0
        is_value = expected_value > 1.0

        # ========== 回収率ベース買い目判定 ==========
        bet_layer = None
        recommended_bet = 0
        min_odds = MIN_PLACE_ODDS_FOR_ROI.get(track_code, 2.0)

        if i == 0:  # 予測1位のみ対象（バックテストと同じ条件）
            if effective_place_odds >= min_odds:
                bet_layer = "roi_buy"
                recommended_bet = calculate_bet_amount(expected_value)
            elif effective_place_odds > 0:
                bet_layer = "watch"
                recommended_bet = 0

        predictions.append({
            "rank": i + 1,
            "number": horse_num,
            "name": row.get('horse_name', '不明'),
            "jockey": row.get('jockey_name', '不明'),
            "prob": round(prob, 3),
            "win_rate": round(win_rate, 1),
            "show_rate": round(show_rate, 1),
            "odds": win_odds,
            "place_odds": place_odds,
            "place_odds_min": place_odds_min,
            "place_odds_max": place_odds_max,
            "expected_value": round(expected_value, 2),
            "is_value": is_value,
            "bet_layer": bet_layer,
            "recommended_bet": recommended_bet
        })

    # 予測ログを保存（誤答分析用）
    metadata = {
        "race_name": race_name,
        "distance": distance,
        "track_condition": df['track_condition'].iloc[0] if 'track_condition' in df.columns else "不明",
        "weather": df['weather'].iloc[0] if 'weather' in df.columns else "不明",
        "field_size": len(df)
    }
    save_prediction_log(race_id, track_code, predictions, metadata)

    # ========== 回収率ベース買い目サマリー（単一レース用） ==========
    betting_picks = {
        "roi_buy": [],
        "watch": [],
        "total_bet": 0,
        "expected_return": 0,
        "min_odds_required": min_odds,
        "strategy": f"複勝オッズ{min_odds}倍以上のみ購入"
    }
    for pred in predictions:
        if pred["bet_layer"] in ["roi_buy", "watch"]:
            pick = {
                "number": pred["number"],
                "name": pred["name"],
                "prob": pred["prob"],
                "place_odds": pred["place_odds"],
                "expected_value": pred["expected_value"],
                "recommended_bet": pred["recommended_bet"],
                "reason": "オッズ条件クリア" if pred["bet_layer"] == "roi_buy" else f"オッズ{min_odds}倍未満"
            }
            betting_picks[pred["bet_layer"]].append(pick)
            if pred["bet_layer"] == "roi_buy":
                betting_picks["total_bet"] += pred["recommended_bet"]
                betting_picks["expected_return"] += pred["recommended_bet"] * pred["expected_value"]

    return {
        "id": race_num,
        "name": race_name,
        "distance": distance,
        "time": start_time,
        "field_size": len(df),
        "predictions": predictions,
        "betting_picks": betting_picks
    }


# ========== 軽量オッズ取得API ==========

class OddsRequest(BaseModel):
    race_id: str
    track_code: str


def get_race_result(race_id: str) -> list:
    """レース結果（着順）を取得（スクレイピング対策済み）"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        soup = fetch_with_retry(url, encoding='EUC-JP', delay=0.2)
        if not soup:
            return []

        results = []
        table = soup.find('table', class_='RaceTable01')
        if not table:
            table = soup.find('table', class_='Result_Table')
        if not table:
            return []

        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 3:
                continue

            rank_text = tds[0].get_text(strip=True)
            if not rank_text.isdigit():
                continue
            rank = int(rank_text)

            # 馬番を取得（tds[2]が馬番、tds[1]は枠番）
            horse_num = None
            if len(tds) >= 3:
                umaban_text = tds[2].get_text(strip=True)
                if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                    horse_num = int(umaban_text)

            if horse_num:
                results.append({"rank": rank, "number": horse_num})

        return sorted(results, key=lambda x: x["rank"])[:3]  # TOP3のみ
    except:
        return []


@app.post("/api/odds")
def get_odds_only(request: OddsRequest):
    """オッズと結果を取得（レース終了時は結果も含む）"""
    race_id = request.race_id
    track_code = request.track_code

    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    scraper = NARScraper(track_code, delay=0.2)

    # 単勝・複勝オッズを一括取得
    all_odds = scraper.get_all_odds(race_id)
    win_odds = all_odds.get('win', {})
    place_odds = all_odds.get('place', {})

    # 結果も取得（終了していれば返る、まだなら空）
    result = get_race_result(race_id)

    return {
        "race_id": race_id,
        "odds": win_odds,
        "place_odds": place_odds,  # 複勝オッズを追加
        "result": result if result else None
    }


# ========== 事前計算済み予測取得API ==========

@app.get("/api/predictions/{date}/{track_code}")
def get_precomputed_predictions(date: str, track_code: str):
    """事前計算済みの予測JSONを取得（層別買い目を動的に追加）"""
    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    predictions_file = BASE_DIR / "predictions" / date / f"{track_code}.json"

    if not predictions_file.exists():
        raise HTTPException(status_code=404, detail="予測データがありません")

    import json
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 回収率ベース買い目を動的に追加（キャッシュファイルに含まれていない場合）
    min_odds = MIN_PLACE_ODDS_FOR_ROI.get(track_code, 2.0)

    for race in data.get("races", []):
        for i, pred in enumerate(race.get("predictions", [])):
            # bet_layerがまだない場合のみ追加
            if "bet_layer" not in pred:
                bet_layer = None
                recommended_bet = 0

                if i == 0:  # 予測1位のみ対象
                    # キャッシュにはリアルタイムオッズがないので、オッズ取得後に再判定が必要
                    place_odds = pred.get("place_odds", 0) or 0
                    expected_value = pred.get("expected_value", 0) or 0

                    if place_odds >= min_odds:
                        bet_layer = "roi_buy"
                        recommended_bet = calculate_bet_amount(expected_value)
                    else:
                        # オッズ未取得 or オッズ不足 → 様子見
                        bet_layer = "watch"
                        recommended_bet = 0

                pred["bet_layer"] = bet_layer
                pred["recommended_bet"] = recommended_bet

    # betting_picksサマリーを追加
    data["betting_picks"] = {
        "roi_buy": [],
        "watch": [],
        "total_bet": 0,
        "expected_return": 0,
        "min_odds_required": min_odds,
        "strategy": f"複勝オッズ{min_odds}倍以上のみ購入"
    }
    for race in data.get("races", []):
        for pred in race.get("predictions", []):
            if pred.get("bet_layer") in ["roi_buy", "watch"]:
                pick = {
                    "race_id": race.get("id"),
                    "race_name": race.get("name"),
                    "race_time": race.get("time"),
                    "number": pred.get("number"),
                    "name": pred.get("name"),
                    "prob": pred.get("prob"),
                    "place_odds": pred.get("place_odds"),
                    "expected_value": pred.get("expected_value", 0),
                    "recommended_bet": pred.get("recommended_bet", 0),
                    "reason": "オッズ条件クリア" if pred["bet_layer"] == "roi_buy" else f"オッズ{min_odds}倍未満 or 未取得"
                }
                data["betting_picks"][pred["bet_layer"]].append(pick)
                if pred["bet_layer"] == "roi_buy":
                    data["betting_picks"]["total_bet"] += pred.get("recommended_bet", 0)
                    data["betting_picks"]["expected_return"] += pred.get("recommended_bet", 0) * pred.get("expected_value", 0)

    return data


@app.get("/api/predictions/{date}")
def list_available_predictions(date: str):
    """指定日の利用可能な予測一覧"""
    predictions_dir = BASE_DIR / "predictions" / date

    if not predictions_dir.exists():
        return {"date": date, "tracks": []}

    available = []
    for f in predictions_dir.glob("*.json"):
        track_code = f.stem
        if track_code in TRACKS:
            available.append({
                "code": track_code,
                "name": TRACKS[track_code]['name'],
                "emoji": TRACKS[track_code]['emoji']
            })

    return {"date": date, "tracks": available}


# ========== 精度評価API ==========

@app.get("/api/accuracy/{date}/{track_code}")
def get_accuracy(date: str, track_code: str):
    """指定日・競馬場の精度データを取得"""
    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    accuracy_file = BASE_DIR / "accuracy" / date / f"{track_code}.json"

    if not accuracy_file.exists():
        raise HTTPException(status_code=404, detail="精度データがありません")

    import json
    with open(accuracy_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


@app.get("/api/accuracy/{date}")
def get_daily_accuracy(date: str):
    """指定日の全体精度サマリーを取得"""
    summary_file = BASE_DIR / "accuracy" / date / "summary.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="精度データがありません")

    import json
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


@app.get("/api/accuracy")
def get_accuracy_history():
    """過去の精度データ一覧を取得"""
    accuracy_dir = BASE_DIR / "accuracy"

    if not accuracy_dir.exists():
        return {"dates": []}

    dates = []
    for d in sorted(accuracy_dir.iterdir(), reverse=True):
        if d.is_dir():
            summary_file = d / "summary.json"
            if summary_file.exists():
                import json
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                dates.append(summary)

    return {"history": dates[:30]}  # 直近30日分


# ========== モデル情報API ==========

@app.get("/api/models/{track_code}")
def get_model_info(track_code: str):
    """モデルのメタデータを取得"""
    if track_code not in TRACKS:
        raise HTTPException(status_code=400, detail="無効な競馬場コード")

    model_path = TRACKS[track_code]['model']
    meta_path = model_path.replace('.pkl', '_meta.json')
    meta_file = BASE_DIR / meta_path

    result = None
    if meta_file.exists():
        import json
        with open(meta_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
    else:
        # メタデータJSONがない場合、pklから読み込み試行
        model_file = BASE_DIR / model_path
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                if 'metadata' in data:
                    result = data['metadata']
            except:
                pass

    if result is None:
        raise HTTPException(status_code=404, detail="モデル情報がありません")

    # フロントエンドが期待するフィールドを追加
    if 'data_count' not in result:
        # CSVからデータ数を取得
        track_name = TRACKS[track_code]['model'].split('_')[1].replace('.pkl', '')
        csv_path = BASE_DIR / f'data/races_{track_name}.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                result['data_count'] = len(df)
                # 日付範囲を取得
                if 'race_id' in df.columns:
                    race_ids = df['race_id'].astype(str)
                    dates = race_ids.str[0:4] + '-' + race_ids.str[6:8] + '-' + race_ids.str[8:10]
                    result['date_range'] = {
                        'from': dates.min(),
                        'to': dates.max()
                    }
            except:
                result['data_count'] = 0
                result['date_range'] = {'from': 'N/A', 'to': 'N/A'}
        else:
            result['data_count'] = 0
            result['date_range'] = {'from': 'N/A', 'to': 'N/A'}

    if 'date_range' not in result:
        result['date_range'] = {'from': 'N/A', 'to': 'N/A'}

    return result


@app.get("/api/models")
def get_all_models_info():
    """全モデルの情報を取得"""
    models_info = []

    for code, info in TRACKS.items():
        model_path = info['model']
        meta_path = model_path.replace('.pkl', '_meta.json')
        meta_file = BASE_DIR / meta_path
        model_file = BASE_DIR / model_path

        model_data = {
            "code": code,
            "name": info['name'],
            "emoji": info['emoji'],
            "model_exists": model_file.exists(),
            "metadata": None
        }

        if meta_file.exists():
            try:
                import json
                with open(meta_file, 'r', encoding='utf-8') as f:
                    model_data["metadata"] = json.load(f)
            except:
                pass

        models_info.append(model_data)

    return {"models": models_info}


# ========== 誤答分析API（SSE対応） ==========

def analyze_get_race_result(race_id: str) -> list:
    """レース結果（着順）を取得（分析用・スクレイピング対策済み）"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        soup = fetch_with_retry(url, encoding='EUC-JP', delay=0.3)
        if not soup:
            return []

        results = []
        table = soup.find('table', class_='RaceTable01')
        if not table:
            table = soup.find('table', class_='Result_Table')
        if not table:
            return []

        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 3:
                continue

            rank_text = tds[0].get_text(strip=True)
            if not rank_text.isdigit():
                continue
            rank = int(rank_text)

            umaban_text = tds[2].get_text(strip=True)
            if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                horse_num = int(umaban_text)
                results.append({"rank": rank, "number": horse_num})

        return sorted(results, key=lambda x: x["rank"])
    except Exception as e:
        print(f"Error fetching result for {race_id}: {e}")
        return []


def compare_prediction(prediction_log: dict, result: list) -> dict:
    """予測と結果を照合"""
    if not result:
        return None

    predictions = prediction_log["predictions"]
    metadata = prediction_log.get("metadata", {})

    pred_top3 = [p["number"] for p in predictions[:3]]
    pred_1st = predictions[0]["number"] if predictions else None
    actual_top3 = [r["number"] for r in result[:3]]
    actual_1st = result[0]["number"] if result else None

    win_hit = (pred_1st == actual_1st)
    show_hit = (pred_1st in actual_top3)

    # 予測1位の馬が実際に何着だったか
    pred_1st_actual_rank = None
    for r in result:
        if r["number"] == pred_1st:
            pred_1st_actual_rank = r["rank"]
            break

    # エラータイプ分類
    error_type = None
    if not show_hit:
        if pred_1st_actual_rank is None:
            error_type = "出走取消"
        elif pred_1st_actual_rank >= 10:
            error_type = "大外れ(10着以下)"
        elif pred_1st_actual_rank >= 6:
            error_type = "中外れ(6-9着)"
        elif pred_1st_actual_rank >= 4:
            error_type = "惜しい(4-5着)"

    return {
        "race_id": prediction_log["race_id"],
        "track_name": prediction_log.get("track_name", "不明"),
        "race_name": metadata.get("race_name", "不明"),
        "pred_1st": pred_1st,
        "actual_1st": actual_1st,
        "win_hit": win_hit,
        "show_hit": show_hit,
        "pred_1st_actual_rank": pred_1st_actual_rank,
        "error_type": error_type,
        "metadata": metadata
    }


async def analyze_stream(date: str):
    """分析をストリーミングで実行"""
    log_dir = BASE_DIR / "prediction_logs" / date

    if not log_dir.exists():
        yield f"data: {json.dumps({'type': 'error', 'message': '予測ログがありません'})}\n\n"
        return

    log_files = list(log_dir.glob("*.json"))
    total = len(log_files)

    if total == 0:
        yield f"data: {json.dumps({'type': 'error', 'message': '予測ログがありません'})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

    comparisons = []
    for i, log_file in enumerate(log_files):
        with open(log_file, 'r', encoding='utf-8') as f:
            prediction_log = json.load(f)

        race_id = prediction_log["race_id"]

        # 進捗を送信
        yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total, 'race_id': race_id})}\n\n"

        # 結果を取得
        result = analyze_get_race_result(race_id)
        if result:
            comparison = compare_prediction(prediction_log, result)
            if comparison:
                comparisons.append(comparison)

        # 少し待機（サーバー負荷軽減）
        await asyncio.sleep(0.3)

    # 集計
    if not comparisons:
        yield f"data: {json.dumps({'type': 'error', 'message': '照合できるデータがありません'})}\n\n"
        return

    # 統計を計算
    total_races = len(comparisons)
    win_hits = sum(1 for c in comparisons if c["win_hit"])
    show_hits = sum(1 for c in comparisons if c["show_hit"])

    # 馬場状態別
    by_track_condition = defaultdict(lambda: {"total": 0, "show_hits": 0})
    # 天気別
    by_weather = defaultdict(lambda: {"total": 0, "show_hits": 0})
    # 距離別
    by_distance = defaultdict(lambda: {"total": 0, "show_hits": 0})
    # エラータイプ
    error_types = defaultdict(int)

    for c in comparisons:
        meta = c.get("metadata", {})

        # 馬場状態
        track_cond = meta.get("track_condition", "不明")
        by_track_condition[track_cond]["total"] += 1
        if c["show_hit"]:
            by_track_condition[track_cond]["show_hits"] += 1

        # 天気
        weather = meta.get("weather", "不明")
        by_weather[weather]["total"] += 1
        if c["show_hit"]:
            by_weather[weather]["show_hits"] += 1

        # 距離
        distance = meta.get("distance", 0)
        if distance < 1400:
            dist_cat = "短距離(<1400m)"
        elif distance < 1800:
            dist_cat = "中距離(1400-1800m)"
        else:
            dist_cat = "長距離(>1800m)"
        by_distance[dist_cat]["total"] += 1
        if c["show_hit"]:
            by_distance[dist_cat]["show_hits"] += 1

        # エラータイプ
        if c.get("error_type"):
            error_types[c["error_type"]] += 1

    # 結果を送信
    result_data = {
        "type": "result",
        "date": date,
        "summary": {
            "total_races": total_races,
            "win_hits": win_hits,
            "win_rate": round(win_hits / total_races * 100, 1) if total_races > 0 else 0,
            "show_hits": show_hits,
            "show_rate": round(show_hits / total_races * 100, 1) if total_races > 0 else 0
        },
        "by_track_condition": {k: v for k, v in by_track_condition.items()},
        "by_weather": {k: v for k, v in by_weather.items()},
        "by_distance": {k: v for k, v in by_distance.items()},
        "error_types": dict(error_types),
        "details": comparisons
    }

    yield f"data: {json.dumps(result_data, ensure_ascii=False)}\n\n"

    # 結果をファイルに保存
    output_dir = BASE_DIR / "analysis_reports" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    yield f"data: {json.dumps({'type': 'complete', 'saved_to': str(output_file)})}\n\n"


@app.get("/api/analyze/{date}")
def analyze_predictions(date: str):
    """予測の誤答分析を実行（通常API版）"""
    log_dir = BASE_DIR / "prediction_logs" / date

    if not log_dir.exists():
        raise HTTPException(status_code=404, detail="予測ログがありません")

    log_files = list(log_dir.glob("*.json"))
    total = len(log_files)

    if total == 0:
        raise HTTPException(status_code=404, detail="予測ログがありません")

    comparisons = []
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            prediction_log = json.load(f)

        race_id = prediction_log["race_id"]

        # 結果を取得
        result = analyze_get_race_result(race_id)
        if result:
            comparison = compare_prediction(prediction_log, result)
            if comparison:
                comparisons.append(comparison)

        # サーバー負荷軽減
        time.sleep(0.3)

    if not comparisons:
        raise HTTPException(status_code=404, detail="照合できるデータがありません")

    # 統計を計算
    total_races = len(comparisons)
    win_hits = sum(1 for c in comparisons if c["win_hit"])
    show_hits = sum(1 for c in comparisons if c["show_hit"])

    # 馬場状態別
    by_track_condition = defaultdict(lambda: {"total": 0, "show_hits": 0})
    by_weather = defaultdict(lambda: {"total": 0, "show_hits": 0})
    by_distance = defaultdict(lambda: {"total": 0, "show_hits": 0})
    error_types = defaultdict(int)

    for c in comparisons:
        meta = c.get("metadata", {})

        track_cond = meta.get("track_condition", "不明")
        by_track_condition[track_cond]["total"] += 1
        if c["show_hit"]:
            by_track_condition[track_cond]["show_hits"] += 1

        weather = meta.get("weather", "不明")
        by_weather[weather]["total"] += 1
        if c["show_hit"]:
            by_weather[weather]["show_hits"] += 1

        distance = meta.get("distance", 0)
        if distance < 1400:
            dist_cat = "短距離(<1400m)"
        elif distance < 1800:
            dist_cat = "中距離(1400-1800m)"
        else:
            dist_cat = "長距離(>1800m)"
        by_distance[dist_cat]["total"] += 1
        if c["show_hit"]:
            by_distance[dist_cat]["show_hits"] += 1

        if c.get("error_type"):
            error_types[c["error_type"]] += 1

    result_data = {
        "date": date,
        "summary": {
            "total_races": total_races,
            "win_hits": win_hits,
            "win_rate": round(win_hits / total_races * 100, 1) if total_races > 0 else 0,
            "show_hits": show_hits,
            "show_rate": round(show_hits / total_races * 100, 1) if total_races > 0 else 0
        },
        "by_track_condition": {k: v for k, v in by_track_condition.items()},
        "by_weather": {k: v for k, v in by_weather.items()},
        "by_distance": {k: v for k, v in by_distance.items()},
        "error_types": dict(error_types),
        "details": comparisons
    }

    # 結果をファイルに保存
    output_dir = BASE_DIR / "analysis_reports" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return result_data


@app.get("/api/analysis/{date}")
def get_analysis_report(date: str):
    """保存済みの分析レポートを取得"""
    report_file = BASE_DIR / "analysis_reports" / date / "report.json"

    if not report_file.exists():
        raise HTTPException(status_code=404, detail="分析レポートがありません")

    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)


# ========== 買い目表示HTML ==========
from fastapi.responses import HTMLResponse

@app.get("/betting/{track_code}/{date}", response_class=HTMLResponse)
def show_betting_picks(track_code: str, date: str):
    """買い目を見やすく表示するHTMLページ"""
    from pydantic import BaseModel

    class TempRequest(BaseModel):
        track_code: str
        date: str

    # 予測を実行
    try:
        request = TempRequest(track_code=track_code, date=date)
        result = predict(PredictRequest(track_code=track_code, date=date))
    except Exception as e:
        return f"<html><body><h1>エラー</h1><p>{str(e)}</p></body></html>"

    track_info = result.get("track", {})
    betting = result.get("betting_picks", {})
    v6_buys = betting.get("v6_buy", [])
    watches = betting.get("watch", [])

    # HTML生成
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{track_info.get('emoji', '')} {track_info.get('name', '')} 買い目 - {date}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.8em;
        }}
        .strategy {{
            text-align: center;
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .strategy .roi {{ color: #4ade80; font-size: 1.2em; font-weight: bold; }}
        .summary {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .summary-item {{
            background: rgba(255,255,255,0.1);
            padding: 15px 25px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-item .value {{ font-size: 1.5em; font-weight: bold; color: #60a5fa; }}
        .section {{ margin-bottom: 30px; }}
        .section-title {{
            font-size: 1.3em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #4ade80;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-title.watch {{ border-bottom-color: #fbbf24; }}
        .pick-card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .pick-card.buy {{
            border-left: 4px solid #4ade80;
            background: rgba(74, 222, 128, 0.1);
        }}
        .pick-card.watch {{
            border-left: 4px solid #fbbf24;
            background: rgba(251, 191, 36, 0.1);
        }}
        .race-info {{
            font-size: 0.9em;
            color: #9ca3af;
            min-width: 80px;
        }}
        .horse-info {{
            flex: 1;
            min-width: 150px;
        }}
        .horse-num {{
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            text-align: center;
            background: #3b82f6;
            border-radius: 50%;
            font-weight: bold;
            margin-right: 10px;
        }}
        .horse-name {{ font-weight: bold; font-size: 1.1em; }}
        .stats {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
            min-width: 60px;
        }}
        .stat-label {{ font-size: 0.75em; color: #9ca3af; }}
        .stat-value {{ font-weight: bold; }}
        .stat-value.high {{ color: #4ade80; }}
        .stat-value.medium {{ color: #60a5fa; }}
        .confidence {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .confidence.high {{ background: #4ade80; color: #000; }}
        .confidence.medium {{ background: #60a5fa; color: #000; }}
        .no-picks {{
            text-align: center;
            padding: 40px;
            color: #9ca3af;
            font-size: 1.1em;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #6b7280;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{track_info.get('emoji', '')} {track_info.get('name', '')} 買い目</h1>
        <p style="text-align:center; color:#9ca3af; margin-bottom:20px;">{date}</p>

        <div class="strategy">
            <p>戦略: <span class="roi">{betting.get('strategy', '')}</span></p>
        </div>

        <div class="summary">
            <div class="summary-item">
                <div class="stat-label">推奨買い目</div>
                <div class="value">{len(v6_buys)}R</div>
            </div>
            <div class="summary-item">
                <div class="stat-label">合計賭け金</div>
                <div class="value">¥{betting.get('total_bet', 0):,}</div>
            </div>
            <div class="summary-item">
                <div class="stat-label">期待リターン</div>
                <div class="value" style="color:#4ade80;">¥{int(betting.get('expected_return', 0)):,}</div>
            </div>
        </div>
"""

    # 買い目セクション
    if v6_buys:
        html += """
        <div class="section">
            <div class="section-title">🎯 買い目（複勝）</div>
"""
        for pick in v6_buys:
            conf_class = "high" if pick.get("confidence") == "高" else "medium"
            html += f"""
            <div class="pick-card buy">
                <div class="race-info">
                    <div>{pick.get('race_id', '')}R</div>
                    <div>{pick.get('race_time', '')}</div>
                </div>
                <div class="horse-info">
                    <span class="horse-num">{pick.get('number', '')}</span>
                    <span class="horse-name">{pick.get('name', '')}</span>
                </div>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">AI確率</div>
                        <div class="stat-value high">{pick.get('prob', 0):.1%}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">確率差</div>
                        <div class="stat-value medium">{pick.get('prob_diff', 0):.1%}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">単勝</div>
                        <div class="stat-value">{pick.get('odds', 0)}倍</div>
                    </div>
                </div>
                <span class="confidence {conf_class}">{pick.get('confidence', '中')}</span>
            </div>
"""
        html += "</div>"
    else:
        html += '<div class="no-picks">🤔 本日は推奨買い目がありません</div>'

    # 様子見セクション
    if watches:
        html += """
        <div class="section">
            <div class="section-title watch">👀 様子見（確率差不足）</div>
"""
        for pick in watches[:5]:  # 最大5件
            html += f"""
            <div class="pick-card watch">
                <div class="race-info">
                    <div>{pick.get('race_id', '')}R</div>
                    <div>{pick.get('race_time', '')}</div>
                </div>
                <div class="horse-info">
                    <span class="horse-num">{pick.get('number', '')}</span>
                    <span class="horse-name">{pick.get('name', '')}</span>
                </div>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">AI確率</div>
                        <div class="stat-value">{pick.get('prob', 0):.1%}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">確率差</div>
                        <div class="stat-value" style="color:#fbbf24;">{pick.get('prob_diff', 0):.1%}</div>
                    </div>
                </div>
                <span style="color:#9ca3af; font-size:0.85em;">{pick.get('reason', '')}</span>
            </div>
"""
        html += "</div>"

    html += f"""
        <div class="footer">
            <p>v6 選択的ベッティング戦略</p>
            <p>期待ROI: 大井105.7% / 川崎114.7%</p>
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/betting", response_class=HTMLResponse)
def betting_index():
    """買い目トップページ"""
    today = datetime.now().strftime("%Y-%m-%d")
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>競馬AI 買い目</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        h1 {{ margin-bottom: 30px; }}
        .tracks {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            max-width: 600px;
            width: 100%;
        }}
        a {{
            display: block;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
            text-decoration: none;
            color: #fff;
            text-align: center;
            transition: all 0.3s;
        }}
        a:hover {{
            background: rgba(255,255,255,0.2);
            transform: translateY(-3px);
        }}
        .emoji {{ font-size: 2em; display: block; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>🏇 競馬AI 買い目</h1>
    <p style="margin-bottom:20px; color:#9ca3af;">{today}</p>
    <div class="tracks">
"""
    for code, info in TRACKS.items():
        html += f'<a href="/betting/{code}/{today}"><span class="emoji">{info["emoji"]}</span>{info["name"]}</a>\n'

    html += """
    </div>
</body>
</html>
"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
