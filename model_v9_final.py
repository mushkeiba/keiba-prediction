# -*- coding: utf-8 -*-
"""
モデル V9 - 最終版
川崎: 前走着順ベース
大井: 騎手成績ベース
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 最適設定
CONFIG = {
    'kawasaki': {
        'feature_type': 'last_rank',
        'threshold': 0.03,  # 閾値を少し上げて精度向上
        'train_ratio': 0.6,
    },
    'ohi': {
        'feature_type': 'jockey',
        'threshold': 0.12,
        'train_ratio': 0.6,
    }
}

def load_data(track):
    """データ読み込み"""
    if track == 'kawasaki':
        df = pd.read_csv('data/races_kawasaki.csv')
    else:
        df = pd.read_csv('data/races_ohi.csv')

    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank'])
    df = df[df['rank'] > 0]
    df['target'] = (df['rank'] <= 3).astype(int)

    return df

def create_features_kawasaki(df):
    """川崎用特徴量: 前走着順"""
    features = pd.DataFrame(index=df.index)
    features['f_last_rank'] = pd.to_numeric(df['last_rank'], errors='coerce').fillna(10)
    features['f_last_rank_good'] = (features['f_last_rank'] <= 3).astype(int)
    return features

def create_features_ohi(df):
    """大井用特徴量: 騎手成績"""
    features = pd.DataFrame(index=df.index)
    features['f_jockey_win'] = pd.to_numeric(df['jockey_win_rate'], errors='coerce').fillna(0)
    features['f_jockey_show'] = pd.to_numeric(df['jockey_show_rate'], errors='coerce').fillna(0)
    features['f_jockey_rank'] = df.groupby('race_id')['jockey_win_rate'].transform(
        lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False)
    ).fillna(8)
    return features

def train_and_save_model(track):
    """モデル訓練・保存"""
    print(f"\n{'='*60}")
    print(f"訓練開始: {track.upper()}")
    print("=" * 60)

    config = CONFIG[track]
    df = load_data(track)
    print(f"データ数: {len(df):,}件")

    # 特徴量作成
    if track == 'kawasaki':
        features = create_features_kawasaki(df)
    else:
        features = create_features_ohi(df)

    print(f"特徴量: {list(features.columns)}")

    # 全データで学習（本番用）
    X = features
    y = df['target']

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)

    # 保存
    os.makedirs('models', exist_ok=True)
    model_path = f'models/model_{track}_v9.pkl'

    model_data = {
        'model': model,
        'feature_columns': list(features.columns),
        'feature_type': config['feature_type'],
        'threshold': config['threshold'],
        'track': track,
        'version': 'v9'
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"保存完了: {model_path}")
    print(f"閾値: {config['threshold']}")

    return model_data

def validate_model(track):
    """モデル検証"""
    print(f"\n{'='*60}")
    print(f"検証: {track.upper()}")
    print("=" * 60)

    config = CONFIG[track]
    df = load_data(track)

    # 特徴量作成
    if track == 'kawasaki':
        features = create_features_kawasaki(df)
    else:
        features = create_features_ohi(df)

    # 時系列分割
    race_ids = df['race_id'].unique()
    train_size = int(len(race_ids) * config['train_ratio'])
    train_races = race_ids[:train_size]
    test_races = race_ids[train_size:]

    train_mask = df['race_id'].isin(train_races)
    test_mask = df['race_id'].isin(test_races)

    # 訓練
    model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
    model.fit(features[train_mask], df.loc[train_mask, 'target'])

    # テスト
    test_df = df[test_mask].copy()
    test_df['pred_prob'] = model.predict_proba(features[test_mask])[:, 1]

    threshold = config['threshold']
    results = []

    for race_id in test_df['race_id'].unique():
        race_df = test_df[test_df['race_id'] == race_id].sort_values('pred_prob', ascending=False)
        if len(race_df) < 2:
            continue

        gap = race_df.iloc[0]['pred_prob'] - race_df.iloc[1]['pred_prob']
        if gap >= threshold:
            top1 = race_df.iloc[0]
            hit = int(top1['rank']) <= 3

            payout = 0
            if hit:
                place_odds = top1.get('place_odds', np.nan)
                if pd.notna(place_odds) and place_odds > 0:
                    payout = float(place_odds) * 100
                else:
                    payout = 150  # デフォルト

            results.append({
                'race_id': race_id,
                'hit': hit,
                'bet': 100,
                'payout': payout
            })

    if results:
        total_bets = len(results)
        total_hits = sum(r['hit'] for r in results)
        hit_rate = total_hits / total_bets
        total_payout = sum(r['payout'] for r in results)
        roi = total_payout / (total_bets * 100)
        profit = total_payout - total_bets * 100

        print(f"テスト期間: {len(test_races)}レース")
        print(f"賭け数: {total_bets}件")
        print(f"的中数: {total_hits}件 ({hit_rate*100:.1f}%)")
        print(f"ROI: {roi*100:.1f}%")
        print(f"収支: {profit:+,.0f}円")

        return {'bets': total_bets, 'hit_rate': hit_rate, 'roi': roi, 'profit': profit}

    return None

def main():
    print("=" * 60)
    print("モデル V9 - 最終版 訓練・検証")
    print("=" * 60)

    results = {}

    for track in ['kawasaki', 'ohi']:
        # 検証
        result = validate_model(track)
        if result:
            results[track] = result

        # 訓練・保存
        train_and_save_model(track)

    # サマリー
    print("\n" + "=" * 60)
    print("【最終結果サマリー】")
    print("=" * 60)

    total_bets = 0
    total_profit = 0

    for track, result in results.items():
        print(f"\n{track.upper()}:")
        print(f"  賭け: {result['bets']}件")
        print(f"  的中率: {result['hit_rate']*100:.1f}%")
        print(f"  ROI: {result['roi']*100:.1f}%")
        print(f"  収支: {result['profit']:+,.0f}円")
        total_bets += result['bets']
        total_profit += result['profit']

    print(f"\n合計: {total_bets}件賭け, 収支 {total_profit:+,.0f}円")

if __name__ == '__main__':
    main()
