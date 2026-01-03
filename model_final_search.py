# -*- coding: utf-8 -*-
"""
最終モデル探索 - 賭け数とROIのバランスを最適化
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data(track):
    """データ読み込み"""
    if track == 'kawasaki':
        df = pd.read_csv('data/races_kawasaki.csv')
    else:
        df = pd.read_csv('data/races_ohi.csv')

    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank'])
    df = df[df['rank'] > 0]

    # place_oddsがあるレースのみ
    races_with_odds = df[df['place_odds'] > 0]['race_id'].unique()
    df = df[df['race_id'].isin(races_with_odds)]
    df['target'] = (df['rank'] <= 3).astype(int)

    return df

def create_all_features(df, feature_type):
    """各種特徴量セット"""
    features = pd.DataFrame(index=df.index)

    if feature_type == 'basic':
        for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
            features[f'f_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    elif feature_type == 'relative':
        for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
            val = pd.to_numeric(df[col], errors='coerce').fillna(0)
            features[f'f_{col}'] = val
            mean = df.groupby('race_id')[col].transform(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).fillna(0)
            features[f'f_{col}_vs_mean'] = val - mean

    elif feature_type == 'full':
        for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
            val = pd.to_numeric(df[col], errors='coerce').fillna(0)
            features[f'f_{col}'] = val
            features[f'f_{col}_rank'] = df.groupby('race_id')[col].transform(
                lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False)
            ).fillna(8)
            mean = df.groupby('race_id')[col].transform(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).fillna(0)
            features[f'f_{col}_vs_mean'] = val - mean

        if 'horse_recent_show_rate' in df.columns:
            features['f_recent_show'] = pd.to_numeric(df['horse_recent_show_rate'], errors='coerce').fillna(0)
        if 'last_rank' in df.columns:
            features['f_last_rank'] = pd.to_numeric(df['last_rank'], errors='coerce').fillna(10)

    elif feature_type == 'last_rank':
        features['f_last_rank'] = pd.to_numeric(df['last_rank'], errors='coerce').fillna(10)
        features['f_last_rank_good'] = (features['f_last_rank'] <= 3).astype(int)

    elif feature_type == 'jockey':
        features['f_jockey_win'] = pd.to_numeric(df['jockey_win_rate'], errors='coerce').fillna(0)
        features['f_jockey_show'] = pd.to_numeric(df['jockey_show_rate'], errors='coerce').fillna(0)
        features['f_jockey_rank'] = df.groupby('race_id')['jockey_win_rate'].transform(
            lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False)
        ).fillna(8)

    elif feature_type == 'combined':
        # 基本
        for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
            features[f'f_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # 前走
        features['f_last_rank'] = pd.to_numeric(df['last_rank'], errors='coerce').fillna(10)
        # 直近
        if 'horse_recent_show_rate' in df.columns:
            features['f_recent_show'] = pd.to_numeric(df['horse_recent_show_rate'], errors='coerce').fillna(0)

    return features

def run_backtest(df, features, threshold, train_ratio):
    """バックテスト"""
    race_ids = df['race_id'].unique()
    train_size = int(len(race_ids) * train_ratio)
    train_races = race_ids[:train_size]
    test_races = race_ids[train_size:]

    train_mask = df['race_id'].isin(train_races)
    test_mask = df['race_id'].isin(test_races)

    if train_mask.sum() < 50 or test_mask.sum() < 50:
        return None

    model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
    model.fit(features[train_mask], df.loc[train_mask, 'target'])

    test_df = df[test_mask].copy()
    test_df['pred_prob'] = model.predict_proba(features[test_mask])[:, 1]

    results = []
    for race_id in test_df['race_id'].unique():
        race_df = test_df[test_df['race_id'] == race_id].sort_values('pred_prob', ascending=False)
        if len(race_df) < 2:
            continue
        gap = race_df.iloc[0]['pred_prob'] - race_df.iloc[1]['pred_prob']
        if gap >= threshold:
            top1 = race_df.iloc[0]
            hit = int(top1['rank']) <= 3
            payout = float(top1['place_odds']) * 100 if hit and top1['place_odds'] > 0 else 0
            results.append({'hit': hit, 'bet': 100, 'payout': payout})

    if len(results) < 5:
        return None

    total_bets = len(results)
    total_hits = sum(r['hit'] for r in results)
    total_payout = sum(r['payout'] for r in results)

    return {
        'bets': total_bets,
        'hits': total_hits,
        'hit_rate': total_hits / total_bets,
        'roi': total_payout / (total_bets * 100),
        'profit': total_payout - total_bets * 100
    }

def main():
    print("=" * 80)
    print("最終モデル探索 - 賭け数とROIのバランス最適化")
    print("=" * 80)

    FEATURE_TYPES = ['basic', 'relative', 'full', 'last_rank', 'jockey', 'combined']
    THRESHOLDS = [0.00, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
    TRAIN_RATIOS = [0.5, 0.6, 0.7]

    all_results = []

    for track in ['kawasaki', 'ohi']:
        print(f"\n{'='*80}")
        print(f"競馬場: {track.upper()}")
        print("=" * 80)

        df = load_data(track)
        print(f"データ: {len(df):,}件 / {df['race_id'].nunique()}レース")

        for feat_type in FEATURE_TYPES:
            features = create_all_features(df, feat_type)

            for threshold in THRESHOLDS:
                for train_ratio in TRAIN_RATIOS:
                    result = run_backtest(df, features, threshold, train_ratio)

                    if result is None:
                        continue

                    result['track'] = track
                    result['feature_type'] = feat_type
                    result['threshold'] = threshold
                    result['train_ratio'] = train_ratio

                    # スコア: ROI * log(賭け数) で賭け数も考慮
                    result['score'] = result['roi'] * np.log1p(result['bets'])
                    all_results.append(result)

    # 結果集計
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("【結果サマリー】ROI 100%以上 かつ 賭け数10以上")
    print("=" * 80)

    good = results_df[(results_df['roi'] >= 1.0) & (results_df['bets'] >= 10)]
    good = good.sort_values('score', ascending=False)

    for track in ['kawasaki', 'ohi']:
        track_good = good[good['track'] == track]
        print(f"\n【{track.upper()}】")
        if len(track_good) == 0:
            print("  該当なし")
        else:
            for i, row in track_good.head(10).iterrows():
                print(f"  {row['feature_type']:10s} | 閾値{row['threshold']:.2f} | 訓練{row['train_ratio']:.0%} | "
                      f"賭け{row['bets']:3d}件 | 的中{row['hit_rate']*100:5.1f}% | "
                      f"ROI {row['roi']*100:6.1f}% | {row['profit']:+,.0f}円")

    # 推奨モデル
    print("\n" + "=" * 80)
    print("【推奨モデル】")
    print("=" * 80)

    for track in ['kawasaki', 'ohi']:
        track_good = good[good['track'] == track]
        if len(track_good) > 0:
            best = track_good.iloc[0]
            print(f"\n{track.upper()}:")
            print(f"  特徴量: {best['feature_type']}")
            print(f"  閾値: {best['threshold']:.2f}")
            print(f"  訓練比率: {best['train_ratio']:.0%}")
            print(f"  結果: {best['bets']}件賭け | 的中{best['hit_rate']*100:.1f}% | ROI {best['roi']*100:.1f}%")

    results_df.to_csv('data/model_final_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n結果保存: data/model_final_results.csv")

if __name__ == '__main__':
    main()
