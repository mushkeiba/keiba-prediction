# -*- coding: utf-8 -*-
"""
モデル改善イテレーション V2
- place_oddsがあるデータのみ使用
- より正確なROI計算
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data(track):
    """データ読み込み - place_oddsがあるデータのみ"""
    if track == 'kawasaki':
        df = pd.read_csv('data/races_kawasaki.csv')
    else:
        df = pd.read_csv('data/races_ohi.csv')

    # 基本クリーニング
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank'])
    df = df[df['rank'] > 0]

    # place_oddsがあるレースのみ
    races_with_odds = df[df['place_odds'].notna()]['race_id'].unique()
    df = df[df['race_id'].isin(races_with_odds)]

    # 目的変数
    df['target'] = (df['rank'] <= 3).astype(int)

    return df

def create_features(df, feature_type):
    """特徴量作成"""
    features = pd.DataFrame(index=df.index)

    base_cols = ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']

    if feature_type == 'basic':
        # 基本のみ
        for col in base_cols:
            if col in df.columns:
                features[f'f_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    elif feature_type == 'rank':
        # ランキング追加
        for col in base_cols:
            if col in df.columns:
                features[f'f_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                features[f'f_{col}_rank'] = df.groupby('race_id')[col].transform(
                    lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False, method='min')
                ).fillna(8)

    elif feature_type == 'relative':
        # 相対値追加
        for col in base_cols:
            if col in df.columns:
                val = pd.to_numeric(df[col], errors='coerce').fillna(0)
                features[f'f_{col}'] = val
                mean = df.groupby('race_id')[col].transform(
                    lambda x: pd.to_numeric(x, errors='coerce').mean()
                ).fillna(0)
                features[f'f_{col}_vs_mean'] = val - mean

    elif feature_type == 'combo':
        # 組み合わせ
        for col in base_cols:
            if col in df.columns:
                features[f'f_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        h_show = pd.to_numeric(df.get('horse_show_rate', 0), errors='coerce').fillna(0)
        j_show = pd.to_numeric(df.get('jockey_show_rate', 0), errors='coerce').fillna(0)
        h_win = pd.to_numeric(df.get('horse_win_rate', 0), errors='coerce').fillna(0)
        j_win = pd.to_numeric(df.get('jockey_win_rate', 0), errors='coerce').fillna(0)

        features['f_show_combo'] = h_show * j_show
        features['f_win_combo'] = h_win * j_win
        features['f_avg_show'] = (h_show + j_show) / 2
        features['f_avg_win'] = (h_win + j_win) / 2

    elif feature_type == 'full':
        # 全部入り
        for col in base_cols:
            if col in df.columns:
                val = pd.to_numeric(df[col], errors='coerce').fillna(0)
                features[f'f_{col}'] = val
                features[f'f_{col}_rank'] = df.groupby('race_id')[col].transform(
                    lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False, method='min')
                ).fillna(8)
                mean = df.groupby('race_id')[col].transform(
                    lambda x: pd.to_numeric(x, errors='coerce').mean()
                ).fillna(0)
                features[f'f_{col}_vs_mean'] = val - mean

        # 追加特徴量
        if 'horse_recent_show_rate' in df.columns:
            features['f_recent_show'] = pd.to_numeric(df['horse_recent_show_rate'], errors='coerce').fillna(0)
        if 'last_rank' in df.columns:
            features['f_last_rank'] = pd.to_numeric(df['last_rank'], errors='coerce').fillna(10)

    return features

def run_backtest(df, features, threshold, train_ratio=0.6):
    """バックテスト実行"""
    # 時系列分割
    race_ids = df['race_id'].unique()
    train_size = int(len(race_ids) * train_ratio)
    train_races = race_ids[:train_size]
    test_races = race_ids[train_size:]

    train_mask = df['race_id'].isin(train_races)
    test_mask = df['race_id'].isin(test_races)

    X_train = features[train_mask]
    y_train = df.loc[train_mask, 'target']
    X_test = features[test_mask]
    test_df = df[test_mask].copy()

    if len(X_train) < 100 or len(X_test) < 50:
        return None

    # モデル学習
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # 予測
    proba = model.predict_proba(X_test)[:, 1]
    test_df['pred_prob'] = proba

    # レースごとに予測
    results = []
    for race_id in test_df['race_id'].unique():
        race_df = test_df[test_df['race_id'] == race_id].copy()
        race_df = race_df.sort_values('pred_prob', ascending=False)

        if len(race_df) < 2:
            continue

        top1_prob = race_df.iloc[0]['pred_prob']
        top2_prob = race_df.iloc[1]['pred_prob']
        gap = top1_prob - top2_prob

        if gap >= threshold:
            top1 = race_df.iloc[0]
            actual_rank = int(top1['rank'])
            hit = actual_rank <= 3

            # 払戻計算
            payout = 0
            if hit:
                place_odds = top1.get('place_odds', np.nan)
                if pd.notna(place_odds):
                    payout = float(place_odds) * 100
                else:
                    payout = 150  # デフォルト

            results.append({
                'race_id': race_id,
                'gap': gap,
                'hit': hit,
                'bet': 100,
                'payout': payout,
                'place_odds': top1.get('place_odds', np.nan)
            })

    if len(results) < 5:
        return None

    results_df = pd.DataFrame(results)
    total_bets = len(results_df)
    total_hits = results_df['hit'].sum()
    hit_rate = total_hits / total_bets
    total_bet_amount = results_df['bet'].sum()
    total_payout = results_df['payout'].sum()
    roi = total_payout / total_bet_amount
    profit = total_payout - total_bet_amount

    return {
        'bets': total_bets,
        'hits': total_hits,
        'hit_rate': hit_rate,
        'roi': roi,
        'profit': profit
    }

def main():
    print("=" * 70)
    print("モデル改善イテレーション V2 - place_oddsありデータのみ")
    print("=" * 70)

    FEATURE_TYPES = ['basic', 'rank', 'relative', 'combo', 'full']
    THRESHOLDS = [0.00, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    TRAIN_RATIOS = [0.5, 0.6, 0.7]

    all_results = []

    for track in ['kawasaki', 'ohi']:
        print(f"\n{'='*70}")
        print(f"競馬場: {track.upper()}")
        print("=" * 70)

        df = load_data(track)
        print(f"place_oddsありデータ数: {len(df):,}件")
        print(f"レース数: {df['race_id'].nunique()}レース")

        for feat_type in FEATURE_TYPES:
            features = create_features(df, feat_type)

            for threshold in THRESHOLDS:
                for train_ratio in TRAIN_RATIOS:
                    result = run_backtest(df, features, threshold, train_ratio)

                    if result is None:
                        continue

                    result['track'] = track
                    result['feature_type'] = feat_type
                    result['threshold'] = threshold
                    result['train_ratio'] = train_ratio
                    all_results.append(result)

                    # ROI100%以上のみ表示
                    if result['roi'] >= 1.0:
                        print(f"★ {feat_type:8s} | 閾値{threshold:.2f} | 訓練{train_ratio:.0%} | "
                              f"賭け{result['bets']:3d} | 的中{result['hit_rate']*100:5.1f}% | "
                              f"ROI {result['roi']*100:6.1f}% | {result['profit']:+,.0f}円")

    # 結果サマリー
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("ROI 100%以上の組み合わせ TOP20")
    print("=" * 70)

    good = results_df[results_df['roi'] >= 1.0].sort_values('roi', ascending=False)

    if len(good) == 0:
        print("ROI 100%以上の組み合わせなし")
        # 最も良いものを表示
        best = results_df.sort_values('roi', ascending=False).head(10)
        print("\n最も良い結果 TOP10:")
        for i, row in best.iterrows():
            print(f"{row['track']:8s} | {row['feature_type']:8s} | 閾値{row['threshold']:.2f} | "
                  f"訓練{row['train_ratio']:.0%} | 賭け{row['bets']:3d} | 的中{row['hit_rate']*100:5.1f}% | "
                  f"ROI {row['roi']*100:6.1f}% | {row['profit']:+,.0f}円")
    else:
        for i, row in good.head(20).iterrows():
            print(f"{row['track']:8s} | {row['feature_type']:8s} | 閾値{row['threshold']:.2f} | "
                  f"訓練{row['train_ratio']:.0%} | 賭け{row['bets']:3d} | 的中{row['hit_rate']*100:5.1f}% | "
                  f"ROI {row['roi']*100:6.1f}% | {row['profit']:+,.0f}円")

    results_df.to_csv('data/model_iteration_v2_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n結果保存: data/model_iteration_v2_results.csv")

if __name__ == '__main__':
    main()
