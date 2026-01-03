# -*- coding: utf-8 -*-
"""
モデル改善イテレーション
- 複数の特徴量セット
- 複数の閾値
- 最適な組み合わせを自動探索
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data(track):
    """データ読み込み"""
    if track == 'kawasaki':
        df = pd.read_csv('data/races_kawasaki.csv')
    else:
        df = pd.read_csv('data/races_ohi.csv')

    # 基本的なクリーニング
    df = df.dropna(subset=['rank'])
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank'])
    df = df[df['rank'] > 0]

    # 目的変数: 3着以内
    df['target'] = (df['rank'] <= 3).astype(int)

    return df

def create_features_v1(df):
    """特徴量セットV1: 基本統計"""
    features = pd.DataFrame(index=df.index)

    for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
        if col in df.columns:
            features[f'f_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'horse_avg_prize' in df.columns:
        features['f_horse_avg_prize'] = pd.to_numeric(df['horse_avg_prize'], errors='coerce').fillna(0)

    return features

def create_features_v2(df):
    """特徴量セットV2: 相対ランキング"""
    features = create_features_v1(df)

    # レース内での相対順位を追加
    for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').fillna(0)
            features[f'f_{col}_rank'] = df.groupby('race_id')[col].transform(
                lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False, method='min')
            ).fillna(8)

    return features

def create_features_v3(df):
    """特徴量セットV3: 相対値（フィールド平均との差）"""
    features = create_features_v1(df)

    for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').fillna(0)
            race_mean = df.groupby('race_id')[col].transform(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).fillna(0)
            features[f'f_{col}_vs_field'] = values - race_mean

    return features

def create_features_v4(df):
    """特徴量セットV4: 全部入り"""
    features = create_features_v1(df)

    # 相対ランキング
    for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').fillna(0)
            features[f'f_{col}_rank'] = df.groupby('race_id')[col].transform(
                lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False, method='min')
            ).fillna(8)
            race_mean = df.groupby('race_id')[col].transform(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).fillna(0)
            features[f'f_{col}_vs_field'] = values - race_mean

    return features

def create_features_v5(df):
    """特徴量セットV5: 馬体重・枠番追加"""
    features = create_features_v4(df)

    if 'horse_weight' in df.columns:
        features['f_horse_weight'] = pd.to_numeric(df['horse_weight'], errors='coerce').fillna(450)

    if 'post_position' in df.columns:
        features['f_post_position'] = pd.to_numeric(df['post_position'], errors='coerce').fillna(5)

    if 'horse_age' in df.columns:
        features['f_horse_age'] = pd.to_numeric(df['horse_age'], errors='coerce').fillna(4)

    return features

def create_features_v6(df):
    """特徴量セットV6: 勝率重視（連対率・複勝率の組み合わせ）"""
    features = pd.DataFrame(index=df.index)

    # 基本
    for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
        if col in df.columns:
            features[f'f_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 組み合わせ特徴量
    if 'horse_show_rate' in df.columns and 'jockey_show_rate' in df.columns:
        h_show = pd.to_numeric(df['horse_show_rate'], errors='coerce').fillna(0)
        j_show = pd.to_numeric(df['jockey_show_rate'], errors='coerce').fillna(0)
        features['f_combined_show'] = h_show * j_show
        features['f_avg_show'] = (h_show + j_show) / 2

    if 'horse_win_rate' in df.columns and 'jockey_win_rate' in df.columns:
        h_win = pd.to_numeric(df['horse_win_rate'], errors='coerce').fillna(0)
        j_win = pd.to_numeric(df['jockey_win_rate'], errors='coerce').fillna(0)
        features['f_combined_win'] = h_win * j_win
        features['f_avg_win'] = (h_win + j_win) / 2

    return features

def create_features_v7(df):
    """特徴量セットV7: ランキング特化"""
    features = pd.DataFrame(index=df.index)

    # ランキングのみ
    for col in ['horse_show_rate', 'horse_win_rate', 'jockey_show_rate', 'jockey_win_rate']:
        if col in df.columns:
            features[f'f_{col}_rank'] = df.groupby('race_id')[col].transform(
                lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False, method='min')
            ).fillna(8)

    # 総合ランキング
    features['f_total_rank'] = features.mean(axis=1)

    return features

def create_features_v8(df):
    """特徴量セットV8: 過去成績の深掘り"""
    features = create_features_v4(df)

    # 馬の出走回数
    if 'horse_races' in df.columns:
        features['f_horse_races'] = pd.to_numeric(df['horse_races'], errors='coerce').fillna(0)

    # 騎手の出走回数
    if 'jockey_races' in df.columns:
        features['f_jockey_races'] = pd.to_numeric(df['jockey_races'], errors='coerce').fillna(0)

    # 経験値スコア
    if 'horse_races' in df.columns and 'horse_win_rate' in df.columns:
        races = pd.to_numeric(df['horse_races'], errors='coerce').fillna(0)
        win_rate = pd.to_numeric(df['horse_win_rate'], errors='coerce').fillna(0)
        features['f_horse_exp_score'] = races * win_rate

    return features

FEATURE_SETS = {
    'V1_基本': create_features_v1,
    'V2_相対ランク': create_features_v2,
    'V3_相対値': create_features_v3,
    'V4_全部入り': create_features_v4,
    'V5_馬体重枠番': create_features_v5,
    'V6_勝率重視': create_features_v6,
    'V7_ランク特化': create_features_v7,
    'V8_経験値': create_features_v8,
}

THRESHOLDS = [0.00, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]

def run_backtest(df, features, threshold):
    """バックテスト実行"""
    # 時系列分割
    race_ids = df['race_id'].unique()
    train_size = int(len(race_ids) * 0.7)
    train_races = race_ids[:train_size]
    test_races = race_ids[train_size:]

    train_mask = df['race_id'].isin(train_races)
    test_mask = df['race_id'].isin(test_races)

    X_train = features[train_mask]
    y_train = df.loc[train_mask, 'target']
    X_test = features[test_mask]
    y_test = df.loc[test_mask, 'target']
    test_df = df[test_mask].copy()

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

    # レースごとに1位予測を取得
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
            hit = int(top1['rank']) <= 3

            # 複勝払戻
            payout = 0
            if hit and 'place_odds' in top1.index:
                try:
                    payout = float(top1['place_odds']) * 100
                except:
                    payout = 150  # デフォルト

            results.append({
                'race_id': race_id,
                'gap': gap,
                'hit': hit,
                'bet': 100,
                'payout': payout
            })

    if not results:
        return {'bets': 0, 'hits': 0, 'hit_rate': 0, 'roi': 0, 'profit': 0}

    results_df = pd.DataFrame(results)
    total_bets = len(results_df)
    total_hits = results_df['hit'].sum()
    hit_rate = total_hits / total_bets if total_bets > 0 else 0
    total_bet_amount = results_df['bet'].sum()
    total_payout = results_df['payout'].sum()
    roi = total_payout / total_bet_amount if total_bet_amount > 0 else 0
    profit = total_payout - total_bet_amount

    return {
        'bets': total_bets,
        'hits': total_hits,
        'hit_rate': hit_rate,
        'roi': roi,
        'profit': profit
    }

def main():
    print("=" * 60)
    print("モデル改善イテレーション開始")
    print("=" * 60)

    all_results = []

    for track in ['kawasaki', 'ohi']:
        print(f"\n{'='*60}")
        print(f"競馬場: {track.upper()}")
        print("=" * 60)

        df = load_data(track)
        print(f"データ数: {len(df):,}件")

        for feat_name, feat_func in FEATURE_SETS.items():
            features = feat_func(df)

            for threshold in THRESHOLDS:
                result = run_backtest(df, features, threshold)
                result['track'] = track
                result['feature_set'] = feat_name
                result['threshold'] = threshold
                all_results.append(result)

                if result['bets'] > 0:
                    print(f"{feat_name} | 閾値{threshold:.2f} | "
                          f"賭け{result['bets']:3d}件 | "
                          f"的中{result['hit_rate']*100:5.1f}% | "
                          f"ROI {result['roi']*100:6.1f}% | "
                          f"収支 {result['profit']:+,.0f}円")

    # 結果をDataFrameに
    results_df = pd.DataFrame(all_results)

    # 最適な組み合わせを探索
    print("\n" + "=" * 60)
    print("最適な組み合わせ TOP10")
    print("=" * 60)

    # 条件: 賭け数20以上、ROI100%以上
    good_results = results_df[(results_df['bets'] >= 20) & (results_df['roi'] >= 1.0)]
    good_results = good_results.sort_values(['roi', 'hit_rate'], ascending=[False, False])

    for i, row in good_results.head(10).iterrows():
        print(f"{row['track']:8s} | {row['feature_set']:12s} | 閾値{row['threshold']:.2f} | "
              f"賭け{row['bets']:3d}件 | 的中{row['hit_rate']*100:5.1f}% | "
              f"ROI {row['roi']*100:6.1f}% | 収支 {row['profit']:+,.0f}円")

    # 結果保存
    results_df.to_csv('data/model_iteration_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n結果を data/model_iteration_results.csv に保存しました")

    return results_df

if __name__ == '__main__':
    main()
