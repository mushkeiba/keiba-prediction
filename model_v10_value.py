# -*- coding: utf-8 -*-
"""
モデル V10 - バリューベット
「勝つか」ではなく「期待値がプラスか」を探す
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data(track):
    if track == 'kawasaki':
        df = pd.read_csv('data/races_kawasaki.csv')
    else:
        df = pd.read_csv('data/races_ohi.csv')

    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank'])
    df = df[df['rank'] > 0]
    df['target'] = (df['rank'] <= 3).astype(int)
    df['race_date'] = pd.to_numeric(df['race_date'])

    # place_oddsを数値化
    df['place_odds'] = pd.to_numeric(df['place_odds'], errors='coerce')

    return df

def create_features(df):
    """特徴量: オッズを使わない純粋な実力指標"""
    f = pd.DataFrame(index=df.index)

    # 馬の実力
    f['horse_show'] = pd.to_numeric(df['horse_show_rate'], errors='coerce').fillna(0)
    f['horse_win'] = pd.to_numeric(df['horse_win_rate'], errors='coerce').fillna(0)
    f['horse_recent'] = pd.to_numeric(df['horse_recent_show_rate'], errors='coerce').fillna(0)
    f['last_rank'] = pd.to_numeric(df['last_rank'], errors='coerce').fillna(10)

    # 騎手の実力
    f['jockey_show'] = pd.to_numeric(df['jockey_show_rate'], errors='coerce').fillna(0)
    f['jockey_win'] = pd.to_numeric(df['jockey_win_rate'], errors='coerce').fillna(0)

    # レース内の相対位置
    for col in ['horse_show_rate', 'jockey_show_rate']:
        f[f'{col}_rank'] = df.groupby('race_id')[col].transform(
            lambda x: pd.to_numeric(x, errors='coerce').rank(ascending=False)
        ).fillna(8)

    return f

def calculate_expected_value(pred_prob, place_odds):
    """期待値計算: 予測確率 × オッズ - 1"""
    if pd.isna(place_odds) or place_odds <= 0:
        return -1  # オッズ不明は賭けない
    return pred_prob * place_odds - 1

def backtest_value_betting(df, train_end, test_start, test_end, ev_threshold=0.0):
    """バリューベット戦略のバックテスト"""

    train_df = df[df['race_date'] <= train_end]
    test_df = df[(df['race_date'] >= test_start) & (df['race_date'] <= test_end)]

    if len(train_df) < 500 or len(test_df) < 100:
        return None

    train_features = create_features(train_df)
    test_features = create_features(test_df)

    # モデル訓練
    model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
    model.fit(train_features, train_df['target'])

    # 予測
    test_df = test_df.copy()
    test_df['pred_prob'] = model.predict_proba(test_features)[:, 1]

    # 期待値計算
    test_df['ev'] = test_df.apply(
        lambda x: calculate_expected_value(x['pred_prob'], x['place_odds']),
        axis=1
    )

    # 各レースで期待値が閾値以上の馬に賭ける
    results = []

    for race_id in test_df['race_id'].unique():
        race_df = test_df[test_df['race_id'] == race_id]

        # 期待値がプラスの馬を探す
        positive_ev = race_df[race_df['ev'] >= ev_threshold]

        for _, horse in positive_ev.iterrows():
            hit = int(horse['rank']) <= 3
            payout = float(horse['place_odds']) * 100 if hit else 0

            results.append({
                'race_id': race_id,
                'date': horse['race_date'],
                'horse': horse['horse_name'],
                'pred_prob': horse['pred_prob'],
                'place_odds': horse['place_odds'],
                'ev': horse['ev'],
                'hit': hit,
                'payout': payout
            })

    return results

def main():
    print("=" * 70)
    print("モデル V10 - バリューベット戦略")
    print("=" * 70)

    for track in ['kawasaki', 'ohi']:
        print(f"\n{'='*70}")
        print(f"競馬場: {track.upper()}")
        print("=" * 70)

        df = load_data(track)

        # place_oddsがあるデータのみ
        df = df[df['place_odds'] > 0]
        print(f"データ: {len(df):,}件 / {df['race_id'].nunique()}レース")
        print(f"期間: {df['race_date'].min()} - {df['race_date'].max()}")

        # 時系列分割
        race_dates = sorted(df['race_date'].unique())
        split_idx = int(len(race_dates) * 0.6)
        train_end = race_dates[split_idx]
        test_start = race_dates[split_idx + 1]
        test_end = race_dates[-1]

        print(f"訓練: ~{train_end}, テスト: {test_start}~{test_end}")

        # 様々な期待値閾値でテスト
        print()
        print("期待値閾値ごとの結果:")
        print("-" * 60)

        for ev_threshold in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
            results = backtest_value_betting(df, train_end, test_start, test_end, ev_threshold)

            if results and len(results) >= 5:
                total_bets = len(results)
                total_hits = sum(r['hit'] for r in results)
                hit_rate = total_hits / total_bets
                total_payout = sum(r['payout'] for r in results)
                total_bet = total_bets * 100
                roi = total_payout / total_bet
                profit = total_payout - total_bet
                avg_ev = np.mean([r['ev'] for r in results])

                mark = '★' if roi >= 1.0 else ' '
                print(f"{mark} EV>={ev_threshold:.2f}: 賭け{total_bets:>4} | 的中{hit_rate*100:>5.1f}% | "
                      f"平均EV{avg_ev:>+.2f} | ROI {roi*100:>6.1f}% | {profit:>+,.0f}円")

if __name__ == '__main__':
    main()
