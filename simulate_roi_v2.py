"""回収率シミュレーション v2 - 単勝オッズから複勝オッズを推定"""
import pandas as pd
import numpy as np
import pickle
import sys
import io
from optimize_v5 import ProcessorV5, TargetEncoderSafe

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def estimate_place_odds_from_win(win_odds):
    """
    単勝オッズから複勝オッズを推定
    実際のデータに基づく経験則:
    - 単勝1.1-2.0倍 → 複勝1.1-1.3倍
    - 単勝2.0-5.0倍 → 複勝1.2-1.8倍
    - 単勝5.0-10.0倍 → 複勝1.5-2.5倍
    - 単勝10倍以上 → 複勝2.0-4.0倍
    """
    if win_odds <= 0:
        return 1.5  # デフォルト

    if win_odds <= 2.0:
        return 1.1 + (win_odds - 1.0) * 0.2
    elif win_odds <= 5.0:
        return 1.3 + (win_odds - 2.0) * 0.17
    elif win_odds <= 10.0:
        return 1.8 + (win_odds - 5.0) * 0.14
    elif win_odds <= 30.0:
        return 2.5 + (win_odds - 10.0) * 0.075
    else:
        return min(4.0 + (win_odds - 30.0) * 0.02, 10.0)


def simulate(track_name):
    """回収率シミュレーション"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} 回収率シミュレーション (実オッズベース)')
    print(f'{"="*60}')

    # データ読み込み
    df = pd.read_csv(f'data/races_{track_name}.csv')

    # 前処理
    processor = ProcessorV5()
    df = processor.process_base(df)

    # 時系列分割
    df = df.sort_values('race_id').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Target Encoding
    te_cols = ['jockey_id', 'trainer_id', 'horse_id']
    te_encoder = TargetEncoderSafe(smoothing=10)
    te_encoder.fit(train_df, te_cols, 'target')
    train_df = te_encoder.transform(train_df, te_cols)
    test_df = te_encoder.transform(test_df, te_cols)

    # 欠損埋め
    for f in processor.features:
        if f not in train_df.columns:
            train_df[f] = 0
            test_df[f] = 0
        train_df[f] = train_df[f].fillna(0)
        test_df[f] = test_df[f].fillna(0)

    # モデル読み込み
    with open(f'models/model_{track_name}.pkl', 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']

    # 予測
    X_test = test_df[processor.features]

    if model['type'] == 'ensemble':
        lgb_pred = model['lgb'].predict_proba(X_test)[:, 1]
        xgb_pred = model['xgb'].predict_proba(X_test)[:, 1]
        pred_prob = (lgb_pred + xgb_pred) / 2
    else:
        pred_prob = model.predict_proba(X_test)[:, 1]

    test_df['pred_prob'] = pred_prob
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_prob'].rank(ascending=False)

    # 2位との確率差
    def calc_prob_diff(group):
        sorted_g = group.sort_values('pred_prob', ascending=False)
        if len(sorted_g) >= 2:
            diff = sorted_g['pred_prob'].iloc[0] - sorted_g['pred_prob'].iloc[1]
            group['prob_diff'] = group['pred_prob'].apply(
                lambda x: diff if x == sorted_g['pred_prob'].iloc[0] else 0
            )
        else:
            group['prob_diff'] = 0
        return group

    test_df = test_df.groupby('race_id', group_keys=False).apply(calc_prob_diff)

    # 単勝オッズから複勝オッズを推定
    test_df['place_odds_est'] = test_df['win_odds'].apply(estimate_place_odds_from_win)

    # 的中フラグ
    test_df['is_place'] = (test_df['rank'] <= 3).astype(int)

    print(f'\nテストデータ: {len(test_df):,}件')
    print(f'レース数: {test_df["race_id"].nunique():,}R')

    # オッズ有効確認
    valid_odds = test_df[test_df['win_odds'] > 0]
    print(f'単勝オッズ有効: {len(valid_odds):,}件 ({len(valid_odds)/len(test_df)*100:.1f}%)')
    print(f'単勝オッズ平均: {valid_odds["win_odds"].mean():.1f}倍')
    print(f'複勝オッズ推定平均: {valid_odds["place_odds_est"].mean():.2f}倍')

    # ===== シミュレーション =====
    results = []

    # 予測1位のみ（オッズありのみ）
    top1 = test_df[(test_df['pred_rank'] == 1) & (test_df['win_odds'] > 0)].copy()

    print(f'\n予測1位（オッズあり）: {len(top1)}レース')

    # 1. フィルターなし
    bets = top1
    hits = bets['is_place'].sum()
    total = len(bets)
    hit_rate = hits / total if total > 0 else 0
    avg_odds = bets['place_odds_est'].mean()
    roi = hit_rate * avg_odds
    results.append({
        'filter': 'All',
        'bets': total,
        'hits': hits,
        'hit_rate': hit_rate,
        'avg_odds': avg_odds,
        'roi': roi
    })

    # 2. 確率差フィルター
    for min_diff in [0.05, 0.10, 0.15]:
        bets = top1[top1['prob_diff'] >= min_diff]
        hits = bets['is_place'].sum()
        total = len(bets)
        hit_rate = hits / total if total > 0 else 0
        avg_odds = bets['place_odds_est'].mean() if total > 0 else 0
        roi = hit_rate * avg_odds
        results.append({
            'filter': f'Diff>={int(min_diff*100)}%',
            'bets': total,
            'hits': hits,
            'hit_rate': hit_rate,
            'avg_odds': avg_odds,
            'roi': roi
        })

    # 3. 単勝オッズフィルター（低オッズ本命除外）
    for min_odds in [3.0, 5.0, 10.0]:
        bets = top1[top1['win_odds'] >= min_odds]
        hits = bets['is_place'].sum()
        total = len(bets)
        hit_rate = hits / total if total > 0 else 0
        avg_odds = bets['place_odds_est'].mean() if total > 0 else 0
        roi = hit_rate * avg_odds
        results.append({
            'filter': f'WinOdds>={min_odds}',
            'bets': total,
            'hits': hits,
            'hit_rate': hit_rate,
            'avg_odds': avg_odds,
            'roi': roi
        })

    # 結果表示
    print(f'\n{"="*65}')
    print(f'{"Filter":<15} {"Bets":>6} {"Hits":>5} {"HitRate":>8} {"PlaceOdds":>10} {"ROI":>8}')
    print(f'{"="*65}')

    for r in results:
        mark = ' OK' if r['roi'] >= 1.0 else '   '
        print(f'{r["filter"]:<15} {r["bets"]:>6} {r["hits"]:>5} {r["hit_rate"]:>7.1%} {r["avg_odds"]:>9.2f}x {r["roi"]:>7.1%}{mark}')

    print(f'{"="*65}')
    print(f'\nOK = ROI >= 100%')

    # 回収率100%に必要なオッズ
    if results[0]['hit_rate'] > 0:
        required = 1.0 / results[0]['hit_rate']
        print(f'\nBreak-even Place Odds: {required:.2f}x (at {results[0]["hit_rate"]:.1%} hit rate)')

    return results


if __name__ == '__main__':
    track = sys.argv[1] if len(sys.argv) > 1 else 'ohi'
    simulate(track)
