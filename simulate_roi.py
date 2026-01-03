"""回収率シミュレーション - フィルター検証"""
import pandas as pd
import numpy as np
import pickle
import sys
import io
from optimize_v5 import ProcessorV5, TargetEncoderSafe, add_previous_race_features_safe

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def estimate_place_odds(show_rate, field_size):
    """
    複勝オッズを推定
    - 控除率約25%を考慮
    - field_sizeに応じて3着払い/2着払いを調整
    """
    if show_rate <= 0:
        return 10.0  # データなしは高オッズ

    # 基本オッズ = 0.75 / 複勝率（控除率25%）
    base_odds = 0.75 / show_rate

    # 最低オッズは1.1倍
    return max(1.1, min(base_odds, 50.0))


def simulate(track_name):
    """回収率シミュレーション"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} 回収率シミュレーション')
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

    # レース内順位
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_prob'].rank(ascending=False)

    # 2位との確率差
    def calc_prob_diff(group):
        sorted_g = group.sort_values('pred_prob', ascending=False)
        if len(sorted_g) >= 2:
            group['prob_diff'] = sorted_g['pred_prob'].iloc[0] - sorted_g['pred_prob'].iloc[1]
        else:
            group['prob_diff'] = 0
        return group

    test_df = test_df.groupby('race_id', group_keys=False).apply(calc_prob_diff)

    # 実オッズを使用（place_odds = win_odds / 3 で取得済み）
    # place_oddsがない場合は推定
    if 'place_odds' in test_df.columns and test_df['place_odds'].sum() > 0:
        test_df['est_place_odds'] = test_df['place_odds'].clip(lower=1.1)
        print(f'実オッズ使用: 平均 {test_df["est_place_odds"].mean():.2f}倍')
    else:
        test_df['est_place_odds'] = test_df.apply(
            lambda x: estimate_place_odds(x['horse_show_rate'], x['field_size']), axis=1
        )
        print('推定オッズ使用')

    # 的中フラグ
    test_df['is_place'] = (test_df['rank'] <= 3).astype(int)

    print(f'\nテストデータ: {len(test_df):,}件')
    print(f'レース数: {test_df["race_id"].nunique():,}R')

    # ===== シミュレーション =====
    results = []

    # 予測1位のみ
    top1 = test_df[test_df['pred_rank'] == 1].copy()

    # 1. フィルターなし
    bets = top1
    hits = bets['is_place'].sum()
    total = len(bets)
    hit_rate = hits / total if total > 0 else 0
    avg_odds = bets['est_place_odds'].mean()
    roi = hit_rate * avg_odds
    results.append({
        'filter': 'なし',
        'bets': total,
        'hits': hits,
        'hit_rate': hit_rate,
        'avg_odds': avg_odds,
        'roi': roi
    })

    # 2. 確率差フィルター
    for min_diff in [0.05, 0.10, 0.15, 0.20]:
        bets = top1[top1['prob_diff'] >= min_diff]
        hits = bets['is_place'].sum()
        total = len(bets)
        hit_rate = hits / total if total > 0 else 0
        avg_odds = bets['est_place_odds'].mean() if total > 0 else 0
        roi = hit_rate * avg_odds
        results.append({
            'filter': f'確率差≥{min_diff:.0%}',
            'bets': total,
            'hits': hits,
            'hit_rate': hit_rate,
            'avg_odds': avg_odds,
            'roi': roi
        })

    # 3. オッズフィルター（低オッズ除外）
    for min_odds in [1.5, 2.0, 2.5, 3.0]:
        bets = top1[top1['est_place_odds'] >= min_odds]
        hits = bets['is_place'].sum()
        total = len(bets)
        hit_rate = hits / total if total > 0 else 0
        avg_odds = bets['est_place_odds'].mean() if total > 0 else 0
        roi = hit_rate * avg_odds
        results.append({
            'filter': f'オッズ≥{min_odds}倍',
            'bets': total,
            'hits': hits,
            'hit_rate': hit_rate,
            'avg_odds': avg_odds,
            'roi': roi
        })

    # 4. 複合フィルター
    for min_diff, min_odds in [(0.10, 2.0), (0.15, 2.0), (0.10, 2.5), (0.15, 2.5)]:
        bets = top1[(top1['prob_diff'] >= min_diff) & (top1['est_place_odds'] >= min_odds)]
        hits = bets['is_place'].sum()
        total = len(bets)
        hit_rate = hits / total if total > 0 else 0
        avg_odds = bets['est_place_odds'].mean() if total > 0 else 0
        roi = hit_rate * avg_odds
        results.append({
            'filter': f'差≥{min_diff:.0%} & オッズ≥{min_odds}',
            'bets': total,
            'hits': hits,
            'hit_rate': hit_rate,
            'avg_odds': avg_odds,
            'roi': roi
        })

    # 結果表示
    print(f'\n{"="*70}')
    print(f'{"フィルター":<20} {"買い目":>6} {"的中":>5} {"的中率":>8} {"平均ｵｯｽﾞ":>8} {"回収率":>8}')
    print(f'{"="*70}')

    for r in results:
        roi_mark = '🔥' if r['roi'] >= 1.0 else '  '
        print(f'{r["filter"]:<20} {r["bets"]:>6} {r["hits"]:>5} {r["hit_rate"]:>7.1%} {r["avg_odds"]:>7.2f}倍 {r["roi"]:>7.1%} {roi_mark}')

    print(f'{"="*70}')
    print(f'\n🔥 = 回収率100%以上（黒字）')
    print(f'※オッズは推定値（実際とは異なる場合あり）')

    # 最良フィルターを特定
    best = max(results, key=lambda x: x['roi'] if x['bets'] >= 50 else 0)
    print(f'\n【推奨】{best["filter"]}')
    print(f'  買い目: {best["bets"]}件 / 的中率: {best["hit_rate"]:.1%} / 回収率: {best["roi"]:.1%}')

    return results


def compare_tracks():
    """大井・川崎比較"""
    print('\n' + '='*70)
    print('大井・川崎 回収率比較')
    print('='*70)

    for track in ['ohi', 'kawasaki']:
        simulate(track)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        simulate(sys.argv[1])
    else:
        compare_tracks()
