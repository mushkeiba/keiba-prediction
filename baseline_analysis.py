"""
ベースライン分析: 市場の効率性を測定
- 人気別の的中率・回収率
- これを超えないと意味がない
"""
import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def analyze_baseline(track_name):
    """人気別の的中率・回収率を分析"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} ベースライン分析')
    print(f'{"="*60}')

    # データ読み込み
    df = pd.read_csv(f'data/races_{track_name}.csv')
    print(f'総レコード数: {len(df):,}')

    # 数値変換
    for col in ['rank', 'win_odds', 'place_odds']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 人気を計算（オッズの低い順）
    df['popularity'] = df.groupby('race_id')['win_odds'].rank(ascending=True)

    # 複勝フラグ
    df['is_place'] = (df['rank'] <= 3).astype(int)

    # 複勝払戻を解析
    def parse_place_odds(x):
        if pd.isna(x):
            return np.nan
        try:
            # "1.5-2.0" のような形式を平均に
            if '-' in str(x):
                parts = str(x).split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(x)
        except:
            return np.nan

    df['place_odds_avg'] = df['place_odds'].apply(parse_place_odds)

    print(f'\n{"="*60}')
    print('【人気別の成績】')
    print(f'{"="*60}')
    print(f'{"人気":>4} | {"件数":>6} | {"的中率":>7} | {"平均配当":>7} | {"回収率":>7} | 判定')
    print('-' * 60)

    results = []
    for pop in range(1, 13):
        pop_df = df[df['popularity'] == pop].copy()
        if len(pop_df) < 100:
            continue

        n = len(pop_df)
        hits = pop_df['is_place'].sum()
        hit_rate = hits / n

        # 回収率計算（複勝）
        # 的中時の払戻を計算
        hit_df = pop_df[pop_df['is_place'] == 1]
        if len(hit_df) > 0 and hit_df['place_odds_avg'].notna().sum() > 0:
            avg_payout = hit_df['place_odds_avg'].mean()
        else:
            avg_payout = np.nan

        if pd.notna(avg_payout):
            roi = hit_rate * avg_payout
        else:
            roi = np.nan

        # 判定
        if pd.notna(roi):
            if roi >= 1.0:
                judge = '✓ プラス'
            elif roi >= 0.9:
                judge = '△ ほぼ均衡'
            else:
                judge = '✗ マイナス'
        else:
            judge = '- データ不足'

        print(f'{pop:>4} | {n:>6,} | {hit_rate*100:>6.1f}% | {avg_payout:>6.2f}倍 | {roi*100:>6.1f}% | {judge}')

        results.append({
            'popularity': pop,
            'count': n,
            'hit_rate': hit_rate,
            'avg_payout': avg_payout,
            'roi': roi
        })

    results_df = pd.DataFrame(results)

    # サマリー
    print(f'\n{"="*60}')
    print('【サマリー】')
    print(f'{"="*60}')

    # 全体の複勝率
    total_place_rate = df['is_place'].mean()
    print(f'全体の複勝率: {total_place_rate*100:.1f}%')

    # 回収率100%超えの人気
    profitable = results_df[results_df['roi'] >= 1.0]
    if len(profitable) > 0:
        print(f'\n回収率100%超えの人気: {profitable["popularity"].tolist()}')
    else:
        print(f'\n回収率100%超えの人気: なし（市場は効率的）')

    # ベストな人気帯
    if len(results_df) > 0 and results_df['roi'].notna().any():
        best = results_df.loc[results_df['roi'].idxmax()]
        print(f'最も効率的な人気: {int(best["popularity"])}人気 (ROI: {best["roi"]*100:.1f}%)')

    return results_df


def analyze_specific_conditions(track_name):
    """特定条件での成績を分析（エッジ探し）"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} エッジ探索')
    print(f'{"="*60}')

    df = pd.read_csv(f'data/races_{track_name}.csv')

    for col in ['rank', 'win_odds', 'place_odds', 'horse_runs', 'last_rank',
                'horse_win_rate', 'horse_show_rate', 'jockey_win_rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['popularity'] = df.groupby('race_id')['win_odds'].rank(ascending=True)
    df['is_place'] = (df['rank'] <= 3).astype(int)

    def parse_place_odds(x):
        if pd.isna(x):
            return np.nan
        try:
            if '-' in str(x):
                parts = str(x).split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(x)
        except:
            return np.nan

    df['place_odds_avg'] = df['place_odds'].apply(parse_place_odds)

    def calc_roi(subset):
        if len(subset) < 50:
            return None, None, None
        hit_rate = subset['is_place'].mean()
        hit_df = subset[subset['is_place'] == 1]
        if len(hit_df) > 0 and hit_df['place_odds_avg'].notna().sum() > 0:
            avg_payout = hit_df['place_odds_avg'].mean()
            roi = hit_rate * avg_payout
        else:
            avg_payout = None
            roi = None
        return hit_rate, avg_payout, roi

    # 仮説検証
    hypotheses = []

    # 仮説1: 前走1着の馬（過大評価されている？）
    if 'last_rank' in df.columns:
        cond = df['last_rank'] == 1
        hr, ap, roi = calc_roi(df[cond])
        if roi:
            hypotheses.append(('前走1着', df[cond].shape[0], hr, roi))

    # 仮説2: 前走10着以下（過小評価されている？）
    if 'last_rank' in df.columns:
        cond = df['last_rank'] >= 10
        hr, ap, roi = calc_roi(df[cond])
        if roi:
            hypotheses.append(('前走10着以下', df[cond].shape[0], hr, roi))

    # 仮説3: 経験豊富（20走以上）
    if 'horse_runs' in df.columns:
        cond = df['horse_runs'] >= 20
        hr, ap, roi = calc_roi(df[cond])
        if roi:
            hypotheses.append(('20走以上', df[cond].shape[0], hr, roi))

    # 仮説4: 初出走〜3走（経験不足）
    if 'horse_runs' in df.columns:
        cond = df['horse_runs'] <= 3
        hr, ap, roi = calc_roi(df[cond])
        if roi:
            hypotheses.append(('3走以下', df[cond].shape[0], hr, roi))

    # 仮説5: 高勝率騎手（15%以上）
    if 'jockey_win_rate' in df.columns:
        cond = df['jockey_win_rate'] >= 0.15
        hr, ap, roi = calc_roi(df[cond])
        if roi:
            hypotheses.append(('騎手勝率15%+', df[cond].shape[0], hr, roi))

    # 仮説6: 5-8人気（中穴）
    cond = (df['popularity'] >= 5) & (df['popularity'] <= 8)
    hr, ap, roi = calc_roi(df[cond])
    if roi:
        hypotheses.append(('5-8人気', df[cond].shape[0], hr, roi))

    # 仮説7: 馬の複勝率30%以上 & 5人気以下
    if 'horse_show_rate' in df.columns:
        cond = (df['horse_show_rate'] >= 0.30) & (df['popularity'] >= 5)
        hr, ap, roi = calc_roi(df[cond])
        if roi:
            hypotheses.append(('複勝率30%+&5人気↓', df[cond].shape[0], hr, roi))

    # 仮説8: 前走1着なのに5人気以下（過小評価？）
    if 'last_rank' in df.columns:
        cond = (df['last_rank'] == 1) & (df['popularity'] >= 5)
        hr, ap, roi = calc_roi(df[cond])
        if roi:
            hypotheses.append(('前走1着&5人気↓', df[cond].shape[0], hr, roi))

    # 結果表示
    print(f'\n{"条件":<20} | {"件数":>6} | {"的中率":>7} | {"回収率":>7} | 判定')
    print('-' * 60)

    for name, n, hr, roi in sorted(hypotheses, key=lambda x: x[3] if x[3] else 0, reverse=True):
        if roi >= 1.0:
            judge = '★ エッジあり'
        elif roi >= 0.9:
            judge = '△ 要検証'
        else:
            judge = ''
        print(f'{name:<20} | {n:>6,} | {hr*100:>6.1f}% | {roi*100:>6.1f}% | {judge}')

    # エッジのある条件を抽出
    edges = [(name, n, hr, roi) for name, n, hr, roi in hypotheses if roi and roi >= 1.0]
    if edges:
        print(f'\n{"="*60}')
        print('【発見したエッジ】')
        print(f'{"="*60}')
        for name, n, hr, roi in edges:
            print(f'  ★ {name}: ROI {roi*100:.1f}% ({n:,}件)')
    else:
        print(f'\n明確なエッジは見つかりませんでした。')
        print('追加の仮説検証が必要です。')

    return hypotheses


if __name__ == '__main__':
    print('='*60)
    print('ベースライン分析 & エッジ探索')
    print('='*60)

    for track in ['kawasaki', 'ohi']:
        analyze_baseline(track)
        analyze_specific_conditions(track)

    print('\n' + '='*60)
    print('分析完了')
    print('='*60)
