"""
バックテストスクリプト
過去データで「この買い方をしてたら儲かった？」を検証
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent


def load_model(track_name):
    """モデルを読み込む"""
    model_path = BASE_DIR / f"models/model_{track_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']


def predict_with_model(model, X):
    """予測（アンサンブル対応）"""
    if isinstance(model, dict):
        model_type = model.get('type', 'ensemble')
        if model_type == 'ensemble':
            lgb_pred = model['lgb'].predict(X)
            xgb_pred = model['xgb'].predict(X)
            return (lgb_pred + xgb_pred) / 2
        elif 'lgb' in model:
            return model['lgb'].predict(X)
        elif 'xgb' in model:
            return model['xgb'].predict(X)
    return model.predict(X)


def prepare_features(df, features):
    """特徴量を準備（不足分は0埋め）"""
    df = df.copy()

    # 基本的なエンコーディング
    if 'sex' in df.columns:
        df['sex_encoded'] = df['sex'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
    else:
        df['sex_encoded'] = 0

    if 'track_condition' in df.columns:
        df['track_condition_encoded'] = df['track_condition'].map(
            {'良': 0, '稍重': 1, '重': 2, '不良': 3}
        ).fillna(0)
    else:
        df['track_condition_encoded'] = 0

    if 'weather' in df.columns:
        df['weather_encoded'] = df['weather'].map(
            {'晴': 0, '曇': 1, '小雨': 2, '雨': 3, '雪': 4}
        ).fillna(0)
    else:
        df['weather_encoded'] = 0

    # 計算特徴量
    if 'weight_carried' in df.columns and 'race_id' in df.columns:
        df['weight_diff'] = df.groupby('race_id')['weight_carried'].transform(lambda x: x - x.mean())
    else:
        df['weight_diff'] = 0

    if 'horse_number' in df.columns and 'field_size' in df.columns:
        df['horse_number_ratio'] = df['horse_number'] / df['field_size'].clip(lower=1)
    else:
        df['horse_number_ratio'] = 0.5

    if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
        df['last_rank_diff'] = df['last_rank'] - df['horse_avg_rank']
    else:
        df['last_rank_diff'] = 0

    if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
        df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False, method='min')
        df['field_avg_win_rate'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
        df['horse_win_rate_vs_field'] = df['horse_win_rate'] - df['field_avg_win_rate']
    else:
        df['win_rate_rank'] = 6
        df['horse_win_rate_vs_field'] = 0

    if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
        df['field_avg_jockey_win_rate'] = df.groupby('race_id')['jockey_win_rate'].transform('mean')
        df['jockey_win_rate_vs_field'] = df['jockey_win_rate'] - df['field_avg_jockey_win_rate']
    else:
        df['jockey_win_rate_vs_field'] = 0

    if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
        df['field_avg_rank'] = df.groupby('race_id')['horse_avg_rank'].transform('mean')
        df['horse_avg_rank_vs_field'] = df['field_avg_rank'] - df['horse_avg_rank']
    else:
        df['horse_avg_rank_vs_field'] = 0

    if 'horse_avg_rank' in df.columns and 'last_rank' in df.columns:
        df['rank_trend'] = df['horse_avg_rank'] - df['last_rank']
    else:
        df['rank_trend'] = 0

    # 馬体重関連
    if 'horse_weight' not in df.columns:
        df['horse_weight'] = 450
    if 'weight_change' in df.columns:
        df['horse_weight_change'] = df['weight_change'].fillna(0)
    else:
        df['horse_weight_change'] = 0

    # 時系列特徴量
    df['days_since_last_race'] = 30
    df['win_streak'] = 0
    df['show_streak'] = 0

    if 'horse_recent_avg_rank' in df.columns:
        df['recent_3_avg_rank'] = df['horse_recent_avg_rank']
    else:
        df['recent_3_avg_rank'] = 10

    if 'horse_avg_rank' in df.columns:
        df['recent_10_avg_rank'] = df['horse_avg_rank']
    else:
        df['recent_10_avg_rank'] = 10

    if 'recent_3_avg_rank' in df.columns and 'horse_avg_rank' in df.columns:
        df['rank_improvement'] = df['horse_avg_rank'] - df['recent_3_avg_rank']
    else:
        df['rank_improvement'] = 0

    # Target Encoding（推論時はグローバル平均）
    df['jockey_id_te'] = 0.08
    df['trainer_id_te'] = 0.08
    df['horse_id_te'] = 0.08

    # 追加特徴量
    if 'horse_win_rate' in df.columns and 'jockey_win_rate' in df.columns:
        df['horse_jockey_synergy'] = df['horse_win_rate'] * df['jockey_win_rate']
    else:
        df['horse_jockey_synergy'] = 0

    if all(c in df.columns for c in ['last_rank', 'field_size', 'horse_recent_avg_rank', 'horse_win_rate']):
        df['form_score'] = (
            0.5 * (1 - df['last_rank'] / df['field_size'].clip(lower=1)) +
            0.3 * (1 - df['horse_recent_avg_rank'] / df['field_size'].clip(lower=1)) +
            0.2 * df['horse_win_rate']
        ).fillna(0)
    else:
        df['form_score'] = 0

    if 'field_size' in df.columns and 'horse_avg_rank' in df.columns:
        df['class_indicator'] = df['field_size'] / (df['horse_avg_rank'] + 1)
    else:
        df['class_indicator'] = 1

    df['horse_win_rate_std'] = 0

    if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
        df['field_strength'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
    else:
        df['field_strength'] = 0.1

    if 'horse_number' in df.columns:
        df['inner_outer'] = df['horse_number'].apply(
            lambda x: 0 if pd.notna(x) and x <= 4 else (2 if pd.notna(x) and x >= 10 else 1)
        )
    else:
        df['inner_outer'] = 1

    if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
        df['avg_rank_percentile'] = df.groupby('race_id')['horse_avg_rank'].rank(pct=True)
    else:
        df['avg_rank_percentile'] = 0.5

    if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
        df['jockey_rank_in_race'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)
    else:
        df['jockey_rank_in_race'] = 6

    if 'win_odds' in df.columns:
        df['odds_implied_prob'] = 1 / (df['win_odds'].clip(lower=1) + 1)
    else:
        df['odds_implied_prob'] = 0.1

    df['distance_fitness'] = 1.0

    if 'weight_carried' in df.columns and 'distance' in df.columns:
        df['weight_per_meter'] = df['weight_carried'] / (df['distance'] / 1000).clip(lower=0.1)
    else:
        df['weight_per_meter'] = 50

    if 'horse_runs' in df.columns and 'horse_show_rate' in df.columns:
        df['experience_score'] = np.log1p(df['horse_runs']) * df['horse_show_rate']
    else:
        df['experience_score'] = 0

    # 不足特徴量を0埋め
    for f in features:
        if f not in df.columns:
            df[f] = 0

    return df


def estimate_odds(df):
    """
    オッズを推定する（実際のオッズがない場合）
    勝率の逆数をベースに、控除率を考慮して計算
    """
    df = df.copy()

    # 実際のオッズがあればそれを使う
    if 'win_odds' in df.columns and (df['win_odds'] > 0).any():
        return df

    # レース内での相対的な強さから推定オッズを計算
    def calc_race_odds(group):
        # 各馬の勝率を正規化して確率に
        win_rates = group['horse_win_rate'].fillna(0.01).clip(lower=0.01)
        # 逆数をとって相対的なオッズを計算
        raw_odds = 1 / win_rates
        # 全馬の確率の合計が1になるように正規化
        total_prob = (1 / raw_odds).sum()
        probs = (1 / raw_odds) / total_prob

        # オッズ = 1/確率 × 0.8（控除率20%）
        estimated_win_odds = (1 / probs) * 0.8
        # 複勝オッズは単勝の約1/3（経験則）
        estimated_place_odds = estimated_win_odds / 3

        group['win_odds'] = estimated_win_odds.clip(lower=1.1, upper=100)
        group['place_odds'] = estimated_place_odds.clip(lower=1.1, upper=30)
        return group

    df = df.groupby('race_id', group_keys=False).apply(calc_race_odds)
    print("  ※オッズは勝率から推定した値を使用")
    return df


def estimate_odds_quiet(df):
    """オッズ推定（出力なし版）"""
    df = df.copy()
    if 'win_odds' in df.columns and (df['win_odds'] > 0).any():
        return df

    def calc_race_odds(group):
        win_rates = group['horse_win_rate'].fillna(0.01).clip(lower=0.01)
        raw_odds = 1 / win_rates
        total_prob = (1 / raw_odds).sum()
        probs = (1 / raw_odds) / total_prob
        estimated_win_odds = (1 / probs) * 0.8
        estimated_place_odds = estimated_win_odds / 3
        group['win_odds'] = estimated_win_odds.clip(lower=1.1, upper=100)
        group['place_odds'] = estimated_place_odds.clip(lower=1.1, upper=30)
        return group

    return df.groupby('race_id', group_keys=False).apply(calc_race_odds)


def run_backtest(track_name, start_date=None, end_date=None,
                 min_prob=0.0, min_prob_diff=0.0, min_race_num=1, min_ev=0.0,
                 bet_type='place'):
    """
    バックテスト実行

    Parameters:
    -----------
    track_name : str - 競馬場名（ohi, kawasaki）
    start_date : str - 開始日（YYYY-MM-DD）
    end_date : str - 終了日（YYYY-MM-DD）
    min_prob : float - 最低予測確率（0-1）
    min_prob_diff : float - 1位と2位の確率差（0-1）
    min_race_num : int - 最低レース番号（8なら8R以降）
    min_ev : float - 最低期待値
    bet_type : str - 賭け方（'place'=複勝, 'win'=単勝）
    """

    # データ読み込み
    csv_path = BASE_DIR / f"data/races_{track_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"データが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)

    # 日付フォーマット対応（YYYYMMDD形式も対応）
    df['race_date'] = pd.to_datetime(df['race_date'], format='%Y%m%d', errors='coerce')
    if df['race_date'].isna().all():
        df['race_date'] = pd.to_datetime(df['race_date'])

    # オッズ推定
    df = estimate_odds(df)

    # 期間フィルター
    if start_date:
        df = df[df['race_date'] >= start_date]
    if end_date:
        df = df[df['race_date'] <= end_date]

    if len(df) == 0:
        print("データがありません")
        return

    # モデル読み込み
    model, features = load_model(track_name)

    # 特徴量準備
    df = prepare_features(df, features)

    # 予測
    X = df[features].fillna(0)
    df['pred_prob'] = predict_with_model(model, X)

    # レースごとの順位付け
    df['pred_rank'] = df.groupby('race_id')['pred_prob'].rank(ascending=False, method='min')

    # 1位と2位の確率差を計算
    def calc_prob_diff(group):
        sorted_probs = group['pred_prob'].sort_values(ascending=False)
        if len(sorted_probs) >= 2:
            return sorted_probs.iloc[0] - sorted_probs.iloc[1]
        return 0

    prob_diffs = df.groupby('race_id').apply(calc_prob_diff)
    df['prob_diff'] = df['race_id'].map(prob_diffs)

    # レース番号抽出
    df['race_num'] = df['race_id'].astype(str).str[-2:].astype(int)

    # オッズ列の確認
    if bet_type == 'place':
        odds_col = 'place_odds'
    else:
        odds_col = 'win_odds'

    if odds_col not in df.columns:
        print(f"⚠️ {odds_col}列がありません")
        return

    # 期待値計算
    df['ev'] = df['pred_prob'] * df[odds_col]

    # フィルター適用（予測1位のみ）
    bets = df[df['pred_rank'] == 1].copy()

    if min_prob > 0:
        bets = bets[bets['pred_prob'] >= min_prob]
    if min_prob_diff > 0:
        bets = bets[bets['prob_diff'] >= min_prob_diff]
    if min_race_num > 1:
        bets = bets[bets['race_num'] >= min_race_num]
    if min_ev > 0:
        bets = bets[bets['ev'] >= min_ev]

    # 的中判定
    if bet_type == 'place':
        bets['is_hit'] = bets['rank'] <= 3  # 3着以内で的中
    else:
        bets['is_hit'] = bets['rank'] == 1  # 1着で的中

    # 結果集計
    total_bets = len(bets)
    if total_bets == 0:
        print("条件に合う買い目がありません")
        return

    hits = bets['is_hit'].sum()
    hit_rate = hits / total_bets * 100

    # 回収率計算（100円均一賭け）
    bet_amount = total_bets * 100
    if bet_type == 'place':
        # 複勝は的中時にオッズ分の払い戻し
        payout = (bets[bets['is_hit']][odds_col] * 100).sum()
    else:
        payout = (bets[bets['is_hit']][odds_col] * 100).sum()

    roi = payout / bet_amount * 100 if bet_amount > 0 else 0

    # 期間情報
    date_from = df['race_date'].min().strftime('%Y-%m-%d')
    date_to = df['race_date'].max().strftime('%Y-%m-%d')

    # 結果表示
    print("\n" + "="*50)
    print(f"📊 バックテスト結果: {track_name.upper()}")
    print("="*50)
    print(f"\n📅 期間: {date_from} 〜 {date_to}")
    print(f"🎯 賭け方: {'複勝' if bet_type == 'place' else '単勝'}")
    print(f"\n── フィルター条件 ──")
    print(f"  予測1位のみ: ✓")
    if min_prob > 0:
        print(f"  最低予測確率: {min_prob*100:.1f}%以上")
    if min_prob_diff > 0:
        print(f"  確率差（1位-2位）: {min_prob_diff*100:.1f}%以上")
    if min_race_num > 1:
        print(f"  レース: {min_race_num}R以降")
    if min_ev > 0:
        print(f"  期待値: {min_ev:.2f}以上")

    print(f"\n── 結果 ──")
    print(f"  買い目数: {total_bets:,}点")
    print(f"  的中数:   {hits:,}点")
    print(f"  的中率:   {hit_rate:.1f}%")
    print(f"\n  投資額:   {bet_amount:,}円")
    print(f"  払戻額:   {payout:,.0f}円")
    print(f"\n  💰 回収率: {roi:.1f}%", end="")
    if roi >= 100:
        print(" 🎉 黒字!")
    else:
        print(f" （あと{100-roi:.1f}%で黒字）")

    # 月別推移
    bets['month'] = bets['race_date'].dt.to_period('M')
    bets['payout'] = bets[odds_col] * bets['is_hit'] * 100

    monthly = bets.groupby('month').agg({
        'is_hit': ['count', 'sum'],
        'payout': 'sum'
    })
    monthly.columns = ['bets', 'hits', 'payout']
    monthly['roi'] = monthly['payout'] / (monthly['bets'] * 100) * 100

    print(f"\n── 月別回収率 ──")
    for month, row in monthly.iterrows():
        bar_len = int(row['roi'] / 10)
        bar = '█' * min(bar_len, 15) + '░' * max(0, 10 - bar_len)
        status = "✓" if row['roi'] >= 100 else " "
        print(f"  {month} {bar} {row['roi']:5.1f}% ({int(row['bets']):3d}点) {status}")

    print("\n" + "="*50)

    return {
        'total_bets': total_bets,
        'hits': hits,
        'hit_rate': hit_rate,
        'bet_amount': bet_amount,
        'payout': payout,
        'roi': roi
    }


def find_best_strategy(track_name, bet_type='place'):
    """最適な戦略を探索"""
    print("\n🔍 最適戦略を探索中...\n")

    results = []

    # いろんな条件を試す
    for min_prob in [0, 0.3, 0.4, 0.5]:
        for min_prob_diff in [0, 0.05, 0.10, 0.15]:
            for min_race_num in [1, 6, 8]:
                for min_ev in [0, 1.0, 1.2, 1.5]:
                    try:
                        result = run_backtest_quiet(
                            track_name,
                            min_prob=min_prob,
                            min_prob_diff=min_prob_diff,
                            min_race_num=min_race_num,
                            min_ev=min_ev,
                            bet_type=bet_type
                        )
                        if result and result['total_bets'] >= 50:  # 最低50件
                            results.append({
                                'min_prob': min_prob,
                                'min_prob_diff': min_prob_diff,
                                'min_race_num': min_race_num,
                                'min_ev': min_ev,
                                **result
                            })
                    except:
                        pass

    if not results:
        print("条件に合う戦略が見つかりませんでした")
        return

    # 回収率でソート
    results = sorted(results, key=lambda x: x['roi'], reverse=True)

    print("🏆 回収率TOP5戦略:\n")
    print(f"{'順位':<4} {'回収率':<8} {'的中率':<8} {'買い目':<8} 条件")
    print("-" * 70)

    for i, r in enumerate(results[:5], 1):
        conditions = []
        if r['min_prob'] > 0:
            conditions.append(f"確率{r['min_prob']*100:.0f}%+")
        if r['min_prob_diff'] > 0:
            conditions.append(f"差{r['min_prob_diff']*100:.0f}%+")
        if r['min_race_num'] > 1:
            conditions.append(f"{r['min_race_num']}R+")
        if r['min_ev'] > 0:
            conditions.append(f"EV{r['min_ev']:.1f}+")

        cond_str = ", ".join(conditions) if conditions else "フィルターなし"

        status = "🎉" if r['roi'] >= 100 else ""
        print(f"{i:<4} {r['roi']:>6.1f}% {r['hit_rate']:>6.1f}% {r['total_bets']:>6}点  {cond_str} {status}")

    print("\n" + "="*50)
    best = results[0]
    print(f"\n💡 推奨設定:")
    print(f"   最低予測確率: {best['min_prob']*100:.0f}%")
    print(f"   確率差: {best['min_prob_diff']*100:.0f}%")
    print(f"   レース: {best['min_race_num']}R以降")
    print(f"   期待値: {best['min_ev']:.1f}以上")
    print(f"\n   → 回収率 {best['roi']:.1f}% が期待できます")


def run_backtest_quiet(track_name, **kwargs):
    """結果表示なしのバックテスト（探索用）"""
    csv_path = BASE_DIR / f"data/races_{track_name}.csv"
    df = pd.read_csv(csv_path)
    df['race_date'] = pd.to_datetime(df['race_date'], format='%Y%m%d', errors='coerce')
    df = estimate_odds_quiet(df)

    model, features = load_model(track_name)
    df = prepare_features(df, features)

    X = df[features].fillna(0)
    df['pred_prob'] = predict_with_model(model, X)
    df['pred_rank'] = df.groupby('race_id')['pred_prob'].rank(ascending=False, method='min')

    def calc_prob_diff(group):
        sorted_probs = group['pred_prob'].sort_values(ascending=False)
        if len(sorted_probs) >= 2:
            return sorted_probs.iloc[0] - sorted_probs.iloc[1]
        return 0

    prob_diffs = df.groupby('race_id').apply(calc_prob_diff)
    df['prob_diff'] = df['race_id'].map(prob_diffs)
    df['race_num'] = df['race_id'].astype(str).str[-2:].astype(int)

    bet_type = kwargs.get('bet_type', 'place')
    odds_col = 'place_odds' if bet_type == 'place' else 'win_odds'

    if odds_col not in df.columns:
        return None

    df['ev'] = df['pred_prob'] * df[odds_col]

    bets = df[df['pred_rank'] == 1].copy()

    min_prob = kwargs.get('min_prob', 0)
    min_prob_diff = kwargs.get('min_prob_diff', 0)
    min_race_num = kwargs.get('min_race_num', 1)
    min_ev = kwargs.get('min_ev', 0)

    if min_prob > 0:
        bets = bets[bets['pred_prob'] >= min_prob]
    if min_prob_diff > 0:
        bets = bets[bets['prob_diff'] >= min_prob_diff]
    if min_race_num > 1:
        bets = bets[bets['race_num'] >= min_race_num]
    if min_ev > 0:
        bets = bets[bets['ev'] >= min_ev]

    total_bets = len(bets)
    if total_bets == 0:
        return None

    if bet_type == 'place':
        bets['is_hit'] = bets['rank'] <= 3
    else:
        bets['is_hit'] = bets['rank'] == 1

    hits = bets['is_hit'].sum()
    hit_rate = hits / total_bets * 100
    bet_amount = total_bets * 100
    payout = (bets[bets['is_hit']][odds_col] * 100).sum()
    roi = payout / bet_amount * 100 if bet_amount > 0 else 0

    return {
        'total_bets': total_bets,
        'hits': hits,
        'hit_rate': hit_rate,
        'bet_amount': bet_amount,
        'payout': payout,
        'roi': roi
    }


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("""
==================================================
          競馬バックテストツール
==================================================
    """)

    # コマンドライン引数
    if len(sys.argv) < 2:
        print("使い方:")
        print("  python backtest.py <競馬場> [オプション]")
        print("")
        print("競馬場: ohi, kawasaki")
        print("")
        print("オプション:")
        print("  --find-best     最適戦略を自動探索")
        print("  --prob X        最低予測確率 X% (例: --prob 40)")
        print("  --diff X        確率差 X% (例: --diff 10)")
        print("  --race X        XR以降 (例: --race 8)")
        print("  --ev X          期待値X以上 (例: --ev 1.5)")
        print("  --win           単勝で検証（デフォルトは複勝）")
        print("")
        print("例:")
        print("  python backtest.py ohi")
        print("  python backtest.py ohi --find-best")
        print("  python backtest.py ohi --prob 40 --diff 10 --race 8")
        sys.exit(1)

    track = sys.argv[1].lower()

    if '--find-best' in sys.argv:
        bet_type = 'win' if '--win' in sys.argv else 'place'
        find_best_strategy(track, bet_type=bet_type)
    else:
        # オプション解析
        min_prob = 0
        min_prob_diff = 0
        min_race_num = 1
        min_ev = 0
        bet_type = 'place'

        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == '--prob' and i + 1 < len(args):
                min_prob = float(args[i + 1]) / 100
                i += 2
            elif args[i] == '--diff' and i + 1 < len(args):
                min_prob_diff = float(args[i + 1]) / 100
                i += 2
            elif args[i] == '--race' and i + 1 < len(args):
                min_race_num = int(args[i + 1])
                i += 2
            elif args[i] == '--ev' and i + 1 < len(args):
                min_ev = float(args[i + 1])
                i += 2
            elif args[i] == '--win':
                bet_type = 'win'
                i += 1
            else:
                i += 1

        run_backtest(
            track,
            min_prob=min_prob,
            min_prob_diff=min_prob_diff,
            min_race_num=min_race_num,
            min_ev=min_ev,
            bet_type=bet_type
        )
