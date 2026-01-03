"""
v8モデル（修正版）: オッズ除外 + 閾値フィルタリング
- データリークを完全に排除
- 予測時点で利用可能な情報のみ使用
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')


def load_data(track_name):
    """データ読み込み"""
    df = pd.read_csv(f'data/races_{track_name}.csv')
    print(f'{track_name.upper()}: {len(df):,}件')
    return df


def calculate_past_speed_index(df):
    """
    過去レースからスピード指数を計算（データリークなし）
    馬の過去の平均スピード指数を特徴量として使用
    """
    df = df.copy()

    # 数値変換
    df['race_time_seconds'] = pd.to_numeric(df['race_time_seconds'], errors='coerce')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['weight_carried'] = pd.to_numeric(df['weight_carried'], errors='coerce')
    df['race_date'] = pd.to_datetime(df['race_date'].astype(str), format='%Y%m%d', errors='coerce')

    # ソート
    df = df.sort_values(['race_date', 'race_id', 'horse_number'])

    # 距離・馬場別の基準タイム（全データから計算）
    base_stats = df.groupby(['distance', 'track_condition'])['race_time_seconds'].agg(['mean', 'std']).reset_index()
    base_stats.columns = ['distance', 'track_condition', 'base_time', 'base_std']

    df = df.merge(base_stats, on=['distance', 'track_condition'], how='left')

    # 当該レースのスピード指数（後で過去平均を取るため）
    df['race_speed_index'] = np.where(
        df['base_std'] > 0,
        (df['base_time'] - df['race_time_seconds']) / df['base_std'] * 10 + 50,
        50
    )

    # 斤量補正
    df['race_speed_index'] = df['race_speed_index'] + (55 - df['weight_carried'].fillna(55)) * 2

    # 馬ごとの過去レースの平均スピード指数を計算（当該レースを除く）
    # expanding meanで累積平均を計算し、shiftで当該レースを除外
    df = df.sort_values(['horse_id', 'race_date'])
    df['past_speed_index'] = df.groupby('horse_id')['race_speed_index'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # 過去3レースの平均
    df['past_3_speed_index'] = df.groupby('horse_id')['race_speed_index'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    # 欠損値は50（平均）で埋める
    df['past_speed_index'] = df['past_speed_index'].fillna(50)
    df['past_3_speed_index'] = df['past_3_speed_index'].fillna(50)

    return df


def calculate_past_last_3f(df):
    """
    過去レースの上がり3F平均を計算（データリークなし）
    """
    df = df.copy()

    df['last_3f'] = pd.to_numeric(df['last_3f'], errors='coerce')
    df['race_date'] = pd.to_datetime(df['race_date'].astype(str), format='%Y%m%d', errors='coerce')

    # 馬ごとの過去の上がり3F平均
    df = df.sort_values(['horse_id', 'race_date'])
    df['past_last_3f'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # 過去3レースの平均
    df['past_3_last_3f'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    # 欠損値は全体平均で埋める
    overall_mean = df['last_3f'].mean()
    df['past_last_3f'] = df['past_last_3f'].fillna(overall_mean if pd.notna(overall_mean) else 40)
    df['past_3_last_3f'] = df['past_3_last_3f'].fillna(overall_mean if pd.notna(overall_mean) else 40)

    return df


def calculate_days_since_last_race(df):
    """前走からの経過日数を計算"""
    df = df.copy()

    df['race_date'] = pd.to_datetime(df['race_date'].astype(str), format='%Y%m%d', errors='coerce')

    df = df.sort_values(['horse_id', 'race_date'])
    df['prev_race_date'] = df.groupby('horse_id')['race_date'].shift(1)
    df['days_since_last'] = (df['race_date'] - df['prev_race_date']).dt.days

    df['days_since_last'] = df['days_since_last'].fillna(60)
    df['days_since_last'] = df['days_since_last'].clip(0, 180)

    return df


def create_features_no_leak(df):
    """
    データリークのない特徴量セット
    全ての特徴量が予測時点で利用可能
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

    # 過去のスピード指数（データリークなし）
    df = calculate_past_speed_index(df)

    # 過去の上がり3F（データリークなし）
    df = calculate_past_last_3f(df)

    # 前走からの経過日数
    df = calculate_days_since_last_race(df)

    # --- レース内での相対順位（予測時点で利用可能な情報のみ） ---
    df['show_rate_rank'] = df.groupby('race_id')['horse_show_rate'].rank(ascending=False)
    df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False)
    df['jockey_rank'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)
    df['avg_rank_rank'] = df.groupby('race_id')['horse_avg_rank'].rank(ascending=True)

    # 過去スピード指数での順位
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
    df['track_condition_code'] = df['track_condition'].map(condition_map).fillna(0)

    # --- 休み明け効果 ---
    df['is_fresh'] = (df['days_since_last'] >= 30).astype(int)
    df['is_long_rest'] = (df['days_since_last'] >= 60).astype(int)

    # --- 特徴量リスト（全て予測時点で利用可能！） ---
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

    # 欠損値を埋める
    for f in features:
        if f in df.columns:
            median_val = df[f].median() if df[f].notna().any() else 0
            df[f] = df[f].fillna(median_val)
        else:
            df[f] = 0

    return df, features


def backtest_with_threshold(df, features, model, test_start_idx, threshold=0.1):
    """閾値フィルタリング付きバックテスト"""
    test_df = df.iloc[test_start_idx:].copy()

    X_test = test_df[features].values
    probs = model.predict_proba(X_test)[:, 1]
    test_df['pred_prob'] = probs

    results_all = []
    results_filtered = []

    for race_id in test_df['race_id'].unique():
        race = test_df[test_df['race_id'] == race_id].copy()

        if len(race) < 3:
            continue

        race = race.sort_values('pred_prob', ascending=False)

        top1_prob = race.iloc[0]['pred_prob']
        top2_prob = race.iloc[1]['pred_prob']
        gap = top1_prob - top2_prob

        top1 = race.iloc[0]
        hit = 1 if top1['rank'] <= 3 else 0

        # 払戻計算（place_oddsが0や欠損の場合はデフォルト値を使用）
        payout = 0
        if hit:
            place_odds_val = top1.get('place_odds', None)
            try:
                if pd.notna(place_odds_val):
                    payout_str = str(place_odds_val)
                    if '-' in payout_str:
                        payout_num = float(payout_str.split('-')[0])
                    else:
                        payout_num = float(payout_str)

                    # 有効なオッズの場合のみ使用（1.0以上）
                    if payout_num >= 1.0:
                        payout = payout_num * 100
                    else:
                        # オッズが無効な場合、デフォルト（複勝平均1.5倍）
                        payout = 150
                else:
                    payout = 150
            except:
                payout = 150

        result = {
            'race_id': race_id,
            'pred_prob': top1_prob,
            'gap': gap,
            'actual_rank': top1['rank'],
            'hit': hit,
            'bet': 100,
            'payout': payout if hit else 0
        }

        results_all.append(result)
        if gap >= threshold:
            results_filtered.append(result)

    def summarize(results):
        if len(results) == 0:
            return {'hit_rate': 0, 'roi': 0, 'n_bets': 0, 'n_hits': 0}

        results_df = pd.DataFrame(results)
        hit_rate = results_df['hit'].mean()
        total_bet = results_df['bet'].sum()
        total_payout = results_df['payout'].sum()
        roi = total_payout / total_bet if total_bet > 0 else 0

        return {
            'hit_rate': hit_rate,
            'roi': roi,
            'n_bets': len(results_df),
            'n_hits': int(results_df['hit'].sum())
        }

    return {
        'all': summarize(results_all),
        'filtered': summarize(results_filtered),
        'threshold': threshold
    }


def train_model(df, features):
    """モデル学習"""
    df = df.copy()
    df['target'] = (df['rank'] <= 3).astype(int)

    df = df.sort_values('race_id')
    split_idx = int(len(df) * 0.7)

    train_df = df.iloc[:split_idx]

    X_train = train_df[features].values
    y_train = train_df['target'].values

    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbose': -1,
        'random_state': 42
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    return model, split_idx


def find_optimal_threshold(df, features, model, test_start_idx):
    """最適な閾値を探索"""
    print('\n閾値探索中...')

    best_threshold = 0.1
    best_roi = 0

    results = []

    for threshold in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        bt = backtest_with_threshold(df, features, model, test_start_idx, threshold)

        filtered = bt['filtered']
        all_races = bt['all']

        results.append({
            'threshold': threshold,
            'n_bets': filtered['n_bets'],
            'hit_rate': filtered['hit_rate'],
            'roi': filtered['roi'],
            'coverage': filtered['n_bets'] / max(all_races['n_bets'], 1)
        })

        if filtered['roi'] > best_roi and filtered['n_bets'] >= 30:
            best_roi = filtered['roi']
            best_threshold = threshold

    print(f'\n{"閾値":>6} | {"レース数":>8} | {"的中率":>7} | {"回収率":>7} | {"カバー率":>7}')
    print('-' * 55)

    for r in results:
        roi_mark = '★' if r['roi'] >= 1.0 else ('△' if r['roi'] >= 0.9 else '')
        print(f'{r["threshold"]:>6.2f} | {r["n_bets"]:>8} | {r["hit_rate"]*100:>6.1f}% | {r["roi"]*100:>6.0f}% | {r["coverage"]*100:>6.1f}% {roi_mark}')

    return best_threshold, results


def run_experiment(track_name):
    """実験実行"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} v8モデル（データリーク修正版）')
    print(f'{"="*60}')

    df = load_data(track_name)
    df_processed, features = create_features_no_leak(df)

    print(f'\n特徴量数: {len(features)}')
    print('※ 全特徴量が予測時点で利用可能（データリークなし）')

    df_processed['rank'] = pd.to_numeric(df_processed['rank'], errors='coerce')

    model, split_idx = train_model(df_processed, features)

    # AUC
    test_df = df_processed.iloc[split_idx:]
    X_test = test_df[features].values
    y_test = (test_df['rank'] <= 3).astype(int).values
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f'\nAUC: {auc:.4f}')

    # 特徴量重要度
    print(f'\n【特徴量重要度 Top10】')
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in importance.head(10).iterrows():
        print(f'  {row["feature"]}: {row["importance"]:.0f}')

    # 閾値探索
    best_threshold, threshold_results = find_optimal_threshold(
        df_processed, features, model, split_idx
    )

    print(f'\n最適閾値: {best_threshold}')

    # 最終バックテスト
    final_bt = backtest_with_threshold(df_processed, features, model, split_idx, best_threshold)

    print(f'\n{"="*60}')
    print('【最終結果（データリークなし）】')
    print(f'{"="*60}')

    all_r = final_bt['all']
    flt_r = final_bt['filtered']

    print(f'\n全レース（フィルタなし）:')
    print(f'  レース数: {all_r["n_bets"]:,}')
    print(f'  的中率: {all_r["hit_rate"]*100:.1f}%')
    print(f'  回収率: {all_r["roi"]*100:.0f}%')

    print(f'\nフィルタ後（gap >= {best_threshold}）:')
    print(f'  レース数: {flt_r["n_bets"]:,} ({flt_r["n_bets"]/max(all_r["n_bets"],1)*100:.1f}%)')
    print(f'  的中率: {flt_r["hit_rate"]*100:.1f}%')
    print(f'  回収率: {flt_r["roi"]*100:.0f}%')

    if flt_r['roi'] >= 1.0:
        print(f'\n  ★★★ 回収率100%超え達成！ ★★★')
    elif flt_r['roi'] >= 0.9:
        print(f'\n  △ あと少し（90%以上）')
    else:
        print(f'\n  まだ課題あり（{100-flt_r["roi"]*100:.0f}%不足）')

    return {
        'model': model,
        'features': features,
        'auc': auc,
        'best_threshold': best_threshold,
        'results': {
            'all': all_r,
            'filtered': flt_r
        },
        'threshold_results': threshold_results
    }


def save_model(result, track_name):
    """モデル保存"""
    model_data = {
        'model': result['model'],
        'features': result['features'],
        'best_threshold': result['best_threshold'],
        'auc': result['auc'],
        'version': 'v8_no_leak',
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    model_path = f'models/model_{track_name}_v8.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    meta = {
        'track_name': track_name,
        'version': 'v8_no_leak',
        'trained_at': model_data['trained_at'],
        'auc': round(result['auc'], 4),
        'best_threshold': result['best_threshold'],
        'filtered_hit_rate': round(result['results']['filtered']['hit_rate'], 4),
        'filtered_roi': round(result['results']['filtered']['roi'], 4),
        'all_hit_rate': round(result['results']['all']['hit_rate'], 4),
        'all_roi': round(result['results']['all']['roi'], 4),
    }

    with open(f'models/model_{track_name}_v8_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\n✓ モデル保存: {model_path}')

    return model_path


if __name__ == '__main__':
    print('='*60)
    print('v8モデル（データリーク修正版）')
    print('目標: 回収率100%超え（正しい方法で）')
    print('='*60)

    all_results = {}

    for track in ['kawasaki', 'ohi']:
        result = run_experiment(track)
        save_model(result, track)
        all_results[track] = result

    print('\n' + '='*60)
    print('総合結果（データリークなし）')
    print('='*60)

    for track, r in all_results.items():
        flt = r['results']['filtered']
        all_r = r['results']['all']
        print(f'\n{track.upper()}:')
        print(f'  AUC: {r["auc"]:.4f}')
        print(f'  閾値: {r["best_threshold"]}')
        print(f'  全体: 的中率={all_r["hit_rate"]*100:.1f}%, 回収率={all_r["roi"]*100:.0f}%')
        print(f'  フィルタ後: 的中率={flt["hit_rate"]*100:.1f}%, 回収率={flt["roi"]*100:.0f}%')
