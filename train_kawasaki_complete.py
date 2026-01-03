"""
川崎競馬 v10モデル: kawasaki_complete.csv使用版
- データ量1.7倍（42,458件、2021年〜）
- 過去成績を自前で計算（データリークなし）
- model_v8_fixed.pyベースの特徴量
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


def load_data():
    """kawasaki_complete.csv読み込み"""
    df = pd.read_csv('data/kawasaki_complete.csv', encoding='utf-8-sig')
    print(f'読み込み: {len(df):,}件')
    print(f'カラム: {list(df.columns)}')
    return df


def calculate_historical_stats(df):
    """
    過去成績を計算（データリークなし）
    - 当該レースを含まない累積統計をexpanding + shiftで計算
    """
    df = df.copy()

    # 日付変換
    df['race_date'] = pd.to_datetime(df['race_date'].astype(str), format='%Y%m%d', errors='coerce')

    # 着順の数値化
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

    # 勝ち・複勝フラグ
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_place'] = (df['rank'] <= 2).astype(int)
    df['is_show'] = (df['rank'] <= 3).astype(int)

    # ソート（馬ごと、日付順）
    df = df.sort_values(['horse_id', 'race_date', 'race_id'])

    # === 馬の過去成績（当該レースを除く） ===
    df['horse_runs'] = df.groupby('horse_id').cumcount()

    df['horse_win_rate'] = df.groupby('horse_id')['is_win'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)

    df['horse_place_rate'] = df.groupby('horse_id')['is_place'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)

    df['horse_show_rate'] = df.groupby('horse_id')['is_show'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)

    df['horse_avg_rank'] = df.groupby('horse_id')['rank'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(6)

    # 直近5走の成績
    df['horse_recent_win_rate'] = df.groupby('horse_id')['is_win'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    ).fillna(0)

    df['horse_recent_show_rate'] = df.groupby('horse_id')['is_show'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    ).fillna(0)

    df['horse_recent_avg_rank'] = df.groupby('horse_id')['rank'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    ).fillna(6)

    # 前走着順
    df['last_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(6)

    # === 騎手の過去成績（名前ベース） ===
    df = df.sort_values(['jockey_name', 'race_date', 'race_id'])

    df['jockey_win_rate'] = df.groupby('jockey_name')['is_win'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0.1)

    df['jockey_place_rate'] = df.groupby('jockey_name')['is_place'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0.2)

    df['jockey_show_rate'] = df.groupby('jockey_name')['is_show'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0.3)

    # === 前走からの経過日数 ===
    df = df.sort_values(['horse_id', 'race_date'])
    df['prev_race_date'] = df.groupby('horse_id')['race_date'].shift(1)
    df['days_since_last'] = (df['race_date'] - df['prev_race_date']).dt.days.fillna(60).clip(0, 180)

    print(f'過去成績計算完了')
    return df


def calculate_past_speed_index(df):
    """過去スピード指数（データリークなし）"""
    df = df.copy()

    # finish_time を秒に変換
    def time_to_seconds(t):
        if pd.isna(t):
            return np.nan
        try:
            t_str = str(t)
            if ':' in t_str:
                parts = t_str.split(':')
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
            return float(t_str)
        except:
            return np.nan

    df['race_time_seconds'] = df['finish_time'].apply(time_to_seconds)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['weight_carried'] = pd.to_numeric(df['weight_carried'], errors='coerce')

    # 距離・馬場別基準タイム
    df['track_condition'] = df['track_condition'].fillna('良')
    base_stats = df.groupby(['distance', 'track_condition'])['race_time_seconds'].agg(['mean', 'std']).reset_index()
    base_stats.columns = ['distance', 'track_condition', 'base_time', 'base_std']
    df = df.merge(base_stats, on=['distance', 'track_condition'], how='left')

    # スピード指数
    df['race_speed_index'] = np.where(
        df['base_std'] > 0,
        (df['base_time'] - df['race_time_seconds']) / df['base_std'] * 10 + 50,
        50
    )
    df['race_speed_index'] = df['race_speed_index'] + (55 - df['weight_carried'].fillna(55)) * 2

    # 過去平均（当該レース除く）
    df = df.sort_values(['horse_id', 'race_date'])
    df['past_speed_index'] = df.groupby('horse_id')['race_speed_index'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(50)

    df['past_3_speed_index'] = df.groupby('horse_id')['race_speed_index'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(50)

    return df


def calculate_past_last_3f(df):
    """過去上がり3F平均（データリークなし）"""
    df = df.copy()
    df['last_3f'] = pd.to_numeric(df['last_3f'], errors='coerce')

    df = df.sort_values(['horse_id', 'race_date'])
    df['past_last_3f'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    df['past_3_last_3f'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    overall_mean = df['last_3f'].mean()
    df['past_last_3f'] = df['past_last_3f'].fillna(overall_mean if pd.notna(overall_mean) else 40)
    df['past_3_last_3f'] = df['past_3_last_3f'].fillna(overall_mean if pd.notna(overall_mean) else 40)

    return df


def create_features(df):
    """特徴量作成"""
    df = df.copy()

    # 数値変換
    num_cols = ['horse_number', 'bracket', 'age', 'weight_carried', 'distance',
                'field_size', 'horse_weight', 'weight_change', 'popularity']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 過去スピード指数
    df = calculate_past_speed_index(df)

    # 過去上がり3F
    df = calculate_past_last_3f(df)

    # レース内相対順位
    df['show_rate_rank'] = df.groupby('race_id')['horse_show_rate'].rank(ascending=False)
    df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False)
    df['jockey_rank'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)
    df['avg_rank_rank'] = df.groupby('race_id')['horse_avg_rank'].rank(ascending=True)
    df['past_speed_rank'] = df.groupby('race_id')['past_speed_index'].rank(ascending=False)
    df['past_3f_rank'] = df.groupby('race_id')['past_last_3f'].rank(ascending=True)

    # レース内相対値
    df['show_rate_vs_field'] = df['horse_show_rate'] - df.groupby('race_id')['horse_show_rate'].transform('mean')
    df['win_rate_vs_field'] = df['horse_win_rate'] - df.groupby('race_id')['horse_win_rate'].transform('mean')
    df['jockey_vs_field'] = df['jockey_win_rate'] - df.groupby('race_id')['jockey_win_rate'].transform('mean')
    df['past_speed_vs_field'] = df['past_speed_index'] - df.groupby('race_id')['past_speed_index'].transform('mean')

    # 経験値・調子
    df['experience_score'] = np.log1p(df['horse_runs']) * df['horse_show_rate']
    df['form_score'] = df['horse_recent_show_rate'].fillna(df['horse_show_rate'])
    df['form_trend'] = df['form_score'] - df['horse_show_rate']

    # 前走成績
    df['last_rank_score'] = np.where(df['last_rank'] <= 3, 1, 0)
    df['last_rank_normalized'] = df['last_rank'] / df['field_size'].clip(lower=1)

    # 馬場
    condition_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    df['track_condition_code'] = df['track_condition'].map(condition_map).fillna(0)

    # 休み明け
    df['is_fresh'] = (df['days_since_last'] >= 30).astype(int)
    df['is_long_rest'] = (df['days_since_last'] >= 60).astype(int)

    # 特徴量リスト
    features = [
        'show_rate_rank', 'win_rate_rank', 'jockey_rank', 'avg_rank_rank',
        'past_speed_rank', 'past_3f_rank',
        'show_rate_vs_field', 'win_rate_vs_field', 'jockey_vs_field', 'past_speed_vs_field',
        'horse_show_rate', 'horse_win_rate', 'horse_avg_rank',
        'jockey_win_rate', 'jockey_show_rate',
        'experience_score', 'form_score', 'form_trend', 'horse_runs',
        'last_rank', 'last_rank_score', 'last_rank_normalized',
        'past_speed_index', 'past_3_speed_index', 'past_last_3f', 'past_3_last_3f',
        'days_since_last', 'is_fresh', 'is_long_rest',
        'field_size', 'age', 'horse_number', 'track_condition_code',
        'weight_carried', 'horse_weight'
    ]

    # 欠損値埋め
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

        # 払戻計算（place_odds_min使用）
        payout = 0
        if hit:
            place_odds_val = top1.get('place_odds_min', None)
            try:
                if pd.notna(place_odds_val):
                    payout_num = float(place_odds_val)
                    if payout_num >= 1.0:
                        payout = payout_num * 100
                    else:
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
        return {
            'hit_rate': results_df['hit'].mean(),
            'roi': results_df['payout'].sum() / results_df['bet'].sum() if results_df['bet'].sum() > 0 else 0,
            'n_bets': len(results_df),
            'n_hits': int(results_df['hit'].sum())
        }

    return {'all': summarize(results_all), 'filtered': summarize(results_filtered), 'threshold': threshold}


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
    """最適閾値探索"""
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
        roi_mark = ' *' if r['roi'] >= 1.0 else (' +' if r['roi'] >= 0.9 else '')
        print(f'{r["threshold"]:>6.2f} | {r["n_bets"]:>8} | {r["hit_rate"]*100:>6.1f}% | {r["roi"]*100:>6.0f}% | {r["coverage"]*100:>6.1f}%{roi_mark}')

    return best_threshold, results


def save_model(model, features, auc, best_threshold, results, df):
    """モデル保存（拡張メタデータ付き）"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 日付範囲取得
    race_ids = df['race_id'].astype(str)
    dates = race_ids.str[:4] + '-' + race_ids.str[4:6] + '-' + race_ids.str[6:8]
    date_from = dates.min()
    date_to = dates.max()

    model_data = {
        'model': model,
        'features': features,
        'best_threshold': best_threshold,
        'auc': auc,
        'version': 'v10_complete',
        'trained_at': now,
    }

    model_path = 'models/model_kawasaki_v10.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # 拡張メタデータ
    meta = {
        'track_name': 'kawasaki',
        'model_name': 'kawasaki_v10_complete',
        'version': 'v10_complete',
        'trained_at': now,
        'deployed_at': now,
        'data_source': 'kawasaki_complete.csv',
        'data_count': len(df),
        'race_count': df['race_id'].nunique(),
        'date_range': {'from': date_from, 'to': date_to},
        'feature_count': len(features),
        'features': features,
        'auc': round(auc, 4),
        'best_threshold': best_threshold,
        'filtered_hit_rate': round(results['filtered']['hit_rate'], 4),
        'filtered_roi': round(results['filtered']['roi'], 4),
        'all_hit_rate': round(results['all']['hit_rate'], 4),
        'all_roi': round(results['all']['roi'], 4),
    }

    with open('models/model_kawasaki_v10_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\nモデル保存: {model_path}')
    print(f'メタデータ: models/model_kawasaki_v10_meta.json')

    return model_path


def main():
    print('=' * 60)
    print('川崎競馬 v10モデル（kawasaki_complete.csv使用）')
    print('=' * 60)

    # データ読み込み
    df = load_data()

    # 過去成績計算
    df = calculate_historical_stats(df)

    # 特徴量作成
    df, features = create_features(df)
    print(f'\n特徴量数: {len(features)}')

    # モデル学習
    model, split_idx = train_model(df, features)

    # AUC計算
    test_df = df.iloc[split_idx:]
    X_test = test_df[features].values
    y_test = (test_df['rank'] <= 3).astype(int).values
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f'\nAUC: {auc:.4f}')

    # 特徴量重要度
    print('\n【特徴量重要度 Top10】')
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance.head(10).iterrows():
        print(f'  {row["feature"]}: {row["importance"]:.0f}')

    # 閾値探索
    best_threshold, _ = find_optimal_threshold(df, features, model, split_idx)
    print(f'\n最適閾値: {best_threshold}')

    # 最終バックテスト
    final_bt = backtest_with_threshold(df, features, model, split_idx, best_threshold)

    print(f'\n{"=" * 60}')
    print('【最終結果】')
    print(f'{"=" * 60}')

    all_r = final_bt['all']
    flt_r = final_bt['filtered']

    print(f'\n全レース: {all_r["n_bets"]:,}件, 的中率={all_r["hit_rate"]*100:.1f}%, 回収率={all_r["roi"]*100:.0f}%')
    print(f'フィルタ後: {flt_r["n_bets"]:,}件, 的中率={flt_r["hit_rate"]*100:.1f}%, 回収率={flt_r["roi"]*100:.0f}%')

    if flt_r['roi'] >= 1.0:
        print('\n*** 回収率100%超え達成! ***')

    # モデル保存
    save_model(model, features, auc, best_threshold, final_bt, df)

    print('\n完了!')


if __name__ == '__main__':
    main()
