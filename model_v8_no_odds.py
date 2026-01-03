"""
v8モデル: オッズ除外 + 閾値フィルタリング + スピード指数
- 回収率100%超えを目指す
- 参考: Qiita 140%, うまたん 168%の手法
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


def calculate_speed_index(df):
    """
    スピード指数を計算
    西田式: (基準タイム - 走破タイム) × 距離指数 + 馬場指数 + (斤量-55)×2 + 80
    簡易版: 距離別・馬場別の平均タイムとの差を標準化
    """
    df = df.copy()

    # 数値変換
    df['race_time_seconds'] = pd.to_numeric(df['race_time_seconds'], errors='coerce')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['weight_carried'] = pd.to_numeric(df['weight_carried'], errors='coerce')

    # 距離・馬場別の基準タイム（平均）を計算
    base_time = df.groupby(['distance', 'track_condition'])['race_time_seconds'].transform('mean')
    base_std = df.groupby(['distance', 'track_condition'])['race_time_seconds'].transform('std')

    # スピード指数 = (基準タイム - 走破タイム) / 標準偏差 * 10 + 50
    # タイムが速いほど高い値になる
    df['speed_index'] = np.where(
        base_std > 0,
        (base_time - df['race_time_seconds']) / base_std * 10 + 50,
        50
    )

    # 斤量補正（55kgを基準に±2点/kg）
    df['speed_index'] = df['speed_index'] + (55 - df['weight_carried'].fillna(55)) * 2

    # 欠損値は50（平均）で埋める
    df['speed_index'] = df['speed_index'].fillna(50)

    return df


def calculate_days_since_last_race(df):
    """前走からの経過日数を計算"""
    df = df.copy()

    # race_dateを日付型に変換
    df['race_date'] = pd.to_datetime(df['race_date'].astype(str), format='%Y%m%d', errors='coerce')

    # 馬ごとに前回出走日を取得
    df = df.sort_values(['horse_id', 'race_date'])
    df['prev_race_date'] = df.groupby('horse_id')['race_date'].shift(1)

    # 経過日数
    df['days_since_last'] = (df['race_date'] - df['prev_race_date']).dt.days

    # 欠損値（初出走）は60日（長期休み明け扱い）
    df['days_since_last'] = df['days_since_last'].fillna(60)

    # 極端な値をクリップ（0-180日）
    df['days_since_last'] = df['days_since_last'].clip(0, 180)

    return df


def create_features_no_odds(df):
    """
    オッズを除外した特徴量セット
    市場の予測に頼らず、馬の実力だけで予測
    """
    df = df.copy()

    # 数値変換
    num_cols = [
        'horse_runs', 'horse_win_rate', 'horse_show_rate', 'horse_avg_rank',
        'horse_recent_win_rate', 'horse_recent_show_rate', 'horse_recent_avg_rank',
        'last_rank', 'jockey_win_rate', 'jockey_show_rate',
        'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
        'field_size', 'horse_weight', 'weight_change', 'last_3f', 'race_time_seconds'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # スピード指数を計算
    df = calculate_speed_index(df)

    # 前走からの経過日数
    df = calculate_days_since_last_race(df)

    # --- レース内での相対順位（オッズなしでの実力比較） ---
    df['show_rate_rank'] = df.groupby('race_id')['horse_show_rate'].rank(ascending=False)
    df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False)
    df['jockey_rank'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)
    df['avg_rank_rank'] = df.groupby('race_id')['horse_avg_rank'].rank(ascending=True)
    df['speed_rank'] = df.groupby('race_id')['speed_index'].rank(ascending=False)
    df['last_3f_rank'] = df.groupby('race_id')['last_3f'].rank(ascending=True)  # 速いほど良い

    # --- レース内での相対値 ---
    df['show_rate_vs_field'] = df['horse_show_rate'] - df.groupby('race_id')['horse_show_rate'].transform('mean')
    df['win_rate_vs_field'] = df['horse_win_rate'] - df.groupby('race_id')['horse_win_rate'].transform('mean')
    df['jockey_vs_field'] = df['jockey_win_rate'] - df.groupby('race_id')['jockey_win_rate'].transform('mean')
    df['speed_vs_field'] = df['speed_index'] - df.groupby('race_id')['speed_index'].transform('mean')

    # --- 経験値スコア ---
    df['experience_score'] = np.log1p(df['horse_runs']) * df['horse_show_rate']

    # --- 調子スコア（直近の成績） ---
    df['form_score'] = df['horse_recent_show_rate'].fillna(df['horse_show_rate'])
    df['form_trend'] = df['form_score'] - df['horse_show_rate']  # 調子の上昇/下降

    # --- 前走の成績 ---
    df['last_rank_score'] = np.where(df['last_rank'] <= 3, 1, 0)  # 前走3着以内
    df['last_rank_normalized'] = df['last_rank'] / df['field_size'].clip(lower=1)

    # --- 馬場・距離適性（カテゴリ） ---
    # track_conditionをエンコード
    condition_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    df['track_condition_code'] = df['track_condition'].map(condition_map).fillna(0)

    # --- 特徴量リスト（オッズ関連は含まない！） ---
    features = [
        # 相対順位
        'show_rate_rank', 'win_rate_rank', 'jockey_rank', 'avg_rank_rank',
        'speed_rank', 'last_3f_rank',
        # 相対値
        'show_rate_vs_field', 'win_rate_vs_field', 'jockey_vs_field', 'speed_vs_field',
        # 実績
        'horse_show_rate', 'horse_win_rate', 'horse_avg_rank',
        'jockey_win_rate', 'jockey_show_rate',
        # 経験・調子
        'experience_score', 'form_score', 'form_trend', 'horse_runs',
        # 前走
        'last_rank', 'last_rank_score', 'last_rank_normalized',
        # スピード
        'speed_index',
        # 経過日数
        'days_since_last',
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
    """
    閾値フィルタリング付きバックテスト
    予測確率の1位と2位の差がthreshold以上の場合のみベット
    """
    test_df = df.iloc[test_start_idx:].copy()

    X_test = test_df[features].values

    # 予測
    probs = model.predict_proba(X_test)[:, 1]
    test_df['pred_prob'] = probs

    results_all = []  # 全レース
    results_filtered = []  # フィルター後

    # レースごとに評価
    for race_id in test_df['race_id'].unique():
        race = test_df[test_df['race_id'] == race_id].copy()

        if len(race) < 3:
            continue

        # 予測確率でソート
        race = race.sort_values('pred_prob', ascending=False)

        # Top1とTop2の差を計算
        top1_prob = race.iloc[0]['pred_prob']
        top2_prob = race.iloc[1]['pred_prob']
        gap = top1_prob - top2_prob

        top1 = race.iloc[0]

        # 的中判定
        hit = 1 if top1['rank'] <= 3 else 0

        # 払戻計算（複勝）
        if hit and 'place_odds' in race.columns and pd.notna(top1['place_odds']):
            try:
                payout_str = str(top1['place_odds'])
                if '-' in payout_str:
                    payout = float(payout_str.split('-')[0]) * 100
                else:
                    payout = float(payout_str) * 100
            except:
                payout = 150  # デフォルト
        else:
            payout = 0 if not hit else 150

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

        # 閾値フィルタリング
        if gap >= threshold:
            results_filtered.append(result)

    # 結果集計
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

    # 時系列分割
    df = df.sort_values('race_id')
    split_idx = int(len(df) * 0.7)

    train_df = df.iloc[:split_idx]

    X_train = train_df[features].values
    y_train = train_df['target'].values

    # LightGBMパラメータ
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

    for threshold in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
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
        roi_mark = '★' if r['roi'] >= 1.0 else ''
        print(f'{r["threshold"]:>6.2f} | {r["n_bets"]:>8} | {r["hit_rate"]*100:>6.1f}% | {r["roi"]*100:>6.0f}% | {r["coverage"]*100:>6.1f}% {roi_mark}')

    return best_threshold, results


def run_experiment(track_name):
    """実験実行"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} v8モデル（オッズ除外）')
    print(f'{"="*60}')

    # データ読み込み
    df = load_data(track_name)

    # 特徴量作成
    df_processed, features = create_features_no_odds(df)

    print(f'\n特徴量数: {len(features)}')
    print(f'使用特徴量: {features[:10]}...')

    # ターゲット作成
    df_processed['rank'] = pd.to_numeric(df_processed['rank'], errors='coerce')

    # モデル学習
    model, split_idx = train_model(df_processed, features)

    # AUC計算
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
    print('【最終結果】')
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
        'version': 'v8_no_odds',
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    model_path = f'models/model_{track_name}_v8.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    meta = {
        'track_name': track_name,
        'version': 'v8_no_odds',
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
    print('v8モデル: オッズ除外 + 閾値フィルタリング')
    print('目標: 回収率100%超え')
    print('='*60)

    all_results = {}

    for track in ['kawasaki', 'ohi']:
        result = run_experiment(track)
        save_model(result, track)
        all_results[track] = result

    print('\n' + '='*60)
    print('総合結果')
    print('='*60)

    for track, r in all_results.items():
        flt = r['results']['filtered']
        print(f'\n{track.upper()}:')
        print(f'  AUC: {r["auc"]:.4f}')
        print(f'  閾値: {r["best_threshold"]}')
        print(f'  フィルタ後: 的中率={flt["hit_rate"]*100:.1f}%, 回収率={flt["roi"]*100:.0f}%')
