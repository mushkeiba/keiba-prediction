"""
自動モデル最適化スクリプト
- 複数のアプローチを自動テスト
- 的中率・回収率でバックテスト
- ベストモデルを自動選択
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


def create_features_v1(df):
    """アプローチ1: シンプル特徴量（レース内相対値重視）"""
    df = df.copy()

    # 数値変換
    num_cols = ['horse_runs', 'horse_win_rate', 'horse_show_rate', 'horse_avg_rank',
                'last_rank', 'jockey_win_rate', 'jockey_show_rate',
                'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
                'field_size', 'horse_weight', 'last_3f', 'win_odds']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # レース内での相対順位（これが重要！）
    df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False)
    df['show_rate_rank'] = df.groupby('race_id')['horse_show_rate'].rank(ascending=False)
    df['jockey_rank'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)
    df['avg_rank_rank'] = df.groupby('race_id')['horse_avg_rank'].rank(ascending=True)

    # レース内での相対値
    df['win_rate_vs_field'] = df['horse_win_rate'] - df.groupby('race_id')['horse_win_rate'].transform('mean')
    df['show_rate_vs_field'] = df['horse_show_rate'] - df.groupby('race_id')['horse_show_rate'].transform('mean')

    # 人気（オッズから計算）
    df['popularity'] = df.groupby('race_id')['win_odds'].rank(ascending=True)

    # 特徴量リスト
    features = [
        'win_rate_rank', 'show_rate_rank', 'jockey_rank', 'avg_rank_rank',
        'win_rate_vs_field', 'show_rate_vs_field',
        'horse_runs', 'horse_number', 'field_size', 'age',
        'popularity'
    ]

    # 欠損埋め
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(df[f].median() if df[f].notna().any() else 0)
        else:
            df[f] = 0

    return df, features


def create_features_v2(df):
    """アプローチ2: 実績ベース（シンプルな実績のみ）"""
    df = df.copy()

    num_cols = ['horse_runs', 'horse_win_rate', 'horse_show_rate', 'horse_avg_rank',
                'last_rank', 'jockey_win_rate', 'jockey_show_rate',
                'field_size', 'age', 'distance', 'horse_weight', 'last_3f', 'win_odds']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 経験値スコア
    df['exp_score'] = np.log1p(df['horse_runs']) * df['horse_show_rate']

    # 調子スコア（直近の成績）
    df['form_score'] = df['horse_recent_show_rate'] if 'horse_recent_show_rate' in df.columns else df['horse_show_rate']

    # 人気
    df['popularity'] = df.groupby('race_id')['win_odds'].rank(ascending=True)

    features = [
        'horse_show_rate', 'horse_avg_rank', 'last_rank',
        'jockey_show_rate', 'exp_score', 'form_score',
        'horse_runs', 'field_size', 'popularity'
    ]

    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(df[f].median() if df[f].notna().any() else 0)
        else:
            df[f] = 0

    return df, features


def create_features_v3(df):
    """アプローチ3: 人気ベース（市場の知恵を活用）"""
    df = df.copy()

    num_cols = ['horse_win_rate', 'horse_show_rate', 'last_rank',
                'jockey_win_rate', 'field_size', 'win_odds', 'last_3f']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 人気（最重要）
    df['popularity'] = df.groupby('race_id')['win_odds'].rank(ascending=True)
    df['odds_implied_prob'] = 1 / df['win_odds'].clip(lower=1)

    # 人気に対する実力の乖離
    df['show_rate_rank'] = df.groupby('race_id')['horse_show_rate'].rank(ascending=False)
    df['value_gap'] = df['show_rate_rank'] - df['popularity']  # 実力>人気なら負の値

    # 上がり3F
    df['last_3f_rank'] = df.groupby('race_id')['last_3f'].rank(ascending=True)

    features = [
        'popularity', 'odds_implied_prob', 'value_gap',
        'horse_show_rate', 'jockey_win_rate', 'last_rank',
        'last_3f_rank', 'field_size'
    ]

    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(df[f].median() if df[f].notna().any() else 0)
        else:
            df[f] = 0

    return df, features


def backtest(df, features, model, test_start_idx, bet_type='show'):
    """バックテスト: 的中率・回収率を計算"""
    test_df = df.iloc[test_start_idx:].copy()

    X_test = test_df[features].values
    y_test = (test_df['rank'] <= 3).astype(int).values

    # 予測
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)

    test_df['pred_prob'] = probs

    results = []

    # レースごとに評価
    for race_id in test_df['race_id'].unique():
        race = test_df[test_df['race_id'] == race_id].copy()

        if len(race) < 3:
            continue

        # 予測確率でソート
        race = race.sort_values('pred_prob', ascending=False)

        # Top1推奨馬
        top1 = race.iloc[0]

        # 的中判定
        hit = 1 if top1['rank'] <= 3 else 0

        # 払戻計算（複勝）
        if hit and 'place_odds' in race.columns and pd.notna(top1['place_odds']):
            try:
                payout = float(str(top1['place_odds']).split('-')[0]) * 100
            except:
                payout = 150  # デフォルト
        else:
            payout = 0 if not hit else 150

        results.append({
            'race_id': race_id,
            'pred_prob': top1['pred_prob'],
            'actual_rank': top1['rank'],
            'hit': hit,
            'bet': 100,
            'payout': payout if hit else 0
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return {'hit_rate': 0, 'roi': 0, 'n_bets': 0}

    hit_rate = results_df['hit'].mean()
    total_bet = results_df['bet'].sum()
    total_payout = results_df['payout'].sum()
    roi = total_payout / total_bet if total_bet > 0 else 0

    return {
        'hit_rate': hit_rate,
        'roi': roi,
        'n_bets': len(results_df),
        'n_hits': results_df['hit'].sum()
    }


def train_and_evaluate(df, feature_func, model_class, model_params, name):
    """学習＆評価"""
    # 特徴量作成
    df_processed, features = feature_func(df.copy())

    # ターゲット
    df_processed['target'] = (df_processed['rank'] <= 3).astype(int)

    # 時系列分割
    df_processed = df_processed.sort_values('race_id')
    split_idx = int(len(df_processed) * 0.7)

    train_df = df_processed.iloc[:split_idx]

    X_train = train_df[features].values
    y_train = train_df['target'].values

    # 学習
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    # バックテスト（直近30%のデータで）
    bt_results = backtest(df_processed, features, model, split_idx)

    # AUCも計算
    test_df = df_processed.iloc[split_idx:]
    X_test = test_df[features].values
    y_test = test_df['target'].values

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)

    return {
        'name': name,
        'auc': auc,
        'hit_rate': bt_results['hit_rate'],
        'roi': bt_results['roi'],
        'n_bets': bt_results['n_bets'],
        'model': model,
        'features': features,
        'feature_func': feature_func
    }


def run_experiments(track_name):
    """全実験を実行"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} 自動最適化')
    print(f'{"="*60}')

    df = load_data(track_name)

    # 実験リスト
    experiments = [
        # アプローチ1: 相対順位重視
        ('v1_LR', create_features_v1, LogisticRegression, {'max_iter': 1000, 'C': 0.1}),
        ('v1_RF', create_features_v1, RandomForestClassifier, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}),
        ('v1_LGB', create_features_v1, lgb.LGBMClassifier, {'n_estimators': 100, 'max_depth': 5, 'verbose': -1}),

        # アプローチ2: 実績ベース
        ('v2_LR', create_features_v2, LogisticRegression, {'max_iter': 1000, 'C': 0.1}),
        ('v2_RF', create_features_v2, RandomForestClassifier, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}),
        ('v2_LGB', create_features_v2, lgb.LGBMClassifier, {'n_estimators': 100, 'max_depth': 5, 'verbose': -1}),

        # アプローチ3: 人気ベース
        ('v3_LR', create_features_v3, LogisticRegression, {'max_iter': 1000, 'C': 0.1}),
        ('v3_RF', create_features_v3, RandomForestClassifier, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}),
        ('v3_LGB', create_features_v3, lgb.LGBMClassifier, {'n_estimators': 100, 'max_depth': 5, 'verbose': -1}),
    ]

    results = []

    print('\n実験中...')
    for name, feat_func, model_class, params in experiments:
        try:
            r = train_and_evaluate(df, feat_func, model_class, params, name)
            results.append(r)
            print(f'  {name}: AUC={r["auc"]:.3f}, 的中率={r["hit_rate"]*100:.1f}%, ROI={r["roi"]*100:.0f}%')
        except Exception as e:
            print(f'  {name}: ERROR - {e}')

    # 結果をソート（的中率重視）
    results.sort(key=lambda x: (x['hit_rate'], x['roi']), reverse=True)

    print(f'\n{"="*60}')
    print('ランキング（的中率重視）')
    print(f'{"="*60}')
    for i, r in enumerate(results[:5], 1):
        print(f'{i}. {r["name"]}: 的中率={r["hit_rate"]*100:.1f}%, ROI={r["roi"]*100:.0f}%, AUC={r["auc"]:.3f}')

    return results


def save_best_model(results, track_name):
    """ベストモデルを保存"""
    if not results:
        print('保存するモデルがありません')
        return

    best = results[0]

    # 特徴量関数の名前を保存
    func_name = best['feature_func'].__name__

    model_data = {
        'model': best['model'],
        'features': best['features'],
        'feature_func_name': func_name,
        'auc': best['auc'],
        'hit_rate': best['hit_rate'],
        'roi': best['roi'],
        'version': 'auto_optimized',
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 保存
    model_path = f'models/model_{track_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # メタデータ
    meta = {
        'track_name': track_name,
        'trained_at': model_data['trained_at'],
        'auc': round(best['auc'], 4),
        'hit_rate': round(best['hit_rate'], 4),
        'roi': round(best['roi'], 4),
        'version': 'auto_optimized',
        'best_approach': best['name'],
    }
    with open(f'models/model_{track_name}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n✓ ベストモデル保存: {model_path}')
    print(f'  アプローチ: {best["name"]}')
    print(f'  的中率: {best["hit_rate"]*100:.1f}%')
    print(f'  ROI: {best["roi"]*100:.0f}%')

    return best


if __name__ == '__main__':
    print('='*60)
    print('自動モデル最適化 - PDCA高速実行')
    print('='*60)

    all_results = {}

    for track in ['kawasaki', 'ohi']:
        results = run_experiments(track)
        best = save_best_model(results, track)
        all_results[track] = best

    print('\n' + '='*60)
    print('最終結果')
    print('='*60)
    for track, r in all_results.items():
        if r:
            print(f'{track.upper()}: {r["name"]} / 的中率={r["hit_rate"]*100:.1f}% / ROI={r["roi"]*100:.0f}%')
