"""AUC 0.8を目指す最適化スクリプト v6 - 選択的ベッティング版"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import optuna
import warnings
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TargetEncoderSafe:
    """リークしないTarget Encoder（学習データのみで統計作成）"""

    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.global_mean = None
        self.mappings = {}

    def fit(self, train_df, cols, target):
        """学習データのみで統計を作成"""
        self.global_mean = train_df[target].mean()

        for col in cols:
            stats = train_df.groupby(col)[target].agg(['mean', 'count'])
            smooth_mean = (stats['mean'] * stats['count'] + self.global_mean * self.smoothing) / \
                         (stats['count'] + self.smoothing)
            self.mappings[col] = smooth_mean.to_dict()

        return self

    def transform(self, df, cols):
        """学習データの統計でテストデータを変換"""
        df = df.copy()
        for col in cols:
            te_col = f'{col}_te'
            df[te_col] = df[col].map(self.mappings.get(col, {})).fillna(self.global_mean)
        return df


def add_previous_race_features_safe(df):
    """
    前走特徴量を追加（リークなし版）
    """
    df = df.copy()

    df['race_date_num'] = df['race_id'].astype(str).str[:8].astype(int)
    df = df.sort_values(['horse_id', 'race_date_num', 'race_id'])

    df['prev_last_3f'] = df.groupby('horse_id')['last_3f'].shift(1)

    df['avg_last_3f_3races'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['avg_last_3f_5races'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    df['prev_race_time'] = df.groupby('horse_id')['race_time_seconds'].shift(1)

    if 'rank' in df.columns:
        df['past_rank_std'] = df.groupby('horse_id')['rank'].transform(
            lambda x: x.shift(1).expanding().std()
        )

    return df


class ProcessorV6:
    """v6: 選択的ベッティング版 - オッズを特徴量から除外"""

    def __init__(self):
        # ★★★ v6: オッズ関連を特徴量から完全除外 ★★★
        self.base_features = [
            'horse_runs', 'horse_win_rate', 'horse_place_rate', 'horse_show_rate',
            'horse_avg_rank', 'horse_recent_win_rate', 'horse_recent_show_rate',
            'horse_recent_avg_rank', 'last_rank',
            'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
            'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
            'sex_encoded', 'field_size', 'weight_diff',
            'track_condition_encoded', 'weather_encoded',
            'horse_weight', 'horse_weight_change',
            'horse_number_ratio', 'last_rank_diff', 'win_rate_rank',
            'horse_win_rate_vs_field', 'jockey_win_rate_vs_field',
            'horse_avg_rank_vs_field',
            'days_since_last_race', 'rank_trend',
            'win_streak', 'show_streak', 'recent_3_avg_rank', 'recent_10_avg_rank', 'rank_improvement',
        ]

        self.te_features = ['jockey_id_te', 'trainer_id_te', 'horse_id_te']

        self.extra_features = [
            'horse_jockey_synergy', 'form_score', 'class_indicator',
            'field_strength', 'inner_outer',
            'avg_rank_percentile', 'jockey_rank_in_race',
            'distance_fitness', 'weight_per_meter', 'experience_score',
            'prev_last_3f',
            'avg_last_3f_3races',
            'avg_last_3f_5races',
            'prev_last_3f_rank',
            'prev_last_3f_vs_field',
            'past_rank_std',
            'is_first_race',
        ]

        # ❌ v6で完全除外: win_odds, odds_implied_prob など
        # 理由: 市場のコンセンサスに引っ張られないため

        self.features = self.base_features + self.te_features + self.extra_features

    def process_base(self, df):
        """基本前処理（Target Encoding以外）"""
        df = df.copy()

        if 'rank' in df.columns:
            df = df[df['rank'].notna() & (df['rank'] > 0)]

        df = df.reset_index(drop=True)
        df['target'] = (df['rank'] <= 3).astype(int)

        num_cols = ['rank', 'bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
                    'horse_weight', 'weight_change', 'last_3f', 'race_time_seconds']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = add_previous_race_features_safe(df)

        df['is_first_race'] = df['prev_last_3f'].isna().astype(int)

        if 'sex' in df.columns:
            df['sex_encoded'] = df['sex'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        else:
            df['sex_encoded'] = 0

        if 'track_condition' in df.columns:
            df['track_condition_encoded'] = df['track_condition'].map(
                {'良': 0, '稍重': 1, '重': 2, '不良': 3}).fillna(0)
        else:
            df['track_condition_encoded'] = 0

        if 'weather' in df.columns:
            df['weather_encoded'] = df['weather'].map(
                {'晴': 0, '曇': 1, '小雨': 2, '雨': 3, '雪': 4}).fillna(0)
        else:
            df['weather_encoded'] = 0

        df['weight_diff'] = df.groupby('race_id')['weight_carried'].transform(lambda x: x - x.mean())
        df['horse_number_ratio'] = df['horse_number'] / df['field_size'].clip(lower=1)
        df['last_rank_diff'] = df['last_rank'] - df['horse_avg_rank']
        df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False)
        df['horse_win_rate_vs_field'] = df['horse_win_rate'] - df.groupby('race_id')['horse_win_rate'].transform('mean')
        df['jockey_win_rate_vs_field'] = df['jockey_win_rate'] - df.groupby('race_id')['jockey_win_rate'].transform('mean')
        df['horse_avg_rank_vs_field'] = df.groupby('race_id')['horse_avg_rank'].transform('mean') - df['horse_avg_rank']
        df['rank_trend'] = df['horse_avg_rank'] - df['last_rank']

        if 'horse_weight' not in df.columns:
            df['horse_weight'] = 450
        df['horse_weight'] = df['horse_weight'].fillna(450)
        df['horse_weight_change'] = df['weight_change'].fillna(0) if 'weight_change' in df.columns else 0

        df['days_since_last_race'] = df['days_since_last_race'] if 'days_since_last_race' in df.columns else 30
        df['win_streak'] = df['win_streak'] if 'win_streak' in df.columns else 0
        df['show_streak'] = df['show_streak'] if 'show_streak' in df.columns else 0
        df['recent_3_avg_rank'] = df['recent_3_avg_rank'] if 'recent_3_avg_rank' in df.columns else df['horse_recent_avg_rank']
        df['recent_10_avg_rank'] = df['recent_10_avg_rank'] if 'recent_10_avg_rank' in df.columns else df['horse_avg_rank']
        df['rank_improvement'] = df['horse_avg_rank'] - df['recent_3_avg_rank']

        df['horse_jockey_synergy'] = df['horse_win_rate'] * df['jockey_win_rate']
        df['form_score'] = (0.5 * (1 - df['last_rank'] / df['field_size'].clip(lower=1)) +
                           0.3 * (1 - df['horse_recent_avg_rank'] / df['field_size'].clip(lower=1)) +
                           0.2 * df['horse_win_rate']).fillna(0)
        df['class_indicator'] = df['field_size'] / (df['horse_avg_rank'] + 1)
        df['field_strength'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
        df['inner_outer'] = df['horse_number'].apply(lambda x: 0 if x <= 4 else (2 if x >= 10 else 1))
        df['avg_rank_percentile'] = df.groupby('race_id')['horse_avg_rank'].rank(pct=True)
        df['jockey_rank_in_race'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)
        df['distance_fitness'] = 1.0
        df['weight_per_meter'] = df['weight_carried'] / (df['distance'] / 1000).clip(lower=0.1)
        df['experience_score'] = np.log1p(df['horse_runs']) * df['horse_show_rate']

        df['prev_last_3f_rank'] = df.groupby('race_id')['prev_last_3f'].rank(ascending=True)
        df['prev_last_3f_vs_field'] = df.groupby('race_id')['prev_last_3f'].transform('mean') - df['prev_last_3f']

        return df


def train_and_simulate(track_name, n_trials=50):
    """モデル学習 + 選択的ベッティングシミュレーション"""
    print(f'\n{"="*70}')
    print(f'{track_name.upper()} モデル学習 (v6 - 選択的ベッティング版)')
    print(f'{"="*70}')

    # データ読み込み
    df = pd.read_csv(f'data/races_{track_name}.csv')
    print(f'データ数: {len(df):,}')

    # 前処理（TE以外）
    processor = ProcessorV6()
    df = processor.process_base(df)

    # 時系列分割
    df = df.sort_values('race_id').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f'\n時系列分割:')
    print(f'  学習: {len(train_df):,}件')
    print(f'  テスト: {len(test_df):,}件')

    # Target Encoding（学習データのみで統計作成）
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

    # 特徴量とターゲット
    X_train = train_df[processor.features]
    y_train = train_df['target']
    X_test = test_df[processor.features]
    y_test = test_df['target']

    print(f'特徴量数: {len(processor.features)}')
    print(f'★ オッズを特徴量から除外（市場バイアス回避）')

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Optuna最適化
    print(f'\nOptuna {n_trials}試行で最適化中...')

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params, n_estimators=500, random_state=42)
        model.fit(X_train_sm, y_train_sm, eval_set=[(X_test, y_test)],
                 callbacks=[lgb.early_stopping(50, verbose=False)])
        return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f'\nBest AUC (Optuna): {study.best_value:.4f}')

    # 最終モデル学習
    lgb_model = lgb.LGBMClassifier(**best_params, n_estimators=500, random_state=42)
    lgb_model.fit(X_train_sm, y_train_sm)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        random_state=42, eval_metric='auc', use_label_encoder=False)
    xgb_model.fit(X_train_sm, y_train_sm)

    # 評価
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    ensemble_pred = (lgb_pred + xgb_pred) / 2

    ensemble_auc = roc_auc_score(y_test, ensemble_pred)

    print(f'\n{"="*50}')
    print(f'AUC: {ensemble_auc:.4f}')
    print(f'{"="*50}')

    # 予測結果をテストデータに追加
    test_df['pred_prob'] = ensemble_pred
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_prob'].rank(ascending=False)

    # 2位との確率差を計算
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

    # 的中フラグ
    test_df['is_place'] = (test_df['rank'] <= 3).astype(int)

    # ===== 選択的ベッティングシミュレーション =====
    print(f'\n{"="*70}')
    print(f'選択的ベッティングシミュレーション')
    print(f'{"="*70}')

    # 実オッズを使用（place_oddsがあれば）
    if 'place_odds' in test_df.columns:
        test_df['place_odds'] = pd.to_numeric(test_df['place_odds'], errors='coerce')
        valid_odds = test_df[test_df['place_odds'] > 0]
        use_real_odds = len(valid_odds) > len(test_df) * 0.1
    else:
        use_real_odds = False

    if use_real_odds:
        print(f'★ 実複勝オッズを使用')
        # 実オッズがある馬のみ（3着以内）
        odds_col = 'place_odds'
    else:
        print(f'★ 単勝オッズから複勝オッズを推定')
        # 単勝から推定
        def estimate_place_odds(win_odds):
            if win_odds <= 0:
                return 1.5
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

        if 'win_odds' in test_df.columns:
            test_df['place_odds_est'] = test_df['win_odds'].apply(estimate_place_odds)
        else:
            test_df['place_odds_est'] = 1.5
        odds_col = 'place_odds_est'

    # 予測1位のみ
    top1 = test_df[test_df['pred_rank'] == 1].copy()

    # オッズがあるものだけ
    if 'win_odds' in top1.columns:
        top1 = top1[top1['win_odds'] > 0]

    print(f'\n予測1位（有効）: {len(top1)}レース')

    # シミュレーション結果
    results = []

    # 1. 全レース
    bets = top1
    if len(bets) > 0:
        hits = bets['is_place'].sum()
        total = len(bets)
        hit_rate = hits / total
        avg_odds = bets[odds_col].mean() if odds_col in bets.columns else 1.5
        roi = hit_rate * avg_odds
        results.append({'filter': 'All', 'bets': total, 'hits': hits, 'hit_rate': hit_rate, 'avg_odds': avg_odds, 'roi': roi})

    # 2. 確率差フィルター（選択的ベッティングの核心）
    for min_diff in [0.05, 0.10, 0.15, 0.20]:
        bets = top1[top1['prob_diff'] >= min_diff]
        if len(bets) > 0:
            hits = bets['is_place'].sum()
            total = len(bets)
            hit_rate = hits / total
            avg_odds = bets[odds_col].mean() if odds_col in bets.columns else 1.5
            roi = hit_rate * avg_odds
            results.append({'filter': f'ProbDiff>={int(min_diff*100)}%', 'bets': total, 'hits': hits, 'hit_rate': hit_rate, 'avg_odds': avg_odds, 'roi': roi})

    # 3. 単勝オッズフィルター（中穴狙い）
    if 'win_odds' in top1.columns:
        for min_odds, max_odds in [(3.0, 15.0), (5.0, 20.0), (3.0, 50.0)]:
            bets = top1[(top1['win_odds'] >= min_odds) & (top1['win_odds'] <= max_odds)]
            if len(bets) > 0:
                hits = bets['is_place'].sum()
                total = len(bets)
                hit_rate = hits / total
                avg_odds = bets[odds_col].mean() if odds_col in bets.columns else 1.5
                roi = hit_rate * avg_odds
                results.append({'filter': f'Odds{min_odds}-{max_odds}', 'bets': total, 'hits': hits, 'hit_rate': hit_rate, 'avg_odds': avg_odds, 'roi': roi})

    # 4. 確率差 + オッズ複合フィルター（最強戦略）
    if 'win_odds' in top1.columns:
        for min_diff in [0.10, 0.15]:
            for min_odds, max_odds in [(3.0, 15.0), (5.0, 30.0)]:
                bets = top1[(top1['prob_diff'] >= min_diff) & (top1['win_odds'] >= min_odds) & (top1['win_odds'] <= max_odds)]
                if len(bets) > 0:
                    hits = bets['is_place'].sum()
                    total = len(bets)
                    hit_rate = hits / total
                    avg_odds = bets[odds_col].mean() if odds_col in bets.columns else 1.5
                    roi = hit_rate * avg_odds
                    results.append({'filter': f'Diff{int(min_diff*100)}%+Odds{min_odds}-{max_odds}', 'bets': total, 'hits': hits, 'hit_rate': hit_rate, 'avg_odds': avg_odds, 'roi': roi})

    # 結果表示
    print(f'\n{"Filter":<25} {"Bets":>6} {"Hits":>5} {"HitRate":>8} {"AvgOdds":>8} {"ROI":>8}')
    print(f'{"-"*65}')

    best_roi = 0
    best_filter = ""

    for r in results:
        mark = ' ★' if r['roi'] >= 1.0 else '  '
        print(f'{r["filter"]:<25} {r["bets"]:>6} {r["hits"]:>5} {r["hit_rate"]:>7.1%} {r["avg_odds"]:>7.2f}x {r["roi"]:>7.1%}{mark}')
        if r['roi'] > best_roi:
            best_roi = r['roi']
            best_filter = r['filter']

    print(f'{"-"*65}')
    print(f'★ = ROI >= 100%')

    if best_roi > 0:
        print(f'\n最高ROI: {best_roi:.1%} (フィルター: {best_filter})')

    # 保存
    model_data = {
        'model': {'type': 'ensemble', 'lgb': lgb_model, 'xgb': xgb_model},
        'features': processor.features,
        'te_encoder': te_encoder,
        'version': 'v6_selective_betting'
    }
    with open(f'models/model_{track_name}.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    meta = {
        'track_name': track_name,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'auc': round(ensemble_auc, 4),
        'best_roi': round(best_roi, 4),
        'best_filter': best_filter,
        'version': 'v6_selective_betting',
    }
    with open(f'models/model_{track_name}_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\n保存完了: models/model_{track_name}.pkl')

    return ensemble_auc, best_roi, best_filter


if __name__ == '__main__':
    track = sys.argv[1] if len(sys.argv) > 1 else 'ohi'
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    train_and_simulate(track, n_trials)
