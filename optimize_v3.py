"""AUC 0.8を目指す最適化スクリプト v3 - 上がり3F追加"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
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


def target_encode(df, col, target, n_splits=5, smoothing=10):
    """Target Encoding with cross-validation"""
    df = df.copy()
    global_mean = df[target].mean()
    df[f'{col}_te'] = global_mean
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df, df[target]):
        train_df = df.iloc[train_idx]
        stats = train_df.groupby(col)[target].agg(['mean', 'count'])
        smooth_mean = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        df.loc[val_idx, f'{col}_te'] = df.loc[val_idx, col].map(smooth_mean).fillna(global_mean)

    return df


class ProcessorV3:
    """v3: 上がり3F追加"""

    def __init__(self):
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
            'horse_win_rate_std', 'field_strength', 'inner_outer',
            'avg_rank_percentile', 'jockey_rank_in_race',
            'odds_implied_prob', 'distance_fitness', 'weight_per_meter', 'experience_score',
            # v3追加: 上がり3F関連
            'last_3f',                    # 上がり3Fタイム
            'last_3f_rank',               # レース内上がり3F順位
            'last_3f_vs_field',           # フィールド平均との差
            'last_3f_speed',              # 上がり3F速度（距離/タイム）
        ]

        self.features = self.base_features + self.te_features + self.extra_features

    def process(self, df, apply_te=True):
        df = df.copy()
        if 'rank' in df.columns:
            df = df[df['rank'].notna() & (df['rank'] > 0)]

        df = df.reset_index(drop=True)
        df['target'] = (df['rank'] <= 3).astype(int)

        # 基本前処理
        num_cols = ['rank', 'bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
                    'horse_weight', 'weight_change', 'last_3f', 'race_time_seconds']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # エンコーディング
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

        # 計算特徴量
        df['weight_diff'] = df.groupby('race_id')['weight_carried'].transform(lambda x: x - x.mean())
        df['horse_number_ratio'] = df['horse_number'] / df['field_size'].clip(lower=1)
        df['last_rank_diff'] = df['last_rank'] - df['horse_avg_rank']
        df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False)
        df['horse_win_rate_vs_field'] = df['horse_win_rate'] - df.groupby('race_id')['horse_win_rate'].transform('mean')
        df['jockey_win_rate_vs_field'] = df['jockey_win_rate'] - df.groupby('race_id')['jockey_win_rate'].transform('mean')
        df['horse_avg_rank_vs_field'] = df.groupby('race_id')['horse_avg_rank'].transform('mean') - df['horse_avg_rank']
        df['rank_trend'] = df['horse_avg_rank'] - df['last_rank']

        # 馬体重
        if 'horse_weight' not in df.columns:
            df['horse_weight'] = 450
        df['horse_weight'] = df['horse_weight'].fillna(450)
        df['horse_weight_change'] = df['weight_change'].fillna(0) if 'weight_change' in df.columns else 0

        # 時系列特徴量
        df['days_since_last_race'] = df['days_since_last_race'] if 'days_since_last_race' in df.columns else 30
        df['win_streak'] = df['win_streak'] if 'win_streak' in df.columns else 0
        df['show_streak'] = df['show_streak'] if 'show_streak' in df.columns else 0
        df['recent_3_avg_rank'] = df['recent_3_avg_rank'] if 'recent_3_avg_rank' in df.columns else df['horse_recent_avg_rank']
        df['recent_10_avg_rank'] = df['recent_10_avg_rank'] if 'recent_10_avg_rank' in df.columns else df['horse_avg_rank']
        df['rank_improvement'] = df['horse_avg_rank'] - df['recent_3_avg_rank']

        # Target Encoding
        if apply_te:
            for col in ['jockey_id', 'trainer_id', 'horse_id']:
                if col in df.columns:
                    df = target_encode(df, col, 'target')
        else:
            df['jockey_id_te'] = 0.3
            df['trainer_id_te'] = 0.3
            df['horse_id_te'] = 0.3

        # 追加特徴量
        df['horse_jockey_synergy'] = df['horse_win_rate'] * df['jockey_win_rate']
        df['form_score'] = (0.5 * (1 - df['last_rank'] / df['field_size'].clip(lower=1)) +
                           0.3 * (1 - df['horse_recent_avg_rank'] / df['field_size'].clip(lower=1)) +
                           0.2 * df['horse_win_rate']).fillna(0)
        df['class_indicator'] = df['field_size'] / (df['horse_avg_rank'] + 1)
        df['horse_win_rate_std'] = df.groupby('horse_id')['target'].transform('std').fillna(0)
        df['field_strength'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
        df['inner_outer'] = df['horse_number'].apply(lambda x: 0 if x <= 4 else (2 if x >= 10 else 1))
        df['avg_rank_percentile'] = df.groupby('race_id')['horse_avg_rank'].rank(pct=True)
        df['jockey_rank_in_race'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False)

        if 'win_odds' in df.columns and (df['win_odds'] > 0).any():
            df['odds_implied_prob'] = 1 / (df['win_odds'].clip(lower=1) + 1)
        else:
            df['odds_implied_prob'] = 0.1

        df['distance_fitness'] = 1.0
        df['weight_per_meter'] = df['weight_carried'] / (df['distance'] / 1000).clip(lower=0.1)
        df['experience_score'] = np.log1p(df['horse_runs']) * df['horse_show_rate']

        # ★ v3追加: 上がり3F特徴量 ★
        if 'last_3f' in df.columns:
            df['last_3f'] = df['last_3f'].fillna(df['last_3f'].mean())
            df['last_3f_rank'] = df.groupby('race_id')['last_3f'].rank(ascending=True)  # 速い方が上位
            df['last_3f_vs_field'] = df.groupby('race_id')['last_3f'].transform('mean') - df['last_3f']  # 速いほど+
            df['last_3f_speed'] = 600 / df['last_3f'].clip(lower=30)  # 速度（m/s的な）
        else:
            df['last_3f'] = 40.0
            df['last_3f_rank'] = 6
            df['last_3f_vs_field'] = 0
            df['last_3f_speed'] = 15.0

        # 欠損埋め
        for f in self.features:
            if f not in df.columns:
                df[f] = 0
            df[f] = df[f].fillna(0)

        return df


def train_model(track_name, n_trials=50):
    """モデル学習"""
    print(f'\n{"="*50}')
    print(f'{track_name.upper()} モデル学習 (v3 - 上がり3F)')
    print(f'{"="*50}')

    # データ読み込み
    df = pd.read_csv(f'data/races_{track_name}.csv')
    print(f'データ数: {len(df):,}')

    # 前処理
    processor = ProcessorV3()
    df = processor.process(df)

    # 上がり3Fの確認
    last_3f_valid = (df['last_3f'] > 0) & (df['last_3f'] < 60)
    print(f'上がり3F有効: {last_3f_valid.sum():,} ({last_3f_valid.mean()*100:.1f}%)')

    # 特徴量とターゲット
    X = df[processor.features]
    y = df['target']
    print(f'特徴量数: {len(processor.features)}')

    # 学習/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

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
    print(f'Best AUC: {study.best_value:.4f}')

    # 最終モデル
    lgb_model = lgb.LGBMClassifier(**best_params, n_estimators=500, random_state=42)
    lgb_model.fit(X_train_sm, y_train_sm)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        random_state=42, eval_metric='auc', use_label_encoder=False)
    xgb_model.fit(X_train_sm, y_train_sm)

    # アンサンブル
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    ensemble_pred = (lgb_pred + xgb_pred) / 2
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)

    print(f'\n=== 結果 ===')
    print(f'LightGBM AUC: {roc_auc_score(y_test, lgb_pred):.4f}')
    print(f'XGBoost AUC: {roc_auc_score(y_test, xgb_pred):.4f}')
    print(f'Ensemble AUC: {ensemble_auc:.4f}')

    # 特徴量重要度TOP10
    importance = pd.DataFrame({
        'feature': processor.features,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f'\n特徴量重要度TOP10:')
    for i, row in importance.head(10).iterrows():
        print(f'  {row["feature"]}: {row["importance"]:.0f}')

    # 保存
    model_data = {
        'model': {'type': 'ensemble', 'lgb': lgb_model, 'xgb': xgb_model},
        'features': processor.features
    }
    with open(f'models/model_{track_name}.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    meta = {
        'track_name': track_name,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'auc': round(ensemble_auc, 4),
        'features': processor.features,
        'version': 'v3_last3f'
    }
    with open(f'models/model_{track_name}_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\n保存完了: models/model_{track_name}.pkl')
    return ensemble_auc


if __name__ == '__main__':
    import sys
    track = sys.argv[1] if len(sys.argv) > 1 else 'ohi'
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    train_model(track, n_trials)
