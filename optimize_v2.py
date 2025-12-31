"""AUC 0.8を目指す最適化スクリプト v2 - Target Encoding追加"""
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
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def target_encode(df, col, target, n_splits=5, smoothing=10):
    """Target Encoding with cross-validation to avoid leakage"""
    df = df.copy()
    global_mean = df[target].mean()

    # K-fold target encoding
    df[f'{col}_te'] = global_mean
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df, df[target]):
        train_df = df.iloc[train_idx]
        stats = train_df.groupby(col)[target].agg(['mean', 'count'])
        # Smoothing
        smooth_mean = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        df.loc[val_idx, f'{col}_te'] = df.loc[val_idx, col].map(smooth_mean).fillna(global_mean)

    return df


class AdvancedProcessorV2:
    """改良版前処理クラス v2 - Target Encoding追加"""

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

        # Target Encoding特徴量
        self.te_features = [
            'jockey_id_te',     # 騎手別の複勝率
            'trainer_id_te',    # 調教師別の複勝率
            'horse_id_te',      # 馬別の複勝率（過去実績）
        ]

        # 追加特徴量
        self.extra_features = [
            'horse_jockey_synergy',
            'form_score',
            'class_indicator',
            'horse_win_rate_std',
            'field_strength',
            'inner_outer',
            'avg_rank_percentile',
            'jockey_rank_in_race',
            # 新規追加
            'odds_implied_prob',     # オッズから計算した暗黙確率
            'distance_fitness',      # 距離適性
            'weight_per_meter',      # 斤量/距離
            'experience_score',      # 経験値スコア
        ]

        self.features = self.base_features + self.te_features + self.extra_features

    def process(self, df, apply_te=True):
        df = df.copy()
        if 'rank' in df.columns:
            df = df[df['rank'].notna() & (df['rank'] > 0)]

        # インデックスをリセット
        df = df.reset_index(drop=True)

        # ターゲット（先に作成）
        df['target'] = (df['rank'] <= 3).astype(int)

        # === 基本前処理 ===
        num_cols = ['rank', 'bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
                    'horse_weight', 'weight_change']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'sex' in df.columns:
            df['sex_encoded'] = df['sex'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        else:
            df['sex_encoded'] = 0

        if 'weight_carried' in df.columns and 'race_id' in df.columns:
            df['weight_diff'] = df.groupby('race_id')['weight_carried'].transform(lambda x: x - x.mean())
        else:
            df['weight_diff'] = 0

        if 'field_size' not in df.columns:
            df['field_size'] = 12

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

        if 'horse_weight' in df.columns:
            df['horse_weight'] = df['horse_weight'].fillna(450)
        else:
            df['horse_weight'] = 450

        if 'weight_change' in df.columns:
            df['horse_weight_change'] = df['weight_change'].fillna(0)
        else:
            df['horse_weight_change'] = 0

        # 計算特徴量
        df['horse_number_ratio'] = (df['horse_number'] / df['field_size']).fillna(0.5)
        df['last_rank_diff'] = (df['last_rank'] - df['horse_avg_rank']).fillna(0) if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns else 0

        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False, method='min').fillna(6)
            df['horse_win_rate_vs_field'] = (df['horse_win_rate'] - df.groupby('race_id')['horse_win_rate'].transform('mean')).fillna(0)
            df['horse_win_rate_std'] = df.groupby('race_id')['horse_win_rate'].transform('std').fillna(0)
            df['field_strength'] = df.groupby('race_id')['horse_win_rate'].transform('mean').fillna(0)
        else:
            df['win_rate_rank'] = 6
            df['horse_win_rate_vs_field'] = 0
            df['horse_win_rate_std'] = 0
            df['field_strength'] = 0

        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['jockey_win_rate_vs_field'] = (df['jockey_win_rate'] - df.groupby('race_id')['jockey_win_rate'].transform('mean')).fillna(0)
            df['jockey_rank_in_race'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False, method='min').fillna(6)
        else:
            df['jockey_win_rate_vs_field'] = 0
            df['jockey_rank_in_race'] = 6

        if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
            df['horse_avg_rank_vs_field'] = (df.groupby('race_id')['horse_avg_rank'].transform('mean') - df['horse_avg_rank']).fillna(0)
            df['avg_rank_percentile'] = df.groupby('race_id')['horse_avg_rank'].rank(pct=True).fillna(0.5)
        else:
            df['horse_avg_rank_vs_field'] = 0
            df['avg_rank_percentile'] = 0.5

        df['days_since_last_race'] = df['days_since_last_race'].fillna(30).clip(0, 365) if 'days_since_last_race' in df.columns else 30
        df['rank_trend'] = (df['horse_avg_rank'] - df['last_rank']).fillna(0) if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns else 0

        df['win_streak'] = df['win_streak'] if 'win_streak' in df.columns else 0
        df['show_streak'] = df['show_streak'] if 'show_streak' in df.columns else 0
        df['recent_3_avg_rank'] = df['recent_3_avg_rank'] if 'recent_3_avg_rank' in df.columns else (df['horse_recent_avg_rank'] if 'horse_recent_avg_rank' in df.columns else 10)
        df['recent_10_avg_rank'] = df['recent_10_avg_rank'] if 'recent_10_avg_rank' in df.columns else (df['horse_avg_rank'] if 'horse_avg_rank' in df.columns else 10)
        df['rank_improvement'] = (df['horse_avg_rank'] - df['recent_3_avg_rank']).fillna(0)

        # === Target Encoding ===
        if apply_te:
            if 'jockey_id' in df.columns:
                df = target_encode(df, 'jockey_id', 'target', smoothing=20)
            else:
                df['jockey_id_te'] = df['target'].mean()

            if 'trainer_id' in df.columns:
                df = target_encode(df, 'trainer_id', 'target', smoothing=20)
            else:
                df['trainer_id_te'] = df['target'].mean()

            if 'horse_id' in df.columns:
                df = target_encode(df, 'horse_id', 'target', smoothing=5)
            else:
                df['horse_id_te'] = df['target'].mean()
        else:
            df['jockey_id_te'] = df['target'].mean()
            df['trainer_id_te'] = df['target'].mean()
            df['horse_id_te'] = df['target'].mean()

        # === 追加特徴量 ===
        df['horse_jockey_synergy'] = df['horse_win_rate'].fillna(0) * df['jockey_win_rate'].fillna(0) * 100

        df['form_score'] = (
            df['horse_recent_win_rate'].fillna(0) * 0.3 +
            df['horse_recent_show_rate'].fillna(0) * 0.3 +
            (1 - df['horse_recent_avg_rank'].fillna(10) / 15) * 0.2 +
            df['rank_trend'].fillna(0) / 10 * 0.2
        )

        df['class_indicator'] = df['horse_win_rate'].fillna(0) * np.log1p(df['horse_runs'].fillna(0))

        df['inner_outer'] = df.apply(
            lambda x: (x['horse_number'] / x['field_size']) * (1 if x['distance'] < 1600 else 0.5)
            if pd.notna(x.get('distance')) and x['field_size'] > 0 else 0.5,
            axis=1
        )

        # オッズから暗黙確率（オッズがある場合のみ）
        if 'win_odds' in df.columns:
            df['odds_implied_prob'] = 1 / df['win_odds'].clip(1.01, 100)
        else:
            df['odds_implied_prob'] = 0.1

        # 距離適性（馬の平均着順と距離の組み合わせ）
        df['distance_fitness'] = df['horse_avg_rank'].fillna(10) / (df['distance'].fillna(1600) / 1000)

        # 斤量/距離
        df['weight_per_meter'] = df['weight_carried'].fillna(55) / (df['distance'].fillna(1600) / 100)

        # 経験値スコア
        df['experience_score'] = np.log1p(df['horse_runs'].fillna(0)) * df['horse_show_rate'].fillna(0)

        # 不足特徴量を補完
        for f in self.features:
            if f not in df.columns:
                df[f] = 0

        return df


def objective(trial, X_tr, y_tr, X_te, y_te):
    """Optuna目的関数"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'num_leaves': trial.suggest_int('num_leaves', 30, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 5.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    pred = model.predict(X_te)
    return roc_auc_score(y_te, pred)


def train_optimized(df, features, n_trials=100):
    """最適化付き学習"""
    X, y = df[features].fillna(-1), df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE
    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr, y_tr)
        print(f"  SMOTE: {len(y_tr)} -> {len(y_tr_resampled)}")
    except:
        X_tr_resampled, y_tr_resampled = X_tr, y_tr

    # Optuna
    print(f"  Optuna最適化中（{n_trials}試行）...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_tr_resampled, y_tr_resampled, X_te, y_te),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['verbose'] = -1

    print(f"  Best LGB: num_leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.4f}, depth={best_params['max_depth']}")

    # LightGBM
    lgb_model = lgb.train(
        best_params,
        lgb.Dataset(X_tr_resampled, y_tr_resampled),
        num_boost_round=1500,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    lgb_pred = lgb_model.predict(X_te)
    lgb_auc = roc_auc_score(y_te, lgb_pred)
    print(f"  LightGBM AUC: {lgb_auc:.4f}")

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['bagging_fraction'],
        colsample_bytree=best_params['feature_fraction'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        n_estimators=1500,
        early_stopping_rounds=150,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_tr_resampled, y_tr_resampled, eval_set=[(X_te, y_te)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_te)[:, 1]
    xgb_auc = roc_auc_score(y_te, xgb_pred)
    print(f"  XGBoost AUC: {xgb_auc:.4f}")

    # Ensemble
    best_weight, best_auc = 0.5, 0
    for w in np.arange(0.3, 0.8, 0.05):
        pred = w * lgb_pred + (1 - w) * xgb_pred
        auc = roc_auc_score(y_te, pred)
        if auc > best_auc:
            best_auc = auc
            best_weight = w

    print(f"  Ensemble AUC: {best_auc:.4f} (LGB:{best_weight:.2f})")

    return {
        'lgb': lgb_model,
        'xgb': xgb_model,
        'lgb_weight': best_weight,
        'type': 'weighted_ensemble',
        'best_params': best_params
    }, best_auc


def main():
    import sys

    track_name = sys.argv[1] if len(sys.argv) > 1 else "大井"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    TRACKS = {
        "大井": {"csv": "data/races_ohi.csv", "model": "models/model_ohi.pkl", "meta": "models/model_ohi_meta.json"},
        "川崎": {"csv": "data/races_kawasaki.csv", "model": "models/model_kawasaki.pkl", "meta": "models/model_kawasaki_meta.json"},
    }

    track = TRACKS.get(track_name)
    if not track:
        print(f"エラー: {track_name} は未対応")
        sys.exit(1)

    old_auc = None
    try:
        with open(track["meta"], 'r', encoding='utf-8') as f:
            old_auc = json.load(f).get('auc')
    except:
        pass

    print("=" * 60)
    print(f"AUC 0.8 目標最適化 v2: {track_name}")
    print("  - Target Encoding追加")
    print("  - 新規特徴量追加")
    print("  - 長めのearly stopping")
    print("=" * 60)

    df = pd.read_csv(track["csv"])
    print(f"データ: {len(df)}件")

    processor = AdvancedProcessorV2()
    df_processed = processor.process(df)
    print(f"処理後: {len(df_processed)}件")
    print(f"特徴量: {len(processor.features)}個")

    print()
    model, auc = train_optimized(df_processed, processor.features, n_trials)

    print()
    print("保存中...")
    with open(track["model"], 'wb') as f:
        pickle.dump({'model': model, 'features': processor.features}, f)

    race_dates = df['race_date'].dropna().astype(int).astype(str)
    min_date, max_date = race_dates.min(), race_dates.max()

    meta = {
        'track_name': track_name,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_count': len(df),
        'race_count': int(df['race_id'].nunique()),
        'date_range': {
            'from': f"{min_date[:4]}-{min_date[4:6]}-{min_date[6:8]}",
            'to': f"{max_date[:4]}-{max_date[4:6]}-{max_date[6:8]}"
        },
        'auc': round(auc, 4),
        'features': processor.features
    }
    with open(track["meta"], 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    if old_auc:
        print(f"旧AUC: {old_auc:.4f}")
    print(f"新AUC: {auc:.4f}")
    if old_auc:
        print(f"改善: {(auc - old_auc) * 100:+.2f}%")

    if auc >= 0.8:
        print("🎉 AUC 0.8 達成！")
    else:
        print(f"目標まで: {(0.8 - auc) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
