"""AUC 0.8を目指す最適化スクリプト"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class AdvancedProcessor:
    """改良版前処理クラス - 追加特徴量あり"""

    def __init__(self):
        self.base_features = [
            'horse_runs', 'horse_win_rate', 'horse_place_rate', 'horse_show_rate',
            'horse_avg_rank', 'horse_recent_win_rate', 'horse_recent_show_rate',
            'horse_recent_avg_rank', 'last_rank',
            'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
            'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
            'sex_encoded', 'track_encoded', 'field_size', 'weight_diff',
            'track_condition_encoded', 'weather_encoded',
            'trainer_encoded', 'horse_weight', 'horse_weight_change',
            'horse_number_ratio', 'last_rank_diff', 'win_rate_rank',
            'horse_win_rate_vs_field', 'jockey_win_rate_vs_field',
            'horse_avg_rank_vs_field',
            'days_since_last_race', 'rank_trend',
            'jockey_track_interaction', 'trainer_distance_interaction', 'jockey_distance_interaction',
            'win_streak', 'show_streak', 'recent_3_avg_rank', 'recent_10_avg_rank', 'rank_improvement',
            'father_win_rate', 'father_show_rate', 'bms_win_rate', 'bms_show_rate',
        ]

        # 追加特徴量
        self.extra_features = [
            # 複合特徴量
            'horse_jockey_synergy',      # 馬勝率 × 騎手勝率
            'form_score',                 # 総合調子スコア
            'class_indicator',            # クラス指標
            # 統計特徴量
            'horse_win_rate_std',         # レース内勝率の標準偏差
            'field_strength',             # 出走メンバーの強さ
            # 位置特徴量
            'inner_outer',                # 内枠/外枠（距離別）
            # ランク特徴量
            'avg_rank_percentile',        # 平均着順のパーセンタイル
            'jockey_rank_in_race',        # レース内騎手ランク
        ]

        self.features = self.base_features + self.extra_features

    def process(self, df):
        df = df.copy()
        if 'rank' in df.columns:
            df = df[df['rank'].notna() & (df['rank'] > 0)]

        # === 基本前処理（既存と同じ） ===
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

        df['track_encoded'] = 0

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

        if 'trainer_id' in df.columns:
            df['trainer_encoded'] = df['trainer_id'].apply(
                lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0
            )
        else:
            df['trainer_encoded'] = 0

        if 'horse_weight' in df.columns:
            df['horse_weight'] = df['horse_weight'].fillna(450)
        else:
            df['horse_weight'] = 450

        if 'weight_change' in df.columns:
            df['horse_weight_change'] = df['weight_change'].fillna(0)
        else:
            df['horse_weight_change'] = 0

        # 計算特徴量
        if 'horse_number' in df.columns and 'field_size' in df.columns:
            df['horse_number_ratio'] = (df['horse_number'] / df['field_size']).fillna(0.5)
        else:
            df['horse_number_ratio'] = 0.5

        if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['last_rank_diff'] = (df['last_rank'] - df['horse_avg_rank']).fillna(0)
        else:
            df['last_rank_diff'] = 0

        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False, method='min').fillna(6)
        else:
            df['win_rate_rank'] = 6

        # 相対特徴量
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['horse_win_rate_vs_field'] = (df['horse_win_rate'] - df.groupby('race_id')['horse_win_rate'].transform('mean')).fillna(0)
        else:
            df['horse_win_rate_vs_field'] = 0

        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['jockey_win_rate_vs_field'] = (df['jockey_win_rate'] - df.groupby('race_id')['jockey_win_rate'].transform('mean')).fillna(0)
        else:
            df['jockey_win_rate_vs_field'] = 0

        if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
            df['horse_avg_rank_vs_field'] = (df.groupby('race_id')['horse_avg_rank'].transform('mean') - df['horse_avg_rank']).fillna(0)
        else:
            df['horse_avg_rank_vs_field'] = 0

        df['days_since_last_race'] = df['days_since_last_race'].fillna(30).clip(0, 365) if 'days_since_last_race' in df.columns else 30

        if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['rank_trend'] = (df['horse_avg_rank'] - df['last_rank']).fillna(0)
        else:
            df['rank_trend'] = 0

        # 交互作用
        if 'jockey_id' in df.columns and 'race_id' in df.columns:
            df['track_code'] = df['race_id'].astype(str).str[4:6]
            df['jockey_track_interaction'] = df.apply(
                lambda x: hash(str(x.get('jockey_id', '')) + str(x.get('track_code', ''))) % 10000, axis=1
            )
        else:
            df['jockey_track_interaction'] = 0

        if 'trainer_id' in df.columns and 'distance' in df.columns:
            df['distance_cat'] = df['distance'].apply(
                lambda d: 'short' if pd.notna(d) and d < 1400 else ('long' if pd.notna(d) and d >= 1800 else 'mid')
            )
            df['trainer_distance_interaction'] = df.apply(
                lambda x: hash(str(x.get('trainer_id', '')) + str(x.get('distance_cat', ''))) % 10000, axis=1
            )
        else:
            df['trainer_distance_interaction'] = 0

        if 'jockey_id' in df.columns and 'distance' in df.columns:
            if 'distance_cat' not in df.columns:
                df['distance_cat'] = df['distance'].apply(
                    lambda d: 'short' if pd.notna(d) and d < 1400 else ('long' if pd.notna(d) and d >= 1800 else 'mid')
                )
            df['jockey_distance_interaction'] = df.apply(
                lambda x: hash(str(x.get('jockey_id', '')) + str(x.get('distance_cat', ''))) % 10000, axis=1
            )
        else:
            df['jockey_distance_interaction'] = 0

        # 時系列
        df['win_streak'] = df['win_streak'] if 'win_streak' in df.columns else 0
        df['show_streak'] = df['show_streak'] if 'show_streak' in df.columns else 0

        if 'recent_3_avg_rank' not in df.columns:
            df['recent_3_avg_rank'] = df['horse_recent_avg_rank'] if 'horse_recent_avg_rank' in df.columns else 10
        if 'recent_10_avg_rank' not in df.columns:
            df['recent_10_avg_rank'] = df['horse_avg_rank'] if 'horse_avg_rank' in df.columns else 10

        if 'recent_3_avg_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['rank_improvement'] = (df['horse_avg_rank'] - df['recent_3_avg_rank']).fillna(0)
        else:
            df['rank_improvement'] = 0

        # 血統
        for col in ['father_win_rate', 'father_show_rate', 'bms_win_rate', 'bms_show_rate']:
            if col not in df.columns:
                df[col] = 0

        # === 追加特徴量 ===

        # 馬×騎手シナジー
        df['horse_jockey_synergy'] = (
            df['horse_win_rate'].fillna(0) * df['jockey_win_rate'].fillna(0) * 100
        )

        # 総合調子スコア
        df['form_score'] = (
            df['horse_recent_win_rate'].fillna(0) * 0.3 +
            df['horse_recent_show_rate'].fillna(0) * 0.3 +
            (1 - df['horse_recent_avg_rank'].fillna(10) / 15) * 0.2 +
            df['rank_trend'].fillna(0) / 10 * 0.2
        )

        # クラス指標（勝率×出走数で経験値）
        df['class_indicator'] = (
            df['horse_win_rate'].fillna(0) * np.log1p(df['horse_runs'].fillna(0))
        )

        # レース内勝率の標準偏差
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['horse_win_rate_std'] = df.groupby('race_id')['horse_win_rate'].transform('std').fillna(0)
        else:
            df['horse_win_rate_std'] = 0

        # 出走メンバーの強さ（平均勝率）
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['field_strength'] = df.groupby('race_id')['horse_win_rate'].transform('mean').fillna(0)
        else:
            df['field_strength'] = 0

        # 内枠/外枠（距離別で有利不利が変わる）
        if 'horse_number' in df.columns and 'field_size' in df.columns and 'distance' in df.columns:
            # 短距離は内枠有利、長距離は外枠も可
            df['inner_outer'] = df.apply(
                lambda x: (x['horse_number'] / x['field_size']) * (1 if x['distance'] < 1600 else 0.5)
                if pd.notna(x['distance']) and x['field_size'] > 0 else 0.5,
                axis=1
            )
        else:
            df['inner_outer'] = 0.5

        # 平均着順のパーセンタイル
        if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
            df['avg_rank_percentile'] = df.groupby('race_id')['horse_avg_rank'].rank(pct=True).fillna(0.5)
        else:
            df['avg_rank_percentile'] = 0.5

        # レース内騎手ランク
        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['jockey_rank_in_race'] = df.groupby('race_id')['jockey_win_rate'].rank(ascending=False, method='min').fillna(6)
        else:
            df['jockey_rank_in_race'] = 6

        # ターゲット
        df['target'] = (df['rank'] <= 3).astype(int)

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
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
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


def train_optimized(df, features, n_trials=50):
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

    # Optuna最適化
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

    print(f"  Best params: num_leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.4f}")

    # LightGBM with best params
    lgb_model = lgb.train(
        best_params,
        lgb.Dataset(X_tr_resampled, y_tr_resampled),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    lgb_pred = lgb_model.predict(X_te)
    lgb_auc = roc_auc_score(y_te, lgb_pred)
    print(f"  LightGBM AUC: {lgb_auc:.4f}")

    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': best_params.get('max_depth', 6),
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params['bagging_fraction'],
        'colsample_bytree': best_params['feature_fraction'],
        'reg_alpha': best_params['reg_alpha'],
        'reg_lambda': best_params['reg_lambda'],
        'random_state': 42,
        'verbosity': 0
    }
    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=1000, early_stopping_rounds=100)
    xgb_model.fit(X_tr_resampled, y_tr_resampled, eval_set=[(X_te, y_te)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_te)[:, 1]
    xgb_auc = roc_auc_score(y_te, xgb_pred)
    print(f"  XGBoost AUC: {xgb_auc:.4f}")

    # Weighted Ensemble
    best_weight = 0.5
    best_ensemble_auc = 0
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ensemble_pred = w * lgb_pred + (1 - w) * xgb_pred
        ensemble_auc = roc_auc_score(y_te, ensemble_pred)
        if ensemble_auc > best_ensemble_auc:
            best_ensemble_auc = ensemble_auc
            best_weight = w

    print(f"  Ensemble AUC: {best_ensemble_auc:.4f} (LGB weight: {best_weight})")

    return {
        'lgb': lgb_model,
        'xgb': xgb_model,
        'lgb_weight': best_weight,
        'type': 'weighted_ensemble',
        'best_params': best_params
    }, best_ensemble_auc


def main():
    import sys

    track_name = sys.argv[1] if len(sys.argv) > 1 else "大井"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    TRACKS = {
        "大井": {"csv": "data/races_ohi.csv", "model": "models/model_ohi.pkl", "meta": "models/model_ohi_meta.json"},
        "川崎": {"csv": "data/races_kawasaki.csv", "model": "models/model_kawasaki.pkl", "meta": "models/model_kawasaki_meta.json"},
    }

    if track_name not in TRACKS:
        print(f"エラー: {track_name} は未対応")
        sys.exit(1)

    track = TRACKS[track_name]

    # 既存AUC取得
    old_auc = None
    try:
        with open(track["meta"], 'r', encoding='utf-8') as f:
            old_auc = json.load(f).get('auc')
    except:
        pass

    print("=" * 60)
    print(f"AUC 0.8 目標最適化: {track_name}")
    print("=" * 60)
    print()

    # CSV読み込み
    df = pd.read_csv(track["csv"])
    print(f"データ: {len(df)}件")

    # 改良版前処理
    processor = AdvancedProcessor()
    df_processed = processor.process(df)
    print(f"処理後: {len(df_processed)}件")
    print(f"特徴量: {len(processor.features)}個（+{len(processor.extra_features)}個追加）")

    # 最適化付き学習
    print()
    model, auc = train_optimized(df_processed, processor.features, n_trials)

    # 保存
    print()
    print("保存中...")
    with open(track["model"], 'wb') as f:
        pickle.dump({'model': model, 'features': processor.features}, f)

    # メタデータ
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
        'features': processor.features,
        'optimization': {
            'n_trials': n_trials,
            'best_params': model['best_params']
        }
    }
    with open(track["meta"], 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("結果")
    print("=" * 60)
    if old_auc:
        print(f"旧AUC: {old_auc:.4f}")
    print(f"新AUC: {auc:.4f}")
    if old_auc:
        print(f"改善: {(auc - old_auc) * 100:+.2f}%")

    if auc >= 0.8:
        print()
        print("🎉 AUC 0.8 達成！")
    else:
        print()
        print(f"目標まであと: {(0.8 - auc) * 100:.2f}%")


if __name__ == "__main__":
    main()
