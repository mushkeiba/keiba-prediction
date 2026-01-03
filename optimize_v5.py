"""AUC 0.8を目指す最適化スクリプト v5 - リーク完全排除版"""
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
            # スムージング
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
    - 必ずshift(1)で「過去」のデータのみ使用
    """
    df = df.copy()

    # race_idから日付を抽出してソート
    df['race_date_num'] = df['race_id'].astype(str).str[:8].astype(int)
    df = df.sort_values(['horse_id', 'race_date_num', 'race_id'])

    # 前走の上がり3F（shift(1)で過去のみ）
    df['prev_last_3f'] = df.groupby('horse_id')['last_3f'].shift(1)

    # 過去N走の平均（shift(1)してからrolling）
    df['avg_last_3f_3races'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['avg_last_3f_5races'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    # 前走のrace_time
    df['prev_race_time'] = df.groupby('horse_id')['race_time_seconds'].shift(1)

    # 過去成績の標準偏差（リーク修正版）
    # ❌ df.groupby('horse_id')['target'].transform('std')  # 未来を含む
    # ✅ shift(1)してからexpanding
    if 'rank' in df.columns:
        df['past_rank_std'] = df.groupby('horse_id')['rank'].transform(
            lambda x: x.shift(1).expanding().std()
        )

    return df


class ProcessorV5:
    """v5: リーク完全排除版"""

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
            'field_strength', 'inner_outer',
            'avg_rank_percentile', 'jockey_rank_in_race',
            'distance_fitness', 'weight_per_meter', 'experience_score',
            # v5: 前走上がり3F（リークなし）
            'prev_last_3f',
            'avg_last_3f_3races',
            'avg_last_3f_5races',
            'prev_last_3f_rank',
            'prev_last_3f_vs_field',
            # v5: 過去成績変動（リークなし）
            'past_rank_std',
            'is_first_race',  # 初出走フラグ
        ]

        # ❌ 削除: horse_win_rate_std（未来の成績を含むリーク）
        # ❌ 削除: odds_implied_prob（レース当日のオッズはリーク）

        self.features = self.base_features + self.te_features + self.extra_features

    def process_base(self, df):
        """基本前処理（Target Encoding以外）"""
        df = df.copy()

        if 'rank' in df.columns:
            df = df[df['rank'].notna() & (df['rank'] > 0)]

        df = df.reset_index(drop=True)
        df['target'] = (df['rank'] <= 3).astype(int)

        # 数値変換
        num_cols = ['rank', 'bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
                    'horse_weight', 'weight_change', 'last_3f', 'race_time_seconds']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 前走特徴量（リークなし版）
        df = add_previous_race_features_safe(df)

        # 初出走フラグ
        df['is_first_race'] = df['prev_last_3f'].isna().astype(int)

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

        # 追加特徴量
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

        # 前走上がり3Fのレース内順位
        df['prev_last_3f_rank'] = df.groupby('race_id')['prev_last_3f'].rank(ascending=True)
        df['prev_last_3f_vs_field'] = df.groupby('race_id')['prev_last_3f'].transform('mean') - df['prev_last_3f']

        return df


def train_model(track_name, n_trials=50):
    """モデル学習（リーク完全排除版）"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} モデル学習 (v5 - リーク完全排除)')
    print(f'{"="*60}')

    # データ読み込み
    df = pd.read_csv(f'data/races_{track_name}.csv')
    print(f'データ数: {len(df):,}')

    # 前処理（TE以外）
    processor = ProcessorV5()
    df = processor.process_base(df)

    # ★★★ 時系列分割（これが正解）★★★
    df = df.sort_values('race_id').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f'\n時系列分割:')
    print(f'  学習: {len(train_df):,}件 (race_id: {train_df["race_id"].min()} ~ {train_df["race_id"].max()})')
    print(f'  テスト: {len(test_df):,}件 (race_id: {test_df["race_id"].min()} ~ {test_df["race_id"].max()})')

    # ★★★ Target Encoding（学習データのみで統計作成）★★★
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
    print(f'学習データ正例率: {y_train.mean():.1%}')
    print(f'テストデータ正例率: {y_test.mean():.1%}')

    # SMOTE（学習データのみに適用）
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f'SMOTE後: {len(X_train_sm):,}件')

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

    lgb_auc = roc_auc_score(y_test, lgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)

    print(f'\n{"="*40}')
    print(f'結果（リーク排除版・これが本当のAUC）')
    print(f'{"="*40}')
    print(f'LightGBM AUC: {lgb_auc:.4f}')
    print(f'XGBoost AUC:  {xgb_auc:.4f}')
    print(f'Ensemble AUC: {ensemble_auc:.4f}')

    # 的中率も計算
    test_df['pred_prob'] = ensemble_pred
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_prob'].rank(ascending=False)

    top1 = test_df[test_df['pred_rank'] == 1]
    win_rate = (top1['rank'] == 1).mean()
    place_rate = (top1['rank'] <= 3).mean()

    print(f'\n予測1位の成績:')
    print(f'  単勝的中率: {win_rate:.1%}')
    print(f'  複勝的中率: {place_rate:.1%}')

    # 特徴量重要度TOP10
    importance = pd.DataFrame({
        'feature': processor.features,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f'\n特徴量重要度TOP10:')
    for _, row in importance.head(10).iterrows():
        print(f'  {row["feature"]}: {row["importance"]:.0f}')

    # 保存
    model_data = {
        'model': {'type': 'ensemble', 'lgb': lgb_model, 'xgb': xgb_model},
        'features': processor.features,
        'te_encoder': te_encoder,  # TEエンコーダーも保存
    }
    with open(f'models/model_{track_name}.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    meta = {
        'track_name': track_name,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'auc': round(ensemble_auc, 4),
        'win_rate': round(win_rate, 4),
        'place_rate': round(place_rate, 4),
        'features': processor.features,
        'version': 'v5_no_leak',
        'train_size': len(train_df),
        'test_size': len(test_df),
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
