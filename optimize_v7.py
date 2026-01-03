"""
v7モデル - 推論時に確実に取れる特徴量のみで学習
+ 確率キャリブレーション付き
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
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
    """リークしないTarget Encoder"""
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.global_mean = None
        self.mappings = {}

    def fit(self, train_df, cols, target):
        self.global_mean = train_df[target].mean()
        for col in cols:
            stats = train_df.groupby(col)[target].agg(['mean', 'count'])
            smooth_mean = (stats['mean'] * stats['count'] + self.global_mean * self.smoothing) / \
                         (stats['count'] + self.smoothing)
            self.mappings[col] = smooth_mean.to_dict()
        return self

    def transform(self, df, cols):
        df = df.copy()
        for col in cols:
            te_col = f'{col}_te'
            df[te_col] = df[col].map(self.mappings.get(col, {})).fillna(self.global_mean)
        return df


def add_previous_race_features_safe(df):
    """前走特徴量を追加（リークなし版）"""
    df = df.copy()
    df['race_date_num'] = df['race_id'].astype(str).str[:8].astype(int)
    df = df.sort_values(['horse_id', 'race_date_num', 'race_id'])

    # 前走の上がり3F
    df['prev_last_3f'] = df.groupby('horse_id')['last_3f'].shift(1)

    # 過去3走・5走の平均
    df['avg_last_3f_3races'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['avg_last_3f_5races'] = df.groupby('horse_id')['last_3f'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    # 前走からの日数
    df['prev_race_date'] = df.groupby('horse_id')['race_date_num'].shift(1)
    df['days_since_last_race'] = df['race_date_num'] - df['prev_race_date']
    df['days_since_last_race'] = df['days_since_last_race'].fillna(30).clip(1, 365)

    # 過去着順の標準偏差
    if 'rank' in df.columns:
        df['past_rank_std'] = df.groupby('horse_id')['rank'].transform(
            lambda x: x.shift(1).expanding().std()
        )
        df['past_rank_std'] = df['past_rank_std'].fillna(3.0)

    return df


class ProcessorV7:
    """
    v7: 推論時に確実に取得できる特徴量のみ使用

    ✓ 取得可能:
      - 馬の過去成績 (horse_runs, horse_win_rate, etc.) → スクレイピング
      - 騎手成績 (jockey_win_rate, etc.) → スクレイピング
      - 出馬表情報 (horse_number, distance, field_size, etc.)
      - 上がり3F (prev_last_3f, avg_last_3f_3races) → スクレイピング実装済み
      - 前走日数 (days_since_last_race) → スクレイピング実装済み

    ✗ 除外:
      - オッズ関連 (市場に引っ張られる)
      - 計算が複雑な派生特徴量
      - 学習時のみ取得可能なデータ
    """

    def __init__(self):
        # ★★★ v7: 推論時に確実に取れる特徴量のみ ★★★
        self.reliable_features = [
            # 馬の基本成績（スクレイピング可能）
            'horse_runs', 'horse_win_rate', 'horse_place_rate', 'horse_show_rate',
            'horse_avg_rank', 'horse_recent_win_rate', 'horse_recent_show_rate',
            'horse_recent_avg_rank', 'last_rank',

            # 騎手成績（スクレイピング可能）
            'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',

            # 出馬表から取得可能
            'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
            'sex_encoded', 'field_size',

            # 馬体重（レース当日取得可能）
            'horse_weight',

            # 上がり3F関連（スクレイピング実装済み）
            'prev_last_3f', 'avg_last_3f_3races', 'avg_last_3f_5races',

            # 前走日数（スクレイピング実装済み）
            'days_since_last_race',

            # 安定性指標
            'past_rank_std',
        ]

        # Target Encoding（スクレイピング不要、モデルに含まれる）
        self.te_features = ['jockey_id_te', 'trainer_id_te', 'horse_id_te']

        # レース内で計算できる相対特徴量
        self.race_relative_features = [
            'horse_win_rate_vs_field',      # 出走馬の中での相対勝率
            'prev_last_3f_rank',            # 出走馬の中での上がり3F順位
            'prev_last_3f_vs_field',        # フィールド平均との差
            'horse_number_ratio',           # 馬番/頭数
            'is_first_race',                # 初出走フラグ
        ]

        self.features = self.reliable_features + self.te_features + self.race_relative_features

        # 除外した特徴量（理由付き）
        self.excluded = {
            'win_odds': '市場バイアス',
            'odds_implied_prob': '市場バイアス',
            'weather_encoded': '当日まで不確定',
            'track_condition_encoded': '当日まで不確定',
            'weight_diff': '前走体重が必要',
            'horse_weight_change': '前走体重が必要',
            'horse_jockey_synergy': '騎手×馬の組み合わせ履歴が必要',
            'form_score': '計算が複雑',
            'class_indicator': 'クラス情報が必要',
            'rank_trend': '信頼性低い',
            'win_streak': '計算誤差が大きい',
            'show_streak': '計算誤差が大きい',
        }

    def process_base(self, df):
        df = df.copy()
        num_cols = ['bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
                    'horse_weight', 'last_3f', 'race_time_seconds']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 前走特徴量を追加
        df = add_previous_race_features_safe(df)

        # 初出走フラグ
        df['is_first_race'] = df['prev_last_3f'].isna().astype(int)

        # 性別エンコード
        if 'sex' in df.columns:
            df['sex_encoded'] = df['sex'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        else:
            df['sex_encoded'] = 0

        return df

    def process(self, df, te_encoder=None):
        df = self.process_base(df)

        # Target Encoding
        if te_encoder is not None:
            te_cols = ['jockey_id', 'trainer_id', 'horse_id']
            df = te_encoder.transform(df, te_cols)
        else:
            for col in self.te_features:
                df[col] = 0.274  # 複勝率の平均

        # フィールド内相対特徴量
        if 'horse_win_rate' in df.columns:
            df['field_avg_win_rate'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
            df['horse_win_rate_vs_field'] = df['horse_win_rate'] - df['field_avg_win_rate']
            df['horse_win_rate_vs_field'] = df['horse_win_rate_vs_field'].fillna(0)

        # 上がり3Fのレース内順位
        if 'prev_last_3f' in df.columns:
            df['prev_last_3f_rank'] = df.groupby('race_id')['prev_last_3f'].rank(ascending=True)
            df['prev_last_3f_vs_field'] = df.groupby('race_id')['prev_last_3f'].transform('mean') - df['prev_last_3f']
            df['prev_last_3f_rank'] = df['prev_last_3f_rank'].fillna(6)
            df['prev_last_3f_vs_field'] = df['prev_last_3f_vs_field'].fillna(0)

        # 馬番比率
        if 'horse_number' in df.columns and 'field_size' in df.columns:
            df['horse_number_ratio'] = df['horse_number'] / df['field_size'].clip(lower=1)
        else:
            df['horse_number_ratio'] = 0.5

        # 欠損値埋め
        defaults = {
            'horse_runs': 0, 'horse_win_rate': 0, 'horse_place_rate': 0, 'horse_show_rate': 0,
            'horse_avg_rank': 10, 'horse_recent_win_rate': 0, 'horse_recent_show_rate': 0,
            'horse_recent_avg_rank': 10, 'last_rank': 10,
            'jockey_win_rate': 0.1, 'jockey_place_rate': 0.2, 'jockey_show_rate': 0.3,
            'horse_number': 5, 'bracket': 4, 'age': 4, 'weight_carried': 55, 'distance': 1600,
            'sex_encoded': 0, 'field_size': 12, 'horse_weight': 470,
            'prev_last_3f': 41.2, 'avg_last_3f_3races': 41.2, 'avg_last_3f_5races': 41.2,
            'days_since_last_race': 30, 'past_rank_std': 3.0, 'is_first_race': 0,
        }

        for col, default in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
            else:
                df[col] = default

        for f in self.features:
            if f not in df.columns:
                df[f] = 0

        return df


def train_model(track_name, n_trials=30):
    """モデル学習（キャリブレーション付き）"""
    print(f'\n{"="*60}')
    print(f'{track_name.upper()} モデル学習 (v7 - 信頼できる特徴量のみ)')
    print(f'{"="*60}')

    # データ読み込み
    df = pd.read_csv(f'data/races_{track_name}.csv')
    print(f'データ件数: {len(df):,}')

    # 前処理
    processor = ProcessorV7()
    df = processor.process_base(df)

    # 複勝ターゲット
    df['target'] = (df['rank'] <= 3).astype(int)
    print(f'複勝率: {df["target"].mean()*100:.1f}%')

    # 上がり3Fの確認
    prev_3f_valid = df['prev_last_3f'].notna() & (df['prev_last_3f'] > 30) & (df['prev_last_3f'] < 55)
    print(f'上がり3F有効: {prev_3f_valid.sum():,} ({prev_3f_valid.mean()*100:.1f}%)')

    # Train/Test分割（時系列）
    df = df.sort_values('race_id')
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f'Train: {len(train_df):,}, Test: {len(test_df):,}')

    # Target Encoding（学習データのみで作成）
    te_encoder = TargetEncoderSafe(smoothing=10)
    te_encoder.fit(train_df, ['jockey_id', 'trainer_id', 'horse_id'], 'target')

    # 特徴量作成
    train_df = processor.process(train_df, te_encoder)
    test_df = processor.process(test_df, te_encoder)

    X_train = train_df[processor.features].values
    y_train = train_df['target'].values
    X_test = test_df[processor.features].values
    y_test = test_df['target'].values

    print(f'特徴量数: {len(processor.features)}')

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f'SMOTE後: {len(X_train_res):,}')

    # Optuna最適化
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': 5,
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
            'verbose': -1,
        }

        train_data = lgb.Dataset(X_train_res, label=y_train_res)
        model = lgb.train(params, train_data, num_boost_round=300)
        pred = model.predict(X_test)
        return roc_auc_score(y_test, pred)

    print('\nOptuna最適化中...')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f'Best AUC: {study.best_value:.4f}')

    # 最終モデル学習
    final_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        **best_params
    }

    train_data = lgb.Dataset(X_train_res, label=y_train_res)
    lgb_model = lgb.train(final_params, train_data, num_boost_round=300)

    # XGBoostも学習
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    )
    xgb_model.fit(X_train_res, y_train_res)

    # アンサンブル予測
    lgb_pred = lgb_model.predict(X_test)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    ensemble_pred = (lgb_pred + xgb_pred) / 2

    # 評価
    auc = roc_auc_score(y_test, ensemble_pred)
    brier = brier_score_loss(y_test, ensemble_pred)

    print(f'\n=== 最終評価 ===')
    print(f'AUC: {auc:.4f}')
    print(f'Brier Score: {brier:.4f} (低いほど良い)')

    # キャリブレーション評価
    print(f'\n=== 確率キャリブレーション ===')
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = ensemble_pred >= thresh
        if mask.sum() > 0:
            actual_rate = y_test[mask].mean()
            print(f'予測 >= {thresh*100:.0f}%: {mask.sum():,}件, 実際の的中率: {actual_rate*100:.1f}%')

    # 特徴量重要度
    importance = pd.DataFrame({
        'feature': processor.features,
        'importance': lgb_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(f'\n=== 特徴量重要度 TOP10 ===')
    for i, row in importance.head(10).iterrows():
        print(f'{row["feature"]}: {row["importance"]:.0f}')

    # モデル保存
    model_data = {
        'model': {
            'type': 'ensemble',
            'lgb': lgb_model,
            'xgb': xgb_model
        },
        'features': processor.features,
        'te_encoder': te_encoder,
        'auc': auc,
        'brier': brier,
        'version': 'v7_reliable_features',
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'excluded_features': list(processor.excluded.keys()),
    }

    model_path = f'models/model_{track_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f'\nモデル保存: {model_path}')

    # メタデータ保存
    meta = {
        'track_name': track_name,
        'trained_at': model_data['trained_at'],
        'auc': round(auc, 4),
        'brier': round(brier, 4),
        'version': 'v7_reliable_features',
        'feature_count': len(processor.features),
    }
    with open(f'models/model_{track_name}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return auc, brier


if __name__ == '__main__':
    print('='*60)
    print('v7モデル学習: 信頼できる特徴量のみ + キャリブレーション評価')
    print('='*60)

    results = {}

    for track in ['kawasaki', 'ohi']:
        auc, brier = train_model(track, n_trials=30)
        results[track] = {'auc': auc, 'brier': brier}

    print('\n' + '='*60)
    print('=== 最終結果 ===')
    print('='*60)
    for track, r in results.items():
        print(f'{track.upper()}: AUC={r["auc"]:.4f}, Brier={r["brier"]:.4f}')
