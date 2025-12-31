"""競馬場モデル再学習（CSVのみ、スクレイピングなし）
使い方: python retrain_kawasaki.py [競馬場名]
例: python retrain_kawasaki.py 川崎
    python retrain_kawasaki.py 大井
"""
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Processor:
    def __init__(self):
        self.features = [
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

    def process(self, df):
        df = df.copy()
        if 'rank' in df.columns:
            df = df[df['rank'].notna() & (df['rank'] > 0)]

        num_cols = ['rank', 'bracket', 'horse_number', 'age', 'weight_carried', 'distance',
                    'field_size', 'horse_runs', 'horse_win_rate', 'horse_place_rate',
                    'horse_show_rate', 'horse_avg_rank', 'horse_recent_win_rate',
                    'horse_recent_show_rate', 'horse_recent_avg_rank', 'last_rank',
                    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
                    'horse_weight', 'weight_change']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # エンコーディング
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
            df['horse_number_ratio'] = df['horse_number'] / df['field_size']
            df['horse_number_ratio'] = df['horse_number_ratio'].fillna(0.5)
        else:
            df['horse_number_ratio'] = 0.5

        if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['last_rank_diff'] = df['last_rank'] - df['horse_avg_rank']
            df['last_rank_diff'] = df['last_rank_diff'].fillna(0)
        else:
            df['last_rank_diff'] = 0

        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['win_rate_rank'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False, method='min')
            df['win_rate_rank'] = df['win_rate_rank'].fillna(df['field_size'] / 2)
        else:
            df['win_rate_rank'] = 6

        # 相対特徴量
        if 'horse_win_rate' in df.columns and 'race_id' in df.columns:
            df['field_avg_win_rate'] = df.groupby('race_id')['horse_win_rate'].transform('mean')
            df['horse_win_rate_vs_field'] = df['horse_win_rate'] - df['field_avg_win_rate']
            df['horse_win_rate_vs_field'] = df['horse_win_rate_vs_field'].fillna(0)
        else:
            df['horse_win_rate_vs_field'] = 0

        if 'jockey_win_rate' in df.columns and 'race_id' in df.columns:
            df['field_avg_jockey_win_rate'] = df.groupby('race_id')['jockey_win_rate'].transform('mean')
            df['jockey_win_rate_vs_field'] = df['jockey_win_rate'] - df['field_avg_jockey_win_rate']
            df['jockey_win_rate_vs_field'] = df['jockey_win_rate_vs_field'].fillna(0)
        else:
            df['jockey_win_rate_vs_field'] = 0

        if 'horse_avg_rank' in df.columns and 'race_id' in df.columns:
            df['field_avg_rank'] = df.groupby('race_id')['horse_avg_rank'].transform('mean')
            df['horse_avg_rank_vs_field'] = df['field_avg_rank'] - df['horse_avg_rank']
            df['horse_avg_rank_vs_field'] = df['horse_avg_rank_vs_field'].fillna(0)
        else:
            df['horse_avg_rank_vs_field'] = 0

        # 休養日数
        if 'days_since_last_race' in df.columns:
            df['days_since_last_race'] = df['days_since_last_race'].fillna(30).clip(0, 365)
        else:
            df['days_since_last_race'] = 30

        # 着順トレンド
        if 'last_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['rank_trend'] = df['horse_avg_rank'] - df['last_rank']
            df['rank_trend'] = df['rank_trend'].fillna(0)
        else:
            df['rank_trend'] = 0

        # 交互作用特徴量
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

        # 時系列強化
        if 'win_streak' not in df.columns:
            df['win_streak'] = 0
        if 'show_streak' not in df.columns:
            df['show_streak'] = 0

        if 'recent_3_avg_rank' not in df.columns:
            if 'horse_recent_avg_rank' in df.columns:
                df['recent_3_avg_rank'] = df['horse_recent_avg_rank']
            else:
                df['recent_3_avg_rank'] = 10
        if 'recent_10_avg_rank' not in df.columns:
            if 'horse_avg_rank' in df.columns:
                df['recent_10_avg_rank'] = df['horse_avg_rank']
            else:
                df['recent_10_avg_rank'] = 10

        if 'recent_3_avg_rank' in df.columns and 'horse_avg_rank' in df.columns:
            df['rank_improvement'] = df['horse_avg_rank'] - df['recent_3_avg_rank']
            df['rank_improvement'] = df['rank_improvement'].fillna(0)
        else:
            df['rank_improvement'] = 0

        # 血統
        for col in ['father_win_rate', 'father_show_rate', 'bms_win_rate', 'bms_show_rate']:
            if col not in df.columns:
                df[col] = 0

        # ターゲット
        df['target'] = (df['rank'] <= 3).astype(int)

        # 不足特徴量を補完
        for f in self.features:
            if f not in df.columns:
                df[f] = 0

        return df


def train_model(df, features):
    X, y = df[features].fillna(-1), df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE
    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr, y_tr)
        print(f"  SMOTE: {len(y_tr)} -> {len(y_tr_resampled)}")
    except Exception as e:
        print(f"  SMOTE skip: {e}")
        X_tr_resampled, y_tr_resampled = X_tr, y_tr

    # LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'num_leaves': 112,
        'learning_rate': 0.067,
        'min_child_samples': 65,
        'reg_alpha': 6.8e-07,
        'reg_lambda': 0.025,
        'feature_fraction': 0.78,
        'bagging_fraction': 0.78,
        'bagging_freq': 3
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb.Dataset(X_tr_resampled, y_tr_resampled),
        500,
        [lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    lgb_pred = lgb_model.predict(X_te)
    lgb_auc = roc_auc_score(y_te, lgb_pred)
    print(f"  LightGBM AUC: {lgb_auc:.4f}")

    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 0
    }
    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=500, early_stopping_rounds=50)
    xgb_model.fit(X_tr_resampled, y_tr_resampled, eval_set=[(X_te, y_te)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_te)[:, 1]
    xgb_auc = roc_auc_score(y_te, xgb_pred)
    print(f"  XGBoost AUC: {xgb_auc:.4f}")

    # Ensemble
    ensemble_pred = (lgb_pred + xgb_pred) / 2
    ensemble_auc = roc_auc_score(y_te, ensemble_pred)
    print(f"  Ensemble AUC: {ensemble_auc:.4f}")

    if ensemble_auc >= max(lgb_auc, xgb_auc):
        print("  -> Ensemble")
        return {'lgb': lgb_model, 'xgb': xgb_model, 'type': 'ensemble'}, ensemble_auc
    elif xgb_auc > lgb_auc:
        print("  -> XGBoost")
        return {'xgb': xgb_model, 'type': 'xgb'}, xgb_auc
    else:
        print("  -> LightGBM")
        return lgb_model, lgb_auc


TRACKS = {
    "大井": {"csv": "data/races_ohi.csv", "model": "models/model_ohi.pkl", "meta": "models/model_ohi_meta.json"},
    "川崎": {"csv": "data/races_kawasaki.csv", "model": "models/model_kawasaki.pkl", "meta": "models/model_kawasaki_meta.json"},
}


def main():
    import sys

    # コマンドライン引数から競馬場名を取得
    if len(sys.argv) < 2:
        track_name = "川崎"  # デフォルト
    else:
        track_name = sys.argv[1]

    if track_name not in TRACKS:
        print(f"エラー: 不明な競馬場 '{track_name}'")
        print(f"利用可能: {', '.join(TRACKS.keys())}")
        sys.exit(1)

    track = TRACKS[track_name]

    # 既存のAUCを取得
    old_auc = None
    try:
        with open(track["meta"], 'r', encoding='utf-8') as f:
            old_meta = json.load(f)
            old_auc = old_meta.get('auc')
    except:
        pass

    print("=" * 50)
    print(f"{track_name}モデル再学習（CSVのみ、スクレイピングなし）")
    print("=" * 50)
    print()

    # CSV読み込み
    df = pd.read_csv(track["csv"])
    print(f"データ: {len(df)}件")

    # 前処理
    processor = Processor()
    df_processed = processor.process(df)
    print(f"処理後: {len(df_processed)}件")
    print(f"特徴量: {len(processor.features)}個")

    # 学習
    print()
    print("学習中...")
    model, auc = train_model(df_processed, processor.features)

    # 保存
    print()
    print("保存中...")
    with open(track["model"], 'wb') as f:
        pickle.dump({'model': model, 'features': processor.features}, f)

    # メタデータ
    race_dates = df['race_date'].dropna().astype(int).astype(str)
    min_date = race_dates.min()
    max_date = race_dates.max()

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
    print("=" * 50)
    print("完了")
    print("=" * 50)
    if old_auc:
        print(f"旧AUC: {old_auc:.4f}")
        print(f"新AUC: {auc:.4f}")
        print(f"改善: {(auc - old_auc) * 100:+.2f}%")
    else:
        print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
