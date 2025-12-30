"""
Optunaによるハイパーパラメータ最適化

使い方:
    python optimize_model.py 大井
    python optimize_model.py 大井 --trials 50  # 試行回数指定

機能:
    - LightGBMのハイパーパラメータを自動最適化
    - 時系列クロスバリデーションで評価
    - 最適パラメータを保存
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from datetime import datetime, timedelta

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Optunaがインストールされていません")
    print("pip install optuna")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent

TRACKS = {
    "大井": {"code": "44", "data": "data/races_ohi.csv"},
    "川崎": {"code": "45", "data": "data/races_kawasaki.csv"},
    "船橋": {"code": "43", "data": "data/races_funabashi.csv"},
    "浦和": {"code": "42", "data": "data/races_urawa.csv"},
}

FEATURES = [
    'horse_runs', 'horse_win_rate', 'horse_place_rate', 'horse_show_rate',
    'horse_avg_rank', 'horse_recent_win_rate', 'horse_recent_show_rate',
    'horse_recent_avg_rank', 'last_rank',
    'jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate',
    'horse_number', 'bracket', 'age', 'weight_carried', 'distance',
    'sex_encoded', 'track_encoded', 'field_size', 'weight_diff',
    'track_condition_encoded', 'weather_encoded',
    'trainer_encoded', 'horse_weight', 'horse_weight_change'
]


def preprocess(df):
    """前処理"""
    df = df.copy()

    if 'rank' in df.columns:
        df = df[df['rank'].notna() & (df['rank'] > 0)]

    # 数値変換
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

    # 馬場状態
    if 'track_condition' in df.columns:
        df['track_condition_encoded'] = df['track_condition'].map(
            {'良': 0, '稍重': 1, '重': 2, '不良': 3}
        ).fillna(0)
    else:
        df['track_condition_encoded'] = 0

    # 天気
    if 'weather' in df.columns:
        df['weather_encoded'] = df['weather'].map(
            {'晴': 0, '曇': 1, '小雨': 2, '雨': 3, '雪': 4}
        ).fillna(0)
    else:
        df['weather_encoded'] = 0

    # 調教師
    if 'trainer_id' in df.columns:
        df['trainer_encoded'] = df['trainer_id'].apply(
            lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0
        )
    else:
        df['trainer_encoded'] = 0

    # 馬体重
    if 'horse_weight' in df.columns:
        df['horse_weight'] = df['horse_weight'].fillna(450)
    else:
        df['horse_weight'] = 450

    if 'weight_change' in df.columns:
        df['horse_weight_change'] = df['weight_change'].fillna(0)
    else:
        df['horse_weight_change'] = 0

    # ターゲット
    if 'rank' in df.columns:
        df['target'] = (df['rank'] <= 3).astype(int)

    # 欠損特徴量を追加
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0

    return df


def evaluate_model(model, test_df):
    """モデル評価"""
    X_test = test_df[FEATURES].fillna(-1)
    test_df = test_df.copy()
    test_df['prob'] = model.predict(X_test)

    # レースごとに複勝的中率を計算
    total_races = 0
    show_hits = 0

    for race_id, race_df in test_df.groupby('race_id'):
        race_df = race_df.sort_values('prob', ascending=False)
        if len(race_df) < 3:
            continue

        total_races += 1
        pred_1st_rank = race_df.iloc[0]['rank']

        if pred_1st_rank <= 3:
            show_hits += 1

    show_rate = show_hits / total_races * 100 if total_races > 0 else 0

    # AUC
    try:
        auc = roc_auc_score(test_df['target'], test_df['prob'])
    except:
        auc = 0.5

    return {'show_rate': show_rate, 'auc': auc, 'total_races': total_races}


def objective(trial, train_df, valid_df):
    """Optunaの目的関数"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }

    X_train = train_df[FEATURES].fillna(-1)
    y_train = train_df['target']
    X_valid = valid_df[FEATURES].fillna(-1)
    y_valid = valid_df['target']

    train_data = lgb.Dataset(X_train, y_train)
    valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0)
        ]
    )

    # 複勝的中率で評価
    metrics = evaluate_model(model, valid_df)
    return metrics['show_rate']


def optimize(df, n_trials=30, train_months=24):
    """ハイパーパラメータ最適化"""
    df = df.sort_values('race_date')
    dates = sorted(df['race_date'].unique())

    latest_dt = datetime.strptime(str(dates[-1]), '%Y%m%d')

    # バリデーション期間: 直近3ヶ月
    valid_start_dt = latest_dt - timedelta(days=90)
    valid_start_date = int(valid_start_dt.strftime('%Y%m%d'))

    # 学習期間: バリデーション前のN ヶ月
    train_start_dt = valid_start_dt - timedelta(days=train_months * 30)
    train_start_date = int(train_start_dt.strftime('%Y%m%d'))

    train_df = df[(df['race_date'] >= train_start_date) & (df['race_date'] < valid_start_date)]
    valid_df = df[df['race_date'] >= valid_start_date]

    print(f"学習データ: {len(train_df)}件")
    print(f"検証データ: {len(valid_df)}件")

    train_processed = preprocess(train_df)
    valid_processed = preprocess(valid_df)

    # Optuna最適化
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, train_processed, valid_processed),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n最良スコア（複勝的中率）: {study.best_value:.1f}%")
    print(f"最良パラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study.best_params, study.best_value


def train_with_best_params(df, best_params, train_months=24):
    """最適パラメータでモデルを学習"""
    df = df.sort_values('race_date')
    dates = sorted(df['race_date'].unique())

    latest_dt = datetime.strptime(str(dates[-1]), '%Y%m%d')
    train_start_dt = latest_dt - timedelta(days=train_months * 30)
    train_start_date = int(train_start_dt.strftime('%Y%m%d'))

    train_df = df[df['race_date'] >= train_start_date]
    train_processed = preprocess(train_df)

    X = train_processed[FEATURES].fillna(-1)
    y = train_processed['target']

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        **best_params
    }

    model = lgb.train(
        params,
        lgb.Dataset(X, y),
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(0)]
    )

    return model


def main():
    if len(sys.argv) < 2:
        print("使い方: python optimize_model.py <競馬場名>")
        print("例: python optimize_model.py 大井")
        print("例: python optimize_model.py 大井 --trials 50")
        sys.exit(1)

    track_name = sys.argv[1]
    n_trials = 30

    # 引数でtrials数を指定
    if '--trials' in sys.argv:
        idx = sys.argv.index('--trials')
        if idx + 1 < len(sys.argv):
            n_trials = int(sys.argv[idx + 1])

    if track_name not in TRACKS:
        print(f"エラー: 競馬場 '{track_name}' は存在しません")
        sys.exit(1)

    track_info = TRACKS[track_name]
    data_path = BASE_DIR / track_info['data']

    if not data_path.exists():
        print(f"エラー: データファイルがありません: {data_path}")
        sys.exit(1)

    print(f"【{track_name}競馬場 ハイパーパラメータ最適化】")
    print(f"データ: {data_path}")
    print(f"試行回数: {n_trials}")
    print()

    # データ読み込み
    df = pd.read_csv(data_path)
    print(f"総データ数: {len(df)}件")

    # 最適化実行
    best_params, best_score = optimize(df, n_trials=n_trials)

    # 最適パラメータでモデルを学習
    print("\n最適パラメータでモデルを学習中...")
    model = train_with_best_params(df, best_params)

    # 保存
    output_path = BASE_DIR / 'models' / f'model_{track_info["code"]}_optimized.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': FEATURES,
            'params': best_params,
            'best_score': best_score,
            'created_at': datetime.now().isoformat()
        }, f)

    print(f"\n最適化モデルを保存: {output_path}")
    print("\n" + "=" * 50)
    print("完了！")


if __name__ == "__main__":
    main()
