#!/usr/bin/env python3
"""
フィルター条件の自動最適化

過去データでバックテストを繰り返し、最高の回収率を達成するフィルター条件を探索
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from datetime import datetime

# プロジェクトルート
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from train import Processor
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def load_data(track_code: str = "44"):
    """データ読み込み"""
    tracks = {
        "44": "data/races_ohi.csv",
        "45": "data/races_kawasaki.csv",
    }
    csv_path = BASE_DIR / tracks.get(track_code, tracks["44"])
    df = pd.read_csv(csv_path)

    # 日付カラム作成
    race_id_str = df["race_id"].astype(str)
    df["date"] = pd.to_datetime(race_id_str.str[:4] + race_id_str.str[6:10], format="%Y%m%d")
    df["race_num"] = race_id_str.str[-2:].astype(int)

    return df


def train_and_predict(df_train, df_test, features):
    """モデル学習と予測"""
    X_train = df_train[features].fillna(-1)
    y_train = df_train["target"]
    X_test = df_test[features].fillna(-1)

    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    return model.predict_proba(X_test)[:, 1]


def evaluate_filters(df_test, probs, params):
    """フィルター条件で評価"""
    min_prob_diff = params["min_prob_diff"]
    min_race_num = params["min_race_num"]
    min_prob = params["min_prob"]

    df_test = df_test.copy()
    df_test["prob"] = probs

    results = []

    for race_id, race_df in df_test.groupby("race_id"):
        race_num = int(str(race_id)[-2:])

        # フィルター1: レース番号
        if race_num < min_race_num:
            continue

        # 確率でソート
        race_df = race_df.sort_values("prob", ascending=False)
        if len(race_df) < 2:
            continue

        first = race_df.iloc[0]
        second = race_df.iloc[1]
        prob_diff = first["prob"] - second["prob"]

        # フィルター2: 確率差
        if prob_diff < min_prob_diff:
            continue

        # フィルター3: 最低確率
        if first["prob"] < min_prob:
            continue

        # 的中判定
        actual_rank = first["rank"]
        win_hit = actual_rank == 1
        show_hit = actual_rank <= 3

        results.append({
            "race_id": race_id,
            "pred_num": first["horse_number"],
            "pred_prob": first["prob"],
            "prob_diff": prob_diff,
            "actual_rank": actual_rank,
            "win_hit": win_hit,
            "show_hit": show_hit,
        })

    if len(results) == 0:
        return {"n_bets": 0, "win_rate": 0, "show_rate": 0, "score": 0}

    results_df = pd.DataFrame(results)
    n_bets = len(results_df)
    win_rate = results_df["win_hit"].mean() * 100
    show_rate = results_df["show_hit"].mean() * 100

    # スコア = 複勝的中率 × √買い目数（買い目が少なすぎるのもNG）
    score = show_rate * np.sqrt(n_bets / 10)

    return {
        "n_bets": n_bets,
        "win_rate": win_rate,
        "show_rate": show_rate,
        "score": score
    }


def objective(trial, df_train, df_test, features):
    """Optuna目的関数"""
    params = {
        "min_prob_diff": trial.suggest_float("min_prob_diff", 0.03, 0.25),
        "min_race_num": trial.suggest_int("min_race_num", 1, 10),
        "min_prob": trial.suggest_float("min_prob", 0.3, 0.7),
    }

    # 予測
    probs = train_and_predict(df_train, df_test, features)

    # 評価
    result = evaluate_filters(df_test, probs, params)

    # 買い目が少なすぎる場合はペナルティ
    if result["n_bets"] < 20:
        return 0

    return result["score"]


def main():
    print("=" * 60)
    print("フィルター条件の自動最適化")
    print("=" * 60)

    # データ読み込み
    print("\n[1] データ読み込み...")
    df = load_data("44")
    print(f"  総レコード: {len(df)}")
    print(f"  レース数: {df['race_id'].nunique()}")

    # 学習/テスト分割（時系列）
    cutoff = "2025-10-01"
    df_train = df[df["date"] < cutoff].copy()
    df_test = df[df["date"] >= cutoff].copy()

    print(f"  学習データ: {len(df_train)}件 (〜{cutoff})")
    print(f"  テストデータ: {len(df_test)}件 ({cutoff}〜)")

    # 前処理
    print("\n[2] 前処理...")
    processor = Processor()
    df_train = processor.process(df_train)
    df_test = processor.process(df_test)
    features = processor.features
    print(f"  特徴量数: {len(features)}")

    # Optuna最適化
    print("\n[3] Optuna最適化 (100試行)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, df_train, df_test, features),
        n_trials=100,
        show_progress_bar=True
    )

    # 結果表示
    print("\n" + "=" * 60)
    print("最適なフィルター条件")
    print("=" * 60)

    best_params = study.best_params
    print(f"  確率差閾値: {best_params['min_prob_diff']:.2%}")
    print(f"  最低レース番号: {best_params['min_race_num']}R以降")
    print(f"  最低確率: {best_params['min_prob']:.1%}")

    # 最適条件で再評価
    probs = train_and_predict(df_train, df_test, features)
    result = evaluate_filters(df_test, probs, best_params)

    print(f"\n【テスト期間の成績】")
    print(f"  買い目数: {result['n_bets']}点")
    print(f"  単勝的中率: {result['win_rate']:.1f}%")
    print(f"  複勝的中率: {result['show_rate']:.1f}%")

    # 比較: フィルターなし
    no_filter = {"min_prob_diff": 0, "min_race_num": 1, "min_prob": 0}
    result_no_filter = evaluate_filters(df_test, probs, no_filter)

    print(f"\n【比較: フィルターなし】")
    print(f"  買い目数: {result_no_filter['n_bets']}点")
    print(f"  単勝的中率: {result_no_filter['win_rate']:.1f}%")
    print(f"  複勝的中率: {result_no_filter['show_rate']:.1f}%")

    print(f"\n【改善効果】")
    print(f"  複勝的中率: {result_no_filter['show_rate']:.1f}% → {result['show_rate']:.1f}% ({result['show_rate'] - result_no_filter['show_rate']:+.1f}pt)")

    # 設定をJSONで保存
    import json
    config = {
        "min_prob_diff": best_params["min_prob_diff"],
        "min_race_num": best_params["min_race_num"],
        "min_prob": best_params["min_prob"],
        "test_period": f"{cutoff}〜",
        "n_bets": result["n_bets"],
        "show_rate": result["show_rate"],
        "optimized_at": datetime.now().isoformat()
    }
    config_path = BASE_DIR / "models" / "optimal_filters.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n設定を保存: {config_path}")


if __name__ == "__main__":
    main()
