#!/usr/bin/env python3
"""
完全バックテスト: 指定日までで学習 → それ以降でテスト

使い方:
  python scripts/backtest_full.py --track 44 --cutoff 2025-12-25
  python scripts/backtest_full.py --track 45 --cutoff 2025-12-25
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# プロジェクトルート
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from train import Processor, train_model

# 競馬場設定
TRACKS = {
    "44": {"name": "大井", "data": "data/races_ohi.csv"},
    "45": {"name": "川崎", "data": "data/races_kawasaki.csv"},
}


def predict_with_model(model, X):
    """モデルで予測（アンサンブル対応）"""
    if isinstance(model, dict):
        model_type = model.get("type", "ensemble")
        if model_type == "ensemble":
            lgb_pred = model["lgb"].predict(X)
            xgb_pred = model["xgb"].predict(X)
            return (lgb_pred + xgb_pred) / 2
        elif model_type == "xgb":
            return model["xgb"].predict(X)
        else:
            return model["lgb"].predict(X)
    return model.predict(X)


def run_full_backtest(track_code: str, cutoff_date: str):
    """完全バックテスト実行"""
    track_info = TRACKS[track_code]
    print(f"\n{'='*60}")
    print(f"完全バックテスト: {track_info['name']}競馬場")
    print(f"学習データ: 〜{cutoff_date}")
    print(f"テストデータ: {cutoff_date}〜")
    print(f"{'='*60}\n")

    # データ読み込み
    data_path = BASE_DIR / track_info["data"]
    if not data_path.exists():
        print(f"[ERROR] データファイルがありません: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"総データ: {len(df)}件")

    # 日付カラムを作成
    race_id_str = df["race_id"].astype(str)
    df["date"] = pd.to_datetime(race_id_str.str[:4] + race_id_str.str[6:10], format="%Y%m%d")
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # データ分割
    cutoff_dt = pd.to_datetime(cutoff_date)
    df_train = df[df["date"] <= cutoff_dt].copy()
    df_test = df[df["date"] > cutoff_dt].copy()

    print(f"学習データ: {len(df_train)}件 ({df_train['date_str'].min()} 〜 {df_train['date_str'].max()})")
    print(f"テストデータ: {len(df_test)}件 ({df_test['date_str'].min()} 〜 {df_test['date_str'].max()})")

    if len(df_train) < 100:
        print(f"[ERROR] 学習データが不足しています")
        return

    if len(df_test) < 10:
        print(f"[ERROR] テストデータが不足しています")
        return

    # 前処理
    processor = Processor()
    df_train_processed = processor.process(df_train)
    df_test_processed = processor.process(df_test)

    print(f"\n【学習中...】")

    # 学習（SMOTEなし、アンサンブルなしで高速化）
    features = processor.features
    X_train = df_train_processed[features].fillna(-1)
    y_train = df_train_processed["target"]

    # LightGBMで学習
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    # 検証用AUC
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    val_pred = lgb_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"検証AUC: {val_auc:.4f}")

    # テストデータで予測
    X_test = df_test_processed[features].fillna(-1)
    df_test_processed["prob"] = lgb_model.predict_proba(X_test)[:, 1]

    # レースごとに集計
    results = []
    for race_id, race_df in df_test_processed.groupby("race_id"):
        race_df = race_df.sort_values("prob", ascending=False)

        # 予測1位
        pred_1st = race_df.iloc[0]
        pred_1st_num = int(pred_1st["horse_number"])
        pred_1st_prob = pred_1st["prob"]

        # 実際の結果
        if "rank" not in race_df.columns:
            continue

        actual_1st = race_df[race_df["rank"] == 1]
        actual_top3 = race_df[race_df["rank"] <= 3]["horse_number"].tolist()

        if len(actual_1st) == 0:
            continue

        actual_1st_num = int(actual_1st.iloc[0]["horse_number"])

        # 的中判定
        win_hit = pred_1st_num == actual_1st_num
        show_hit = pred_1st_num in actual_top3

        results.append({
            "race_id": race_id,
            "date": str(race_df.iloc[0]["date"].date()),
            "pred_1st": pred_1st_num,
            "pred_prob": pred_1st_prob,
            "actual_1st": actual_1st_num,
            "actual_top3": actual_top3,
            "win_hit": win_hit,
            "show_hit": show_hit,
        })

    if len(results) == 0:
        print("[ERROR] 結果データがありません")
        return

    # 集計
    results_df = pd.DataFrame(results)
    total_races = len(results_df)
    win_hits = results_df["win_hit"].sum()
    show_hits = results_df["show_hit"].sum()

    # 結果表示
    print(f"\n{'='*60}")
    print(f"【バックテスト結果】{track_info['name']}競馬場")
    print(f"{'='*60}")
    print(f"  学習期間: 〜{cutoff_date}")
    print(f"  テスト期間: {results_df['date'].min()} 〜 {results_df['date'].max()}")
    print(f"  テストレース数: {total_races}")
    print()
    print(f"【1位予測の成績】")
    print(f"  単勝的中: {win_hits}/{total_races} ({win_hits/total_races*100:.1f}%)")
    print(f"  複勝的中: {show_hits}/{total_races} ({show_hits/total_races*100:.1f}%)")
    print()

    # 日別成績
    print(f"【日別成績】")
    for date, day_df in results_df.groupby("date"):
        day_races = len(day_df)
        day_win = day_df["win_hit"].sum()
        day_show = day_df["show_hit"].sum()
        print(f"  {date}: {day_races}R, 単勝 {day_win}/{day_races} ({day_win/day_races*100:.0f}%), 複勝 {day_show}/{day_races} ({day_show/day_races*100:.0f}%)")

    return {
        "track": track_info["name"],
        "cutoff": cutoff_date,
        "total_races": total_races,
        "win_rate": win_hits / total_races * 100,
        "show_rate": show_hits / total_races * 100,
        "val_auc": val_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="完全バックテスト")
    parser.add_argument("--track", help="競馬場コード (44=大井, 45=川崎, all=両方)")
    parser.add_argument("--cutoff", default="2025-12-25", help="学習データのカットオフ日 (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.track == "all" or args.track is None:
        tracks = ["44", "45"]
    else:
        tracks = [args.track]

    results = []
    for track in tracks:
        if track in TRACKS:
            result = run_full_backtest(track, args.cutoff)
            if result:
                results.append(result)

    # サマリー
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("【サマリー】")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['track']}: 単勝 {r['win_rate']:.1f}%, 複勝 {r['show_rate']:.1f}%, AUC {r['val_auc']:.4f}")


if __name__ == "__main__":
    main()
