#!/usr/bin/env python3
"""
バックテスト: 学習データに含まれない日付でモデルを検証

使い方:
  python scripts/backtest.py --track 44 --days 7
  python scripts/backtest.py --track 45 --start 2025-12-26 --end 2025-12-31
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# プロジェクトルート
BASE_DIR = Path(__file__).parent.parent

# 競馬場設定
TRACKS = {
    "44": {"name": "大井", "model": "models/model_ohi.pkl", "data": "data/races_ohi.csv"},
    "45": {"name": "川崎", "model": "models/model_kawasaki.pkl", "data": "data/races_kawasaki.csv"},
}


def load_model(track_code: str):
    """モデルと特徴量を読み込み"""
    if track_code not in TRACKS:
        print(f"[ERROR] 不明な競馬場コード: {track_code}")
        return None, None, None

    model_path = BASE_DIR / TRACKS[track_code]["model"]
    meta_path = model_path.parent / (model_path.stem + "_meta.json")

    if not model_path.exists():
        print(f"[ERROR] モデルファイルがありません: {model_path}")
        return None, None, None

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    features = data["features"]

    # メタデータから学習期間を取得
    training_end = None
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            training_end = meta.get("date_range", {}).get("to") or meta.get("training_period", {}).get("end")

    return model, features, training_end


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


def run_backtest(track_code: str, start_date: str, end_date: str):
    """バックテスト実行"""
    print(f"\n{'='*60}")
    print(f"バックテスト: {TRACKS[track_code]['name']}競馬場")
    print(f"期間: {start_date} 〜 {end_date}")
    print(f"{'='*60}\n")

    # モデル読み込み
    model, features, training_end = load_model(track_code)
    if model is None:
        return

    print(f"モデル学習期間終了: {training_end}")

    # 学習データに含まれる日付をチェック
    if training_end:
        training_end_dt = datetime.strptime(training_end, "%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if start_dt <= training_end_dt:
            print(f"\n[WARNING] テスト期間が学習データと重複しています！")
            print(f"  学習データ終了: {training_end}")
            print(f"  テスト開始: {start_date}")
            print(f"  → 正しい検証のため、{training_end} より後の日付を使用してください\n")

    # データ読み込み
    data_path = BASE_DIR / TRACKS[track_code]["data"]
    if not data_path.exists():
        print(f"[ERROR] データファイルがありません: {data_path}")
        return

    df = pd.read_csv(data_path)

    # 日付フィルタ (race_id形式: YYYYTTMMDDNN → 年4桁 + 競馬場2桁 + 月日4桁 + レース番号2桁)
    race_id_str = df["race_id"].astype(str)
    df["date"] = pd.to_datetime(race_id_str.str[:4] + race_id_str.str[6:10], format="%Y%m%d")
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if len(df) == 0:
        print(f"[ERROR] 指定期間のデータがありません")
        return

    # 欠損特徴量を補完
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # 予測
    X = df[features].fillna(-1)
    df["prob"] = predict_with_model(model, X)

    # レースごとに集計
    results = []
    for race_id, race_df in df.groupby("race_id"):
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

        # オッズ取得
        win_odds = float(pred_1st.get("win_odds", 0) or 0)
        place_odds = float(pred_1st.get("place_odds", 0) or 0)
        if place_odds == 0:
            place_odds = win_odds / 3 if win_odds > 0 else 0

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
            "win_odds": win_odds,
            "place_odds": place_odds,
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

    # ROI計算（単勝）
    bet_per_race = 100
    total_bet = total_races * bet_per_race
    win_payout = results_df[results_df["win_hit"]]["win_odds"].sum() * bet_per_race
    win_roi = (win_payout / total_bet) * 100 if total_bet > 0 else 0

    # ROI計算（複勝）
    show_payout = results_df[results_df["show_hit"]]["place_odds"].sum() * bet_per_race
    show_roi = (show_payout / total_bet) * 100 if total_bet > 0 else 0

    # 結果表示
    print(f"【検証結果】")
    print(f"  レース数: {total_races}")
    print(f"  テスト期間: {results_df['date'].min()} 〜 {results_df['date'].max()}")
    print()
    print(f"【1位予測の成績】")
    print(f"  単勝的中: {win_hits}/{total_races} ({win_hits/total_races*100:.1f}%)")
    print(f"  複勝的中: {show_hits}/{total_races} ({show_hits/total_races*100:.1f}%)")
    print()
    print(f"【回収率】(1レース{bet_per_race}円ずつ賭けた場合)")
    print(f"  単勝回収率: {win_roi:.1f}%")
    print(f"  複勝回収率: {show_roi:.1f}%")
    print()

    # 日別成績
    print(f"【日別成績】")
    for date, day_df in results_df.groupby("date"):
        day_races = len(day_df)
        day_win = day_df["win_hit"].sum()
        day_show = day_df["show_hit"].sum()
        day_win_payout = day_df[day_df["win_hit"]]["win_odds"].sum() * bet_per_race
        day_roi = (day_win_payout / (day_races * bet_per_race)) * 100
        print(f"  {date}: {day_races}R, 単勝{day_win}/{day_races}, 複勝{day_show}/{day_races}, ROI {day_roi:.0f}%")

    # 学習データとの重複警告
    if training_end:
        overlap = results_df[results_df["date"] <= training_end]
        if len(overlap) > 0:
            print(f"\n[WARNING] {len(overlap)}レースが学習データと重複（結果の信頼性低）")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="競馬予測モデルのバックテスト")
    parser.add_argument("--track", required=True, help="競馬場コード (44=大井, 45=川崎)")
    parser.add_argument("--start", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="過去N日間をテスト")

    args = parser.parse_args()

    # 日付設定
    if args.end:
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if args.start:
        start_date = args.start
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=args.days)
        start_date = start_dt.strftime("%Y-%m-%d")

    run_backtest(args.track, start_date, end_date)


if __name__ == "__main__":
    main()
