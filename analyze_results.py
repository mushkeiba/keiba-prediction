"""
予測結果の照合・分析スクリプト

使い方:
    python analyze_results.py <日付>
    python analyze_results.py 2025-12-30

機能:
    1. 予測ログと実際の結果を照合
    2. 正解/不正解をラベリング
    3. 誤答パターンを分析してレポート出力
"""

import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import urllib3
urllib3.disable_warnings()

BASE_DIR = Path(__file__).resolve().parent


def get_race_result(race_id: str) -> list:
    """レース結果（着順）を取得"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        r = session.get(url, timeout=10, verify=False)
        r.encoding = 'EUC-JP'
        soup = BeautifulSoup(r.text, 'lxml')

        results = []
        table = soup.find('table', class_='RaceTable01')
        if not table:
            table = soup.find('table', class_='Result_Table')
        if not table:
            return []

        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 3:
                continue

            rank_text = tds[0].get_text(strip=True)
            if not rank_text.isdigit():
                continue
            rank = int(rank_text)

            # 馬番を取得（tds[2]）
            umaban_text = tds[2].get_text(strip=True)
            if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                horse_num = int(umaban_text)
                results.append({"rank": rank, "number": horse_num})

        return sorted(results, key=lambda x: x["rank"])
    except Exception as e:
        print(f"  Error fetching result for {race_id}: {e}")
        return []


def compare_prediction_with_result(prediction_log: dict, result: list) -> dict:
    """予測と結果を照合"""
    if not result:
        return None

    predictions = prediction_log["predictions"]
    metadata = prediction_log.get("metadata", {})

    # 予測TOP3の馬番
    pred_top3 = [p["number"] for p in predictions[:3]]
    pred_1st = predictions[0]["number"] if predictions else None

    # 実際のTOP3の馬番
    actual_top3 = [r["number"] for r in result[:3]]
    actual_1st = result[0]["number"] if result else None

    # 判定
    win_hit = (pred_1st == actual_1st)  # 単勝的中
    show_hit = (pred_1st in actual_top3)  # 複勝的中
    top3_exact = (set(pred_top3) == set(actual_top3))  # 三連複的中

    # 予測1位の馬が実際に何着だったか
    pred_1st_actual_rank = None
    for r in result:
        if r["number"] == pred_1st:
            pred_1st_actual_rank = r["rank"]
            break

    return {
        "race_id": prediction_log["race_id"],
        "track_code": prediction_log["track_code"],
        "track_name": prediction_log.get("track_name", "不明"),
        "metadata": metadata,
        "pred_top3": pred_top3,
        "actual_top3": actual_top3,
        "pred_1st_prob": predictions[0]["prob"] if predictions else 0,
        "pred_1st_odds": predictions[0].get("odds", 0) if predictions else 0,
        "win_hit": win_hit,
        "show_hit": show_hit,
        "top3_exact": top3_exact,
        "pred_1st_actual_rank": pred_1st_actual_rank,
        "error_type": classify_error(predictions, result, metadata) if not show_hit else None
    }


def classify_error(predictions, result, metadata) -> str:
    """エラーの種類を分類"""
    if not predictions or not result:
        return "unknown"

    pred_1st = predictions[0]["number"]
    pred_1st_prob = predictions[0]["prob"]

    # 予測1位の実際の着順を取得
    actual_rank = None
    for r in result:
        if r["number"] == pred_1st:
            actual_rank = r["rank"]
            break

    # 分類
    if actual_rank is None:
        return "horse_not_in_result"  # 出走取消など
    elif actual_rank >= 10:
        return "big_miss"  # 大外れ（10着以下）
    elif actual_rank >= 6:
        return "moderate_miss"  # 中外れ（6-9着）
    elif actual_rank >= 4:
        return "near_miss"  # 惜しい（4-5着）
    else:
        return "unknown"


def analyze_errors(comparisons: list) -> dict:
    """誤答パターンを分析"""
    analysis = {
        "total_races": len(comparisons),
        "win_hits": sum(1 for c in comparisons if c["win_hit"]),
        "show_hits": sum(1 for c in comparisons if c["show_hit"]),
        "top3_exact": sum(1 for c in comparisons if c["top3_exact"]),
        "by_track_condition": defaultdict(lambda: {"total": 0, "show_hits": 0}),
        "by_weather": defaultdict(lambda: {"total": 0, "show_hits": 0}),
        "by_distance": defaultdict(lambda: {"total": 0, "show_hits": 0}),
        "by_prob_range": defaultdict(lambda: {"total": 0, "show_hits": 0}),
        "error_types": defaultdict(int)
    }

    for c in comparisons:
        meta = c.get("metadata", {})

        # 馬場状態別
        track_cond = meta.get("track_condition", "不明")
        analysis["by_track_condition"][track_cond]["total"] += 1
        if c["show_hit"]:
            analysis["by_track_condition"][track_cond]["show_hits"] += 1

        # 天気別
        weather = meta.get("weather", "不明")
        analysis["by_weather"][weather]["total"] += 1
        if c["show_hit"]:
            analysis["by_weather"][weather]["show_hits"] += 1

        # 距離別（短距離/中距離/長距離）
        distance = meta.get("distance", 0)
        if distance < 1400:
            dist_cat = "短距離(<1400m)"
        elif distance < 1800:
            dist_cat = "中距離(1400-1800m)"
        else:
            dist_cat = "長距離(>1800m)"
        analysis["by_distance"][dist_cat]["total"] += 1
        if c["show_hit"]:
            analysis["by_distance"][dist_cat]["show_hits"] += 1

        # 確率帯別
        prob = c.get("pred_1st_prob", 0)
        if prob >= 0.5:
            prob_cat = "高確率(>=50%)"
        elif prob >= 0.35:
            prob_cat = "中確率(35-50%)"
        else:
            prob_cat = "低確率(<35%)"
        analysis["by_prob_range"][prob_cat]["total"] += 1
        if c["show_hit"]:
            analysis["by_prob_range"][prob_cat]["show_hits"] += 1

        # エラータイプ集計
        if c.get("error_type"):
            analysis["error_types"][c["error_type"]] += 1

    return analysis


def print_report(analysis: dict, date_str: str):
    """分析レポートを出力"""
    print("\n" + "=" * 60)
    print(f"  誤答分析レポート: {date_str}")
    print("=" * 60)

    total = analysis["total_races"]
    if total == 0:
        print("データがありません")
        return

    win_rate = analysis["win_hits"] / total * 100
    show_rate = analysis["show_hits"] / total * 100

    print(f"\n【全体成績】")
    print(f"  レース数: {total}")
    print(f"  単勝的中: {analysis['win_hits']}/{total} ({win_rate:.1f}%)")
    print(f"  複勝的中: {analysis['show_hits']}/{total} ({show_rate:.1f}%)")
    print(f"  三連複的中: {analysis['top3_exact']}/{total}")

    # 馬場状態別
    print(f"\n【馬場状態別の複勝的中率】")
    for cond, data in sorted(analysis["by_track_condition"].items()):
        if data["total"] > 0:
            rate = data["show_hits"] / data["total"] * 100
            indicator = "!!" if rate < show_rate - 10 else ""
            print(f"  {cond}: {data['show_hits']}/{data['total']} ({rate:.1f}%) {indicator}")

    # 天気別
    print(f"\n【天気別の複勝的中率】")
    for weather, data in sorted(analysis["by_weather"].items()):
        if data["total"] > 0:
            rate = data["show_hits"] / data["total"] * 100
            indicator = "!!" if rate < show_rate - 10 else ""
            print(f"  {weather}: {data['show_hits']}/{data['total']} ({rate:.1f}%) {indicator}")

    # 距離別
    print(f"\n【距離別の複勝的中率】")
    for dist, data in sorted(analysis["by_distance"].items()):
        if data["total"] > 0:
            rate = data["show_hits"] / data["total"] * 100
            indicator = "!!" if rate < show_rate - 10 else ""
            print(f"  {dist}: {data['show_hits']}/{data['total']} ({rate:.1f}%) {indicator}")

    # 確率帯別
    print(f"\n【予測確率帯別の複勝的中率】")
    for prob_cat, data in sorted(analysis["by_prob_range"].items()):
        if data["total"] > 0:
            rate = data["show_hits"] / data["total"] * 100
            print(f"  {prob_cat}: {data['show_hits']}/{data['total']} ({rate:.1f}%)")

    # エラータイプ
    if analysis["error_types"]:
        print(f"\n【外れパターン】")
        for err_type, count in sorted(analysis["error_types"].items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count}件")

    print("\n" + "=" * 60)
    print("  !! = 平均より10%以上低い（改善ポイント）")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("使い方: python analyze_results.py <日付>")
        print("例: python analyze_results.py 2025-12-30")
        sys.exit(1)

    date_str = sys.argv[1]
    log_dir = BASE_DIR / "prediction_logs" / date_str

    if not log_dir.exists():
        print(f"予測ログが見つかりません: {log_dir}")
        sys.exit(1)

    print(f"予測ログを読み込み中: {log_dir}")

    # 予測ログを読み込み
    comparisons = []
    for log_file in log_dir.glob("*.json"):
        with open(log_file, 'r', encoding='utf-8') as f:
            prediction_log = json.load(f)

        race_id = prediction_log["race_id"]
        print(f"  {race_id} の結果を取得中...")

        # 結果を取得
        result = get_race_result(race_id)
        if result:
            comparison = compare_prediction_with_result(prediction_log, result)
            if comparison:
                comparisons.append(comparison)

    if not comparisons:
        print("照合できるデータがありません")
        sys.exit(1)

    # 分析
    analysis = analyze_errors(comparisons)

    # レポート出力
    print_report(analysis, date_str)

    # 結果をJSONに保存
    output_dir = BASE_DIR / "analysis_reports" / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "report.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        # defaultdictをdictに変換
        analysis_dict = {
            "date": date_str,
            "total_races": analysis["total_races"],
            "win_hits": analysis["win_hits"],
            "show_hits": analysis["show_hits"],
            "top3_exact": analysis["top3_exact"],
            "by_track_condition": dict(analysis["by_track_condition"]),
            "by_weather": dict(analysis["by_weather"]),
            "by_distance": dict(analysis["by_distance"]),
            "by_prob_range": dict(analysis["by_prob_range"]),
            "error_types": dict(analysis["error_types"]),
            "comparisons": comparisons
        }
        json.dump(analysis_dict, f, ensure_ascii=False, indent=2)

    print(f"\nレポート保存: {output_file}")


if __name__ == "__main__":
    main()
