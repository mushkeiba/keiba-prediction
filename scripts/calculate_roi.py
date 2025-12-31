"""
回収率計算スクリプト

使い方:
    python calculate_roi.py 2025-12-30
    python calculate_roi.py 2025-12-30 --all  # 過去全日分

機能:
    - 予測ログと実際の結果を照合
    - 実際のオッズを使って回収率を計算
    - 単勝・複勝それぞれの成績を出力
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

BASE_DIR = Path(__file__).resolve().parent


def get_race_result(race_id: str) -> dict:
    """レース結果を取得"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        r = session.get(url, timeout=30, verify=False)
        r.encoding = 'EUC-JP'
        soup = BeautifulSoup(r.text, 'lxml')

        results = {}
        table = soup.find('table', class_='RaceTable01')
        if table:
            for tr in table.find_all('tr')[1:]:  # ヘッダーをスキップ
                tds = tr.find_all('td')
                if len(tds) >= 3:
                    rank_text = tds[0].get_text(strip=True)
                    if rank_text.isdigit():
                        rank = int(rank_text)
                        horse_num_text = tds[2].get_text(strip=True)
                        if horse_num_text.isdigit():
                            horse_num = int(horse_num_text)
                            results[horse_num] = rank

        return results
    except Exception as e:
        print(f"  結果取得エラー ({race_id}): {e}")
        return {}


def calculate_roi_for_date(date_str: str) -> dict:
    """指定日の回収率を計算"""
    log_dir = BASE_DIR / "prediction_logs" / date_str

    if not log_dir.exists():
        return {"error": f"ログがありません: {date_str}"}

    log_files = list(log_dir.glob("*.json"))
    if not log_files:
        return {"error": f"ログファイルがありません: {date_str}"}

    # 結果を集計
    stats = {
        "date": date_str,
        "total_races": 0,
        "win": {"bets": 0, "hits": 0, "payout": 0},
        "show": {"bets": 0, "hits": 0, "payout": 0},
        "value_win": {"bets": 0, "hits": 0, "payout": 0},  # 妙味馬のみ
        "value_show": {"bets": 0, "hits": 0, "payout": 0},
        "details": []
    }

    print(f"\n【{date_str}】回収率計算中...")
    print(f"  ログファイル数: {len(log_files)}")

    for log_file in sorted(log_files):
        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        race_id = log_data["race_id"]
        predictions = log_data["predictions"]

        if not predictions:
            continue

        # 結果を取得
        results = get_race_result(race_id)
        if not results:
            continue

        stats["total_races"] += 1

        # AI予測1位の馬
        pred_1st = predictions[0]
        horse_num = pred_1st["number"]
        odds = pred_1st.get("odds", 0)
        is_value = pred_1st.get("is_value", False)

        actual_rank = results.get(horse_num, 99)

        # 単勝（100円賭け）
        stats["win"]["bets"] += 100
        if actual_rank == 1 and odds > 0:
            stats["win"]["hits"] += 1
            stats["win"]["payout"] += int(odds * 100)

        # 複勝（100円賭け） - 複勝オッズは単勝の約1/3と仮定
        show_odds = odds / 3 if odds > 0 else 1.2
        show_odds = max(show_odds, 1.1)  # 最低1.1倍
        stats["show"]["bets"] += 100
        if actual_rank <= 3 and odds > 0:
            stats["show"]["hits"] += 1
            stats["show"]["payout"] += int(show_odds * 100)

        # 妙味馬のみの成績
        if is_value:
            stats["value_win"]["bets"] += 100
            if actual_rank == 1 and odds > 0:
                stats["value_win"]["hits"] += 1
                stats["value_win"]["payout"] += int(odds * 100)

            stats["value_show"]["bets"] += 100
            if actual_rank <= 3:
                stats["value_show"]["hits"] += 1
                stats["value_show"]["payout"] += int(show_odds * 100)

        # 詳細記録
        stats["details"].append({
            "race_id": race_id,
            "horse_num": horse_num,
            "horse_name": pred_1st.get("name", "不明"),
            "prob": pred_1st.get("prob", 0),
            "odds": odds,
            "is_value": is_value,
            "actual_rank": actual_rank,
            "win_hit": actual_rank == 1,
            "show_hit": actual_rank <= 3
        })

    return stats


def print_report(stats: dict):
    """レポートを出力"""
    if "error" in stats:
        print(f"エラー: {stats['error']}")
        return

    print("\n" + "=" * 60)
    print(f"  回収率レポート: {stats['date']}")
    print("=" * 60)

    print(f"\n対象レース数: {stats['total_races']}")

    # 全体
    print("\n【全予測】")
    for bet_type, name in [("win", "単勝"), ("show", "複勝")]:
        data = stats[bet_type]
        if data["bets"] > 0:
            hit_rate = data["hits"] / (data["bets"] / 100) * 100
            roi = data["payout"] / data["bets"] * 100
            print(f"  {name}: {data['hits']}的中 / {data['bets']//100}レース "
                  f"({hit_rate:.1f}%) → 回収率 {roi:.0f}%")

    # 妙味馬のみ
    if stats["value_win"]["bets"] > 0:
        print("\n【妙味馬のみ】")
        for bet_type, name in [("value_win", "単勝"), ("value_show", "複勝")]:
            data = stats[bet_type]
            if data["bets"] > 0:
                hit_rate = data["hits"] / (data["bets"] / 100) * 100
                roi = data["payout"] / data["bets"] * 100
                print(f"  {name}: {data['hits']}的中 / {data['bets']//100}レース "
                      f"({hit_rate:.1f}%) → 回収率 {roi:.0f}%")

    # 結論
    print("\n" + "-" * 60)
    show_roi = stats["show"]["payout"] / stats["show"]["bets"] * 100 if stats["show"]["bets"] > 0 else 0
    if show_roi >= 100:
        print(f"結論: 💰 黒字！ (複勝回収率 {show_roi:.0f}%)")
    elif show_roi >= 80:
        print(f"結論: 📊 惜しい (複勝回収率 {show_roi:.0f}%)")
    else:
        print(f"結論: 📉 要改善 (複勝回収率 {show_roi:.0f}%)")
    print("=" * 60)


def save_report(stats: dict):
    """レポートをJSONで保存"""
    if "error" in stats:
        return

    output_dir = BASE_DIR / "roi_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{stats['date']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nレポート保存: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("使い方: python calculate_roi.py <日付>")
        print("例: python calculate_roi.py 2025-12-30")
        print("例: python calculate_roi.py --all  # 全日分")
        sys.exit(1)

    if sys.argv[1] == "--all":
        # 全日分を計算
        log_base = BASE_DIR / "prediction_logs"
        if not log_base.exists():
            print("予測ログがありません")
            sys.exit(1)

        all_stats = []
        for date_dir in sorted(log_base.iterdir()):
            if date_dir.is_dir():
                stats = calculate_roi_for_date(date_dir.name)
                if "error" not in stats:
                    all_stats.append(stats)
                    print_report(stats)
                    save_report(stats)

        # サマリー
        if all_stats:
            print("\n" + "=" * 60)
            print("  全期間サマリー")
            print("=" * 60)
            total_win_bets = sum(s["win"]["bets"] for s in all_stats)
            total_win_payout = sum(s["win"]["payout"] for s in all_stats)
            total_show_bets = sum(s["show"]["bets"] for s in all_stats)
            total_show_payout = sum(s["show"]["payout"] for s in all_stats)

            if total_win_bets > 0:
                print(f"単勝回収率: {total_win_payout / total_win_bets * 100:.0f}%")
            if total_show_bets > 0:
                print(f"複勝回収率: {total_show_payout / total_show_bets * 100:.0f}%")

    else:
        date_str = sys.argv[1]
        stats = calculate_roi_for_date(date_str)
        print_report(stats)
        save_report(stats)


if __name__ == "__main__":
    main()
