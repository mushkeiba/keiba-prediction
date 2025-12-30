#!/usr/bin/env python3
"""
予測精度を評価するスクリプト
予測結果と実際のレース結果を比較して的中率を計算
"""
import json
import sys
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
import re
import urllib3

# SSL警告を抑制
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# プロジェクトのルートディレクトリ
BASE_DIR = Path(__file__).resolve().parent

# 競馬場コード
TRACKS = {
    "44": "大井", "45": "川崎", "43": "船橋", "42": "浦和",
    "30": "門別", "35": "盛岡", "36": "水沢", "46": "金沢",
    "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路",
    "54": "高知", "55": "佐賀"
}


def fetch_race_result(race_id: str) -> dict:
    """レース結果を取得"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"

    try:
        time.sleep(0.5)
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        r = session.get(url, verify=False)  # SSL検証をスキップ
        r.encoding = 'EUC-JP'
        soup = BeautifulSoup(r.text, 'lxml')

        results = []

        # 結果テーブルを探す
        table = soup.find('table', class_='RaceTable01')
        if not table:
            table = soup.find('table', class_='Result_Table')
        if not table:
            for t in soup.find_all('table'):
                if t.find('a', href=re.compile(r'/horse/')):
                    table = t
                    break

        if not table:
            return {"error": "結果テーブルが見つかりません"}

        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 3:
                continue

            # 着順を取得（最初のtd）
            rank_text = tds[0].get_text(strip=True)
            if not rank_text.isdigit():
                continue
            rank = int(rank_text)

            # 馬番を取得（通常2-3番目のtd）
            horse_num = None
            for td in tds[1:4]:
                text = td.get_text(strip=True)
                if text.isdigit() and 1 <= int(text) <= 18:
                    horse_num = int(text)
                    break

            # 馬名を取得
            horse_name = None
            horse_link = tr.find('a', href=re.compile(r'/horse/\d+'))
            if horse_link:
                horse_name = horse_link.get_text(strip=True)

            # 単勝オッズを取得（払戻金テーブルから後で取得）

            if horse_num:
                results.append({
                    "rank": rank,
                    "number": horse_num,
                    "name": horse_name or "不明"
                })

        # 単勝払戻金を取得
        win_odds = None
        payout_table = soup.find('table', class_='Payout_Detail_Table')
        if payout_table:
            for tr in payout_table.find_all('tr'):
                th = tr.find('th')
                if th and '単勝' in th.get_text():
                    tds = tr.find_all('td')
                    if len(tds) >= 2:
                        payout_text = tds[1].get_text(strip=True)
                        payout_match = re.search(r'([\d,]+)', payout_text)
                        if payout_match:
                            payout = int(payout_match.group(1).replace(',', ''))
                            win_odds = payout / 100

        return {
            "race_id": race_id,
            "results": sorted(results, key=lambda x: x["rank"]),
            "win_odds": win_odds
        }

    except Exception as e:
        return {"error": str(e)}


def evaluate_race(prediction: dict, result: dict) -> dict:
    """1レースの予測精度を評価"""
    if "error" in result or not result.get("results"):
        return {"error": "結果取得失敗"}

    pred_rankings = prediction.get("predictions", [])
    actual_results = result.get("results", [])

    if not pred_rankings or not actual_results:
        return {"error": "データ不足"}

    # 予測1位の馬番
    pred_1st = pred_rankings[0]["number"] if pred_rankings else None
    # 予測TOP3の馬番
    pred_top3 = set(p["number"] for p in pred_rankings[:3])

    # 実際の1着の馬番
    actual_1st = actual_results[0]["number"] if actual_results else None
    # 実際のTOP3の馬番
    actual_top3 = set(r["number"] for r in actual_results[:3])

    # 的中判定
    win_hit = pred_1st == actual_1st  # 単勝的中
    show_hit = pred_1st in actual_top3  # 複勝的中（予測1位が3着以内）

    # TOP3一致数
    top3_matches = len(pred_top3 & actual_top3)

    # 回収率計算（予測1位に100円賭けた場合）
    roi = 0
    if win_hit and result.get("win_odds"):
        roi = result["win_odds"] * 100  # 払戻金

    return {
        "race_id": prediction.get("race_id", ""),
        "race_num": prediction.get("id", ""),
        "pred_1st": pred_1st,
        "actual_1st": actual_1st,
        "win_hit": win_hit,
        "show_hit": show_hit,
        "top3_matches": top3_matches,
        "bet": 100,
        "payout": roi,
        "win_odds": result.get("win_odds", 0)
    }


def evaluate_track(date_str: str, track_code: str) -> dict:
    """1競馬場の予測精度を評価"""
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    # 予測ファイルを読み込み
    pred_file = BASE_DIR / "predictions" / date_formatted / f"{track_code}.json"
    if not pred_file.exists():
        return {"error": f"予測ファイルがありません: {pred_file}"}

    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    track_name = predictions.get("track", {}).get("name", TRACKS.get(track_code, "不明"))
    races = predictions.get("races", [])

    print(f"\n評価中: {track_name} ({len(races)}レース)")

    evaluations = []
    total_bet = 0
    total_payout = 0
    win_hits = 0
    show_hits = 0

    for race in races:
        race_id = race.get("race_id", "")
        if not race_id:
            continue

        print(f"  {race['id']}R: ", end="")

        # 結果取得
        result = fetch_race_result(race_id)
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            continue

        # 評価
        eval_result = evaluate_race(race, result)
        if "error" in eval_result:
            print(f"[ERROR] {eval_result['error']}")
            continue

        evaluations.append(eval_result)
        total_bet += eval_result["bet"]
        total_payout += eval_result["payout"]

        if eval_result["win_hit"]:
            win_hits += 1
            print(f"[HIT!] Tansho! (Payout: {eval_result['payout']}yen)")
        elif eval_result["show_hit"]:
            show_hits += 1
            print(f"[OK] Fukusho (Pred 1st -> Actual {eval_result['top3_matches']}th)")
        else:
            print(f"[MISS] Pred:{eval_result['pred_1st']} -> Actual:{eval_result['actual_1st']}")

    # 集計
    race_count = len(evaluations)
    if race_count == 0:
        return {"error": "評価可能なレースがありません"}

    summary = {
        "date": date_formatted,
        "track_code": track_code,
        "track_name": track_name,
        "race_count": race_count,
        "win_hits": win_hits,
        "show_hits": show_hits,
        "win_rate": round(win_hits / race_count * 100, 1),
        "show_rate": round((win_hits + show_hits) / race_count * 100, 1),
        "total_bet": total_bet,
        "total_payout": total_payout,
        "roi": round(total_payout / total_bet * 100, 1) if total_bet > 0 else 0,
        "evaluations": evaluations
    }

    return summary


def main():
    # 使い方: python evaluate_predictions.py [日付] [競馬場コード]
    # 例: python evaluate_predictions.py 2025-12-30 44

    if len(sys.argv) < 2:
        print("使い方: python evaluate_predictions.py <日付> [競馬場コード]")
        print("例: python evaluate_predictions.py 2025-12-30 44")
        return

    date_str = sys.argv[1].replace("-", "")
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    # 競馬場コード指定（オプション）
    track_codes = sys.argv[2:] if len(sys.argv) > 2 else []

    # 出力ディレクトリ
    output_dir = BASE_DIR / "accuracy" / date_formatted
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"予測精度評価: {date_formatted}")

    # 対象の競馬場を決定
    if track_codes:
        targets = [(code, TRACKS.get(code, "不明")) for code in track_codes if code in TRACKS]
    else:
        # 予測ファイルがある競馬場を自動検出
        pred_dir = BASE_DIR / "predictions" / date_formatted
        if not pred_dir.exists():
            print(f"予測ディレクトリがありません: {pred_dir}")
            return
        targets = []
        for f in pred_dir.glob("*.json"):
            code = f.stem
            if code in TRACKS:
                targets.append((code, TRACKS[code]))

    if not targets:
        print("評価対象の競馬場がありません")
        return

    # 全体集計用
    all_summaries = []

    for track_code, track_name in targets:
        summary = evaluate_track(date_str, track_code)

        if "error" not in summary:
            # 個別保存
            output_file = output_dir / f"{track_code}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n保存: {output_file}")

            all_summaries.append(summary)

            # 結果表示
            print(f"\n{'='*40}")
            print(f"[RESULT] {track_name}")
            print(f"{'='*40}")
            print(f"レース数: {summary['race_count']}")
            print(f"単勝的中: {summary['win_hits']}回 ({summary['win_rate']}%)")
            print(f"複勝的中: {summary['win_hits'] + summary['show_hits']}回 ({summary['show_rate']}%)")
            print(f"投資: {summary['total_bet']}円 → 回収: {summary['total_payout']}円")
            print(f"回収率: {summary['roi']}%")

    # 全体サマリー保存
    if all_summaries:
        total_races = sum(s["race_count"] for s in all_summaries)
        total_win_hits = sum(s["win_hits"] for s in all_summaries)
        total_show_hits = sum(s["show_hits"] for s in all_summaries)
        total_bet = sum(s["total_bet"] for s in all_summaries)
        total_payout = sum(s["total_payout"] for s in all_summaries)

        daily_summary = {
            "date": date_formatted,
            "tracks": [s["track_name"] for s in all_summaries],
            "total_races": total_races,
            "total_win_hits": total_win_hits,
            "total_show_hits": total_show_hits,
            "win_rate": round(total_win_hits / total_races * 100, 1) if total_races > 0 else 0,
            "show_rate": round((total_win_hits + total_show_hits) / total_races * 100, 1) if total_races > 0 else 0,
            "total_bet": total_bet,
            "total_payout": total_payout,
            "roi": round(total_payout / total_bet * 100, 1) if total_bet > 0 else 0
        }

        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(daily_summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*50}")
        print(f"[TOTAL] {date_formatted}")
        print(f"{'='*50}")
        print(f"競馬場: {', '.join(daily_summary['tracks'])}")
        print(f"レース数: {total_races}")
        print(f"単勝的中率: {daily_summary['win_rate']}%")
        print(f"複勝的中率: {daily_summary['show_rate']}%")
        print(f"回収率: {daily_summary['roi']}%")


if __name__ == "__main__":
    main()
