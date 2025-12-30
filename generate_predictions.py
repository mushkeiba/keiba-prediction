#!/usr/bin/env python3
"""
毎朝実行: 全競馬場の予測を事前計算してJSONに保存
オッズ以外の情報（馬名、騎手、AI予測確率など）を保存
"""
import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# api/main.py から必要なクラスをインポート
sys.path.insert(0, str(Path(__file__).parent))
from api.main import NARScraper, Processor, TRACKS, BASE_DIR


def load_model(track_code: str):
    """モデル読み込み"""
    if track_code not in TRACKS:
        return None, None

    model_path = BASE_DIR / TRACKS[track_code]['model']
    if model_path.exists():
        with open(model_path, 'rb') as f:
            d = pickle.load(f)
        return d['model'], d['features']
    return None, None


def generate_predictions_for_track(track_code: str, date_str: str) -> dict:
    """1競馬場の予測を生成"""
    model, model_features = load_model(track_code)
    if model is None:
        print(f"  モデルなし: {TRACKS[track_code]['name']}")
        return None

    scraper = NARScraper(track_code, delay=0.5)
    processor = Processor()

    # レース一覧取得
    race_ids = scraper.get_race_list_by_date(date_str)
    if not race_ids:
        print(f"  レースなし: {TRACKS[track_code]['name']}")
        return None

    print(f"  {len(race_ids)}レース検出: {TRACKS[track_code]['name']}")

    races = []
    for rid in sorted(race_ids):
        df = scraper.get_race_data(rid)
        if df is None:
            continue

        df = scraper.enrich_data(df)
        df = processor.process(df)

        # 予測実行
        X = df[model_features].fillna(-1)
        df['prob'] = model.predict(X)
        df['pred_rank'] = df['prob'].rank(ascending=False, method='min').astype(int)
        df = df.sort_values('prob', ascending=False)

        # レース情報
        race_num = rid[-2:]
        race_name = df['race_name'].iloc[0] if 'race_name' in df.columns else f"{race_num}R"
        distance = int(df['distance'].iloc[0]) if 'distance' in df.columns else 0
        start_time = df['start_time'].iloc[0] if 'start_time' in df.columns else ""

        predictions = []
        for i, (_, row) in enumerate(df.iterrows()):
            horse_num = int(row['horse_number']) if pd.notna(row.get('horse_number')) else 0
            prob = float(row['prob'])

            raw_win_rate = float(row.get('horse_win_rate') or 0)
            raw_show_rate = float(row.get('horse_show_rate') or 0)

            predictions.append({
                "rank": i + 1,
                "number": horse_num,
                "name": row.get('horse_name', '不明'),
                "jockey": row.get('jockey_name', '不明'),
                "prob": round(prob, 3),
                "win_rate": round(raw_win_rate * 100, 1),
                "show_rate": round(raw_show_rate * 100, 1),
            })

        races.append({
            "id": race_num,
            "race_id": rid,  # フルIDも保存（オッズ取得用）
            "name": race_name,
            "distance": distance,
            "time": start_time,
            "field_size": len(df),
            "predictions": predictions
        })

        print(f"    {race_num}R: {len(predictions)}頭")

    if not races:
        return None

    return {
        "track": {
            "code": track_code,
            "name": TRACKS[track_code]['name'],
            "emoji": TRACKS[track_code]['emoji']
        },
        "generated_at": datetime.now().isoformat(),
        "races": races
    }


def main():
    # 日付指定（引数 or 今日）
    if len(sys.argv) >= 2:
        date_str = sys.argv[1].replace("-", "")
    else:
        date_str = datetime.now().strftime("%Y%m%d")

    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    print(f"予測生成: {date_formatted}")

    # 出力ディレクトリ
    output_dir = BASE_DIR / "predictions" / date_formatted
    output_dir.mkdir(parents=True, exist_ok=True)

    # 全競馬場を処理
    for track_code, track_info in TRACKS.items():
        print(f"\n処理中: {track_info['name']} ({track_code})")

        result = generate_predictions_for_track(track_code, date_str)
        if result:
            output_file = output_dir / f"{track_code}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  保存: {output_file}")

    print(f"\n完了: {output_dir}")


if __name__ == "__main__":
    main()
