"""
モデル評価スクリプト（バックテスト）

使い方:
    python evaluate_model.py <競馬場名>
    python evaluate_model.py 大井

機能:
    - 時系列でデータを分割（古いデータで学習、新しいデータでテスト）
    - 複数の学習期間を比較（6ヶ月/1年/2年/3年）
    - 回収率・的中率・AUCを計算
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent

TRACKS = {
    "大井": {"code": "44", "data": "data/races_ohi.csv"},
    "川崎": {"code": "45", "data": "data/races_kawasaki.csv"},
    "船橋": {"code": "43", "data": "data/races_funabashi.csv"},
    "浦和": {"code": "42", "data": "data/races_urawa.csv"},
    "門別": {"code": "30", "data": "data/races_monbetsu.csv"},
    "盛岡": {"code": "35", "data": "data/races_morioka.csv"},
    "水沢": {"code": "36", "data": "data/races_mizusawa.csv"},
    "金沢": {"code": "46", "data": "data/races_kanazawa.csv"},
    "笠松": {"code": "47", "data": "data/races_kasamatsu.csv"},
    "名古屋": {"code": "48", "data": "data/races_nagoya.csv"},
    "園田": {"code": "50", "data": "data/races_sonoda.csv"},
    "姫路": {"code": "51", "data": "data/races_himeji.csv"},
    "高知": {"code": "54", "data": "data/races_kochi.csv"},
    "佐賀": {"code": "55", "data": "data/races_saga.csv"},
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


def train_model(df):
    """モデル学習"""
    X = df[FEATURES].fillna(-1)
    y = df['target']

    model = lgb.train(
        {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        },
        lgb.Dataset(X, y),
        500,
        callbacks=[lgb.log_evaluation(0)]
    )

    return model


def evaluate_model(model, test_df):
    """モデル評価（回収率・的中率計算）"""
    X_test = test_df[FEATURES].fillna(-1)
    test_df = test_df.copy()
    test_df['prob'] = model.predict(X_test)

    results = {
        'total_races': 0,
        'win_bets': 0,
        'win_hits': 0,
        'win_payout': 0,
        'show_bets': 0,
        'show_hits': 0,
    }

    # レースごとに評価
    for race_id, race_df in test_df.groupby('race_id'):
        race_df = race_df.sort_values('prob', ascending=False)

        if len(race_df) < 3:
            continue

        results['total_races'] += 1

        # 予測1位の馬
        pred_1st = race_df.iloc[0]
        pred_1st_rank = pred_1st['rank']

        # 単勝: 予測1位に100円賭け
        results['win_bets'] += 100
        if pred_1st_rank == 1:
            results['win_hits'] += 1
            # 単勝オッズは持っていないので、暫定的に人気から推定
            # 実際のオッズがあればそれを使う
            # ここでは的中したら仮に300円返ってくると仮定（オッズ3倍）
            results['win_payout'] += 300

        # 複勝: 予測1位が3着以内
        results['show_bets'] += 100
        if pred_1st_rank <= 3:
            results['show_hits'] += 1

    # AUC計算
    try:
        auc = roc_auc_score(test_df['target'], test_df['prob'])
    except:
        auc = 0.5

    # 統計計算
    total = results['total_races']
    if total > 0:
        win_rate = results['win_hits'] / total * 100
        show_rate = results['show_hits'] / total * 100
        win_roi = results['win_payout'] / results['win_bets'] * 100 if results['win_bets'] > 0 else 0
    else:
        win_rate = show_rate = win_roi = 0

    return {
        'total_races': total,
        'win_rate': win_rate,
        'show_rate': show_rate,
        'win_roi': win_roi,
        'auc': auc
    }


def backtest(df, train_months_list, test_months=3):
    """バックテスト実行"""
    from datetime import datetime, timedelta

    # 日付でソート
    df = df.sort_values('race_date')

    # 日付範囲を取得（実際のカレンダー日付で計算）
    dates = sorted(df['race_date'].unique())
    if len(dates) < 50:
        print(f"データが少なすぎます: {len(dates)}日分")
        return []

    # 最新日付と最古日付を取得
    latest_date = dates[-1]  # YYYYMMDD形式
    oldest_date = dates[0]

    # 日付を datetime に変換
    latest_dt = datetime.strptime(str(latest_date), '%Y%m%d')
    oldest_dt = datetime.strptime(str(oldest_date), '%Y%m%d')

    # テスト期間: 直近N ヶ月（実際のカレンダー）
    test_start_dt = latest_dt - timedelta(days=test_months * 30)
    test_start_date = int(test_start_dt.strftime('%Y%m%d'))

    test_df = df[df['race_date'] >= test_start_date]

    print(f"\n【テスト期間】")
    print(f"  {test_start_date} 〜 {latest_date}")
    print(f"  レース数: {test_df['race_id'].nunique()}")
    print(f"  データ数: {len(test_df)}")

    results = []

    for train_months in train_months_list:
        # 学習開始日 = テスト開始日 - 学習期間
        train_start_dt = test_start_dt - timedelta(days=train_months * 30)
        train_start_date = int(train_start_dt.strftime('%Y%m%d'))

        # データが足りるか確認
        if train_start_dt < oldest_dt:
            print(f"\n[{train_months}ヶ月] データ不足でスキップ（{train_start_date} < {oldest_date}）")
            continue

        train_end_dt = test_start_dt - timedelta(days=1)
        train_end_date = int(train_end_dt.strftime('%Y%m%d'))

        train_df = df[(df['race_date'] >= train_start_date) & (df['race_date'] < test_start_date)]

        print(f"\n【{train_months}ヶ月モデル】")
        print(f"  学習期間: {train_start_date} 〜 {train_end_date}")
        print(f"  学習データ: {len(train_df)}件")

        if len(train_df) < 1000:
            print(f"  → データ不足でスキップ")
            continue

        # 前処理
        train_processed = preprocess(train_df)
        test_processed = preprocess(test_df)

        # 学習
        model = train_model(train_processed)

        # 評価
        metrics = evaluate_model(model, test_processed)
        metrics['train_months'] = train_months
        metrics['train_count'] = len(train_df)

        results.append(metrics)

        print(f"  結果:")
        print(f"    単勝的中率: {metrics['win_rate']:.1f}%")
        print(f"    複勝的中率: {metrics['show_rate']:.1f}%")
        print(f"    AUC: {metrics['auc']:.3f}")

    return results


def print_comparison(results):
    """結果比較表示"""
    if not results:
        print("\n結果がありません")
        return

    print("\n" + "=" * 60)
    print("  モデル比較結果")
    print("=" * 60)

    print("\n┌─────────┬──────────┬──────────┬──────────┐")
    print("│ 学習期間 │ 単勝的中 │ 複勝的中 │   AUC    │")
    print("├─────────┼──────────┼──────────┼──────────┤")

    best_show = max(results, key=lambda x: x['show_rate'])

    for r in results:
        is_best = r == best_show
        marker = " ★" if is_best else "  "
        print(f"│ {r['train_months']:>4}ヶ月 │ {r['win_rate']:>6.1f}% │ {r['show_rate']:>6.1f}%{marker}│ {r['auc']:>8.3f} │")

    print("└─────────┴──────────┴──────────┴──────────┘")
    print("\n★ = 複勝的中率が最も高いモデル")

    print(f"\n【推奨】{best_show['train_months']}ヶ月分のデータで学習")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("使い方: python evaluate_model.py <競馬場名>")
        print("例: python evaluate_model.py 大井")
        print("\n利用可能な競馬場:")
        for name in TRACKS.keys():
            print(f"  - {name}")
        sys.exit(1)

    track_name = sys.argv[1]

    if track_name not in TRACKS:
        print(f"エラー: 競馬場 '{track_name}' は存在しません")
        sys.exit(1)

    track_info = TRACKS[track_name]
    data_path = BASE_DIR / track_info['data']

    if not data_path.exists():
        print(f"エラー: データファイルがありません: {data_path}")
        sys.exit(1)

    print(f"【{track_name}競馬場 モデル評価】")
    print(f"データ: {data_path}")

    # データ読み込み
    df = pd.read_csv(data_path)
    print(f"総データ数: {len(df)}件")

    # 比較する学習期間（月）
    train_months_list = [6, 12, 18, 24, 30, 36]

    # バックテスト実行
    results = backtest(df, train_months_list, test_months=3)

    # 結果表示
    print_comparison(results)


if __name__ == "__main__":
    main()
