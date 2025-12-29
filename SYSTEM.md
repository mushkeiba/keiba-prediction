# 地方競馬AI予測システム

## システム構成

```
┌─────────────────────────┐     ┌─────────────────────────┐
│   フロントエンド         │     │   バックエンド           │
│   (Next.js + shadcn/ui) │ ──→ │   (FastAPI + LightGBM)  │
│                         │     │                         │
│   Vercel               │     │   Render                │
└─────────────────────────┘     └─────────────────────────┘
```

## URL一覧

| サービス | URL |
|---------|-----|
| **Webアプリ** | https://keiba-web-chi.vercel.app |
| **API** | https://keiba-prediction.onrender.com |
| **API確認** | https://keiba-prediction.onrender.com/api/tracks |

## GitHubリポジトリ

| リポジトリ | URL | 用途 |
|-----------|-----|------|
| keiba-prediction | https://github.com/mushkeiba/keiba-prediction | バックエンド（API + モデル） |
| keiba-web | https://github.com/mushkeiba/keiba-web | フロントエンド（Next.js） |

## ローカルパス

| プロジェクト | パス |
|-------------|------|
| keiba-prediction | `C:\Users\morimoto_sei\Documents\GitHub\keiba-prediction` |
| keiba-web | `C:\Users\morimoto_sei\Documents\GitHub\keiba-web` |

## 技術スタック

### フロントエンド (keiba-web)
- Next.js 15
- TypeScript
- Tailwind CSS v4
- shadcn/ui コンポーネント
- ダークモード対応

### バックエンド (keiba-prediction)
- FastAPI
- LightGBM（予測モデル）
- BeautifulSoup4（スクレイピング）
- Docker（デプロイ用）

## 対応競馬場（14場）

| コード | 競馬場 | モデル |
|--------|--------|--------|
| 44 | 大井 | model_v2.pkl (有) |
| 45 | 川崎 | 未作成 |
| 43 | 船橋 | 未作成 |
| 42 | 浦和 | 未作成 |
| 30 | 門別 | 未作成 |
| 35 | 盛岡 | 未作成 |
| 36 | 水沢 | 未作成 |
| 46 | 金沢 | 未作成 |
| 47 | 笠松 | 未作成 |
| 48 | 名古屋 | 未作成 |
| 50 | 園田 | 未作成 |
| 51 | 姫路 | 未作成 |
| 54 | 高知 | 未作成 |
| 55 | 佐賀 | 未作成 |

## デプロイ情報

### Vercel（フロントエンド）
- アカウント: mushkeiba (GitHub連携)
- プラン: Hobby（無料）
- 環境変数: `NEXT_PUBLIC_API_URL=https://keiba-prediction.onrender.com`

### Render（バックエンド）
- アカウント: steam87mushroom@gmail.com
- プラン: Free
- 無料枠: 750時間/月、100GB帯域幅
- 自動デプロイ: On Commit

## ローカル開発

### フロントエンド起動
```bash
cd C:\Users\morimoto_sei\Documents\GitHub\keiba-web
npm run dev
# http://localhost:3000
```

### バックエンド起動
```bash
cd C:\Users\morimoto_sei\Documents\GitHub\keiba-prediction
pip install -r requirements.txt
uvicorn api.main:app --reload
# http://localhost:8000
```

## 今後のTODO

- [ ] 他の競馬場のモデル作成（GitHub Actionsで自動化設定済み）
- [ ] モデル精度の改善
- [ ] 予測履歴機能
- [ ] ユーザー認証（任意）

## 注意事項

- Renderの無料プランは15分アクセスがないとスリープする（初回アクセス時に50秒程度かかる）
- モデルファイル（.pkl）はGitHubにコミット済み
- netkeibaへのスクレイピングは適度な間隔（0.5秒）を空けている
