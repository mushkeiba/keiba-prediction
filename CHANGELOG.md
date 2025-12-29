# CHANGELOG

## 2025-12-29
### 機能追加: FastAPI バックエンド
**変更内容**:
- `api/main.py` を新規作成
  - Next.jsフロントエンドから呼び出せるREST APIを実装
  - エンドポイント:
    - `GET /api/tracks` - 競馬場一覧取得
    - `POST /api/predict` - 予測実行
  - CORS対応
  - 既存のスクレイパー・前処理ロジックを流用

- `requirements.txt` にFastAPI/uvicorn依存関係を追加
  ```
  fastapi>=0.104.0
  uvicorn>=0.24.0
  ```

### 機能追加: デプロイ設定
**変更内容**:
- `Dockerfile` を新規作成 - Docker コンテナ化
- `render.yaml` を新規作成 - Render.com デプロイ設定
