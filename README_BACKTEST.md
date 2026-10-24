# 30K MACD 回測機制 (backtest_30k.py / event_study_30k.py)

## 硬規矩 (不可妥協)

- **訊號與出場判定與實盤同一份邏輯**：訊號直接呼叫 `scanner.evaluate_signal()`；
  出場逐項移植 `scanner.update_and_check_positions()` 的 5 種判定與順序
  (硬停損/保本線 → TP2 → TP1 減碼 50% + 停損移至成本+0.5% → 死叉破 20MA →
  盤後時間停損)。上一款 0050 期貨的教訓：回測出場 ≠ 實盤出場 = 假數字。
- **只用已收盤 K**：歷史資料全為已收盤 K；yfinance 即時尾巴沿用
  `scanner.drop_incomplete_last_bar()`。
- **進場價 = 訊號 K 收盤** (實盤 daemon 在 K 收盤後數分鐘內市價進場之近似)，
  出場檢查自下一根開始，不用同一根判定出場。
- **不改實盤參數**：本目錄程式只讀取 `scanner.py` 的常數與函數，
  不修改 `scanner.py` 的訊號/出場邏輯。

## 已知與實盤的差異 (誠實揭露)

1. **成本口徑**：回測用買 0.1425% / 賣 0.4425% (合計 0.585%)；
   實盤帳本用 `FEE_TAX_RATE = 0.4%` 單次。回測較保守 0.185pp。
2. **一檔一部位**：回測一檔同時最多一部位，持有期間新訊號忽略；
   實盤 daemon 會以新訊號覆寫既有部位 (不檢查 active)。回測較保守。
3. **無每掃描 5 檔上限**：實盤盤中每 3 分鐘掃描取 score 前 5 檔進場；
   回測取全部訊號 (標準組合回測做法)。交易筆數會高於實盤，須對照解讀。
4. **時間停損用日曆天**：沿用實盤 `(現日 - 進場日).days >= 5`，週五進場
   週三即滿 5 日 (實際僅 3 交易日)，與實盤同樣的寬鬆定義。
5. **視窗近似**：`evaluate_signal` 以 trailing 200 根 30K 視窗呼叫 (非全歷史
   expanding window)，EMA 截斷誤差經 selftest 驗證 < 0.2% (指標值相對誤差)。
6. **日線暖機**：訊號需日線 >= 65 根才放行 (使 MA60 為真 60 日線)；
   實盤用 6mo 日線 (~120 根)，條件等價。
7. **現貨 vs 期貨**：本回測為現貨股價；若未來用於期貨標的，基差/滑價未建模。

## 資料源

### yfinance (管線驗證 / 事件研究初版)
- 30m 上限 60 天，僅能驗近期；日線用 6mo 1d。
- `python3 backtest_30k.py --source yfinance --stocks 50 --mode both`
- `python3 event_study_30k.py --source yfinance --stocks 50`

### FinMind (決策級，2020-2026)
- 資料集：**TaiwanStockKBar** (2026-10-24 已驗證為分鐘 K 資料集名稱；
  `TaiwanStockMinuteTick` / `TaiwanStockMinute` / `TaiwanStockTick` 皆不存在，
  回 422)。
- 憑證：走 surrogate 機制 (custom.finmind)，見
  `~/workspace/custom-connectors/custom.finmind/CONNECTOR.md`；
  **絕對不要把 token 寫進程式或環境變數**。
- **帳戶等級限制 (2026-10-24 已確認)**：目前註冊帳戶等級為 register，
  `TaiwanStockKBar` 回 HTTP 400「Your level is register. Please update your
  user level.」——分鐘資料需 Sponsor 等級。升級後即可執行：
  `python3 backtest_30k.py --source finmind --start 2020-01-01 --end 2026-10-24 --stocks 50`
- 快取：`data_cache/30k_<code>.csv` (30K 聚合結果)，斷點續跑、升級後不需重抓。
- 分鐘 K 聚合：台股時段 09:00–13:30 切 9 根 (09:00/09:30/…/13:00)，
  bucket 標籤 = K 棒開始時間 (與 yfinance 30m index 口徑一致)。
- 限制：FinMind 分鐘價是否除權息調整尚未驗證 (升級後首跑需核對一檔除權息
  個股)；現今成分池回測 2020 年有**倖存者偏誤** (當年下市/被剔除者不在池內)，
  長視窗數字須打折解讀。

## 產出

- `backtest_30k_results.json`：逐筆交易 (進場/出場/原因/淨報酬%) +
  分模式摘要 (勝率/賺賠比/累計算術和/MDD/分年/分出場原因)。
  累計為算術和 (非複利)，MDD 為依出場順序累積曲線的最大回落 (pp)。
- `event_study_30k_results.json`：A / ENTRY / S3 三組訊號在 +5/+10/+20 根
  30K 的平均報酬、勝率、中位數，附同股隨機基準對照。

## 情境三 (S3) 操作化定義

簡報：「零軸下金叉後往零軸靠攏、再度金叉且柱翻上零軸 = 強勢攻擊」。
程式定義：最近 30 根內 OSC 由負翻正至少 2 次；第一次翻正時 DIF < 0；
最新一次翻正即為訊號根且 DIF > 0、OSC > 0；另要求日線多頭濾網
(日收 > 日 MA20/60 且日 MA20 向上)。不要求量增/突破/RSI，
以便與 A 型對照時看出「型態本身」有無邊際。S3 僅觀察，不進場。

## 自我檢查

`python3 backtest_30k.py --selftest`：停損 / TP1 保本 / TP2 / 死叉
四種出場判定 + 200 根視窗指標等價性 (合成資料，不連網)。
