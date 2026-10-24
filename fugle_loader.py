#!/usr/bin/env python3
"""fugle_loader.py — Fugle 30K 資料源 + FinMind 除權息還原 (2026-10-24)

資料源特性 (已實測)：
- Fugle 分 K 歷史自 2023-05-23 起，2020 崩盤與 2022 空頭沒有資料
  (覆蓋缺口，判決時必須揭露)。
- Fugle 30K 一天有 10 根 (09:00~13:30)，多出 13:30 收盤集合競價 bar；
  實盤 scanner 用 yfinance 30m (一天 9 根，末根 13:00)，為與實盤口徑一致，
  本 loader 只取 09:00~13:00 起始時間的 9 根，13:30 bar 排除 (殘差，見 README)。
- Fugle volume 單位為張，轉成股 (x1000) 以對齊 yfinance 口徑。
  量能條件皆為相對倍數 (VOL_MA5 倍數)，單位不影響訊號；轉換僅為口徑一致。
- Fugle 分 K 無還原股價：以 FinMind TaiwanStockDividend (免費層可用)
  做向後還原 (backward adjustment)，見 apply_dividend_adjustment()。
- 憑證：Fugle 走 surrogate (custom.fugle)，FinMind 走 surrogate
  (custom.finmind)；憑證絕不在環境變數或檔案中。
- 限速：Fugle 基本帳號 60 次/分鐘，本 loader 每呼叫間隔 1.2 秒，
  遇 429 退避 65 秒重試 (最多 5 次)。

快取：data_cache/fugle_30k_<code>.csv (還原後 30K) +
data_cache/fugle_meta_<code>.json (覆蓋區間與除權息事件數)。
快取覆蓋所求區間時直接使用，否則整段重抓 (每檔約 7 次呼叫)。
不發送任何 Telegram。
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import timedelta

import pandas as pd

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
FUGLE_PACE_SEC = 1.2
MAX_RETRIES = 5


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def _fugle_get(url):
    """單次 Fugle GET (surrogate 憑證、限速、429 退避)。回傳 parsed JSON。"""
    last_err = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(url)
        add_surrogate_to_request(req, "custom.fugle", allowed_hosts=["api.fugle.tw"])
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code == 429:
                wait = 65 * (attempt + 1)
                print(f"  Fugle 429，退避 {wait} 秒後重試 ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                last_err = RuntimeError(f"Fugle 429: {body}")
                continue
            raise RuntimeError(f"Fugle HTTP {e.code}: {body}")
        time.sleep(FUGLE_PACE_SEC)
    raise last_err


def fetch_fugle_30k(code, start, end):
    """抓 Fugle 30K (未還原)。每段 6 個月 (<1 年上限)。
    回傳 DF (index=Asia/Taipei 30K 開始時間，OHLCV，Volume 單位為股)。"""
    frames = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        seg_end = min(cur + pd.DateOffset(months=6) - pd.Timedelta(days=1), end_ts)
        q = urllib.parse.urlencode({
            "timeframe": "30",
            "from": cur.strftime("%Y-%m-%d"),
            "to": seg_end.strftime("%Y-%m-%d"),
        })
        url = (f"https://api.fugle.tw/marketdata/v1.0/stock/historical/candles/"
               f"{code}?{q}")
        payload = _fugle_get(url)
        rows = payload.get("data") or []
        if rows:
            d = pd.DataFrame(rows)
            d["ts"] = pd.to_datetime(d["date"])
            d = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                  "close": "Close", "volume": "Volume"})
            d["Volume"] = d["Volume"] * 1000  # 張 -> 股，對齊 yfinance 口徑
            d = d.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]]
            frames.append(d)
        time.sleep(FUGLE_PACE_SEC)
        cur = seg_end + pd.Timedelta(days=1)
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Taipei")
    else:
        df.index = df.index.tz_convert("Asia/Taipei")
    # 只取 09:00~13:00 起始時間的 9 根 (排除 13:30 收盤集合競價 bar，與實盤一致)
    df = df[(df.index.hour * 60 + df.index.minute >= 540) &
            (df.index.hour * 60 + df.index.minute <= 780)]
    return df


def fetch_finmind_dividends(code, start, end):
    """FinMind TaiwanStockDividend (免費層可用)。
    回傳事件 list：(ex_date, cash_per_share, stock_ratio)，其中
    cash_per_share = CashEarningsDistribution + CashStatutorySurplus (元/股)，
    stock_ratio = StockEarningsDistribution / 10 (元/股換算為股/股，
    台灣股票股利慣用「元」報價，1 元 = 0.1 股；此換算為約定，見 README)。
    ex_date 優先用 Cash/StockExDividendTradingDate，缺失時退回 date 欄位。"""
    q = urllib.parse.urlencode({
        "dataset": "TaiwanStockDividend", "data_id": code,
        "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
    })
    req = urllib.request.Request(f"https://api.finmindtrade.com/api/v4/data?{q}")
    add_surrogate_to_request(req, "custom.finmind", allowed_hosts=["api.finmindtrade.com"])
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"FinMind Dividend {code} HTTP {e.code}: {body}")
    if payload.get("status") != 200:
        raise RuntimeError(f"FinMind Dividend {code} 回應異常: {str(payload)[:300]}")
    events = []
    for r in payload.get("data") or []:
        cash = float(r.get("CashEarningsDistribution") or 0) + float(r.get("CashStatutorySurplus") or 0)
        stock_val = float(r.get("StockEarningsDistribution") or 0) + float(r.get("StockStatutorySurplus") or 0)
        stock_ratio = stock_val / 10.0
        cash_ex = r.get("CashExDividendTradingDate") or ""
        stock_ex = r.get("StockExDividendTradingDate") or ""
        base_date = r.get("date") or ""
        if cash > 0:
            events.append((str(cash_ex or base_date)[:10], cash, 0.0, "cash"))
        if stock_ratio > 0:
            events.append((str(stock_ex or base_date)[:10], 0.0, stock_ratio, "stock"))
    # 同一除權息日若現金與股票分開列，合併為單一事件
    merged = {}
    for ex_date, cash, ratio, kind in events:
        if not ex_date:
            continue
        c, s = merged.get(ex_date, (0.0, 0.0))
        merged[ex_date] = (c + cash, s + ratio)
    return sorted(merged.items())


def apply_dividend_adjustment(df30, events):
    """向後還原：對每個除權息事件，以事件前一交易日收盤計調整因子，
    事件日之前的全部 K 棒 OHLC 乘以累積因子 (Volume 不動)。
    因子：現金 (prev_close - cash) / prev_close；股票 1 / (1 + ratio)。
    回傳 (adjusted_df, report)。事件前無收盤價 (事件早於資料起點) 則跳過並計數。"""
    if df30 is None or df30.empty or not events:
        return df30, {"cash_events": 0, "stock_events": 0, "skipped": 0}
    df = df30.copy()
    close = df["Close"]
    dates = pd.Series(df.index.date, index=df.index)
    n_cash = n_stock = skipped = 0
    factors = []  # (ex_date, factor)
    for ex_date, (cash, ratio) in events:
        ex = pd.Timestamp(ex_date).date()
        prior = close[dates < ex]
        if prior.empty:
            skipped += 1
            continue
        prev_close = float(prior.iloc[-1])
        if prev_close <= 0:
            skipped += 1
            continue
        f = 1.0
        if cash > 0:
            f *= max((prev_close - cash) / prev_close, 0.01)
            n_cash += 1
        if ratio > 0:
            f *= 1.0 / (1.0 + ratio)
            n_stock += 1
        factors.append((ex, f))
    if not factors:
        return df, {"cash_events": 0, "stock_events": 0, "skipped": skipped}
    # 每根 K 的累積因子 = 所有 ex_date > 該根日期的因子乘積
    cum = pd.Series(1.0, index=df.index)
    for ex, f in factors:
        cum[dates < ex] *= f
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] * cum
    return df, {"cash_events": n_cash, "stock_events": n_stock, "skipped": skipped}


def load_fugle_30k(code, start, end, use_cache=True):
    """回傳 (df30_adjusted, report)。快取覆蓋所求區間時直接使用。"""
    cache_path = os.path.join(CACHE_DIR, f"fugle_30k_{code}.csv")
    meta_path = os.path.join(CACHE_DIR, f"fugle_meta_{code}.json")
    start_ts = pd.Timestamp(start).tz_localize("Asia/Taipei")
    end_ts = pd.Timestamp(end).tz_localize("Asia/Taipei")
    if use_cache and os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path, encoding="utf-8"))
            if (pd.Timestamp(meta["start"]).tz_localize("Asia/Taipei") <= start_ts and
                    pd.Timestamp(meta["end"]).tz_localize("Asia/Taipei") >= end_ts):
                c = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if c.index.tz is None:
                    c.index = c.index.tz_localize("Asia/Taipei")
                print(f"  {code} Fugle 快取命中 {len(c)} 根 (還原後)")
                return c, meta.get("div_report", {})
        except Exception as e:
            print(f"  {code} Fugle 快取讀取失敗，重新下載: {e}")
    raw = fetch_fugle_30k(code, start, end)
    if raw is None or raw.empty:
        return None, {}
    time.sleep(1.0)  # FinMind 限速保護
    try:
        events = fetch_finmind_dividends(code, start, end)
    except Exception as e:
        print(f"  {code} 除權息資料失敗，改用未還原序列: {e}")
        events, report = [], {"cash_events": 0, "stock_events": 0, "skipped": -1}
        adj = raw
    else:
        adj, report = apply_dividend_adjustment(raw, events)
    os.makedirs(CACHE_DIR, exist_ok=True)
    adj.to_csv(cache_path)
    json.dump({"start": str(pd.Timestamp(start).date()), "end": str(pd.Timestamp(end).date()),
               "bars": len(adj), "div_report": report},
              open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return adj, report
