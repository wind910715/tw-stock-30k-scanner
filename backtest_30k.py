#!/usr/bin/env python3
"""backtest_30k.py — 30K MACD 策略回測引擎 (v1, 2026-10-24)

硬規矩 (上一款 0050 期貨的教訓)：訊號與出場判定必須與實盤同一份邏輯。
- 訊號：直接呼叫 scanner.py 的 evaluate_signal() (intraday / eod 兩種模式)。
- 出場：逐項移植 scanner.py update_and_check_positions() 的 5 種出場判定順序
  (硬停損/保本線 -> TP2 -> TP1 減碼 50% + 停損移至成本+0.5% -> 死叉破 20MA ->
  盤後時間停損)，只改成本口徑為買 0.1425% / 賣 0.4425% (實盤帳本用 FEE_TAX_RATE
  0.4% 單次，兩者差 0.185pp，見 README；回測採較保守的 0.585%)。
- 進場價 = 訊號 K 收盤 (實盤 daemon 在 K 收盤後數分鐘內以市價進場之近似)，
  出場檢查自訊號 K 的「下一根」開始，不用同一根 K 判定出場 (無前視)。
- 只用已收盤 K：歷史資料全為已收盤 K；yfinance 即時尾巴沿用
  scanner.drop_incomplete_last_bar()。

資料源：
- yfinance：30m 上限 60 天，僅供管線驗證與事件研究初版。
- FinMind：dataset TaiwanStockKBar (2026-10-24 已驗證為分鐘 K 資料集名稱)，
  憑證走 surrogate (custom.finmind)，見 custom.finmind/CONNECTOR.md。
  注意：目前註冊帳戶等級為 register，FinMind 回傳 400 要求升級 Sponsor
  才能取分鐘資料；2020-2026 長視窗需升級後才能執行，本程式已備妥
  斷點續跑快取 (data_cache/30k_<code>.csv)，升級後直接續跑不需重抓。
- 日線 MA 濾網：yfinance 模式用 yf 1d 日線；FinMind / Fugle 模式由 30K 聚合日收盤。
- Fugle：分 K 歷史自 2023-05-23 起 (2020 崩盤與 2022 空頭無資料，判決須揭露)；
  一天 10 根含 13:30 收盤集合競價 bar，本程式只取 09:00~13:00 的 9 根
  以對齊實盤 yfinance 口徑；分 K 無還原股價，以 FinMind
  TaiwanStockDividend 做向後還原 (見 fugle_loader.py)。
  用法：python3 backtest_30k.py --source fugle --start 2023-05-23 --end 2026-10-22 --stocks 50

股票池：沿用 scanner.fetch_all_market_candidates() 的量能篩選
(個股 >=1500 張 / ETF >=500 張)，取成交量前 N 檔 (預設 50)，失敗時退回
內建流動性名單。注意：現今成分池回測 2020 年有倖存者偏誤，見 README。

用法：
    python3 backtest_30k.py --source yfinance --stocks 50 --mode both
    python3 backtest_30k.py --source finmind --start 2020-01-01 --end 2026-10-24 --stocks 50
    python3 backtest_30k.py --selftest   # 退出引擎單元檢查 (合成 K 棒)
不發送任何 Telegram。
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scanner  # noqa: E402  (實盤同一份邏輯，禁止複寫訊號/出場規則)

TW_TZ = timezone(timedelta(hours=8))
BUY_COST = 0.001425
SELL_COST = 0.004425
CAPITAL_PER_TRADE = scanner.CAPITAL_PER_TRADE
WINDOW = 200          # evaluate_signal  trailing window (EMA 截斷誤差 < 1e-3，相對誤差見 selftest)
WARMUP_DAILY = 65     # 日線需 65 根以上才讓 MA60 為真 60 日線 (貼近實盤 6mo 日線條件)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

FALLBACK_UNIVERSE = [  # TWSE/TPEx API 失敗時的流動性備援名單 (code, kind)
    ("2330", "STOCK"), ("2317", "STOCK"), ("2454", "STOCK"), ("2303", "STOCK"),
    ("2308", "STOCK"), ("2881", "STOCK"), ("2882", "STOCK"), ("2891", "STOCK"),
    ("2886", "STOCK"), ("1301", "STOCK"), ("1303", "STOCK"), ("2002", "STOCK"),
    ("2412", "STOCK"), ("2603", "STOCK"), ("1216", "STOCK"), ("3008", "STOCK"),
    ("2379", "STOCK"), ("2382", "STOCK"), ("3231", "STOCK"), ("3661", "STOCK"),
    ("3034", "STOCK"), ("3037", "STOCK"), ("2344", "STOCK"), ("2408", "STOCK"),
    ("2327", "STOCK"), ("2357", "STOCK"), ("2395", "STOCK"), ("6669", "STOCK"),
    ("3017", "STOCK"), ("2609", "STOCK"), ("2207", "STOCK"), ("2912", "STOCK"),
    ("4938", "STOCK"), ("3711", "STOCK"), ("2301", "STOCK"), ("2345", "STOCK"),
    ("2368", "STOCK"), ("3443", "STOCK"), ("3045", "STOCK"), ("3533", "STOCK"),
    ("9910", "STOCK"), ("1101", "STOCK"), ("2105", "STOCK"), ("2542", "STOCK"),
    ("1907", "STOCK"), ("1605", "STOCK"), ("1513", "STOCK"), ("1504", "STOCK"),
    ("0050", "ETF"), ("0056", "ETF"),
]


# ---------------------------------------------------------------- data layer
def get_universe(n=50):
    """沿用實盤量能篩選，取成交量前 n 檔；API 失敗退回內建名單。"""
    try:
        cands = scanner.fetch_all_market_candidates()
        items = sorted(cands.items(), key=lambda kv: kv[1].get("lots", 0), reverse=True)[:n]
        out = []
        for tk, info in items:
            code = tk.split(".")[0]
            out.append({"ticker": tk, "code": code, "name": info["name"], "kind": info["kind"]})
        if len(out) >= 10:
            print(f"股票池：TWSE/TPEx 量能篩選取前 {len(out)} 檔")
            return out
    except Exception as e:
        print(f"⚠️ 全市場篩選失敗，退回內建名單: {e}")
    out = []
    for code, kind in FALLBACK_UNIVERSE[:n]:
        sfx = ".TW"
        out.append({"ticker": f"{code}{sfx}", "code": code, "name": code, "kind": kind})
    print(f"股票池：內建備援名單 {len(out)} 檔 (未區分上市/上櫃，.TWO 標的可能誤用 .TW)")
    return out


def aggregate_minutes_to_30k(min_df):
    """分鐘 K -> 30K。台股時段 09:00-13:30，切 9 根 (09:00, 09:30, ..., 13:00)。
    bucket 標籤 = K 棒開始時間 (與 yfinance 30m index 口徑一致)。"""
    if min_df is None or min_df.empty:
        return None
    df = min_df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Taipei")
    else:
        df.index = df.index.tz_convert("Asia/Taipei")
    df = df.between_time("09:00", "13:29")
    if df.empty:
        return None
    bucket = df.index.floor("30min")
    g = df.groupby(bucket)
    out = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(),
        "Low": g["Low"].min(), "Close": g["Close"].last(),
        "Volume": g["Volume"].sum(),
    })
    out.index.name = "Datetime"
    return out.sort_index()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def fetch_finmind_kbar(stock_id, start, end):
    """FinMind TaiwanStockKBar 分鐘 K (surrogate 憑證，redirect 關閉)。
    按月分段抓，失敗整段報錯不靜默。回傳分鐘 DF (index=時間, OHLCV)。
    欄位以 FinMind KBar 慣用名稱做寬容對應；對不上直接報錯。"""
    sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
    from dynamic_credentials import add_surrogate_to_request
    frames = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        nxt = (cur + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
        seg_end = min(nxt, end_ts)
        q = urllib.parse.urlencode({
            "dataset": "TaiwanStockKBar", "data_id": stock_id,
            "start_date": cur.strftime("%Y-%m-%d"), "end_date": seg_end.strftime("%Y-%m-%d"),
        })
        req = urllib.request.Request(f"https://api.finmindtrade.com/api/v4/data?{q}")
        add_surrogate_to_request(req, "custom.finmind", allowed_hosts=["api.finmindtrade.com"])
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"FinMind {stock_id} {cur.date()}~{seg_end.date()} HTTP {e.code}: {body}")
        if payload.get("status") != 200:
            raise RuntimeError(f"FinMind {stock_id} 回應異常: {str(payload)[:300]}")
        rows = payload.get("data") or []
        if rows:
            d = pd.DataFrame(rows)
            tcol = next((c for c in ["Time", "time", "datetime", "Datetime"] if c in d.columns), None)
            dcol = "date" if "date" in d.columns else None
            if tcol and dcol:
                d["ts"] = pd.to_datetime(d[dcol].astype(str) + " " + d[tcol].astype(str))
            elif tcol:
                d["ts"] = pd.to_datetime(d[tcol])
            elif dcol:
                d["ts"] = pd.to_datetime(d[dcol])
            else:
                raise RuntimeError(f"FinMind KBar 欄位無法對應: {list(d.columns)}")
            ren = {}
            for want, alts in [("Open", ["open"]), ("High", ["max", "high"]),
                               ("Low", ["min", "low"]), ("Close", ["close"]),
                               ("Volume", ["Trading_Volume", "volume", "TradingVolume"])]:
                for a in alts:
                    if a in d.columns:
                        ren[a] = want
                        break
            d = d.rename(columns=ren).set_index("ts")
            d = d[["Open", "High", "Low", "Close", "Volume"]]
            frames.append(d)
        cur = seg_end + pd.Timedelta(days=1)
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def load_stock_30k(info, source, start, end, use_cache=True):
    """回傳 (df30, daily_df)。df30 index 為 30K 開始時間 (Asia/Taipei)。
    FinMind 模式先讀快取再補抓缺段；yfinance 模式直接下載。"""
    code, ticker = info["code"], info["ticker"]
    cache_path = os.path.join(CACHE_DIR, f"30k_{code}.csv")
    if source == "finmind" and use_cache and os.path.exists(cache_path):
        try:
            c = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if c.index.tz is None:
                c.index = c.index.tz_localize("Asia/Taipei")
            if len(c) > 0 and c.index.min() <= pd.Timestamp(start).tz_localize("Asia/Taipei"):
                print(f"  {code} 快取命中 {len(c)} 根")
                return c, daily_from_30k(c)
        except Exception as e:
            print(f"  {code} 快取讀取失敗，重新下載: {e}")
    if source == "finmind":
        mins = fetch_finmind_kbar(code, start, end)
        df30 = aggregate_minutes_to_30k(mins) if mins is not None else None
        if df30 is not None and use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            df30.to_csv(cache_path)
        return df30, (daily_from_30k(df30) if df30 is not None else None)
    if source == "fugle":
        from fugle_loader import load_fugle_30k
        df30, report = load_fugle_30k(code, start, end, use_cache=use_cache)
        if df30 is not None:
            print(f"  {code} Fugle 除權息事件 {report}")
        return df30, (daily_from_30k(df30) if df30 is not None else None)
    # yfinance (與實盤一致：group_by="ticker"，再用實盤 extract_ticker_df 萃取)
    import yfinance as yf
    batch = yf.download([ticker], period="60d", interval="30m", group_by="ticker",
                        progress=False, auto_adjust=False)
    if batch is None or batch.empty:
        return None, None
    data = scanner.extract_ticker_df(batch, ticker)
    data = scanner.drop_incomplete_last_bar(data)
    daily_batch = yf.download([ticker], period="6mo", interval="1d", group_by="ticker",
                              progress=False, auto_adjust=False)
    daily = None
    if daily_batch is not None and not daily_batch.empty:
        daily = scanner.extract_ticker_df(daily_batch, ticker)
    return data, daily


def daily_from_30k(df30):
    """由 30K 聚合日收盤 (供日線 MA 濾網)。"""
    if df30 is None or df30.empty:
        return None
    idx = df30.index.tz_convert("Asia/Taipei") if df30.index.tz is not None else df30.index.tz_localize("Asia/Taipei")
    d = pd.DataFrame({"Close": df30["Close"].values}, index=idx)
    return d.resample("1D").last().dropna()


# ---------------------------------------------------------------- exit engine (port of update_and_check_positions)
def simulate_position(df_ind, entry_idx, sig):
    """逐根移植實盤 5 種出場判定 (順序與價格邏輯與實盤一致)。
    df_ind 需先經 scanner.calc_30k_indicators (含 MA20/OSC)。
    進場價 = 訊號 K 收盤；出場檢查自下一根開始。
    回傳 dict(net_pct, exit_idx, exit_reason, censored)。"""
    entry = float(sig["close"])
    stop = float(sig["stop"])
    tp1 = float(sig["tp1"])
    tp2 = float(sig["tp2"])
    entry_date = df_ind.index[entry_idx].date()
    tp1_hit = False
    remain = 1.0
    realized_price = 0.0  # 已實現價格報酬貢獻 (權重調整後)
    n = len(df_ind)
    # 日內最後一根集合 (供時間停損，僅盤後判定)
    dates = pd.Series(df_ind.index.date, index=range(n))
    last_bar_of_day = set()
    for i in range(n):
        if i == n - 1 or dates.iloc[i + 1] != dates.iloc[i]:
            last_bar_of_day.add(i)
    for j in range(entry_idx + 1, n):
        row = df_ind.iloc[j]
        close, low, high = float(row["Close"]), float(row["Low"]), float(row["High"])
        # 1. 硬停損 / 保本線
        if close <= stop or low <= stop:
            exit_px = stop if low <= stop else close
            net = realized_price + remain * (exit_px / entry - 1) - BUY_COST - SELL_COST
            return {"net_pct": round(net * 100, 4), "exit_idx": j,
                    "exit_reason": "TP1後保本止盈清倉" if tp1_hit else "觸及硬停損清倉", "censored": False}
        # 2. TP2
        if close >= tp2 or high >= tp2:
            exit_px = tp2 if high >= tp2 else close
            net = realized_price + remain * (exit_px / entry - 1) - BUY_COST - SELL_COST
            return {"net_pct": round(net * 100, 4), "exit_idx": j,
                    "exit_reason": "達陣TP2波段目標清倉", "censored": False}
        # 3. TP1 首次：減碼 50%，停損移至成本+0.5%
        if not tp1_hit and (close >= tp1 or high >= tp1):
            exec_tp1 = tp1 if high >= tp1 else close
            realized_price += 0.5 * (exec_tp1 / entry - 1)
            tp1_hit = True
            remain = 0.5
            stop = round(entry * 1.005, 2)
            continue
        # 4. 死叉破 20MA
        osc_now = float(row["OSC"])
        osc_prev = float(df_ind.iloc[j - 1]["OSC"])
        ma20 = float(row["MA20"])
        if osc_now < 0 and osc_prev >= 0 and close < ma20:
            net = realized_price + remain * (close / entry - 1) - BUY_COST - SELL_COST
            return {"net_pct": round(net * 100, 4), "exit_idx": j,
                    "exit_reason": "30K死叉破20MA清倉", "censored": False}
        # 5. 時間停損 (僅盤後日終判定，日曆天 >=5 且未達 TP1 且漲幅 <1%)
        if j in last_bar_of_day and not tp1_hit:
            days_held = (df_ind.index[j].date() - entry_date).days
            if days_held >= scanner.MAX_HOLD_DAYS and (close / entry - 1) < 0.01:
                net = realized_price + remain * (close / entry - 1) - BUY_COST - SELL_COST
                return {"net_pct": round(net * 100, 4), "exit_idx": j,
                        "exit_reason": "滿5日動能停滯清倉", "censored": False}
    return {"net_pct": None, "exit_idx": None, "exit_reason": "未出場(censored)", "censored": True}


# ---------------------------------------------------------------- backtest driver
def run_book_while(df30, daily_df, info, mode):
    """run_book 的 while 版本 (正確跳過持有期間)。"""
    df_ind = scanner.calc_30k_indicators(df30)
    if df_ind is None or len(df_ind) < 70:
        return []
    daily_close = None
    if daily_df is not None and "Close" in daily_df:
        daily_close = daily_df["Close"].dropna()
    if daily_close is None:
        return []
    trades = []
    n = len(df_ind)
    i = 0
    while i < n:
        bar_date = df_ind.index[i].date()
        dmask = daily_close.index.date <= bar_date
        if int(dmask.sum()) < WARMUP_DAILY:
            i += 1
            continue
        daily_slice_df = pd.DataFrame({"Close": daily_close[dmask].iloc[-200:]})
        window = df30.iloc[max(0, i - WINDOW + 1):i + 1]
        sig = None
        if mode == "eod":
            if i + 1 < n and df_ind.index[i + 1].date() == bar_date:
                i += 1
                continue
            res = scanner.evaluate_signal(window, info["ticker"], info["name"], info["kind"],
                                          mode="eod", daily_df=daily_slice_df)
            if res and res.get("type") == "A":
                sig = res
        else:
            res = scanner.evaluate_signal(window, info["ticker"], info["name"], info["kind"],
                                          mode="intraday", daily_df=daily_slice_df)
            if res and res.get("type") == "ENTRY":
                sig = res
        if sig is None:
            i += 1
            continue
        out = simulate_position(df_ind, i, sig)
        trades.append({
            "ticker": info["ticker"], "code": info["code"], "name": info["name"],
            "mode": mode, "sig_type": sig["type"],
            "entry_date": df_ind.index[i].strftime("%Y-%m-%d %H:%M"),
            "entry": sig["close"], "stop": sig["stop"], "tp1": sig["tp1"], "tp2": sig["tp2"],
            "exit_date": df_ind.index[out["exit_idx"]].strftime("%Y-%m-%d %H:%M") if out["exit_idx"] is not None else None,
            "net_pct": out["net_pct"], "exit_reason": out["exit_reason"], "censored": out["censored"],
        })
        i = (out["exit_idx"] + 1) if out["exit_idx"] is not None else n
    return trades


def summarize(trades):
    """統計：勝率 / 賺賠比 / 累計 (算術和) / MDD (依出場順序累積曲線) / 分年 / 分原因。"""
    done = [t for t in trades if not t["censored"] and t["net_pct"] is not None]
    wins = [t["net_pct"] for t in done if t["net_pct"] > 0]
    losses = [t["net_pct"] for t in done if t["net_pct"] <= 0]
    curve = [0.0]
    for t in sorted(done, key=lambda x: x["exit_date"] or ""):
        curve.append(curve[-1] + t["net_pct"])
    peak, mdd = 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    by_year = {}
    for t in done:
        y = t["entry_date"][:4]
        by_year.setdefault(y, []).append(t["net_pct"])
    by_reason = {}
    for t in done:
        by_reason.setdefault(t["exit_reason"], []).append(t["net_pct"])
    return {
        "n": len(done), "censored": len(trades) - len(done),
        "win_rate": round(len(wins) / len(done) * 100, 2) if done else 0.0,
        "mean_pct": round(float(np.mean([t["net_pct"] for t in done])) , 4) if done else 0.0,
        "rr": round(float(np.mean(wins)) / abs(float(np.mean(losses))) , 4) if wins and losses else 0.0,
        "total_pct": round(sum(t["net_pct"] for t in done), 2),
        "total_ntd": round(sum(t["net_pct"] / 100 * CAPITAL_PER_TRADE for t in done)),
        "mdd_pp": round(mdd, 2),
        "by_year": {y: {"n": len(v), "mean": round(float(np.mean(v)), 3), "total": round(sum(v), 2)}
                    for y, v in sorted(by_year.items())},
        "by_reason": {r: {"n": len(v), "mean": round(float(np.mean(v)), 3)} for r, v in by_reason.items()},
    }


def print_summary(title, s):
    print(f"\n=== {title} ===")
    print(f"交易 {s['n']} 筆 (未完 {s['censored']} 筆)｜勝率 {s['win_rate']}%｜平均 {s['mean_pct']}%｜"
          f"RR {s['rr']}｜累計 {s['total_pct']}% ({s['total_ntd']:+,} 元 @20萬/筆)｜MDD {s['mdd_pp']}pp")
    for y, v in s["by_year"].items():
        print(f"  {y}: {v['n']} 筆 平均 {v['mean']}% 累計 {v['total']}%")
    for r, v in s["by_reason"].items():
        print(f"  {r}: {v['n']} 筆 平均 {v['mean']}%")


# ---------------------------------------------------------------- selftest
def selftest():
    """退出引擎 + 視窗等價性檢查 (合成資料，不連網)。"""
    # 確定性平坦序列 (收盤恆 100)，退出邏輯的每一分支可精確控制
    days = pd.bdate_range("2026-01-02", periods=20)
    bars = []
    for d in days:
        for hh, mm in [(9, 0), (9, 30), (10, 0), (10, 30), (11, 0), (11, 30),
                       (12, 0), (12, 30), (13, 0)]:
            bars.append(pd.Timestamp(d.year, d.month, d.day, hh, mm, tz="Asia/Taipei"))
    idx = pd.DatetimeIndex(bars)
    n = len(idx)
    df = pd.DataFrame({"Open": 100.0, "High": 100.3, "Low": 99.7,
                       "Close": 100.0, "Volume": 1000}, index=idx)
    df_ind = scanner.calc_30k_indicators(df)
    assert df_ind is not None and len(df_ind) > 70
    sig = {"close": 100.0, "stop": 98.0, "tp1": 103.5, "tp2": 108.0}
    e = 69
    # 1. 停損觸發 (低點貫穿停損 -> 以停損價出場)
    df2 = df_ind.copy()
    df2.iloc[e + 1, df2.columns.get_loc("Low")] = 97.0
    r = simulate_position(df2, e, sig)
    assert r["exit_reason"] == "觸及硬停損清倉" and abs(r["net_pct"] - (-2.585)) < 0.01, r
    # 2. TP1 後保本線 (TP1 減碼 50%，隨後跌破保本 100.5)
    df3 = df_ind.copy()
    df3.iloc[e + 1, df3.columns.get_loc("High")] = 104.0
    df3.iloc[e + 2, df3.columns.get_loc("High")] = 102.5
    df3.iloc[e + 2, df3.columns.get_loc("Low")] = 101.0
    df3.iloc[e + 2, df3.columns.get_loc("Close")] = 102.0
    df3.iloc[e + 3, df3.columns.get_loc("Low")] = 100.4
    df3.iloc[e + 3, df3.columns.get_loc("Close")] = 100.4
    r = simulate_position(df3, e, sig)
    assert r["exit_reason"] == "TP1後保本止盈清倉" and abs(r["net_pct"] - 1.415) < 0.01, r
    # 3. TP2
    df4 = df_ind.copy()
    df4.iloc[e + 1, df4.columns.get_loc("High")] = 109.0
    r = simulate_position(df4, e, sig)
    assert r["exit_reason"] == "達陣TP2波段目標清倉" and abs(r["net_pct"] - 7.415) < 0.01, r
    # 4. 死叉破 20MA (OSC 由正翻負且收盤 < MA20)
    df5 = df_ind.copy()
    df5.iloc[e, df5.columns.get_loc("OSC")] = 0.1
    df5.iloc[e + 1, df5.columns.get_loc("OSC")] = -0.1
    df5.iloc[e + 1, df5.columns.get_loc("Close")] = 99.0
    df5.iloc[e + 1, df5.columns.get_loc("Low")] = 98.7
    df5.iloc[e + 1, df5.columns.get_loc("High")] = 99.3
    r = simulate_position(df5, e, sig)
    assert r["exit_reason"] == "30K死叉破20MA清倉", r
    # 5. 視窗等價：隨機漫步序列，完整視窗 vs 200 根 trailing 視窗指標誤差
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    dfr = pd.DataFrame({"Open": close, "High": close + 0.3, "Low": close - 0.3,
                        "Close": close, "Volume": np.full(n, 1000)}, index=idx)
    full = scanner.calc_30k_indicators(dfr)
    win = scanner.calc_30k_indicators(dfr.iloc[-200:])
    for col in ["MA20", "MA60", "DIF", "MACD", "OSC", "RSI"]:
        a, b = full[col].iloc[-1], win[col].iloc[-1]
        assert abs(a - b) <= max(0.01, abs(a) * 0.002), (col, a, b)
    print("selftest 全過：停損/TP1保本/TP2/死叉判定正確；200 根視窗指標與完整視窗誤差 <0.2%")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yfinance", "finmind", "fugle"], default="yfinance")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-24")
    ap.add_argument("--stocks", type=int, default=50)
    ap.add_argument("--mode", choices=["intraday", "eod", "both"], default="both")
    ap.add_argument("--out", default="backtest_30k_results.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    uni = get_universe(args.stocks)
    modes = ["intraday", "eod"] if args.mode == "both" else [args.mode]
    all_trades = []
    for info in uni:
        try:
            df30, daily = load_stock_30k(info, args.source, args.start, args.end)
        except Exception as e:
            print(f"  {info['code']} 資料失敗，跳過: {e}")
            continue
        if df30 is None or len(df30) < 70:
            print(f"  {info['code']} 資料不足，跳過")
            continue
        print(f"  {info['code']} {len(df30)} 根 30K")
        for m in modes:
            all_trades.extend(run_book_while(df30, daily, info, m))
    results = {
        "generated_at": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "source": args.source, "start": args.start, "end": args.end,
        "universe_n": len(uni), "modes": modes,
        "costs": {"buy": BUY_COST, "sell": SELL_COST},
        "trades": all_trades,
    }
    SPLIT_DATE = "2025-05-01"  # 前後半切分：H1 2023-05~2025-04，H2 2025-05~2026-10
    for m in modes:
        s = summarize([t for t in all_trades if t["mode"] == m])
        results[f"summary_{m}"] = s
        print_summary(f"30K 回測 ({args.source}, {m})", s)
        for h, subset in (("h1", [t for t in all_trades if t["mode"] == m and t["entry_date"] < SPLIT_DATE]),
                          ("h2", [t for t in all_trades if t["mode"] == m and t["entry_date"] >= SPLIT_DATE])):
            sh = summarize(subset)
            results[f"summary_{m}_{h}"] = sh
            print_summary(f"30K 回測 ({args.source}, {m}, {h} {'<' if h == 'h1' else '>='} {SPLIT_DATE})", sh)
    s_all = summarize(all_trades)
    results["summary_all"] = s_all
    print_summary(f"30K 回測 ({args.source}, 合計)", s_all)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果已寫入 {args.out} ({len(all_trades)} 筆含未完)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
