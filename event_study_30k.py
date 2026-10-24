#!/usr/bin/env python3
"""event_study_30k.py — 30K 訊號事件研究 (v1, 2026-10-24)

不跑帳本，只驗訊號品質：抓每次訊號，算訊號後 5 / 10 / 20 根 30K 的報酬
分布、勝率、平均報酬，並以同股隨機日取樣做同期基準對照。
訊號一律呼叫實盤 scanner.evaluate_signal() (A 型 / 盤中 ENTRY)；
情境三為本程式自訂的觀察訊號 (不進場，僅對照)，定義見 detect_s3()。

情境三 (簡報：零軸下金叉後往零軸靠攏、再度金叉且柱翻上零軸 = 強勢攻擊)：
操作化定義 — 最近 30 根內 OSC 由負翻正 (DIF 上穿 MACD) 至少 2 次；
其中第一次翻正時 DIF < 0 (零軸之下金叉)；最新一次翻正即為訊號根，
且訊號根 DIF > 0、OSC > 0；另要求日線多頭濾網 (與 evaluate_signal 相同：
日收 > 日 MA20/60 且日 MA20 向上)。不要求量增 / 突破 / RSI，
以便與 A 型對照時看出「型態本身」有無邊際。

用法：
    python3 event_study_30k.py --source yfinance --stocks 50
    python3 event_study_30k.py --source finmind --start 2020-01-01 --end 2026-10-24
不發送任何 Telegram。
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scanner  # noqa: E402
from backtest_30k import (  # noqa: E402
    BUY_COST, SELL_COST, WARMUP_DAILY, WINDOW, get_universe, load_stock_30k,
)

TW_TZ = scanner.TW_TZ
HORIZONS = (5, 10, 20)
ROUNDTRIP = BUY_COST + SELL_COST


def detect_s3(df_ind, daily_close, i):
    """情境三偵測 (i 為訊號根位置)。回傳 True/False。"""
    if i < 31:
        return False
    osc = df_ind["OSC"].iloc[i - 30:i + 1]
    dif = df_ind["DIF"].iloc[i - 30:i + 1]
    crosses = []  # 向上翻正的位置 (相對於視窗)
    for k in range(1, len(osc)):
        if osc.iloc[k - 1] <= 0 < osc.iloc[k]:
            crosses.append(k)
    if len(crosses) < 2:
        return False
    if crosses[-1] != len(osc) - 1:
        return False  # 最新一次翻正必須是訊號根
    first_dif = float(dif.iloc[crosses[0]])
    if first_dif >= 0:
        return False  # 第一次翻正須在零軸之下
    if float(dif.iloc[-1]) <= 0 or float(osc.iloc[-1]) <= 0:
        return False
    # 日線多頭濾網 (與 evaluate_signal 相同)
    if daily_close is None:
        return False
    bar_date = df_ind.index[i].date()
    d = daily_close[daily_close.index.date <= bar_date]
    if len(d) < WARMUP_DAILY:
        return False
    ma20 = d.rolling(20).mean()
    ma60 = d.rolling(60).mean()
    if len(ma20) < 2 or np.isnan(ma60.iloc[-1]):
        return False
    if float(d.iloc[-1]) < float(ma20.iloc[-1]) or float(d.iloc[-1]) < float(ma60.iloc[-1]):
        return False
    if float(ma20.iloc[-1]) < float(ma20.iloc[-2]):
        return False
    return True


def collect_signals(df30, daily_df, info):
    """回傳 [(idx, sig_type)]：A / ENTRY / S3。A 與 ENTRY 用實盤 evaluate_signal；
    S3 用 detect_s3。同一根同一股若多種訊號重疊，各自記錄 (事件研究口徑)。"""
    df_ind = scanner.calc_30k_indicators(df30)
    if df_ind is None or len(df_ind) < 70:
        return [], df_ind
    daily_close = None
    if daily_df is not None and "Close" in daily_df:
        daily_close = daily_df["Close"].dropna()
    if daily_close is None:
        return [], df_ind
    sigs = []
    n = len(df_ind)
    for i in range(n):
        bar_date = df_ind.index[i].date()
        dmask = daily_close.index.date <= bar_date
        if int(dmask.sum()) < WARMUP_DAILY:
            continue
        daily_slice_df = pd.DataFrame({"Close": daily_close[dmask].iloc[-200:]})
        window = df30.iloc[max(0, i - WINDOW + 1):i + 1]
        # 盤中 ENTRY：每根判定
        res = scanner.evaluate_signal(window, info["ticker"], info["name"], info["kind"],
                                      mode="intraday", daily_df=daily_slice_df)
        if res and res.get("type") == "ENTRY":
            sigs.append((i, "ENTRY"))
        # EOD A：僅當日最後一根
        if i + 1 >= n or df_ind.index[i + 1].date() != bar_date:
            res = scanner.evaluate_signal(window, info["ticker"], info["name"], info["kind"],
                                          mode="eod", daily_df=daily_slice_df)
            if res and res.get("type") == "A":
                sigs.append((i, "A"))
        # 情境三：每根判定
        if detect_s3(df_ind, daily_close, i):
            sigs.append((i, "S3"))
    return sigs, df_ind


def fwd_stats(df_ind, idxs, horizons=HORIZONS):
    """給定訊號位置，算各 horizon 的報酬分布 (gross 與 net)。"""
    out = {}
    close = df_ind["Close"]
    n = len(df_ind)
    for h in horizons:
        rets = []
        for i in idxs:
            if i + h < n:
                rets.append(float(close.iloc[i + h] / close.iloc[i] - 1) * 100)
        if rets:
            arr = np.array(rets)
            out[str(h)] = {
                "n": len(rets),
                "win_rate": round(float((arr > 0).mean() * 100), 2),
                "mean": round(float(arr.mean()), 4),
                "median": round(float(np.median(arr)), 4),
                "mean_net": round(float(arr.mean()) - ROUNDTRIP * 100, 4),
            }
    return out


def run_event_study(df30, daily_df, info, seed=42):
    sigs, df_ind = collect_signals(df30, daily_df, info)
    if df_ind is None:
        return {}
    rng = np.random.default_rng(seed)
    result = {}
    n = len(df_ind)
    for sig_type in ("A", "ENTRY", "S3"):
        idxs = [i for i, t in sigs if t == sig_type]
        if not idxs:
            continue
        base_idx = rng.integers(WARMUP_DAILY, n, size=len(idxs)).tolist()
        result[sig_type] = {
            "signals": len(idxs),
            "fwd": fwd_stats(df_ind, idxs),
            "baseline_random": fwd_stats(df_ind, base_idx),
        }
    return result


def merge_stats(all_results):
    """跨股彙總：把各股的 forward returns 重新彙總 (需原始報酬序列)。
    簡化版：以各股 n 加權平均 mean / win_rate。"""
    merged = {}
    for sig_type in ("A", "ENTRY", "S3"):
        per = [r[sig_type] for r in all_results if sig_type in r]
        if not per:
            continue
        m = {"signals": sum(p["signals"] for p in per), "fwd": {}, "baseline_random": {}}
        for key in ("fwd", "baseline_random"):
            for h in ("5", "10", "20"):
                rows = [p[key][h] for p in per if h in p[key]]
                if not rows:
                    continue
                tot = sum(r["n"] for r in rows)
                m[key][h] = {
                    "n": tot,
                    "win_rate": round(sum(r["win_rate"] * r["n"] for r in rows) / tot, 2),
                    "mean": round(sum(r["mean"] * r["n"] for r in rows) / tot, 4),
                    "median": round(float(np.median([r["median"] for r in rows])), 4),
                    "mean_net": round(sum(r["mean_net"] * r["n"] for r in rows) / tot, 4),
                }
        merged[sig_type] = m
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yfinance", "finmind"], default="yfinance")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-24")
    ap.add_argument("--stocks", type=int, default=50)
    ap.add_argument("--out", default="event_study_30k_results.json")
    args = ap.parse_args()
    uni = get_universe(args.stocks)
    all_results = []
    for info in uni:
        try:
            df30, daily = load_stock_30k(info, args.source, args.start, args.end)
        except Exception as e:
            print(f"  {info['code']} 資料失敗，跳過: {e}")
            continue
        if df30 is None or len(df30) < 70:
            print(f"  {info['code']} 資料不足，跳過")
            continue
        r = run_event_study(df30, daily, info)
        counts = {k: v["signals"] for k, v in r.items()}
        print(f"  {info['code']} 訊號 {counts}")
        all_results.append(r)
    merged = merge_stats(all_results)
    out = {
        "generated_at": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "source": args.source, "start": args.start, "end": args.end,
        "universe_n": len(uni), "results": merged,
    }
    for sig_type, m in merged.items():
        print(f"\n=== 事件研究 {sig_type} (共 {m['signals']} 訊號) ===")
        for h in ("5", "10", "20"):
            f, b = m["fwd"].get(h), m["baseline_random"].get(h)
            if f and b:
                print(f"  +{h}根: 平均 {f['mean']:+.3f}% (淨 {f['mean_net']:+.3f}%) 勝率 {f['win_rate']}%"
                      f" ｜ 隨機基準 平均 {b['mean']:+.3f}% 勝率 {b['win_rate']}%")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果已寫入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
