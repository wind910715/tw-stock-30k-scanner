import os
import re
import sys
import json
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ==========================================================
# 核心策略參數設定 (已針對勝率與實戰滑價優化)
# ==========================================================
MIN_STOCK_VOL_LOTS = 1500   # 個股最低日成交量門檻 (張)：提高至 1500 張確保盤中流動性
MIN_ETF_VOL_LOTS = 500      # ETF 最低日成交量門檻 (張)
ZERO_BOUND_PCT = 0.012      # 30K DIF 距離 0 軸容許範圍 (收斂至股價 1.2% 內)
VOL_MULTIPLIER = 1.5        # 觸發當根 30K 成交量需大於 5均量的 1.5 倍 (真點火)
MIN_STOP_PCT = 0.015        # 最小停損呼吸空間 (1.5%)，避免 -0.6% 被雜訊洗出場
TP1_PCT = 0.035             # 第一階段停利目標 (+3.5%)
BATCH_SIZE = 60             # 批次下載檔數
POSITIONS_FILE = "positions.json"
TW_TZ = timezone(timedelta(hours=8))

# 保底核心股票型與主動式 ETF (已排除債券B與平衡型T)
CORE_ETFS = {
    "0050.TW": "元大台灣50", "0056.TW": "元大高股息", "006208.TW": "富邦台50",
    "00878.TW": "國泰永續高股息", "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息",
    "00881.TW": "國泰台灣5G+", "00891.TW": "中信關鍵半導體", "00830.TW": "國泰費城半導體",
    "00757.TW": "統一FANG+", "00662.TW": "富邦NASDAQ", "00631L.TW": "元大台灣50正2",
    "00675L.TW": "富邦臺灣加權正2", "00663L.TW": "國泰臺灣加權正2", "00685L.TW": "群益臺灣加權正2",
    "00980A.TW": "主動式ETF-00980A", "00981A.TW": "主動式ETF-00981A",
    "00982A.TW": "主動式ETF-00982A", "00990A.TW": "主動式ETF-00990A", "00991A.TW": "主動式ETF-00991A"
}

def now_tw():
    return datetime.now(TW_TZ)

def is_valid_code(code):
    """僅保留 4 碼個股，以及純股票/主動式(A)/正2槓桿(L)之 ETF；嚴格排除債券(B)、平衡(T)、反向(R)"""
    code = str(code).strip()
    if re.match(r'^[1-9]\d{3}$', code):
        return "STOCK"
    if re.match(r'^00\d{2,4}[AL]?$', code):
        return "ETF"
    return None

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_positions(positions):
    try:
        with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(positions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 儲存部位狀態失敗: {e}")

def fetch_all_market_candidates():
    """從證交所與櫃買中心抓取符合日均量門檻之上市櫃個股與動能 ETF"""
    candidates = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    for tk, name in CORE_ETFS.items():
        candidates[tk] = {"name": name, "kind": "ETF", "lots": 9999}

    try:
        r_twse = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers=headers, timeout=15)
        if r_twse.status_code == 200:
            for row in r_twse.json():
                code = str(row.get("Code", "")).strip()
                name = str(row.get("Name", "")).strip()
                vol_str = str(row.get("TradeVolume", "0")).replace(",", "").strip()
                kind = is_valid_code(code)
                if kind and vol_str.isdigit():
                    lots = int(vol_str) // 1000
                    if (kind == "STOCK" and lots >= MIN_STOCK_VOL_LOTS) or (kind == "ETF" and lots >= MIN_ETF_VOL_LOTS):
                        candidates[f"{code}.TW"] = {"name": name, "kind": kind, "lots": lots}
    except Exception as e:
        print(f"⚠️ TWSE 讀取異常: {e}")

    try:
        r_tpex = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes", headers=headers, timeout=15)
        if r_tpex.status_code == 200:
            for row in r_tpex.json():
                code = str(row.get("SecuritiesCompanyCode", "")).strip()
                name = str(row.get("CompanyName", "")).strip()
                vol_str = str(row.get("TradingShares", "0")).replace(",", "").strip()
                kind = is_valid_code(code)
                if kind and vol_str.isdigit():
                    lots = int(vol_str) // 1000
                    if (kind == "STOCK" and lots >= MIN_STOCK_VOL_LOTS) or (kind == "ETF" and lots >= MIN_ETF_VOL_LOTS):
                        candidates[f"{code}.TWO"] = {"name": name, "kind": kind, "lots": lots}
    except Exception as e:
        print(f"⚠️ TPEx 讀取異常: {e}")

    print(f"✅ 完成全市場流動性初篩，共載入 {len(candidates)} 檔活水標的")
    return candidates

def calc_30k_indicators(df):
    """計算 30K 均線、MACD 與日線濾網指標"""
    df = df.dropna(subset=['Close', 'Open', 'High', 'Low']).copy()
    if len(df) < 65:
        return None
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['VOL_MA5'] = df['Volume'].rolling(window=5).mean()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['OSC'] = df['DIF'] - df['MACD']
    return df

def evaluate_signal(raw_df, ticker, name, kind, mode="eod"):
    """
    高勝率優化版訊號判定：
    1. 日線收盤 > 日線20MA 且 日線20MA 向上
    2. 30K 20MA > 60MA 且 60MA 向上助漲
    3. 30K DIF 位於 0 軸附近 (1.2%內) 且 OSC 由綠轉紅 (或剛翻紅第1~2根)
    4. 當根 30K 為實體紅 K (Close > Open)，成交量 >= 5均量 * 1.5 倍，且突破前 12 根 30K 高點 (微VCP突破)
    """
    try:
        df = calc_30k_indicators(raw_df)
        if df is None:
            return None

        daily_close = df['Close'].resample('1D').last().dropna()
        ma_win = min(20, len(daily_close))
        if ma_win < 10:
            return None
        daily_ma20_series = daily_close.rolling(ma_win).mean()
        daily_ma20 = float(daily_ma20_series.iloc[-1])
        daily_ma20_prev = float(daily_ma20_series.iloc[-2]) if len(daily_ma20_series) >= 2 else daily_ma20
        current_close = float(df['Close'].iloc[-1])

        # 大週期濾網：股價在日線 20MA 之上且月線向上
        if current_close < daily_ma20 or daily_ma20 < daily_ma20_prev:
            return None

        last = df.iloc[-1]
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]

        ma20_now, ma60_now = float(last['MA20']), float(last['MA60'])
        ma60_prev = float(prev1['MA60'])
        dif_now, osc_now = float(last['DIF']), float(last['OSC'])
        osc_prev1, osc_prev2 = float(prev1['OSC']), float(prev2['OSC'])

        if np.isnan(ma60_now) or ma60_now <= 0:
            return None

        # 動態停損保護：取 60MA 下方 0.5% 與 進場價下方 1.5% 較寬者，避免過窄停損被洗
        stop_by_ma = ma60_now * 0.995
        stop_by_pct = current_close * (1.0 - MIN_STOP_PCT)
        stop_price = round(min(stop_by_ma, stop_by_pct), 2)
        risk_pct = round((current_close - stop_price) / current_close * 100, 2)
        tp1_price = round(current_close * (1.0 + TP1_PCT), 2)
        tp2_price = round(current_close + (current_close - stop_price) * 3.0, 2)

        code = ticker.split('.')[0]
        ma_gap_pct = abs(ma20_now - ma60_now) / current_close
        dif_zero_pct = abs(dif_now) / current_close
        tightness_score = round((ma_gap_pct + dif_zero_pct) * 100, 2)
        near_zero = dif_zero_pct <= ZERO_BOUND_PCT

        # 突破前 12 根 30K 收盤高點 (確認脫離盤整小平台)
        prev_12_high = float(df['Close'].iloc[-13:-1].max())
        is_breakout = current_close >= prev_12_high
        is_bull_bar = float(last['Close']) > float(last['Open']) and float(last['Close']) >= (float(last['High']) + float(last['Low'])) / 2
        ma60_rising = ma60_now >= ma60_prev

        if mode == "intraday":
            # 盤中即時監控：只抓「當下最新 1~2 根 30K 剛金叉翻紅 + 帶量突破平台」的第一時間訊號
            just_turned_red = (osc_now > 0) and (osc_prev1 <= 0 or osc_prev2 <= 0)
            ma_just_crossed = (ma20_now > ma60_now) and (ma_gap_pct <= 0.01)
            vol_ok = float(last['Volume']) >= float(last['VOL_MA5']) * VOL_MULTIPLIER or float(prev1['Volume']) >= float(prev1['VOL_MA5']) * VOL_MULTIPLIER

            if near_zero and ma_just_crossed and ma60_rising and just_turned_red and is_bull_bar and is_breakout and vol_ok:
                return {
                    "type": "ENTRY", "kind": kind, "ticker": ticker, "code": code, "name": name,
                    "close": round(current_close, 2), "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                    "stop": stop_price, "risk": risk_pct, "tp1": tp1_price, "tp2": tp2_price,
                    "status": "🚨【盤中即時突破進場】", "score": tightness_score
                }
            return None

        else:
            # 17:00 盤後選股模式：抓今日盤中 (近 9 根 30K) 完成金叉翻紅且尾盤守穩紅柱之強勢股
            recent_9 = df.iloc[-9:]
            had_neg_osc = (df['OSC'].iloc[-10:-1] <= 0).any()
            had_ma_cross = (df['MA20'].iloc[-10:-1] <= df['MA60'].iloc[-10:-1]).any() or (ma_gap_pct <= 0.008)
            vol_spike = (recent_9['Volume'] >= recent_9['VOL_MA5'] * VOL_MULTIPLIER).any()

            if near_zero and (ma20_now > ma60_now) and ma60_rising and (osc_now > 0) and (had_neg_osc or had_ma_cross) and is_breakout and is_bull_bar and vol_spike:
                return {
                    "type": "A", "kind": kind, "ticker": ticker, "code": code, "name": name,
                    "close": round(current_close, 2), "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                    "stop": stop_price, "risk": risk_pct, "tp1": tp1_price, "tp2": tp2_price,
                    "status": "🔥帶量突破金叉", "score": tightness_score
                }

            # 條件 B：明日早盤準金叉觀察 (均線差距 0.8% 內 + 綠柱連續收斂準備翻紅)
            if near_zero and (ma_gap_pct <= 0.008) and ma60_rising and (osc_now > osc_prev1 > osc_prev2) and (-0.004 * current_close <= osc_now <= 0):
                return {
                    "type": "B", "kind": kind, "ticker": ticker, "code": code, "name": name,
                    "close": round(current_close, 2), "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                    "stop": stop_price, "risk": risk_pct, "tp1": tp1_price, "tp2": tp2_price,
                    "status": "⏳準金叉蓄勢", "score": tightness_score
                }
            return None
    except Exception:
        return None

def check_active_positions(positions):
    """盤中每輪優先檢查已觸發進場之持倉部位是否觸及停損、停利(TP1)或30K MACD死叉"""
    if not positions:
        return [], positions

    tickers = list(positions.keys())
    alerts = []
    try:
        batch_data = yf.download(tickers, period="1mo", interval="30m", group_by="ticker", threads=True, progress=False)
        for tk in list(positions.keys()):
            pos = positions[tk]
            df_raw = batch_data[tk] if len(tickers) > 1 and tk in batch_data else batch_data
            df = calc_30k_indicators(df_raw)
            if df is None or df.empty:
                continue

            last = df.iloc[-1]
            prev1 = df.iloc[-2]
            curr_price = round(float(last['Close']), 2)
            curr_low = round(float(last['Low']), 2)
            curr_high = round(float(last['High']), 2)
            ma20_now = round(float(last['MA20']), 2)
            ma60_now = round(float(last['MA60']), 2)
            osc_now = float(last['OSC'])
            osc_prev = float(prev1['OSC'])

            entry_p = pos['entry']
            stop_p = pos['stop']
            tp1_p = pos['tp1']
            tp1_hit = pos.get('tp1_hit', False)
            pnl_pct = round((curr_price - entry_p) / entry_p * 100, 2)

            # 1. 檢查是否跌破硬停損 (或保本停損)
            if curr_price <= stop_p or curr_low <= stop_p:
                tag = "🛡️【觸及保本防線出場】" if tp1_hit else "🛑【觸發硬性停損出場】"
                alerts.append(
                    f"{tag} {pos['code']} {pos['name']}\n"
                    f"• 現價：{curr_price} (觸及防線：{stop_p})\n"
                    f"• 進場成本：{entry_p}｜預估損益：{pnl_pct}%\n"
                    f"• 動作：請立即執行停損/保本出場，收回資金！"
                )
                del positions[tk]
                continue

            # 2. 檢查是否首次達到第一停利目標 TP1 (+3.5%)
            if not tp1_hit and (curr_price >= tp1_p or curr_high >= tp1_p):
                positions[tk]['tp1_hit'] = True
                positions[tk]['stop'] = entry_p  # 自動將停損上移至成本價 (Breakeven)
                alerts.append(
                    f"💰【達標第一停利 TP1！】{pos['code']} {pos['name']}\n"
                    f"• 現價：{curr_price} (突破目標價：{tp1_p}｜帳面 +{pnl_pct}%)\n"
                    f"• 動作：建議先減碼 50% 落袋為安！\n"
                    f"• 風控升級：剩餘 50% 部位停損已自動上移至成本價 `{entry_p}`，立於不敗之地！"
                )
                continue

            # 3. 檢查是否出現 30K MACD 死叉 (紅柱轉綠柱) 且跌破 30K 20MA
            if (osc_now < 0 and osc_prev >= 0) and (curr_price < ma20_now):
                alerts.append(
                    f"⚠️【30K MACD 死叉出場訊號】{pos['code']} {pos['name']}\n"
                    f"• 現價：{curr_price}｜進場價：{entry_p} (損益：{pnl_pct}%)\n"
                    f"• 指標：30K 柱狀體由紅翻綠且跌破 20MA ({ma20_now})\n"
                    f"• 動作：短線動能轉弱，建議獲利了結或減碼出場（波段極限防守 60MA：{ma60_now}）。"
                )
                del positions[tk]
    except Exception as e:
        print(f"⚠️ 持倉監控檢查異常: {e}")

    return alerts, positions

def send_telegram_chunks(full_text):
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    print("================ 訊息預覽 ================")
    print(full_text)
    print("==========================================")
    if not token:
        return

    chunks, current_chunk = [], ""
    for line in full_text.split("\n"):
        if len(current_chunk) + len(line) + 1 > 3500:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += ("\n" + line if current_chunk else line)
    if current_chunk:
        chunks.append(current_chunk)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    target_chat_id = chat_id

    for idx, chunk in enumerate(chunks):
        res = requests.post(url, json={"chat_id": target_chat_id, "text": chunk}, timeout=15)
        if res.status_code != 200 and idx == 0:
            upd = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15).json()
            if upd.get("ok") and upd.get("result"):
                for item in reversed(upd["result"]):
                    msg_obj = item.get("message") or item.get("edited_message") or item.get("my_chat_member")
                    if msg_obj and "chat" in msg_obj:
                        target_chat_id = msg_obj["chat"]["id"]
                        break
                res = requests.post(url, json={"chat_id": target_chat_id, "text": chunk}, timeout=15)
        print(f"Telegram 推播狀態碼: {res.status_code}")

def run_intraday_daemon():
    """週一至週五 08:30 ~ 13:40 盤中即時監控常駐引擎"""
    print(f"🚀 啟動盤中即時監控引擎，目前台灣時間：{now_tw().strftime('%Y-%m-%d %H:%M:%S')}")
    candidates = fetch_all_market_candidates()
    ticker_list = list(candidates.keys())
    positions = load_positions()
    notified_today = set(positions.keys())

    # 開盤前 08:30~09:00 先發送盤前監控就緒與目前持倉摘要
    open_msg = [
        f"🔔 【盤中即時監控系統已就緒 (08:30-13:40)】",
        f"🕒 啟動時間：{now_tw().strftime('%Y-%m-%d %H:%M')}",
        f"📡 今日鎖定全市場高流動性活水池：{len(ticker_list)} 檔",
        f"💼 目前追蹤中持倉部位：{len(positions)} 檔"
    ]
    if positions:
        for tk, p in positions.items():
            open_msg.append(f"  • {p['code']} {p['name']}｜成本：{p['entry']}｜防守：{p['stop']}｜目標：{p['tp1']}")
    open_msg.append("\n⏳ 09:00 開盤後將每 3 分鐘自動掃描進場、停損與停利訊號！")
    send_telegram_chunks("\n".join(open_msg))

    # 持續循環監控直到台灣時間 13:40
    while True:
        curr = now_tw()
        hhmm = curr.hour * 100 + curr.minute
        if hhmm > 1340:
            print("🏁 已過 13:40 收盤監控時間，盤中引擎正常結束。")
            break

        # 08:30~09:00 等待開盤
        if hhmm < 900:
            time.sleep(30)
            continue

        print(f"🔍 [{curr.strftime('%H:%M:%S')}] 執行盤中即時掃描...")
        # 1. 優先監控既有部位是否觸發停損/停利
        exit_alerts, positions = check_active_positions(positions)
        if exit_alerts:
            save_positions(positions)
            alert_msg = f"⚡ 【盤中持倉停損／停利即時警報】 ({curr.strftime('%H:%M')})\n\n" + "\n\n".join(exit_alerts)
            send_telegram_chunks(alert_msg)

        # 2. 掃描全市場是否有新觸發「30K 金叉 + MACD 零軸翻紅 + 帶量突破」之標的
        new_entries = []
        for i in range(0, len(ticker_list), BATCH_SIZE):
            batch = [t for t in ticker_list[i:i + BATCH_SIZE] if t not in notified_today]
            if not batch:
                continue
            try:
                batch_data = yf.download(batch, period="1mo", interval="30m", group_by="ticker", threads=True, progress=False)
                for tk in batch:
                    info = candidates[tk]
                    df_tk = batch_data[tk] if len(batch) > 1 and tk in batch_data else batch_data
                    res = evaluate_signal(df_tk, tk, info["name"], info["kind"], mode="intraday")
                    if res:
                        new_entries.append(res)
                        notified_today.add(tk)
                        positions[tk] = {
                            "code": res["code"], "name": res["name"], "entry": res["close"],
                            "stop": res["stop"], "tp1": res["tp1"], "tp2": res["tp2"],
                            "tp1_hit": False, "time": curr.strftime("%Y-%m-%d %H:%M")
                        }
            except Exception as e:
                print(f"Intraday batch error: {e}")

        if new_entries:
            new_entries.sort(key=lambda x: x["score"])
            # 若同一瞬間過多標的，精選收斂度最緊密的前 5 檔推播並建倉追蹤
            top_entries = new_entries[:5]
            save_positions(positions)
            entry_lines = [f"🚨 【盤中策略觸發！第一時間進場通知】 ({curr.strftime('%H:%M')})\n"]
            for r in top_entries:
                entry_lines.append(
                    f"🔥 *{r['code']} {r['name']}* {r['status']}\n"
                    f"• 建議進場參考價：`{r['close']}` (突破近12根30K平台)\n"
                    f"• 30K 均線 (20/60MA)：`{r['ma20']}` / `{r['ma60']}`\n"
                    f"• 🛑 硬性停損價：`{r['stop']}` (風險 `-{r['risk']}%`，破線無條件走)\n"
                    f"• 💰 第一停利 TP1：`{r['tp1']}` (+3.5% 減碼半倉、停損移至保本)\n"
                    f"• 🎯 波段停利 TP2：`{r['tp2']}` (或守 30K 20MA / MACD死叉出清)"
                )
            send_telegram_chunks("\n\n".join(entry_lines))

        # 每 180 秒 (3 分鐘) 巡邏一次
        time.sleep(180)

def run_eod_report():
    """每日 17:00 盤後全市場選股日報 + 持倉總結"""
    tw_time = now_tw().strftime("%Y-%m-%d %H:%M")
    candidates = fetch_all_market_candidates()
    ticker_list = list(candidates.keys())
    positions = load_positions()

    etf_matches, stock_a, stock_b = [], [], []
    for i in range(0, len(ticker_list), BATCH_SIZE):
        batch = ticker_list[i:i + BATCH_SIZE]
        try:
            batch_data = yf.download(batch, period="1mo", interval="30m", group_by="ticker", threads=True, progress=False)
            for tk in batch:
                info = candidates[tk]
                df_tk = batch_data[tk] if len(batch) > 1 and tk in batch_data else batch_data
                res = evaluate_signal(df_tk, tk, info["name"], info["kind"], mode="eod")
                if res:
                    if res["kind"] == "ETF":
                        etf_matches.append(res)
                    elif res["type"] == "A":
                        stock_a.append(res)
                    else:
                        stock_b.append(res)
        except Exception as e:
            print(f"EOD Batch error: {e}")

    etf_matches.sort(key=lambda x: (x["type"], x["score"]))
    stock_a.sort(key=lambda x: x["score"])
    stock_b.sort(key=lambda x: x["score"])

    # 自動將今日尾盤確認的精選前 5 強納入明日盤中停損停利追蹤池
    for r in (stock_a[:5] + etf_matches[:2]):
        tk = r["ticker"]
        if tk not in positions:
            positions[tk] = {
                "code": r["code"], "name": r["name"], "entry": r["close"],
                "stop": r["stop"], "tp1": r["tp1"], "tp2": r["tp2"],
                "tp1_hit": False, "time": tw_time
            }
    save_positions(positions)

    msg_lines = [
        f"📊 【全市場高勝率優化版 30K 盤後選股日報】",
        f"🕒 結算時間：{tw_time}｜掃描活水標的：{len(ticker_list)} 檔",
        f"*(已剔除債券/平衡型ETF，並加上12根30K平台突破＋實體紅K濾網)*\n"
    ]

    msg_lines.append(f"🛡️ 一、【動能＆主動式 ETF 專區】(共 {len(etf_matches)} 檔)")
    if etf_matches:
        for r in etf_matches[:8]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}]\n"
                f"  收盤：{r['close']}｜停損：{r['stop']} (-{r['risk']}%)｜TP1：{r['tp1']}｜TP2：{r['tp2']}"
            )
    else:
        msg_lines.append("• 今日無符合嚴格突破條件之動能 ETF")
    msg_lines.append("")

    msg_lines.append(f"🎯 二、【上市櫃個股：今日帶量突破金叉】(共 {len(stock_a)} 檔，精選前 12 強)")
    if stock_a:
        for r in stock_a[:12]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}]\n"
                f"  收盤：{r['close']}｜30K(20/60MA)：{r['ma20']}/{r['ma60']}\n"
                f"  🛑停損：{r['stop']} (-{r['risk']}%)｜💰TP1：{r['tp1']}｜🎯TP2：{r['tp2']}"
            )
    else:
        msg_lines.append("• 今日無符合嚴格突破條件之個股")
    msg_lines.append("")

    msg_lines.append(f"👀 三、【明日早盤準金叉蓄勢觀察】(共 {len(stock_b)} 檔，精選前 10 強)")
    if stock_b:
        for r in stock_b[:10]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}]\n"
                f"  收盤：{r['close']}｜明日突破轉強價：{r['ma20']}｜硬停損：{r['stop']} (-{r['risk']}%)"
            )
    else:
        msg_lines.append("• 今日無準金叉收斂個股")

    msg_lines.append("\n💡 明日 08:30 盤中監控引擎將自動啟動，即時追蹤上述部位之停損與停利！")
    send_telegram_chunks("\n".join(msg_lines))

if __name__ == "__main__":
    mode = "eod"
    if len(sys.argv) > 1 and sys.argv[1] in ["intraday", "eod"]:
        mode = sys.argv[1]
    else:
        # 若未指定參數，依台灣時間自動判斷：08:00~13:45 執行盤中監控，其餘時間執行盤後日報
        hhmm = now_tw().hour * 100 + now_tw().minute
        mode = "intraday" if 800 <= hhmm <= 1345 else "eod"

    if mode == "intraday":
        run_intraday_daemon()
    else:
        run_eod_report()
