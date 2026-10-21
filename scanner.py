import os
import re
import sys
import json
import time
import subprocess
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ==========================================================
# 核心策略與資金帳本參數設定 (主人可隨時微調)
# ==========================================================
CAPITAL_PER_TRADE = 200000  # 預設每筆交易投入本金 (新台幣 200,000 元，用於計算賺賠金額)
FEE_TAX_RATE = 0.004        # 台股單次買賣來回摩擦成本 (證交稅 0.3% + 雙邊手續費約 0.1% = 0.4%)
MIN_STOCK_VOL_LOTS = 1500   # 個股最低日成交量門檻 (張)
MIN_ETF_VOL_LOTS = 500      # ETF 最低日成交量門檻 (張)
ZERO_BOUND_PCT = 0.012      # 30K DIF 距離 0 軸容許範圍 (股價 1.2% 內)
VOL_MULTIPLIER = 1.5        # 觸發當根 30K 成交量需大於 5均量的 1.5 倍
MIN_STOP_PCT = 0.015        # 最小停損呼吸空間 (1.5%)
MAX_STOP_PCT = 0.038        # 最大容許初始停損上限 (3.8%)，超過代表乖離過大不追高
TP1_PCT = 0.035             # 第一階段停利 (+3.5% 減碼 50% 並移至保本+0.5%)
TP2_PCT = 0.080             # 第二階段波段目標 (+8.0% 全數止盈)
MAX_HOLD_DAYS = 5           # 時間停損天數 (滿 5 天未達 TP1 且漲幅 < 1% 自動清倉換股)
BATCH_SIZE = 60
STATE_FILE = "positions.json"
TW_TZ = timezone(timedelta(hours=8))

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
    code = str(code).strip()
    if re.match(r'^[1-9]\d{3}$', code):
        return "STOCK"
    if re.match(r'^00\d{2,4}[AL]?$', code):
        return "ETF"
    return None

def load_portfolio():
    """載入持倉與歷史清倉帳本 (自動相容舊版格式)"""
    default_state = {"active": {}, "closed_history": []}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "active" in data and "closed_history" in data:
                return data
            # 相容舊版直接存放持倉字典之結構
            active_migrated = {}
            for tk, v in data.items():
                if isinstance(v, dict) and "entry" in v:
                    v.setdefault("capital", CAPITAL_PER_TRADE)
                    v.setdefault("remain_ratio", 0.5 if v.get("tp1_hit") else 1.0)
                    v.setdefault("realized_pnl_ntd", 0)
                    v.setdefault("tp2", round(v["entry"] * (1.0 + TP2_PCT), 2))
                    active_migrated[tk] = v
            return {"active": active_migrated, "closed_history": []}
        except Exception as e:
            print(f"⚠️ 讀取帳本異常，初始化新帳本: {e}")
    return default_state

def save_portfolio(state, git_sync=False):
    """儲存持倉與清倉紀錄，並可選擇立即同步回 GitHub 儲存庫"""
    try:
        # 僅保留最近 200 筆歷史清倉紀錄，避免檔案無限膨脹
        state["closed_history"] = state.get("closed_history", [])[-200:]
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        if git_sync:
            subprocess.run(["git", "config", "user.name", "stock-monitor-bot"], check=False)
            subprocess.run(["git", "config", "user.email", "bot@users.noreply.github.com"], check=False)
            subprocess.run(["git", "add", STATE_FILE], check=False)
            diff_res = subprocess.run(["git", "diff", "--staged", "--quiet"], check=False)
            if diff_res.returncode != 0:
                subprocess.run(["git", "commit", "-m", "Auto-sync portfolio state [skip ci]"], check=False)
                subprocess.run(["git", "pull", "--rebase", "-X", "theirs"], check=False)
                subprocess.run(["git", "push"], check=False)
    except Exception as e:
        print(f"⚠️ 儲存帳本失敗: {e}")

def fetch_all_market_candidates():
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
    """計算 30K 20/60MA、MACD(12,26,9) 與 RSI(14)"""
    if df is None or df.empty:
        return None
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

    # 計算 30K RSI(14) 防止過度追高
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = df['RSI'].fillna(50.0)
    return df

def evaluate_signal(raw_df, ticker, name, kind, mode="eod"):
    """
    高勝率＆高賺賠比策略核心：
    1. 大週期保護：日線收盤 > 日線20MA 且 > 日線60MA，且日線20MA向上
    2. 30K 趨勢與零軸：20MA > 60MA，60MA 斜率向上，DIF 位於 0 軸 1.2% 內，OSC 由綠翻紅
    3. K棒品質與動能甜蜜區：實體紅K、上影線 <= K棒全長35%、30K RSI 介於 50~72、突破前12根30K高點
    4. 風控盈虧比把關：初始停損風險介於 1.5% ~ 3.8% 之間 (超過 3.8% 不追高)
    """
    try:
        df = calc_30k_indicators(raw_df)
        if df is None:
            return None

        daily_close = df['Close'].resample('1D').last().dropna()
        if len(daily_close) < 15:
            return None
        ma20_d_series = daily_close.rolling(min(20, len(daily_close))).mean()
        ma60_d_series = daily_close.rolling(min(60, len(daily_close))).mean()
        daily_ma20 = float(ma20_d_series.iloc[-1])
        daily_ma20_prev = float(ma20_d_series.iloc[-2]) if len(ma20_d_series) >= 2 else daily_ma20
        daily_ma60 = float(ma60_d_series.iloc[-1])
        current_close = float(df['Close'].iloc[-1])

        # 日線多頭共振濾網
        if current_close < daily_ma20 or current_close < daily_ma60 or daily_ma20 < daily_ma20_prev:
            return None

        last = df.iloc[-1]
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]

        ma20_now, ma60_now = float(last['MA20']), float(last['MA60'])
        ma60_prev = float(prev1['MA60'])
        dif_now, osc_now = float(last['DIF']), float(last['OSC'])
        osc_prev1, osc_prev2 = float(prev1['OSC']), float(prev2['OSC'])
        rsi_now = float(last['RSI'])

        if np.isnan(ma60_now) or ma60_now <= 0:
            return None

        # 計算硬停損與兩階段停利
        stop_by_ma = ma60_now * 0.995
        stop_by_pct = current_close * (1.0 - MIN_STOP_PCT)
        stop_price = round(min(stop_by_ma, stop_by_pct), 2)
        risk_pct = round((current_close - stop_price) / current_close * 100, 2)

        # 若停損距離超過 3.8%，代表已大幅脫離 60MA 成本區，直接過濾不追高以保護賺賠比
        if risk_pct > (MAX_STOP_PCT * 100):
            return None

        tp1_price = round(max(current_close * (1.0 + TP1_PCT), current_close + (current_close - stop_price) * 1.8), 2)
        tp2_price = round(max(current_close * (1.0 + TP2_PCT), current_close + (current_close - stop_price) * 3.5), 2)

        code = ticker.split('.')[0]
        ma_gap_pct = abs(ma20_now - ma60_now) / current_close
        dif_zero_pct = abs(dif_now) / current_close
        tightness_score = round((ma_gap_pct + dif_zero_pct) * 100, 2)
        near_zero = dif_zero_pct <= ZERO_BOUND_PCT

        # K 棒品質：實體紅 K 且上影線不過長
        bar_range = max(float(last['High']) - float(last['Low']), 0.0001)
        upper_wick_ratio = (float(last['High']) - float(last['Close'])) / bar_range
        is_bull_bar = (float(last['Close']) > float(last['Open'])) and (upper_wick_ratio <= 0.38)
        prev_12_high = float(df['Close'].iloc[-13:-1].max())
        is_breakout = current_close >= prev_12_high
        ma60_rising = ma60_now >= ma60_prev
        rsi_sweet = 50.0 <= rsi_now <= 73.0

        if mode == "intraday":
            just_turned_red = (osc_now > 0) and (osc_prev1 <= 0 or osc_prev2 <= 0)
            ma_just_crossed = (ma20_now > ma60_now) and (ma_gap_pct <= 0.01)
            vol_ok = float(last['Volume']) >= float(last['VOL_MA5']) * VOL_MULTIPLIER or float(prev1['Volume']) >= float(prev1['VOL_MA5']) * VOL_MULTIPLIER

            if near_zero and ma_just_crossed and ma60_rising and just_turned_red and is_bull_bar and is_breakout and vol_ok and rsi_sweet:
                return {
                    "type": "ENTRY", "kind": kind, "ticker": ticker, "code": code, "name": name,
                    "close": round(current_close, 2), "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                    "stop": stop_price, "risk": risk_pct, "tp1": tp1_price, "tp2": tp2_price,
                    "rsi": round(rsi_now, 1), "status": "🚨【盤中帶量突破進場】", "score": tightness_score
                }
            return None

        else:
            recent_9 = df.iloc[-9:]
            had_neg_osc = (df['OSC'].iloc[-10:-1] <= 0).any()
            had_ma_cross = (df['MA20'].iloc[-10:-1] <= df['MA60'].iloc[-10:-1]).any() or (ma_gap_pct <= 0.008)
            vol_spike = (recent_9['Volume'] >= recent_9['VOL_MA5'] * VOL_MULTIPLIER).any()

            if near_zero and (ma20_now > ma60_now) and ma60_rising and (osc_now > 0) and (had_neg_osc or had_ma_cross) and is_breakout and is_bull_bar and vol_spike and rsi_sweet:
                return {
                    "type": "A", "kind": kind, "ticker": ticker, "code": code, "name": name,
                    "close": round(current_close, 2), "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                    "stop": stop_price, "risk": risk_pct, "tp1": tp1_price, "tp2": tp2_price,
                    "rsi": round(rsi_now, 1), "status": "🔥帶量突破金叉", "score": tightness_score
                }

            if near_zero and (ma_gap_pct <= 0.008) and ma60_rising and (osc_now > osc_prev1 > osc_prev2) and (-0.004 * current_close <= osc_now <= 0):
                return {
                    "type": "B", "kind": kind, "ticker": ticker, "code": code, "name": name,
                    "close": round(current_close, 2), "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                    "stop": stop_price, "risk": risk_pct, "tp1": tp1_price, "tp2": tp2_price,
                    "rsi": round(rsi_now, 1), "status": "⏳準金叉蓄勢", "score": tightness_score
                }
            return None
    except Exception:
        return None

def update_and_check_positions(state, is_eod=False):
    """
    檢查並更新所有持倉部位之最新價格、未實現損益，並判定是否觸發：
    1. 硬停損 / 保本防線出場
    2. TP1 (+3.5% / 1.8R) 減碼 50% 鎖利並將停損移至「成本+0.5%」
    3. TP2 (+8.0% / 3.5R) 波段目標全數止盈
    4. 30K MACD 死叉 + 跌破 20MA 出場
    5. 滿 5 日動能停滯之時間停損 (僅於盤後結算執行)
    """
    active = state.get("active", {})
    closed_history = state.get("closed_history", [])
    if not active:
        return [], state

    tickers = list(active.keys())
    alerts = []
    now_str = now_tw().strftime("%Y-%m-%d %H:%M")
    today_date = now_tw().strftime("%Y-%m-%d")

    try:
        batch_data = yf.download(tickers, period="1mo", interval="30m", group_by="ticker", threads=True, progress=False)
        for tk in list(active.keys()):
            pos = active[tk]
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

            entry_p = float(pos['entry'])
            stop_p = float(pos['stop'])
            tp1_p = float(pos['tp1'])
            tp2_p = float(pos.get('tp2', round(entry_p * (1.0 + TP2_PCT), 2)))
            capital = float(pos.get('capital', CAPITAL_PER_TRADE))
            remain_ratio = float(pos.get('remain_ratio', 1.0))
            realized_ntd = float(pos.get('realized_pnl_ntd', 0.0))
            tp1_hit = bool(pos.get('tp1_hit', False))

            # 更新持倉最新現價與未實現損益 (供 17:00 日報展示)
            pos['last_price'] = curr_price
            gross_ret = (curr_price - entry_p) / entry_p
            net_ret_rem = gross_ret - FEE_TAX_RATE
            unrealized_ntd = round(capital * remain_ratio * net_ret_rem)
            total_pos_ntd = round(realized_ntd + unrealized_ntd)
            total_pos_pct = round(total_pos_ntd / capital * 100, 2)
            pos['unrealized_ntd'] = unrealized_ntd
            pos['total_pnl_ntd'] = total_pos_ntd
            pos['total_pnl_pct'] = total_pos_pct

            # 情況 1：觸及硬停損 或 TP1 後的保本防線
            if curr_price <= stop_p or curr_low <= stop_p:
                exit_exec_p = stop_p if curr_low <= stop_p else curr_price
                net_ret_exit = ((exit_exec_p - entry_p) / entry_p) - FEE_TAX_RATE
                exit_ntd = round(capital * remain_ratio * net_ret_exit)
                final_ntd = round(realized_ntd + exit_ntd)
                final_pct = round(final_ntd / capital * 100, 2)
                reason = "🛡️TP1後保本止盈清倉" if tp1_hit else "🛑觸及硬停損清倉"

                closed_record = {
                    "ticker": tk, "code": pos["code"], "name": pos["name"],
                    "entry": entry_p, "exit": exit_exec_p,
                    "entry_time": pos.get("time", ""), "exit_time": now_str, "exit_date": today_date,
                    "reason": reason, "net_pnl_pct": final_pct, "net_pnl_ntd": final_ntd
                }
                closed_history.append(closed_record)
                alerts.append(
                    f"{reason}｜*{pos['code']} {pos['name']}*\n"
                    f"• 進場價：`{entry_p}` ➔ 出場價：`{exit_exec_p}`\n"
                    f"• 實質淨損益（含已鎖利、扣稅費）：`{final_pct:+.2f}%`（`{final_ntd:+,} 元`）"
                )
                del active[tk]
                continue

            # 情況 2：達陣 TP2 波段大目標 (+8% 或 3.5R) 全數獲利清倉
            if curr_price >= tp2_p or curr_high >= tp2_p:
                exit_exec_p = tp2_p if curr_high >= tp2_p else curr_price
                net_ret_exit = ((exit_exec_p - entry_p) / entry_p) - FEE_TAX_RATE
                exit_ntd = round(capital * remain_ratio * net_ret_exit)
                final_ntd = round(realized_ntd + exit_ntd)
                final_pct = round(final_ntd / capital * 100, 2)
                reason = "🎯達陣TP2波段目標清倉"

                closed_history.append({
                    "ticker": tk, "code": pos["code"], "name": pos["name"],
                    "entry": entry_p, "exit": exit_exec_p,
                    "entry_time": pos.get("time", ""), "exit_time": now_str, "exit_date": today_date,
                    "reason": reason, "net_pnl_pct": final_pct, "net_pnl_ntd": final_ntd
                })
                alerts.append(
                    f"🎯【恭喜達陣 TP2 波段目標清倉！】*{pos['code']} {pos['name']}*\n"
                    f"• 進場價：`{entry_p}` ➔ 止盈價：`{exit_exec_p}`\n"
                    f"• 實質總淨獲利（扣稅費）：`{final_pct:+.2f}%`（`{final_ntd:+,} 元`）"
                )
                del active[tk]
                continue

            # 情況 3：首次達陣 TP1 (+3.5% 或 1.8R) -> 減碼 50% 鎖利，停損上移至「成本+0.5%」
            if not tp1_hit and (curr_price >= tp1_p or curr_high >= tp1_p):
                exec_tp1 = tp1_p if curr_high >= tp1_p else curr_price
                net_ret_half = ((exec_tp1 - entry_p) / entry_p) - FEE_TAX_RATE
                half_ntd = round(capital * 0.5 * net_ret_half)
                new_stop = round(entry_p * 1.005, 2)  # 鎖住稅費後淨賺保本線

                pos['tp1_hit'] = True
                pos['remain_ratio'] = 0.5
                pos['realized_pnl_ntd'] = half_ntd
                pos['stop'] = new_stop
                pos['total_pnl_ntd'] = half_ntd + round(capital * 0.5 * (((curr_price - entry_p) / entry_p) - FEE_TAX_RATE))
                pos['total_pnl_pct'] = round(pos['total_pnl_ntd'] / capital * 100, 2)

                alerts.append(
                    f"💰【達標 TP1 減碼 50% 鎖利！】*{pos['code']} {pos['name']}*\n"
                    f"• 進場價：`{entry_p}` ➔ 減碼價：`{exec_tp1}`\n"
                    f"• 已實現入袋（半倉扣稅費）：`+{half_ntd:,} 元`\n"
                    f"• 🛡️ 剩餘 50% 部位防守線已上移至：`{new_stop}`（確保全單必賺），續抱挑戰 TP2 `{tp2_p}`！"
                )
                continue

            # 情況 4：出現 30K MACD 死叉 且 跌破 30K 20MA -> 動能轉弱清倉
            if (osc_now < 0 and osc_prev >= 0) and (curr_price < ma20_now):
                net_ret_exit = ((curr_price - entry_p) / entry_p) - FEE_TAX_RATE
                exit_ntd = round(capital * remain_ratio * net_ret_exit)
                final_ntd = round(realized_ntd + exit_ntd)
                final_pct = round(final_ntd / capital * 100, 2)
                reason = "⚠️30K死叉破20MA清倉"

                closed_history.append({
                    "ticker": tk, "code": pos["code"], "name": pos["name"],
                    "entry": entry_p, "exit": curr_price,
                    "entry_time": pos.get("time", ""), "exit_time": now_str, "exit_date": today_date,
                    "reason": reason, "net_pnl_pct": final_pct, "net_pnl_ntd": final_ntd
                })
                alerts.append(
                    f"⚠️【30K 死叉破線清倉】*{pos['code']} {pos['name']}*\n"
                    f"• 進場價：`{entry_p}` ➔ 清倉價：`{curr_price}` (跌破30K 20MA `{ma20_now}`)\n"
                    f"• 實質總淨損益（扣稅費）：`{final_pct:+.2f}%`（`{final_ntd:+,} 元`）"
                )
                del active[tk]
                continue

            # 情況 5：盤後檢查「時間停損」(持倉超過 5 天未達 TP1 且漲幅 < 1%)
            if is_eod and not tp1_hit:
                try:
                    entry_dt = datetime.strptime(pos.get("time", "")[:10], "%Y-%m-%d")
                    days_held = (now_tw().replace(tzinfo=None) - entry_dt).days
                    if days_held >= MAX_HOLD_DAYS and gross_ret < 0.01:
                        net_ret_exit = gross_ret - FEE_TAX_RATE
                        final_ntd = round(capital * remain_ratio * net_ret_exit)
                        final_pct = round(final_ntd / capital * 100, 2)
                        reason = "⏱️滿5日動能停滯清倉"
                        closed_history.append({
                            "ticker": tk, "code": pos["code"], "name": pos["name"],
                            "entry": entry_p, "exit": curr_price,
                            "entry_time": pos.get("time", ""), "exit_time": now_str, "exit_date": today_date,
                            "reason": reason, "net_pnl_pct": final_pct, "net_pnl_ntd": final_ntd
                        })
                        alerts.append(
                            f"⏱️【時間停損清倉換股】*{pos['code']} {pos['name']}*\n"
                            f"• 持倉逾 {days_held} 日動能停滯｜進場：`{entry_p}` ➔ 清倉：`{curr_price}`\n"
                            f"• 實質淨損益：`{final_pct:+.2f}%`（`{final_ntd:+,} 元`）"
                        )
                        del active[tk]
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ 持倉更新檢查異常: {e}")

    state["active"] = active
    state["closed_history"] = closed_history
    return alerts, state

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
    print(f"🚀 啟動盤中即時監控引擎：{now_tw().strftime('%Y-%m-%d %H:%M:%S')}")
    candidates = fetch_all_market_candidates()
    ticker_list = list(candidates.keys())
    state = load_portfolio()
    notified_today = set(state["active"].keys())

    open_msg = [
        f"🔔 【盤中即時監控引擎已啟動 (08:30-13:40)】",
        f"🕒 時間：{now_tw().strftime('%Y-%m-%d %H:%M')}｜監控活水池：{len(ticker_list)} 檔",
        f"💼 目前追蹤中持倉：{len(state['active'])} 檔 (單筆基準本金：{CAPITAL_PER_TRADE:,} 元)"
    ]
    for tk, p in state["active"].items():
        ratio_str = "50%(已鎖利)" if p.get("tp1_hit") else "100%"
        open_msg.append(f"  • {p['code']} {p['name']} [{ratio_str}]｜成本：{p['entry']}｜防守：{p['stop']}｜TP1/TP2：{p['tp1']}/{p.get('tp2','-')}")
    open_msg.append("\n⏳ 09:00 開盤後每 3 分鐘自動掃描進場、停損與停利訊號！")
    send_telegram_chunks("\n".join(open_msg))

    while True:
        curr = now_tw()
        hhmm = curr.hour * 100 + curr.minute
        if hhmm > 1340:
            print("🏁 已過 13:40 收盤監控時間，盤中引擎正常結束。")
            break
        if hhmm < 900:
            time.sleep(30)
            continue

        print(f"🔍 [{curr.strftime('%H:%M:%S')}] 執行盤中掃描...")
        exit_alerts, state = update_and_check_positions(state, is_eod=False)
        if exit_alerts:
            save_portfolio(state, git_sync=True)
            alert_msg = f"⚡ 【盤中持倉停損／停利即時警報】 ({curr.strftime('%H:%M')})\n\n" + "\n\n".join(exit_alerts)
            send_telegram_chunks(alert_msg)

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
            except Exception as e:
                print(f"Intraday batch error: {e}")

        if new_entries:
            new_entries.sort(key=lambda x: x["score"])
            top_entries = new_entries[:5]
            for r in top_entries:
                tk = r["ticker"]
                state["active"][tk] = {
                    "code": r["code"], "name": r["name"], "kind": r["kind"],
                    "entry": r["close"], "last_price": r["close"],
                    "stop": r["stop"], "tp1": r["tp1"], "tp2": r["tp2"],
                    "tp1_hit": False, "remain_ratio": 1.0,
                    "capital": CAPITAL_PER_TRADE, "realized_pnl_ntd": 0,
                    "unrealized_ntd": 0, "total_pnl_ntd": 0, "total_pnl_pct": 0.0,
                    "time": curr.strftime("%Y-%m-%d %H:%M")
                }
            save_portfolio(state, git_sync=True)

            entry_lines = [f"🚨 【盤中策略觸發！第一時間進場通知】 ({curr.strftime('%H:%M')})\n"]
            for r in top_entries:
                entry_lines.append(
                    f"🔥 *{r['code']} {r['name']}* {r['status']}\n"
                    f"• 建議進場價：`{r['close']}` (30K RSI：`{r['rsi']}`)\n"
                    f"• 30K 均線 (20/60MA)：`{r['ma20']}` / `{r['ma60']}`\n"
                    f"• 🛑 硬性停損價：`{r['stop']}` (風險 `-{r['risk']}%`，破線無條件走)\n"
                    f"• 💰 第一停利 TP1：`{r['tp1']}` (減碼50%鎖利，停損移至保本)\n"
                    f"• 🎯 波段停利 TP2：`{r['tp2']}` (或守30K 20MA / MACD死叉清倉)"
                )
            send_telegram_chunks("\n\n".join(entry_lines))

        time.sleep(180)

def run_eod_report():
    """每日 17:00 盤後報告：含目前持倉損益、今日已清倉損益、以及今日新選股清單"""
    tw_now = now_tw()
    tw_time = tw_now.strftime("%Y-%m-%d %H:%M")
    today_str = tw_now.strftime("%Y-%m-%d")

    candidates = fetch_all_market_candidates()
    ticker_list = list(candidates.keys())
    state = load_portfolio()

    # 1. 先結算既有持倉今日收盤狀態與觸發清倉紀錄
    eod_exit_alerts, state = update_and_check_positions(state, is_eod=True)

    # 2. 掃描全市場今日新符合之標的
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

    # 組合 17:00 盤後日報 (先列出既有持倉與今日清倉，再列新選股)
    msg_lines = [
        f"📊 【17:00 全市場 30K 策略盤後日報＆持倉帳本】",
        f"🕒 結算時間：{tw_time}｜單筆基準本金：{CAPITAL_PER_TRADE:,} 元\n"
    ]

    # 專區壹：目前持有中部位＆未實現賺賠
    active_pos = state.get("active", {})
    msg_lines.append(f"💼 壹、【目前持倉標的＆未實現賺賠】(共 {len(active_pos)} 檔)")
    if active_pos:
        total_unrealized_ntd = 0
        for tk, p in active_pos.items():
            curr_p = p.get("last_price", p["entry"])
            pnl_pct = p.get("total_pnl_pct", 0.0)
            pnl_ntd = p.get("total_pnl_ntd", 0)
            total_unrealized_ntd += pnl_ntd
            ratio_tag = "剩50%倉(已過TP1鎖利)" if p.get("tp1_hit") else "100%全倉"
            icon = "🟢" if pnl_ntd >= 0 else "🔴"
            msg_lines.append(
                f"{icon} *{p['code']} {p['name']}* [{ratio_tag}]\n"
                f"  進場：`{p['entry']}` ➔ 現價：`{curr_p}`｜防守線：`{p['stop']}`\n"
                f"  目前賺賠：`{pnl_pct:+.2f}%`（`{pnl_ntd:+,} 元`，已含稅費估算）"
            )
        msg_lines.append(f"👉 *目前持倉合計未實現淨損益：`{total_unrealized_ntd:+,} 元`*")
    else:
        msg_lines.append("• 目前無留倉部位（空手觀望中）")
    msg_lines.append("")

    # 專區貳：今日已清倉標的＆實質實現損益
    today_closed = [c for c in state.get("closed_history", []) if c.get("exit_date") == today_str]
    msg_lines.append(f"🧾 貳、【今日已清倉標的＆實質賺賠紀錄】(共 {len(today_closed)} 筆)")
    if today_closed:
        today_realized_ntd = sum(c["net_pnl_ntd"] for c in today_closed)
        for c in today_closed:
            icon = "💰" if c["net_pnl_ntd"] >= 0 else "🛑"
            msg_lines.append(
                f"{icon} *{c['code']} {c['name']}* [{c['reason']}]\n"
                f"  進場：`{c['entry']}` ➔ 清倉：`{c['exit']}`\n"
                f"  實質淨賺賠：`{c['net_pnl_pct']:+.2f}%`（`{c['net_pnl_ntd']:+,} 元`，已扣0.4%稅費）"
            )
        msg_lines.append(f"👉 *今日已清倉實現淨損益合計：`{today_realized_ntd:+,} 元`*")
    else:
        msg_lines.append("• 今日無觸發清倉出場之標的")
    msg_lines.append("")

    # 將今日新選出之精選前 5 強個股與前 2 強 ETF 納入明日持倉追蹤池
    new_added = 0
    for r in (stock_a[:5] + [e for e in etf_matches if e["type"] == "A"][:2]):
        tk = r["ticker"]
        if tk not in state["active"]:
            state["active"][tk] = {
                "code": r["code"], "name": r["name"], "kind": r["kind"],
                "entry": r["close"], "last_price": r["close"],
                "stop": r["stop"], "tp1": r["tp1"], "tp2": r["tp2"],
                "tp1_hit": False, "remain_ratio": 1.0,
                "capital": CAPITAL_PER_TRADE, "realized_pnl_ntd": 0,
                "unrealized_ntd": 0, "total_pnl_ntd": 0, "total_pnl_pct": 0.0,
                "time": tw_time
            }
            new_added += 1
    save_portfolio(state, git_sync=True)

    # 專區參：今日新觸發 ETF
    msg_lines.append(f"🛡️ 參、【今日新觸發：動能＆主動式 ETF】(共 {len(etf_matches)} 檔)")
    if etf_matches:
        for r in etf_matches[:6]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}]\n"
                f"  收盤：{r['close']}｜停損：{r['stop']} (-{r['risk']}%)｜TP1：{r['tp1']}｜TP2：{r['tp2']}"
            )
    else:
        msg_lines.append("• 今日無符合條件之動能 ETF")
    msg_lines.append("")

    # 專區肆：今日新觸發上市櫃個股
    msg_lines.append(f"🎯 肆、【今日新觸發：個股帶量突破金叉】(共 {len(stock_a)} 檔，精選前 10 強)")
    if stock_a:
        for r in stock_a[:10]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}｜RSI:{r['rsi']}]\n"
                f"  收盤：{r['close']}｜30K(20/60MA)：{r['ma20']}/{r['ma60']}\n"
                f"  🛑停損：{r['stop']} (-{r['risk']}%)｜💰TP1：{r['tp1']}｜🎯TP2：{r['tp2']}"
            )
    else:
        msg_lines.append("• 今日無符合嚴格突破條件之個股")
    msg_lines.append("")

    # 專區伍：明日早盤準金叉蓄勢觀察
    msg_lines.append(f"👀 伍、【明日早盤準金叉蓄勢觀察】(共 {len(stock_b)} 檔，精選前 8 強)")
    if stock_b:
        for r in stock_b[:8]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}]\n"
                f"  收盤：{r['close']}｜明日突破價：{r['ma20']}｜硬停損：{r['stop']} (-{r['risk']}%)"
            )
    else:
        msg_lines.append("• 今日無準金叉收斂個股")

    msg_lines.append(f"\n📌 系統已自動將今日精選前 {new_added} 檔新建倉標的寫入追蹤帳本，明日 08:30 繼續即時監控！")
    send_telegram_chunks("\n".join(msg_lines))

def run_weekly_review():
    """每週五 18:00 統計當週進出績效與策略復盤診斷報告"""
    tw_now = now_tw()
    tw_time = tw_now.strftime("%Y-%m-%d %H:%M")
    # 計算本週一日期
    monday_dt = (tw_now - timedelta(days=tw_now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    monday_str = monday_dt.strftime("%Y-%m-%d")

    state = load_portfolio()
    _, state = update_and_check_positions(state, is_eod=False)
    save_portfolio(state, git_sync=True)

    active_pos = state.get("active", {})
    all_closed = state.get("closed_history", [])
    weekly_closed = [c for c in all_closed if c.get("exit_date", "") >= monday_str]

    total_trades = len(weekly_closed)
    wins = [c for c in weekly_closed if c["net_pnl_ntd"] > 0]
    losses = [c for c in weekly_closed if c["net_pnl_ntd"] <= 0]
    win_rate = round(len(wins) / total_trades * 100, 1) if total_trades > 0 else 0.0

    realized_ntd = sum(c["net_pnl_ntd"] for c in weekly_closed)
    unrealized_ntd = sum(p.get("total_pnl_ntd", 0) for p in active_pos.values())
    combined_ntd = realized_ntd + unrealized_ntd

    avg_win_pct = round(np.mean([c["net_pnl_pct"] for c in wins]), 2) if wins else 0.0
    avg_loss_pct = round(abs(np.mean([c["net_pnl_pct"] for c in losses])), 2) if losses else 0.0
    rr_ratio = round(avg_win_pct / avg_loss_pct, 2) if avg_loss_pct > 0 else (9.99 if avg_win_pct > 0 else 0.0)

    gross_profit = sum(c["net_pnl_ntd"] for c in wins)
    gross_loss = abs(sum(c["net_pnl_ntd"] for c in losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (9.99 if gross_profit > 0 else 0.0)

    lines = [
        f"🏆 【每週五 18:00 台股 30K 策略績效總體檢週報】",
        f"📅 統計區間：{monday_str} ～ {tw_now.strftime('%Y-%m-%d')}",
        f"💰 單筆基準本金：{CAPITAL_PER_TRADE:,} 元（均已扣除 0.4% 交易稅費）\n",
        f"📈 *一、【本週核心績效指標總覽】*",
        f"• 本週已清倉筆數：`{total_trades} 筆`（獲利 `{len(wins)}` 筆 ／ 虧損 `{len(losses)}` 筆）",
        f"• 實戰勝率 (Win Rate)：`{win_rate}%`",
        f"• 平均單筆獲利：`+{avg_win_pct}%` ｜ 平均單筆虧損：`-{avg_loss_pct}%`",
        f"• 實戰賺賠比 (R/R Ratio)：`{rr_ratio}` ｜ 獲利因子 (PF)：`{profit_factor}`",
        f"• 💵 *本週已實現總淨損益：`{realized_ntd:+,} 元`*",
        f"• 💼 *目前留倉未實現損益：`{unrealized_ntd:+,} 元`（共 {len(active_pos)} 檔）*",
        f"• 🔥 *本週合計總績效（已實現+未實現）：`{combined_ntd:+,} 元`*\n"
    ]

    lines.append(f"🧾 *二、【本週已清倉全明細】*(共 {total_trades} 筆)")
    if weekly_closed:
        for c in weekly_closed:
            tag = "✅" if c["net_pnl_ntd"] > 0 else "❌"
            lines.append(
                f"{tag} *{c['code']} {c['name']}* [{c['reason']}]\n"
                f"  進出場：`{c['entry']}` ➔ `{c['exit']}` ({c['exit_date']})\n"
                f"  淨損益：`{c['net_pnl_pct']:+.2f}%`（`{c['net_pnl_ntd']:+,} 元`）"
            )
    else:
        lines.append("• 本週尚無已完成清倉之交易紀錄")
    lines.append("")

    lines.append(f"💼 *三、【跨週末留倉部位狀態】*(共 {len(active_pos)} 檔)")
    if active_pos:
        for tk, p in active_pos.items():
            lines.append(
                f"• *{p['code']} {p['name']}*｜進場：`{p['entry']}`｜收盤：`{p.get('last_price', p['entry'])}`\n"
                f"  防守：`{p['stop']}`｜帳面損益：`{p.get('total_pnl_pct', 0.0):+.2f}%`（`{p.get('total_pnl_ntd', 0):+,} 元`）"
            )
    else:
        lines.append("• 目前空手無留倉部位")
    lines.append("")

    # AI 自動策略參數復盤建議
    lines.append("🧠 *四、【本週策略自動診斷與調優建議】*")
    if total_trades == 0:
        lines.append("• 本週無已清倉樣本，請觀察留倉部位下週表現。")
    elif win_rate < 45.0:
        lines.append("• ⚠️ 診斷：本週勝率偏低，顯示盤勢震盪假突破較多。建議下週可將 `VOL_MULTIPLIER` 由 1.5 調高至 1.8，或只做日線 RS 最強前 10% 標的。")
    elif rr_ratio < 1.5:
        lines.append("• ⚠️ 診斷：本週勝率穩定但賺賠比偏低，建議可將 `TP1_PCT` 由 3.5% 微調至 4.0%，讓獲利奔跑空間拉大。")
    else:
        lines.append("• 🌟 診斷：本週策略勝率與賺賠比處於高期望值健康區間，建議維持現有參數紀律執行！")

    lines.append("\n💬 *主人可將此份週報直接貼給 Gemini，我們一起進行每週復盤與參數進化！*")
    send_telegram_chunks("\n".join(lines))

if __name__ == "__main__":
    mode = "eod"
    if len(sys.argv) > 1 and sys.argv[1] in ["intraday", "eod", "weekly"]:
        mode = sys.argv[1]
    else:
        curr = now_tw()
        hhmm = curr.hour * 100 + curr.minute
        if curr.weekday() == 4 and hhmm >= 1740:
            mode = "weekly"
        elif 800 <= hhmm <= 1345:
            mode = "intraday"
        else:
            mode = "eod"

    if mode == "intraday":
        run_intraday_daemon()
    elif mode == "weekly":
        run_weekly_review()
    else:
        run_eod_report()
