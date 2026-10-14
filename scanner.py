Python
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ==========================================
# 1. 設定觀察母體池 (上市加 .TW / 上櫃加 .TWO)
# 主人可隨時在此字典新增關注的成分股或強勢股
# ==========================================
WATCHLIST = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2383.TW": "台光電", "3711.TW": "日月光投控", "2317.TW": "鴻海",
    "2382.TW": "廣達", "3231.TW": "緯創", "6669.TW": "緯穎",
    "3017.TW": "奇鋐", "3324.TWO": "雙鴻", "3037.TW": "欣興",
    "8046.TW": "南電", "3189.TW": "景碩", "6274.TWO": "台燿",
    "2368.TW": "金像電", "3661.TW": "世芯-KY", "3443.TW": "創意",
    "3008.TW": "大立光", "2603.TW": "長榮", "2609.TW": "陽明",
    "2615.TW": "萬海", "2618.TW": "長榮航", "2891.TW": "中信金",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2409.TW": "友達",
    "3450.TW": "聯鈞", "4979.TWO": "華星光", "3163.TWO": "波若威"
}

# 策略參數設定
ZERO_BOUND_PCT = 0.008  # 定義 MACD 的 DIF 在 0 軸附近容許範圍 (股價的 0.8% 內)
VOL_MULTIPLIER = 1.3    # 點火紅 K 成交量需大於 30K 5均量的倍數

def analyze_ticker(ticker, name):
    try:
        # 抓取近 60 天之 30 分鐘 K 線資料
        df = yf.Ticker(ticker).history(period="60d", interval="30m")
        if df.empty or len(df) < 120:
            return None

        # 計算日線級別 20MA 濾網 (將 30K 收盤價重取樣為日線收盤價)
        daily_close = df['Close'].resample('1D').last().dropna()
        if len(daily_close) < 20:
            return None
        daily_ma20 = daily_close.rolling(20).mean().iloc[-1]
         daily_ma20_prev = daily_close.rolling(20).mean().iloc[-2]
        current_close = df['Close'].iloc[-1]

        # 大週期濾網：日收盤價需在日線 20MA 之上，且日線 20MA 向上
        if not (current_close > daily_ma20 and daily_ma20 >= daily_ma20_prev):
            return None

        # 計算 30K 的 20MA、60MA 與 5均量
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['VOL_MA5'] = df['Volume'].rolling(window=5).mean()

        # 計算 30K 的 MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = ema12 - ema26
        df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD']  # 柱狀體 (負為綠柱，正為紅柱)

        # 取最後 4 根 30K (約當日後半場走勢) 進行判定
        last = df.iloc[-1]
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]

        # 檢核 DIF 是否在 0 軸附近
        near_zero = abs(last['DIF']) <= (current_close * ZERO_BOUND_PCT)
        if not near_zero:
            return None

        # 條件 A【已點火金叉】：20MA > 60MA (近4根內剛金叉或糾結向上) 且 OSC 剛由綠轉紅
        ma_cross = (last['MA20'] > last['MA60']) and (
            (prev1['MA20'] <= prev1['MA60']) or 
            (prev2['MA20'] <= prev2['MA60']) or 
            ((last['MA20'] - last['MA60']) / current_close <= 0.008)
        )
        osc_turn_red = (last['OSC'] > 0) and (prev1['OSC'] <= 0 or prev2['OSC'] <= 0)
        vol_ok = last['Volume'] >= (last['VOL_MA5'] * VOL_MULTIPLIER)

        # 條件 B【準金叉蓄勢】：20MA 逼近或剛站上 60MA，且綠柱連續收斂即將翻紅 (明日早盤觀察)
        ma_tight = abs(last['MA20'] - last['MA60']) / current_close <= 0.006 and last['MA20'] >= prev1['MA20']
        osc_ready = (-0.003 * current_close <= last['OSC'] <= 0) and (last['OSC'] > prev1['OSC'] > prev2['OSC'])

        # 硬性停損設於 30K 60MA 下方 0.5%
        stop_price = round(last['MA60'] * 0.995, 2)
        risk_pct = round((current_close - stop_price) / current_close * 100, 2)
        code = ticker.split('.')[0]

        if ma_cross and osc_turn_red:
            status = "🔥【今日已金叉翻紅】" if vol_ok else "✅【尾盤金叉翻紅(溫和量)】"
            return {
                "type": "A", "code": code, "name": name, "close": round(current_close, 2),
                "ma20": round(last['MA20'], 2), "ma60": round(last['MA60'], 2),
                "stop": stop_price, "risk": risk_pct, "status": status
            }
        elif ma_tight and osc_ready:
            return {
                "type": "B", "code": code, "name": name, "close": round(current_close, 2),
                "ma20": round(last['MA20'], 2), "ma60": round(last['MA60'], 2),
                "stop": stop_price, "risk": risk_pct, "status": "⏳【準金叉綠柱收斂】"
            }
        return None
    except Exception as e:
        print(f"Error scanning {ticker}: {e}")
        return None

def send_telegram(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("未設定 Telegram 金鑰，直接印出結果：\n", message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload, timeout=15)

if __name__ == "__main__":
    tw_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    results_a, results_b = [], []

    for ticker, name in WATCHLIST.items():
        res = analyze_ticker(ticker, name)
        if res:
            if res["type"] == "A":
                results_a.append(res)
            else:
                results_b.append(res)

    msg_lines = [f"📊 *【台股 30K 均線金叉＋MACD零軸選股日報】*\n🕒 結算時間：{tw_time}\n"]

    if not results_a and not results_b:
        msg_lines.append("今日觀察池無符合「日線20MA之上 ＋ 30K零軸金叉/準金叉」條件之標的，建議保持耐心觀望。")
    else:
        if results_a:
            msg_lines.append("🎯 *一、已觸發訊號（30K金叉＋綠轉紅）*")
            for r in results_a:
                msg_lines.append(
                    f"• *{r['code']} {r['name']}* {r['status']}\n"
                    f"  收盤：`{r['close']}`｜30K(20/60MA)：`{r['ma20']}` / `{r['ma60']}`\n"
                    f"  硬停損價：`{r['stop']}` (預估風險 `{r['risk']}%`)"
                )
            msg_lines.append("")
        if results_b:
            msg_lines.append("👀 *二、明日早盤觀察（30K糾結＋綠柱收斂準翻紅）*")
            for r in results_b:
                msg_lines.append(
                    f"• *{r['code']} {r['name']}* {r['status']}\n"
                    f"  收盤：`{r['close']}`｜明日突破轉強價：`{r['ma20']}`\n"
                    f"  硬停損價：`{r['stop']}` (預估風險 `{r['risk']}%`)"
                )

    msg_lines.append("\n💡 *出場紀律：跌破硬停損無條件出場；獲利後首次30K MACD死叉減碼50%，剩餘守30K之60MA。*")
    final_msg = "\n".join(msg_lines)
    send_telegram(final_msg)
