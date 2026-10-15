import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# 擴充至 60 檔高流動性核心觀察池 (涵蓋權值、AI伺服器、IC載板、CPO光通訊、航運、金控與主動式ETF核心股)
WATCHLIST = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "2308.TW": "台達電", "2383.TW": "台光電",
    "3711.TW": "日月光投控", "2317.TW": "鴻海", "2382.TW": "廣達", "3231.TW": "緯創",
    "6669.TW": "緯穎", "3017.TW": "奇鋐", "3324.TWO": "雙鴻", "3037.TW": "欣興",
    "8046.TW": "南電", "3189.TW": "景碩", "6274.TWO": "台燿", "2368.TW": "金像電",
    "3661.TW": "世芯-KY", "3443.TW": "創意", "3008.TW": "大立光", "2357.TW": "華碩",
    "2376.TW": "技嘉", "2356.TW": "英業達", "2301.TW": "光寶科", "2327.TW": "國巨",
    "2345.TW": "智邦", "3533.TW": "嘉澤", "3653.TW": "健策", "2059.TW": "川湖",
    "6415.TW": "矽力*-KY", "5269.TW": "祥碩", "6488.TWO": "環球晶", "3105.TWO": "穩懋",
    "2409.TW": "友達", "3481.TW": "群創", "3450.TW": "聯鈞", "4979.TWO": "華星光",
    "3163.TWO": "波若威", "3363.TWO": "上詮", "4908.TWO": "前鼎", "6442.TW": "光聖",
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2618.TW": "長榮航",
    "2610.TW": "華航", "2891.TW": "中信金", "2881.TW": "富邦金", "2882.TW": "國泰金",
    "2886.TW": "兆豐金", "2884.TW": "玉山金", "5880.TW": "合庫金", "1519.TW": "華城",
    "1513.TW": "中興電", "1503.TW": "士電", "1504.TW": "東元", "2002.TW": "中鋼",
    "1101.TW": "台泥", "2912.TW": "統一超", "1216.TW": "統一", "2207.TW": "和泰車"
}

ZERO_BOUND_PCT = 0.015  # 放寬 30K DIF 在 0 軸附近容許範圍至股價的 1.5% 內
VOL_MULTIPLIER = 1.2    # 當日最大點火 30K 量達 5均量的 1.2 倍

def analyze_ticker(ticker, name):
    try:
        df = yf.Ticker(ticker).history(period="60d", interval="30m")
        if df is None or df.empty or len(df) < 100:
            return None

        # 計算日線級別 20MA 濾網
        daily_close = df['Close'].resample('1D').last().dropna()
        if len(daily_close) < 20:
            return None

        daily_ma20 = float(daily_close.rolling(20).mean().iloc[-1])
        current_close = float(df['Close'].iloc[-1])

        # 大週期保護：目前收盤價需位於日線 20MA (月線) 之上
        if current_close < daily_ma20:
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
        df['OSC'] = df['DIF'] - df['MACD']

        last = df.iloc[-1]
        prev1 = df.iloc[-2]
        # 取得今日全天 9 根 30K (09:00~13:30) 與昨日尾盤
        recent_9 = df.iloc[-9:]
        before_today = df.iloc[-11:-8]

        ma20_now = float(last['MA20'])
        ma60_now = float(last['MA60'])
        dif_now = float(last['DIF'])
        osc_now = float(last['OSC'])
        osc_prev = float(prev1['OSC'])

        stop_price = round(ma60_now * 0.995, 2)
        risk_pct = round((current_close - stop_price) / current_close * 100, 2)
        code = ticker.split('.')[0]

        # 計算「30K 均線糾結度 + 零軸距離」綜合評分 (越小代表越接近零軸金叉點)
        ma_gap_pct = abs(ma20_now - ma60_now) / current_close
        dif_zero_pct = abs(dif_now) / current_close
        tightness_score = round((ma_gap_pct + dif_zero_pct) * 100, 2)

        near_zero = dif_zero_pct <= ZERO_BOUND_PCT

        # 條件 A【今日盤中已金叉翻紅】：
        # 收盤 20MA > 60MA 且 OSC > 0，並在過去 9 根 30K (今日盤中) 內曾發生 OSC 由負轉正或 20/60MA 金叉
        had_negative_osc = (df['OSC'].iloc[-10:-1] <= 0).any()
        had_ma_cross = (df['MA20'].iloc[-10:-1] <= df['MA60'].iloc[-10:-1]).any() or (ma_gap_pct <= 0.012)
        vol_spike = (recent_9['Volume'] >= recent_9['VOL_MA5'] * VOL_MULTIPLIER).any()

        if near_zero and (ma20_now > ma60_now) and (osc_now > 0) and (had_negative_osc or had_ma_cross):
            status = "🔥【今日帶量金叉翻紅】" if vol_spike else "✅【今日30K金叉翻紅】"
            return {
                "type": "A", "code": code, "name": name, "close": round(current_close, 2),
                "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                "stop": stop_price, "risk": risk_pct, "status": status, "score": tightness_score
            }

        # 條件 B【準金叉蓄勢】：20MA 與 60MA 差距在 1% 內，且綠柱正在收斂或剛翻紅
        if near_zero and (ma_gap_pct <= 0.01) and (osc_now >= osc_prev) and (-0.005 * current_close <= osc_now <= 0.002 * current_close):
            return {
                "type": "B", "code": code, "name": name, "close": round(current_close, 2),
                "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                "stop": stop_price, "risk": risk_pct, "status": "⏳【準金叉綠柱收斂】", "score": tightness_score
            }

        # 條件 C【備用雷達池】：日線多頭且 30K 20MA >= 60MA (或微幅糾結)，供無標的時列出最接近前 5 名
        if current_close >= ma60_now and osc_now >= osc_prev:
            return {
                "type": "C", "code": code, "name": name, "close": round(current_close, 2),
                "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                "stop": stop_price, "risk": risk_pct, "status": f"📡【均線收斂度 {tightness_score}%】", "score": tightness_score
            }

        return None
    except Exception as e:
        print(f"Error scanning {ticker}: {e}")
        return None

def send_telegram(message):
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    print("================ 選股結果預覽 ================")
    print(message)
    print("==============================================")
    if not token or not chat_id:
        print("⚠️ 尚未偵測到 TELEGRAM_TOKEN 或 TELEGRAM_CHAT_ID")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    res = requests.post(url, json=payload, timeout=15)
    print(f"Telegram 推播狀態碼: {res.status_code}")
    if res.status_code != 200:
        print(f"⚠️ Telegram 錯誤詳細原因: {res.text}")
        print("👉 排解提示：請確認已用手機點開新機器人按 Start，且 TELEGRAM_CHAT_ID 為純數字個人 ID。")

if __name__ == "__main__":
    tw_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    results_a, results_b, results_c = [], [], []

    for ticker, name in WATCHLIST.items():
        res = analyze_ticker(ticker, name)
        if res:
            if res["type"] == "A":
                results_a.append(res)
            elif res["type"] == "B":
                results_b.append(res)
            elif res["type"] == "C":
                results_c.append(res)

    results_a.sort(key=lambda x: x["score"])
    results_b.sort(key=lambda x: x["score"])
    results_c.sort(key=lambda x: x["score"])

    msg_lines = [f"📊 【台股 30K 均線金叉＋MACD零軸選股日報】\n🕒 結算時間：{tw_time}\n"]

    if results_a:
        msg_lines.append("🎯 一、今日已觸發訊號（30K金叉＋綠轉紅）")
        for r in results_a:
            msg_lines.append(
                f"• {r['code']} {r['name']} {r['status']}\n"
                f"  收盤：{r['close']}｜30K(20/60MA)：{r['ma20']} / {r['ma60']}\n"
                f"  硬停損價：{r['stop']} (預估風險 {r['risk']}%)"
            )
        msg_lines.append("")

    if results_b:
        msg_lines.append("👀 二、明日早盤觀察（30K糾結＋綠柱收斂準翻紅）")
        for r in results_b:
            msg_lines.append(
                f"• {r['code']} {r['name']} {r['status']}\n"
                f"  收盤：{r['close']}｜明日突破轉強價：{r['ma20']}\n"
                f"  硬停損價：{r['stop']} (預估風險 {r['risk']}%)"
            )
        msg_lines.append("")

    if not results_a and not results_b:
        msg_lines.append("📌 今日無完全觸發零軸金叉之標的，以下為【30K均線最糾結＋動能轉強】前 5 名觀察股：")
        for r in results_c[:5]:
            msg_lines.append(
                f"• {r['code']} {r['name']} {r['status']}\n"
                f"  收盤：{r['close']}｜30K(20/60MA)：{r['ma20']} / {r['ma60']}\n"
                f"  硬停損價：{r['stop']} (預估風險 {r['risk']}%)"
            )

    msg_lines.append("\n💡 出場紀律：跌破硬停損無條件出場；獲利後首次30K MACD死叉減碼50%，剩餘守30K之60MA。")
    final_msg = "\n".join(msg_lines)
    send_telegram(final_msg)
