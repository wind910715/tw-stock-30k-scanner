import os
import re
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ==========================================================
# 策略與流動性濾網參數設定 (主人可隨時微調)
# ==========================================================
MIN_STOCK_VOL_LOTS = 1200  # 個股最低日成交量門檻 (張)：過濾掉無量殭屍股
MIN_ETF_VOL_LOTS = 300     # ETF 最低日成交量門檻 (張)
ZERO_BOUND_PCT = 0.015     # 30K MACD 之 DIF 在 0 軸附近容許範圍 (股價的 1.5% 內)
VOL_MULTIPLIER = 1.2       # 當日點火 30K 成交量達 5均量的倍數
BATCH_SIZE = 60            # 每次批次下載 30K 的檔數

# 保底核心 ETF 與熱門主動式 ETF 清單 (確保即使證交所 API 假日維護也能 100% 掃描)
CORE_ETFS = {
    "0050.TW": "元大台灣50", "0056.TW": "元大高股息", "006208.TW": "富邦台50",
    "00878.TW": "國泰永續高股息", "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息",
    "00940.TW": "元大台灣價值高息", "00939.TW": "統一台灣高息動能", "00713.TW": "元大台灣高息低波",
    "00918.TW": "大華優利高填息30", "00915.TW": "凱基優選高股息30", "00881.TW": "國泰台灣5G+",
    "00891.TW": "中信關鍵半導體", "00830.TW": "國泰費城半導體", "00757.TW": "統一FANG+",
    "00662.TW": "富邦NASDAQ", "00631L.TW": "元大台灣50正2", "00675L.TW": "富邦臺灣加權正2",
    "00663L.TW": "國泰臺灣加權正2", "00685L.TW": "群益臺灣加權正2", "00980A.TW": "主動式ETF-00980A",
    "00981A.TW": "主動式ETF-00981A", "00982A.TW": "主動式ETF-00982A", "00990A.TW": "主動式ETF-00990A",
    "00991A.TW": "主動式ETF-00991A", "00679B.TWO": "元大美債20年", "00687B.TWO": "國泰20年美債"
}

def is_valid_code(code):
    """判斷是否為合法的台股 4 碼個股或 00 開頭之 ETF (含 A 主動式 / L 槓桿 / B 債券)，排除權證"""
    code = str(code).strip()
    if re.match(r'^[1-9]\d{3}$', code):
        return "STOCK"
    if re.match(r'^00\d{2,4}[A-Z]?$', code):
        return "ETF"
    return None

def fetch_all_market_candidates():
    """從證交所(TWSE)與櫃買中心(TPEx)官方 OpenAPI 抓取全市場上市櫃個股與 ETF，並依成交量過濾"""
    candidates = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1. 先載入保底核心 ETF
    for tk, name in CORE_ETFS.items():
        candidates[tk] = {"name": name, "kind": "ETF"}

    # 2. 抓取上市 (TWSE) 全市場收盤表
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
                        candidates[f"{code}.TW"] = {"name": name, "kind": kind}
        print(f"✅ 已載入上市(TWSE)符合流動性標的，目前累計候選池：{len(candidates)} 檔")
    except Exception as e:
        print(f"⚠️ TWSE OpenAPI 讀取異常: {e}")

    # 3. 抓取上櫃 (TPEx) 全市場收盤表
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
                        candidates[f"{code}.TWO"] = {"name": name, "kind": kind}
        print(f"✅ 已載入上櫃(TPEx)符合流動性標的，全市場總掃描池：{len(candidates)} 檔")
    except Exception as e:
        print(f"⚠️ TPEx OpenAPI 讀取異常: {e}")

    return candidates

def evaluate_30k_df(df, ticker, name, kind):
    """對單一標的之 30K DataFrame 執行 20/60MA 金叉 + MACD 零軸綠轉紅判定"""
    try:
        df = df.dropna(subset=['Close']).copy()
        if len(df) < 65:
            return None

        # 計算日線級別 20MA 濾網
        daily_close = df['Close'].resample('1D').last().dropna()
        ma_win = min(20, len(daily_close))
        daily_ma20 = float(daily_close.rolling(ma_win).mean().iloc[-1])
        current_close = float(df['Close'].iloc[-1])

        if current_close < daily_ma20:
            return None

        # 計算 30K 之 20MA、60MA 與 5均量
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['VOL_MA5'] = df['Volume'].rolling(window=5).mean()

        # 計算 30K 之 MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = ema12 - ema26
        df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD']

        last = df.iloc[-1]
        prev1 = df.iloc[-2]
        recent_9 = df.iloc[-9:]

        ma20_now = float(last['MA20'])
        ma60_now = float(last['MA60'])
        dif_now = float(last['DIF'])
        osc_now = float(last['OSC'])
        osc_prev = float(prev1['OSC'])

        if np.isnan(ma60_now) or ma60_now <= 0:
            return None

        stop_price = round(ma60_now * 0.995, 2)
        risk_pct = round((current_close - stop_price) / current_close * 100, 2)
        code = ticker.split('.')[0]

        ma_gap_pct = abs(ma20_now - ma60_now) / current_close
        dif_zero_pct = abs(dif_now) / current_close
        tightness_score = round((ma_gap_pct + dif_zero_pct) * 100, 2)
        near_zero = dif_zero_pct <= ZERO_BOUND_PCT

        # 條件 A【今日盤中 30K 已金叉翻紅】
        had_negative_osc = (df['OSC'].iloc[-10:-1] <= 0).any()
        had_ma_cross = (df['MA20'].iloc[-10:-1] <= df['MA60'].iloc[-10:-1]).any() or (ma_gap_pct <= 0.012)
        vol_spike = (recent_9['Volume'] >= recent_9['VOL_MA5'] * VOL_MULTIPLIER).any()

        if near_zero and (ma20_now > ma60_now) and (osc_now > 0) and (had_negative_osc or had_ma_cross):
            status = "🔥帶量金叉翻紅" if vol_spike else "✅30K金叉翻紅"
            return {
                "type": "A", "kind": kind, "code": code, "name": name, "close": round(current_close, 2),
                "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                "stop": stop_price, "risk": risk_pct, "status": status, "score": tightness_score
            }

        # 條件 B【準金叉蓄勢】：30K 20/60MA 極度糾結且綠柱收斂
        if near_zero and (ma_gap_pct <= 0.01) and (osc_now >= osc_prev) and (-0.005 * current_close <= osc_now <= 0.002 * current_close):
            return {
                "type": "B", "kind": kind, "code": code, "name": name, "close": round(current_close, 2),
                "ma20": round(ma20_now, 2), "ma60": round(ma60_now, 2),
                "stop": stop_price, "risk": risk_pct, "status": "⏳準金叉綠柱收斂", "score": tightness_score
            }
        return None
    except Exception:
        return None

def send_telegram_chunks(full_text):
    """支援超過 4000 字自動分段推播，並內建真實 Chat ID 自動偵測救援機制"""
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    print("================ 選股結果預覽 ================")
    print(full_text)
    print("==============================================")
    if not token:
        print("⚠️ 尚未設定 TELEGRAM_TOKEN")
        return

    # 將長訊息按行切分為不超過 3500 字的區塊，避免超過 Telegram 4096 字上限
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
            print(f"⚠️ 原設定 CHAT_ID 推播回報 ({res.status_code}): {res.text}")
            print("🔄 啟動自動偵測真實 Chat ID 機制...")
            upd = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15).json()
            if upd.get("ok") and upd.get("result"):
                for item in reversed(upd["result"]):
                    msg_obj = item.get("message") or item.get("edited_message") or item.get("my_chat_member")
                    if msg_obj and "chat" in msg_obj:
                        target_chat_id = msg_obj["chat"]["id"]
                        break
                res = requests.post(url, json={"chat_id": target_chat_id, "text": chunk}, timeout=15)
        print(f"第 {idx+1}/{len(chunks)} 則 Telegram 推播狀態碼: {res.status_code}")

if __name__ == "__main__":
    tw_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    candidates = fetch_all_market_candidates()
    ticker_list = list(candidates.keys())

    etf_matches, stock_a, stock_b = [], [], []

    # 批次多執行緒高速下載 30K 數據
    for i in range(0, len(ticker_list), BATCH_SIZE):
        batch = ticker_list[i:i + BATCH_SIZE]
        try:
            batch_data = yf.download(batch, period="1mo", interval="30m", group_by="ticker", threads=True, progress=False)
            for tk in batch:
                info = candidates[tk]
                df_tk = batch_data[tk] if len(batch) > 1 and tk in batch_data else batch_data
                res = evaluate_30k_df(df_tk, tk, info["name"], info["kind"])
                if res:
                    if res["kind"] == "ETF":
                        etf_matches.append(res)
                    elif res["type"] == "A":
                        stock_a.append(res)
                    else:
                        stock_b.append(res)
        except Exception as e:
            print(f"Batch {i} error: {e}")

    # 依均線與零軸收斂緊密度排序 (越緊密排越前)
    etf_matches.sort(key=lambda x: (x["type"], x["score"]))
    stock_a.sort(key=lambda x: x["score"])
    stock_b.sort(key=lambda x: x["score"])

    msg_lines = [
        f"📊 【全市場台股＆ETF 30K金叉＋MACD零軸日報】",
        f"🕒 結算時間：{tw_time}｜掃描活水標的：{len(ticker_list)} 檔\n"
    ]

    # 1. ETF 專區 (含主動式與被動式 ETF)
    msg_lines.append(f"🛡️ 一、【ETF 專區（含主動式/市值/主題型）】(共 {len(etf_matches)} 檔)")
    if etf_matches:
        for r in etf_matches[:10]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}]\n"
                f"  收盤：{r['close']}｜30K(20/60MA)：{r['ma20']}/{r['ma60']}｜停損：{r['stop']} (-{r['risk']}%)"
            )
    else:
        msg_lines.append("• 今日無剛觸發 30K 零軸金叉之 ETF")
    msg_lines.append("")

    # 2. 個股已觸發金叉專區 (精選收斂度最佳前 15 強)
    msg_lines.append(f"🎯 二、【上市櫃個股：今日已金叉翻紅】(共 {len(stock_a)} 檔，精選前 15 強)")
    if stock_a:
        for r in stock_a[:15]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}]\n"
                f"  收盤：{r['close']}｜30K(20/60MA)：{r['ma20']}/{r['ma60']}｜停損：{r['stop']} (-{r['risk']}%)"
            )
    else:
        msg_lines.append("• 今日無剛觸發 30K 零軸金叉之個股")
    msg_lines.append("")

    # 3. 個股明日早盤準金叉專區 (精選前 10 強)
    msg_lines.append(f"👀 三、【上市櫃個股：明日準金叉觀察】(共 {len(stock_b)} 檔，精選前 10 強)")
    if stock_b:
        for r in stock_b[:10]:
            msg_lines.append(
                f"• {r['code']} {r['name']} [{r['status']}]\n"
                f"  收盤：{r['close']}｜突破價：{r['ma20']}｜停損：{r['stop']} (-{r['risk']}%)"
            )
    else:
        msg_lines.append("• 今日無準金叉收斂個股")

    msg_lines.append("\n💡 紀律：跌破硬停損(30K之60MA)無條件出場；獲利後首次30K MACD死叉減碼50%，剩餘守60MA。")
    send_telegram_chunks("\n".join(msg_lines))
