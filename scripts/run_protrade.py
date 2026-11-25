#!/usr/bin/env python3
"""
ProTrade 6-in-1 Self-Evolving Quant System Runner (Phase 0 - Phase 6)
Designed for GitHub Actions + GitHub Pages autonomous daily execution.
"""

import os
import sys
import json
import math
import traceback
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types

ET = ZoneInfo("America/New_York")
BKK = ZoneInfo("Asia/Bangkok")
DATA_DIR = "data"
PROMPT_VERSION = "v3.0"


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def get_expected_asof(holidays):
    now_et = datetime.now(ET)
    check = now_et.date()
    for i in range(14):
        d_str = check.isoformat()
        if check.weekday() < 5 and d_str not in holidays:
            if i == 0 and (now_et.hour * 60 + now_et.minute) < (16 * 60 + 30):
                check -= timedelta(days=1)
                continue
            return d_str
        check -= timedelta(days=1)
    return now_et.date().isoformat()


def add_trading_days(start_str, n_days, holidays):
    cur = date.fromisoformat(start_str)
    added = 0
    while added < n_days:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur.isoformat() not in holidays:
            added += 1
    return cur.isoformat()


def count_trading_days_between(start_str, end_str, holidays):
    if not start_str or not end_str or start_str >= end_str:
        return 0
    cur = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    cnt = 0
    while cur < end and cnt < 120:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur.isoformat() not in holidays:
            cnt += 1
    return cnt


def compute_wilder_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None


def compute_wilder_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None


def evaluate_decision_rule_j(item, prices_all, trading_days_sorted, neutral_band_mult):
    d_str = item["date"]
    t_str = item["target_date"]
    ticker = item["ticker"]
    bias = item["bias"]
    entry_close = float(item["entry_close"])
    inv = item.get("invalidation")
    atr14 = float(item.get("atr14") or 0)
    h = int(str(item.get("horizon", "3d")).replace("d", ""))

    window_days = [d for d in trading_days_sorted if d_str < d <= t_str]
    if not window_days:
        return None

    for s in window_days:
        day_prices = prices_all.get(s, {}).get(ticker)
        if not day_prices or len(day_prices) < 2:
            return {"outcome": "n/a", "outcome_close": None, "outcome_date": s}
        o_px, c_px = float(day_prices[0]), float(day_prices[1])

        if bias == "bull" and inv is not None:
            if o_px < float(inv):
                return {"outcome": "miss", "outcome_close": round(o_px, 4), "outcome_date": s}
            if c_px < float(inv):
                return {"outcome": "miss", "outcome_close": round(c_px, 4), "outcome_date": s}
        elif bias == "bear" and inv is not None:
            if o_px > float(inv):
                return {"outcome": "miss", "outcome_close": round(o_px, 4), "outcome_date": s}
            if c_px > float(inv):
                return {"outcome": "miss", "outcome_close": round(c_px, 4), "outcome_date": s}

    target_px = prices_all.get(t_str, {}).get(ticker)
    if not target_px or len(target_px) < 2:
        return {"outcome": "n/a", "outcome_close": None, "outcome_date": t_str}

    final_close = float(target_px[1])
    if bias == "bull":
        res = "hit" if final_close > entry_close else "miss"
    elif bias == "bear":
        res = "hit" if final_close < entry_close else "miss"
    else:
        pct_move = abs(final_close / entry_close - 1.0) * 100.0
        band = neutral_band_mult * ((atr14 / entry_close) * 100.0) * math.sqrt(h)
        res = "hit" if pct_move <= band else "miss"

    return {"outcome": res, "outcome_close": round(final_close, 4), "outcome_date": t_str}


def record_failure(lessons, phase, reason):
    now_iso = datetime.now(BKK).isoformat(timespec="seconds")
    failed = lessons.setdefault("failed_runs", [])
    failed.append({"date": now_iso, "phase": phase, "reason": str(reason)[:300]})
    lessons["failed_runs"] = failed[-20:]
    save_json(f"{DATA_DIR}/lessons.json", lessons)


def main():
    force_run = os.getenv("FORCE_RUN", "false").lower() == "true"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

    watchlist_cfg = load_json(f"{DATA_DIR}/watchlist.json", {
        "default": ["SPY", "QQQ", "NVDA", "AAPL"],
        "benchmarks": ["SPY", "QQQ"],
        "sectors": {},
        "holidays": [],
        "half_days": []
    })
    config = load_json(f"{DATA_DIR}/config.json", {
        "config_version": 1,
        "params": {
            "analysis_trigger_atr": 1.0,
            "neutral_band_mult": 0.3,
            "invalidation_atr_mult": 0.5,
            "sr_days": 10,
            "min_n_eff": 30,
            "max_rules": 10,
            "max_new_rules_per_7d": 3,
            "max_searches_total": 100,
            "param_change_gap_pp": 5,
            "full_refresh_days": 5,
            "risk_per_trade_pct": 1.0,
            "max_position_pct": 15,
            "sector_concentration_pct": 40
        },
        "bounds": {
            "analysis_trigger_atr": [0.5, 2.0],
            "neutral_band_mult": [0.2, 0.5],
            "invalidation_atr_mult": [0.3, 1.0],
            "sr_days": [5, 20]
        },
        "params_log": []
    })
    heartbeat = load_json(f"{DATA_DIR}/heartbeat.json", {})
    lessons = load_json(f"{DATA_DIR}/lessons.json", {"rules": [], "last_review": "2026-01-01", "failed_runs": []})
    prev_analysis = load_json(f"{DATA_DIR}/analysis.json", {"tickers": {}})

    holidays = watchlist_cfg.get("holidays", [])
    expected_asof = get_expected_asof(holidays)
    now_bkk_iso = datetime.now(BKK).isoformat(timespec="seconds")

    # Phase 0: Preflight
    if not force_run and heartbeat.get("last_ok_asof") == expected_asof and heartbeat.get("status") == "ok":
        heartbeat["last_checked"] = now_bkk_iso
        heartbeat["expected_asof"] = expected_asof
        heartbeat["status"] = "skipped"
        heartbeat["note"] = "ไม่มีวันทำการใหม่ (Skipped)"
        save_json(f"{DATA_DIR}/heartbeat.json", heartbeat)
        print(f"[Phase 0] Skipped: {expected_asof} already completed.")
        return

    if not api_key:
        record_failure(lessons, "Phase 0", "Missing GEMINI_API_KEY secret")
        heartbeat.update({"last_checked": now_bkk_iso, "expected_asof": expected_asof, "status": "failed", "note": "Missing GEMINI_API_KEY"})
        save_json(f"{DATA_DIR}/heartbeat.json", heartbeat)
        sys.exit(1)

    # Phase 1: Price Data & Technical Indicators
    tickers_default = watchlist_cfg.get("default", [])
    benchmarks = watchlist_cfg.get("benchmarks", ["SPY", "QQQ"])
    all_symbols = list(dict.fromkeys(tickers_default + benchmarks))
    params = config["params"]
    sr_days = int(params.get("sr_days", 10))
    inv_mult = float(params.get("invalidation_atr_mult", 0.5))
    trig_atr = float(params.get("analysis_trigger_atr", 1.0))
    full_refresh_days = int(params.get("full_refresh_days", 5))

    metrics_by_ticker = {}
    monthly_price_updates = {}

    for sym in all_symbols:
        try:
            df = yf.download(sym, period="4mo", interval="1d", auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
            df = df[df.index <= expected_asof]

            if df.empty or expected_asof not in df.index:
                metrics_by_ticker[sym] = {"status": "ไม่มีข้อมูล", "reason": f"ไม่พบราคาปิด ณ {expected_asof} จาก Yahoo Finance"}
                continue

            for d_idx, row in df.tail(45).iterrows():
                ym = d_idx[:7]
                monthly_price_updates.setdefault(ym, {}).setdefault(d_idx, {})[sym] = [
                    round(float(row["Open"]), 4),
                    round(float(row["Close"]), 4)
                ]

            close_now = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else close_now
            pct_chg = ((close_now / prev_close) - 1.0) * 100.0 if prev_close else 0.0

            if abs(pct_chg) > 25.0:
                metrics_by_ticker[sym] = {"status": "ไม่มีข้อมูล", "reason": f"ราคาเปลี่ยนแปลงผิดปกติ {pct_chg:.1f}% เกิน 25% รอตรวจสอบ"}
                continue

            sma20 = float(df["Close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
            sma50 = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
            rsi14 = compute_wilder_rsi(df["Close"], 14) if len(df) >= 15 else None
            atr14 = compute_wilder_atr(df, 14) if len(df) >= 15 else None

            vol_now = float(df["Volume"].iloc[-1])
            vol_avg20 = float(df["Volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else vol_now
            vol_ratio = round(vol_now / vol_avg20, 2) if vol_avg20 > 0 else 1.0

            recent_sr = df.tail(sr_days)
            support = round(float(recent_sr["Low"].min()), 2)
            resistance = round(float(recent_sr["High"].max()), 2)
            if support >= resistance:
                resistance = round(support + (atr14 or 1.0), 2)

            move_atr = round(abs(close_now - prev_close) / atr14, 2) if atr14 and atr14 > 0 else 0.0
            prev_entry = prev_analysis.get("tickers", {}).get(sym, {})
            last_full = prev_entry.get("last_full_date")
            days_since_full = count_trading_days_between(last_full, expected_asof, holidays) if last_full else 999

            mode = "full" if (move_atr >= trig_atr or days_since_full >= full_refresh_days or not last_full) else "short"

            metrics_by_ticker[sym] = {
                "status": "ok",
                "close": round(close_now, 2),
                "prev_close": round(prev_close, 2),
                "pct_change": round(pct_chg, 2),
                "sma20": round(sma20, 2) if sma20 else None,
                "sma50": round(sma50, 2) if sma50 else None,
                "rsi14": round(rsi14, 2) if rsi14 else None,
                "atr14": round(atr14, 2) if atr14 else None,
                "move_atr": move_atr,
                "vol_ratio_20d": vol_ratio,
                "support": support,
                "resistance": resistance,
                "bull_invalidation": round(support - inv_mult * (atr14 or 0), 2),
                "bear_invalidation": round(resistance + inv_mult * (atr14 or 0), 2),
                "analysis_mode": mode,
                "last_full_date": expected_asof if mode == "full" else last_full,
                "prev_entry": prev_entry if mode == "short" else None,
                "source": f"Yahoo Finance ({expected_asof} close)"
            }
        except Exception as e:
            metrics_by_ticker[sym] = {"status": "ไม่มีข้อมูล", "reason": f"ดึงข้อมูลไม่สำเร็จ: {str(e)[:120]}"}

    valid_count = sum(1 for s in tickers_default if metrics_by_ticker.get(s, {}).get("status") == "ok")
    if valid_count == 0:
        heartbeat.update({
            "last_checked": now_bkk_iso,
            "expected_asof": expected_asof,
            "status": "skipped",
            "note": f"ข้อมูลราคายังไม่อัปเดต ณ {expected_asof}"
        })
        save_json(f"{DATA_DIR}/heartbeat.json", heartbeat)
        print(f"[Phase 1] Market close data for {expected_asof} not available yet. Skipped.")
        return

    prices_all = {}
    for ym, days_map in monthly_price_updates.items():
        existing_p = load_json(f"{DATA_DIR}/prices/prices_{ym}.json", {})
        for d_k, sym_map in days_map.items():
            existing_p.setdefault(d_k, {}).update(sym_map)
        monthly_price_updates[ym] = existing_p
        prices_all.update(existing_p)

    trading_days_sorted = sorted(prices_all.keys())

    # Phase 3.1: ตัดสินคำทำนายที่ครบกำหนดด้วย Decision Rule J
    ym_cur = expected_asof[:7]
    prev_month_dt = date.fromisoformat(expected_asof).replace(day=1) - timedelta(days=1)
    ym_prev = prev_month_dt.strftime("%Y-%m")

    hist_by_ym = {
        ym_prev: load_json(f"{DATA_DIR}/history/history_{ym_prev}.json", []),
        ym_cur: load_json(f"{DATA_DIR}/history/history_{ym_cur}.json", [])
    }

    decided_today = []
    yesterday_summary = {}
    neutral_band_mult = float(params.get("neutral_band_mult", 0.3))

    for ym_key, records in hist_by_ym.items():
        for rec in records:
            if rec.get("outcome") is None and rec.get("target_date") <= expected_asof:
                verdict = evaluate_decision_rule_j(rec, prices_all, trading_days_sorted, neutral_band_mult)
                if verdict:
                    rec.update(verdict)
                    decided_today.append(rec)
                    yesterday_summary[rec["ticker"]] = {
                        "call": f"{rec['bias']} {rec['horizon']}",
                        "result": verdict["outcome"],
                        "note": f"ออกที่ {verdict['outcome_close']} ({verdict['outcome_date']})" if verdict["outcome_close"] else "ข้อมูลขาดช่วง"
                    }

    active_rules = [r for r in lessons.get("rules", []) if r.get("expires", "9999-12-31") >= expected_asof]
    lessons["rules"] = active_rules

    # Phase 2, 3.4, 4, 5: เรียก Gemini API + Google Search Grounding
    # Fix: API quota exceeded workaround (don't fail the whole run if 429 occurs, fall back to previous data)
    final_tickers = {}
    new_history_entries = []
    has_missing = False
    ai_out = {}
    meta_status = "ok"

    try:
        client = genai.Client(api_key=api_key)

        prompt_payload = {
            "asof": expected_asof,
            "watchlist": tickers_default,
            "sectors": watchlist_cfg.get("sectors", {}),
            "params": params,
            "active_rules": active_rules,
            "decided_today_count": {
                "hit": sum(1 for x in decided_today if x["outcome"] == "hit"),
                "miss": sum(1 for x in decided_today if x["outcome"] == "miss"),
                "n_a": sum(1 for x in decided_today if x["outcome"] == "n/a")
            },
            "yesterday_summary": yesterday_summary,
            "market_metrics": metrics_by_ticker
        }

        system_instruction = f"""
คุณคือ ProTrade 6-in-1 Quant Analyst, Macro Sentinel, Critic และ Portfolio Risk Manager (PROMPT_VERSION={PROMPT_VERSION})
หน้าที่ของคุณคือวิเคราะห์ข้อมูลตลาดจริง ณ วันที่ {expected_asof} และตอบกลับเป็น JSON ที่ถูกต้องตามสเปกเท่านั้น ห้ามมีข้อความนอก JSON

กฎเหล็กที่ต้องปฏิบัติตามอย่างเคร่งครัด:
1. ห้ามเดาตัวเลขราคา แนวรับ แนวต้าน หรือ Invalidation ใหม่ ให้ใช้ค่าที่คำนวณไว้ให้ใน `market_metrics` เท่านั้น:
   - ถ้าเลือก bias = "bull" -> levels.invalidation ต้องเท่ากับ `bull_invalidation` (และต้องน้อยกว่า `close`)
   - ถ้าเลือก bias = "bear" -> levels.invalidation ต้องเท่ากับ `bear_invalidation` (และต้องมากกว่า `close`)
   - ถ้าเลือก bias = "neutral" -> levels.invalidation ต้องเป็น `null`
   - levels.support = `support`, levels.resistance = `resistance`
2. หุ้นที่มี `status == "ไม่มีข้อมูล"` ใน `market_metrics` ให้ใส่ใน `tickers` เป็น `{{"status": "ไม่มีข้อมูล", "reason": "..."}}` เท่านั้น
3. บังคับใช้กฎใน `active_rules` อย่างเคร่งครัด และอ้างอิงรหัสกฎเช่น `[R001]` ในเหตุผล
4. ใช้ Google Search ค้นข่าวเศรษฐกิจมหภาค บอนด์ยีลด์ นโยบาย Fed และข่าว/งบของหุ้นโหมด `full` เพื่อระบุ `catalyst` และ `long_term` พร้อมแหล่งอ้างอิงจริง
5. ข้อจำกัดความยาว:
   - `market_review.macro_md`: ภาษาไทย ไม่เกิน 150 คำ แยกหัวข้อ **ข้อเท็จจริง** และ **การตีความ**
   - `market_review.critic_md`: ภาษาไทย ไม่เกิน 100 คำ สรุปผลตัดสินวันนี้ (hit/miss/n/a)
   - `market_review.analyst_md`: ภาษาไทย ไม่เกิน 100 คำ สรุปภาพรวมการถกเถียง Bull vs Skeptic
   - `market_review.risk_md`: ภาษาไทย ไม่เกิน 120 คำ วิเคราะห์ความกระจุกตัวรายเซกเตอร์ ขนาด Position Sizing และเทียบ SPY/QQQ
   - `reasons`: ไม่เกิน 3 ข้อ (ระบุ `(Bull)` หรือ `(Skeptic)` นำหน้าตามมุมมอง), `risks`: ไม่เกิน 2 ข้อ
   - `confidence`: จำนวนเต็ม 0-100, `horizon`: ต้องเป็น `"1d"`, `"3d"` หรือ `"5d"` เท่านั้น

รูปแบบ JSON Output ที่บังคับ:
{{
  "market_review": {{
    "macro_md": "...",
    "analyst_md": "...",
    "critic_md": "...",
    "risk_md": "..."
  }},
  "tickers": {{
    "SYMBOL": {{
      "close": 180.0,
      "source": "Yahoo Finance (YYYY-MM-DD close)",
      "atr14": 4.1,
      "move_atr": 0.4,
      "analysis_mode": "full",
      "last_full_date": "YYYY-MM-DD",
      "bias": "bull",
      "confidence": 65,
      "horizon": "3d",
      "levels": {{"support": 175.0, "resistance": 186.0, "invalidation": 172.9}},
      "reasons": ["(Bull) ..."],
      "risks": ["(Skeptic) ..."],
      "catalyst": "...",
      "long_term": {{"view": "...", "zone": "..."}},
      "yesterday": {{"call": "...", "result": "...", "note": "..."}}
    }}
  }}
}}
"""
        response = client.models.generate_content(
            model=model_name,
            contents=json.dumps(prompt_payload, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        ai_out = json.loads(raw_text)
    
    except Exception as e:
        print(f"[Warning] AI Analysis failed or rate limited: {e}")
        meta_status = "failed"
        ai_out = {"market_review": {}, "tickers": {}}

    # Phase 6: Schema & Rule Gate Validation
    for sym in tickers_default:
        m = metrics_by_ticker.get(sym, {})
        if m.get("status") != "ok":
            has_missing = True
            final_tickers[sym] = {"status": "ไม่มีข้อมูล", "reason": m.get("reason", "ไม่มีข้อมูล")}
            continue

        t_ai = ai_out.get("tickers", {}).get(sym)
        if not t_ai or t_ai.get("status") == "ไม่มีข้อมูล":
            has_missing = True
            
            # Fallback to previous data if API rate limited
            prev_e = prev_analysis.get("tickers", {}).get(sym)
            if prev_e and prev_e.get("status") != "ไม่มีข้อมูล" and meta_status == "failed":
                # Copy prev data but update close price
                fallback_entry = prev_e.copy()
                fallback_entry["close"] = m["close"]
                fallback_entry["source"] = f"Yahoo Finance ({expected_asof} close) [AI Fallback]"
                final_tickers[sym] = fallback_entry
            else:
                final_tickers[sym] = {"status": "ไม่มีข้อมูล", "reason": "AI ไม่สามารถสรุปข้อมูลได้ (API Rate Limit หรือเกิดข้อผิดพลาด)"}
            continue

        bias = t_ai.get("bias", "neutral")
        if bias not in ("bull", "bear", "neutral"):
            bias = "neutral"

        conf = int(max(0, min(100, round(float(t_ai.get("confidence", 50))))))
        horizon = t_ai.get("horizon", "3d")
        if horizon not in ("1d", "3d", "5d"):
            horizon = "3d"

        close_px = m["close"]
        sup = m["support"]
        res = m["resistance"]
        if bias == "bull":
            inv = m["bull_invalidation"]
            if inv >= close_px:
                inv = round(close_px - max(0.5, (m["atr14"] or 1.0) * 0.5), 2)
        elif bias == "bear":
            inv = m["bear_invalidation"]
            if inv <= close_px:
                inv = round(close_px + max(0.5, (m["atr14"] or 1.0) * 0.5), 2)
        else:
            inv = None

        reasons = [str(x) for x in (t_ai.get("reasons") or [])][:3]
        risks = [str(x) for x in (t_ai.get("risks") or [])][:2]

        prev_e = m.get("prev_entry") or {}
        if m["analysis_mode"] == "short" and prev_e.get("long_term"):
            risks = prev_e.get("risks", risks)[:2]
            catalyst = prev_e.get("catalyst", t_ai.get("catalyst", "ไม่มีข้อมูล"))
            long_term = prev_e.get("long_term", t_ai.get("long_term", {"view": "ไม่มีข้อมูล", "zone": "ไม่มีข้อมูล"}))
        else:
            catalyst = t_ai.get("catalyst") or "ไม่มีข้อมูล"
            long_term = t_ai.get("long_term") or {"view": "ไม่มีข้อมูล", "zone": "ไม่มีข้อมูล"}

        final_tickers[sym] = {
            "close": close_px,
            "source": m["source"],
            "atr14": m["atr14"],
            "move_atr": m["move_atr"],
            "analysis_mode": m["analysis_mode"],
            "last_full_date": m["last_full_date"],
            "bias": bias,
            "confidence": conf,
            "horizon": horizon,
            "levels": {"support": sup, "resistance": res, "invalidation": inv},
            "reasons": reasons,
            "risks": risks,
            "catalyst": catalyst,
            "long_term": long_term,
            "yesterday": yesterday_summary.get(sym, t_ai.get("yesterday", {"call": "", "result": "n/a", "note": "ไม่มีรายการครบกำหนด"}))
        }

        h_days = int(horizon.replace("d", ""))
        new_history_entries.append({
            "date": expected_asof,
            "ticker": sym,
            "bias": bias,
            "confidence": conf,
            "horizon": horizon,
            "entry_close": close_px,
            "invalidation": inv,
            "atr14": m["atr14"],
            "target_date": add_trading_days(expected_asof, h_days, holidays),
            "prompt_version": PROMPT_VERSION,
            "config_version": config.get("config_version", 1),
            "outcome": None,
            "outcome_close": None,
            "outcome_date": None
        })

    # ลำดับการเขียนไฟล์ตาม Phase 6
    for ym, p_data in monthly_price_updates.items():
        save_json(f"{DATA_DIR}/prices/prices_{ym}.json", p_data)

    existing_keys = {(x["date"], x["ticker"]) for x in hist_by_ym[ym_cur]}
    for entry in new_history_entries:
        if (entry["date"], entry["ticker"]) not in existing_keys:
            hist_by_ym[ym_cur].append(entry)
            existing_keys.add((entry["date"], entry["ticker"]))

    for ym_key, records in hist_by_ym.items():
        if records:
            save_json(f"{DATA_DIR}/history/history_{ym_key}.json", records)

    save_json(f"{DATA_DIR}/lessons.json", lessons)
    save_json(f"{DATA_DIR}/config.json", config)

    if meta_status != "failed" and has_missing:
        meta_status = "partial"
        
    analysis_doc = {
        "meta": {
            "asof": expected_asof,
            "generated_at": now_bkk_iso,
            "version": "v3.1.0",
            "prompt_version": PROMPT_VERSION,
            "config_version": config.get("config_version", 1),
            "model": model_name,
            "status": meta_status
        },
        "market_review": ai_out.get("market_review", prev_analysis.get("market_review", {
            "macro_md": "ไม่มีข้อมูล",
            "analyst_md": "ไม่มีข้อมูล",
            "critic_md": "ไม่มีข้อมูล",
            "risk_md": "ไม่มีข้อมูล"
        })),
        "tickers": final_tickers
    }
    save_json(f"{DATA_DIR}/analysis.json", analysis_doc)

    heartbeat.update({
        "last_checked": now_bkk_iso,
        "expected_asof": expected_asof,
        "last_ok_asof": expected_asof if meta_status != "failed" else heartbeat.get("last_ok_asof"),
        "status": meta_status,
        "note": f"Completed {valid_count}/{len(tickers_default)} tickers via GitHub Actions ({model_name})" if meta_status != "failed" else "API Rate Limited, using fallback data"
    })
    save_json(f"{DATA_DIR}/heartbeat.json", heartbeat)
    print(f"[Success] ProTrade run completed for {expected_asof} (status={meta_status}).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        lessons_err = load_json(f"{DATA_DIR}/lessons.json", {"rules": [], "last_review": "2026-01-01", "failed_runs": []})
        record_failure(lessons_err, "Runtime", str(exc))
        hb_err = load_json(f"{DATA_DIR}/heartbeat.json", {})
        hb_err.update({
            "last_checked": datetime.now(BKK).isoformat(timespec="seconds"),
            "status": "failed",
            "note": f"Error: {str(exc)[:150]}"
        })
        save_json(f"{DATA_DIR}/heartbeat.json", hb_err)
        sys.exit(1)
