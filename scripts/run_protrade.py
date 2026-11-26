#!/usr/bin/env python3
"""
ProTrade 6-in-1 Self-Evolving Quant System Runner (Phase 0 - Phase 6)
v4.2.5: Macro 1 + Micro 4, Auto-Fallback, Search Grounding Toggle, Error Tracking
"""

import os
import sys
import json
import math
import time
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
PROMPT_VERSION = "v4.2"
SYSTEM_VERSION = "v4.2.5-Macro1-Micro4"

MAX_MACRO_CALLS = 1
MAX_MICRO_CALLS = 4
SLEEP_SECONDS = 15

# Global dictionary to track unusable models for the current run
unusable_models = {}

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

def parse_json_response(resp_text):
    txt = (resp_text or "").strip()
    if "{" in txt and "}" in txt:
        start_idx = txt.find("{")
        end_idx = txt.rfind("}") + 1
        txt = txt[start_idx:end_idx]
    return json.loads(txt)

def call_gemini_with_fallback(client, models_chain, contents, sys_instruction, allow_search=True):
    """Calls Gemini with automatic model fallback and smart search toggling."""
    last_err = None
    global unusable_models
    
    for idx, mdl in enumerate(models_chain):
        if mdl in unusable_models:
            print(f"    [Skip] Model '{mdl}' is marked unusable for this run: {unusable_models[mdl]}")
            continue
            
        current_allow_search = allow_search
        
        while True:
            try:
                tools = [{"google_search": {}}] if current_allow_search else None
                resp = client.models.generate_content(
                    model=mdl,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        tools=tools,
                        temperature=0.2,
                    ),
                )
                parsed = parse_json_response(resp.text)
                return parsed, mdl, current_allow_search
                
            except Exception as e:
                err_str = str(e)
                print(f"    [Error] Model '{mdl}' (search={current_allow_search}) failed: {err_str[:120]}")
                
                if "429" in err_str:
                    if current_allow_search:
                        print(f"    [Fallback] Retrying '{mdl}' without search grounding...")
                        current_allow_search = False
                        continue
                    else:
                        print(f"    [Fallback] Model '{mdl}' hit 429 even without search. Marking unusable.")
                        unusable_models[mdl] = "429_exhausted"
                        last_err = e
                        time.sleep(3)
                        break
                        
                elif "404" in err_str or "400" in err_str or "403" in err_str:
                    print(f"    [Fallback] Model '{mdl}' invalid or access denied. Marking unusable.")
                    unusable_models[mdl] = "invalid_or_denied"
                    last_err = e
                    break
                    
                elif "503" in err_str:
                    print(f"    [Retry] Model '{mdl}' overloaded (503). Not blocking model, trying next.")
                    last_err = e
                    time.sleep(3)
                    break
                    
                else:
                    last_err = e
                    break
                    
    raise last_err

def chunk_list(items, n_chunks):
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    k, m = divmod(len(items), n_chunks)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]

def main():
    force_run = os.getenv("FORCE_RUN", "false").lower() == "true"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    primary_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    use_search_env = os.getenv("USE_SEARCH", "false").lower() == "true"
    
    fallback_env = os.getenv("GEMINI_FALLBACK_MODELS", "").strip()
    if fallback_env:
        fallback_models = [m.strip() for m in fallback_env.split(",") if m.strip()]
    else:
        fallback_models = [
            "gemini-3-flash",
            "gemini-4-hydrogen",
            "gemini-3.5-flash-lite",
        ]
        
    models_chain = list(dict.fromkeys([primary_model] + fallback_models))

    watchlist_cfg = load_json(f"{DATA_DIR}/watchlist.json", {
        "default": ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "NVDA", "SKHY", "OKLO", "SMR", "LLY", "SCHD", "VGT", "BSX", "WDC", "DY", "CCJ", "GRAB"],
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
        }
    })
    heartbeat = load_json(f"{DATA_DIR}/heartbeat.json", {})
    lessons = load_json(f"{DATA_DIR}/lessons.json", {"rules": [], "last_review": "2026-01-01", "failed_runs": []})
    prev_analysis = load_json(f"{DATA_DIR}/analysis.json", {"tickers": {}})

    holidays = watchlist_cfg.get("holidays", [])
    expected_asof = get_expected_asof(holidays)
    now_bkk_iso = datetime.now(BKK).isoformat(timespec="seconds")

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

    client = genai.Client(api_key=api_key)
    prev_rev = prev_analysis.get("market_review", {})
    meta_status = "ok"
    macro_calls_used = 0
    micro_calls_used = 0
    models_used = []
    search_used_in_macro = False
    
    print(f"[M4] Running Macro Phase ({MAX_MACRO_CALLS} Request, Fallback Chain={models_chain}, Initial Search={use_search_env})...")
    macro_out = {
        "macro_md": prev_rev.get("macro_md", "ไม่มีข้อมูล"),
        "analyst_md": prev_rev.get("analyst_md", "ไม่มีข้อมูล"),
        "critic_md": prev_rev.get("critic_md", "ไม่มีข้อมูล"),
        "risk_md": prev_rev.get("risk_md", "ไม่มีข้อมูล"),
        "search_grounding": False
    }

    try:
        benchmark_snap = {b: metrics_by_ticker.get(b) for b in benchmarks if b in metrics_by_ticker}
        sector_moves = {}
        for sym in tickers_default:
            sec = watchlist_cfg.get("sectors", {}).get(sym, "Other")
            m = metrics_by_ticker.get(sym, {})
            if m.get("status") == "ok":
                sector_moves.setdefault(sec, []).append({"ticker": sym, "pct_change": m["pct_change"], "move_atr": m["move_atr"]})

        macro_payload = {
            "asof": expected_asof,
            "benchmarks": benchmark_snap,
            "sector_moves": sector_moves,
            "risk_params": {
                "risk_per_trade_pct": params.get("risk_per_trade_pct", 1.0),
                "max_position_pct": params.get("max_position_pct", 15),
                "sector_concentration_pct": params.get("sector_concentration_pct", 40),
            },
            "decided_today_count": {
                "hit": sum(1 for x in decided_today if x["outcome"] == "hit"),
                "miss": sum(1 for x in decided_today if x["outcome"] == "miss"),
                "n_a": sum(1 for x in decided_today if x["outcome"] == "n/a"),
            },
            "yesterday_summary": yesterday_summary,
            "active_rules": active_rules,
        }
        
        search_prompt_add = " ห้ามอ้างอิงข่าวถ้าไม่มีข้อมูลชัดเจน ให้วิเคราะห์จากตัวเลขราคาและ Sector แทน" if not use_search_env else ""
        
        macro_sys = f"""
คุณคือทีม 4 เอเจนต์ภาพรวม (Macro Sentinel, Lead Analyst, Critic และ Portfolio Risk Manager) (PROMPT_VERSION={PROMPT_VERSION}-Macro1)
หน้าที่: สรุปภาพรวมเศรษฐกิจมหภาคสหรัฐฯ ณ วันที่ {expected_asof} และวิเคราะห์พอร์ต 17 ตัว{search_prompt_add}

ข้อจำกัด:
- `macro_md`: ภาษาไทย ไม่เกิน 150 คำ แยกหัวข้อ **ข้อเท็จจริง** และ **การตีความ**
- `analyst_md`: ภาษาไทย ไม่เกิน 100 คำ สรุปธีมการถกเถียงภาพรวมระหว่างฝั่ง Bull และ Skeptic วันนี้
- `critic_md`: ภาษาไทย ไม่เกิน 100 คำ สรุปผลตัดสินวันนี้ (hit/miss/n/a) จุดบอด และการบังคับใช้กฎ `[R...]`
- `risk_md`: ภาษาไทย ไม่เกิน 120 คำ วิเคราะห์ความเสี่ยงกระจุกตัวรายเซกเตอร์

ตอบเป็น JSON ล้วน:
{{
  "macro_md": "...",
  "analyst_md": "...",
  "critic_md": "...",
  "risk_md": "..."
}}
"""
        m_json, used_mdl, final_search_state = call_gemini_with_fallback(
            client,
            models_chain,
            json.dumps(macro_payload, ensure_ascii=False),
            macro_sys,
            allow_search=use_search_env
        )
        for k in ("macro_md", "analyst_md", "critic_md", "risk_md"):
            if m_json.get(k):
                macro_out[k] = m_json[k]
        macro_out["search_grounding"] = final_search_state
        search_used_in_macro = final_search_state
        macro_calls_used = 1
        models_used.append(used_mdl)
        print(f"  [Macro 1/1] Completed using model '{used_mdl}', search_grounding={final_search_state}.")
    except Exception as e:
        print(f"  [Warning] Macro 1/1 failed across all models: {e}")
        meta_status = "partial"

    final_tickers = {}
    new_history_entries = []
    has_missing = False

    valid_symbols = []
    for sym in tickers_default:
        m = metrics_by_ticker.get(sym, {})
        if m.get("status") != "ok":
            has_missing = True
            final_tickers[sym] = {"status": "ไม่มีข้อมูล", "reason": m.get("reason", "ไม่มีข้อมูล")}
        else:
            valid_symbols.append(sym)

    batches = chunk_list(valid_symbols, MAX_MICRO_CALLS)
    print(f"[M4] Running Micro Phase ({len(batches)} Batches covering {len(valid_symbols)} valid tickers, sleep={SLEEP_SECONDS}s)...")

    for b_idx, batch_syms in enumerate(batches, start=1):
        print(f"  [Micro Batch {b_idx}/{len(batches)}] Tickers: {batch_syms} (waiting {SLEEP_SECONDS}s for 5 RPM limit)...")
        time.sleep(SLEEP_SECONDS)

        try:
            batch_metrics = {}
            for sym in batch_syms:
                m = metrics_by_ticker[sym]
                batch_metrics[sym] = {
                    "sector": watchlist_cfg.get("sectors", {}).get(sym, "Unknown"),
                    "close": m["close"],
                    "pct_change": m["pct_change"],
                    "sma20": m["sma20"],
                    "sma50": m["sma50"],
                    "rsi14": m["rsi14"],
                    "atr14": m["atr14"],
                    "move_atr": m["move_atr"],
                    "vol_ratio_20d": m["vol_ratio_20d"],
                    "support": m["support"],
                    "resistance": m["resistance"],
                    "bull_invalidation": m["bull_invalidation"],
                    "bear_invalidation": m["bear_invalidation"],
                    "analysis_mode": m["analysis_mode"],
                    "yesterday": yesterday_summary.get(sym),
                }

            micro_payload = {
                "asof": expected_asof,
                "macro_summary": macro_out["macro_md"][:450],
                "tickers": batch_metrics,
                "active_rules": active_rules,
            }
            
            micro_search_prompt_add = " ห้ามค้นหาข่าวใหม่ หากไม่มีข้อมูลให้วิเคราะห์ด้วย Technical และ Sector จากข้อมูลที่ส่งให้เท่านั้น" if not search_used_in_macro else " ใช้ Google Search ค้นหาข่าวล่าสุดเฉพาะเจาะจงของหุ้นแต่ละตัวประกอบการตัดสินใจ"
            
            micro_sys = f"""
คุณคือ Quant Analyst (Bull vs Skeptic) วิเคราะห์หุ้นชุด {batch_syms} ณ วันที่ {expected_asof} (PROMPT_VERSION={PROMPT_VERSION}-MicroBatch{b_idx})
หน้าที่:{micro_search_prompt_add}

กฎเหล็ก:
1. ห้ามเดาตัวเลข levels เอง ให้ใช้ค่าจาก tickers[SYM] เท่านั้น:
   - ถ้า bias="bull": invalidation = bull_invalidation
   - ถ้า bias="bear": invalidation = bear_invalidation
   - ถ้า bias="neutral": invalidation = null
2. บังคับใช้กฎใน active_rules และอ้างอิงรหัสกฎเช่น [R001] หากเข้าเงื่อนไข
3. ตอบเป็น JSON ล้วน โดยมี key เป็นชื่อหุ้นทุกตัวใน {batch_syms} (horizon ต้องเป็น 1d, 3d, หรือ 5d):
{{
  "TICKER": {{
    "bias": "bull/bear/neutral",
    "confidence": 80,
    "horizon": "3d",
    "reasons": ["(Bull) ...", "(Skeptic) ..."],
    "risks": ["(Skeptic) ...", "(Bull) ..."],
    "catalyst": "ปัจจัยเร่ง...",
    "long_term": {{"view": "...", "zone": "..."}}
  }}
}}
"""
            batch_ai, used_mdl, final_batch_search = call_gemini_with_fallback(
                client,
                models_chain,
                json.dumps(micro_payload, ensure_ascii=False),
                micro_sys,
                allow_search=search_used_in_macro
            )
            micro_calls_used += 1
            if used_mdl not in models_used:
                models_used.append(used_mdl)
            print(f"    -> Batch {b_idx}/{len(batches)} succeeded via '{used_mdl}', search={final_batch_search}.")

            for sym in batch_syms:
                m = metrics_by_ticker[sym]
                t_ai = batch_ai.get(sym) if isinstance(batch_ai, dict) else None
                if not isinstance(t_ai, dict):
                    has_missing = True
                    meta_status = "partial"
                    final_tickers[sym] = {"status": "ไม่มีข้อมูล", "reason": "AI ไม่ได้ส่งผลวิเคราะห์ของหุ้นตัวนี้กลับมาในชุด"}
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
                    "catalyst": t_ai.get("catalyst", "ไม่มีข้อมูล"),
                    "long_term": t_ai.get("long_term", {"view": "ไม่มีข้อมูล", "zone": "ไม่มีข้อมูล"}),
                    "search_grounding": final_batch_search,
                    "yesterday": yesterday_summary.get(sym, {"call": "", "result": "n/a", "note": "ไม่มีรายการครบกำหนด"}),
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
                    "prompt_version": f"{PROMPT_VERSION}-MicroBatch",
                    "config_version": config.get("config_version", 1),
                    "outcome": None,
                    "outcome_close": None,
                    "outcome_date": None,
                })

        except Exception as e:
            print(f"  [Warning] Micro Batch {b_idx}/{len(batches)} failed across all models: {e}")
            has_missing = True
            meta_status = "partial"
            for sym in batch_syms:
                m = metrics_by_ticker[sym]
                prev_e = prev_analysis.get("tickers", {}).get(sym)
                if prev_e and prev_e.get("status") != "ไม่มีข้อมูล":
                    fallback_entry = prev_e.copy()
                    fallback_entry["close"] = m["close"]
                    fallback_entry["source"] = f"Yahoo Finance ({expected_asof} close) [AI Fallback]"
                    final_tickers[sym] = fallback_entry
                else:
                    final_tickers[sym] = {"status": "ไม่มีข้อมูล", "reason": "AI ไม่สามารถสรุปข้อมูลได้ (API Rate Limit หรือเกิดข้อผิดพลาด)"}

    ordered_final_tickers = {sym: final_tickers[sym] for sym in tickers_default if sym in final_tickers}

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

    if macro_calls_used == 0 and micro_calls_used == 0:
        meta_status = "failed"
    if meta_status != "failed" and has_missing:
        meta_status = "partial"

    active_model_label = ", ".join(models_used) if models_used else primary_model

    analysis_doc = {
        "meta": {
            "asof": expected_asof,
            "generated_at": now_bkk_iso,
            "version": SYSTEM_VERSION,
            "prompt_version": PROMPT_VERSION,
            "config_version": config.get("config_version", 1),
            "model": active_model_label,
            "api_calls": {
                "macro": macro_calls_used,
                "max_macro": MAX_MACRO_CALLS,
                "micro": micro_calls_used,
                "max_micro": MAX_MICRO_CALLS,
            },
            "status": meta_status,
        },
        "market_review": macro_out,
        "tickers": ordered_final_tickers,
    }
    save_json(f"{DATA_DIR}/analysis.json", analysis_doc)

    heartbeat.update({
        "last_checked": now_bkk_iso,
        "expected_asof": expected_asof,
        "last_ok_asof": expected_asof if meta_status != "failed" else heartbeat.get("last_ok_asof"),
        "status": meta_status,
        "note": f"Macro {macro_calls_used}/{MAX_MACRO_CALLS} + Micro {micro_calls_used}/{MAX_MICRO_CALLS} ({len(ordered_final_tickers)} tickers, models={active_model_label})",
    })
    save_json(f"{DATA_DIR}/heartbeat.json", heartbeat)
    print(f"[Success] ProTrade run completed for {expected_asof} (Macro={macro_calls_used}/{MAX_MACRO_CALLS}, Micro={micro_calls_used}/{MAX_MICRO_CALLS}, status={meta_status}).")

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
            "note": f"Error: {str(exc)[:150]}",
        })
        save_json(f"{DATA_DIR}/heartbeat.json", hb_err)
        sys.exit(1)
