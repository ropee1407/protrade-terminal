# ProTrade | 6-in-1 Self-Evolving Quant Terminal (GitHub Edition)

ระบบวิเคราะห์หุ้นอเมริกาและพอร์ตจำลองแบบอัตโนมัติ 100% บน **GitHub Actions + GitHub Pages** (ไม่ต้องเปิดคอมพิวเตอร์ทิ้งไว้)

## โครงสร้างไฟล์
- `index.html` / `protrade_terminal.html` — หน้าเว็บ Dashboard (M2 Dynamic JSON Loader)
- `.github/workflows/protrade_daily.yml` — ตั้งเวลาอัตโนมัติทุกวัน **05:00 น. เวลาไทย (22:00 UTC)** หลังตลาดหุ้นสหรัฐฯ ปิด
- `scripts/run_protrade.py` — สคริปต์รัน Phase 0–6 (ดึงราคาจริงจาก Yahoo Finance, คำนวณ SMA/RSI/ATR, ตัดสินผลด้วยกฎ J, เรียก Gemini API + Google Search, ตรวจ Schema และบันทึกไฟล์)
- `data/` — ฐานข้อมูล JSON (`analysis.json`, `watchlist.json`, `config.json`, `lessons.json`, `heartbeat.json`, `prices/`, `history/`)

## วิธีติดตั้งบน GitHub (ทำครั้งเดียวจบ)
1. **สร้าง Repository ใหม่บน GitHub** (ตั้งชื่อเช่น `protrade-terminal` จะเป็น Public หรือ Private ก็ได้)
2. **อัปโหลดไฟล์ทั้งหมดในโฟลเดอร์นี้ขึ้น Repository** (รวมโฟลเดอร์ `.github/`, `scripts/`, `data/` และไฟล์ `index.html`)
3. **ใส่ Gemini API Key ใน GitHub Secrets:**
   - ไปที่เมนู **Settings** > **Secrets and variables** > **Actions** > กด **New repository secret**
   - ชื่อ (`Name`): `GEMINI_API_KEY`
   - ค่า (`Secret`): วาง Gemini API Key ของคุณ แล้วกด **Add secret**
4. **เปิดหน้าเว็บผ่าน GitHub Pages:**
   - ไปที่เมนู **Settings** > **Pages**
   - ตรง **Build and deployment > Source** เลือก `Deploy from a branch`
   - เลือก Branch `main` โฟลเดอร์ `/ (root)` แล้วกด **Save**
   - รอประมาณ 1–2 นาที จะได้ลิงก์เว็บไซต์ของคุณ (เช่น `https://<username>.github.io/protrade-terminal/`) เปิดดูได้ทั้งบนมือถือและคอมพิวเตอร์
5. **ทดสอบสั่งรันทันที:**
   - ไปที่แท็บ **Actions** > คลิก **ProTrade Daily Autonomous Quant Run** > กดปุ่ม **Run workflow** (ตั้ง `force_run` เป็น `true` เพื่อทดสอบรันรอบแรกทันที)
