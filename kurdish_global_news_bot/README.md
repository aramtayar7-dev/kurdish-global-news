# Kurdish Global News Bot (ڕۆبۆتی هەواڵی جیهانیی کوردی)
### Autonomous 24/7 International News Aggregation & Telegram Publishing Platform

---

## 1. Project Overview & Mission

**Kurdish Global News Bot** is an autonomous, production-grade 24/7 international news monitoring and Telegram publishing platform written in Python. It continuously discovers breaking and developing news worldwide from multiple independent sources, evaluates credibility and importance, filters duplicates and syndicated copies, translates news into natural **Central Kurdish (Sorani)**, and **immediately publishes** eligible stories to a designated Telegram channel.

### Key Operating Principles
* **Discovery Cycle:** Checks news sources every **60 seconds** continuously (24/7).
* **Immediate Publication:** Stories are evaluated, translated, and published **immediately** upon qualification. There is **no hourly schedule**, **no delayed batch accumulation**, and **no artificial cooldown**.
* **Central Kurdish (Sorani) Localization:** Professional grammatical accuracy, preserving political/military terminology, personal names, countries, dates, numbers, and original URLs.
* **Resilient Architecture:** Full crash recovery and transaction safety on SQLite WAL mode; source failures are isolated without stopping the pipeline; automatic Windows recovery.

---

## 2. Global News Coverage (20 Categories)

The platform actively aggregates news across 20 global news domains:

1. **International Politics** (سیاسەتی نێودەوڵەتی)
2. **World Leaders & Governments** (سەرکردە و حکومەتەکانی جیهان)
3. **Wars & Armed Conflicts** (شەڕ و ململانێ چەکدارییەکان)
4. **Military & Defense** (پەرەسەندنی سەربازی و بەرگری)
5. **Diplomacy & International Relations** (دیپلۆماسی و پەیوەندییە نێودەوڵەتییەکان)
6. **Global Economy & Financial Markets** (ئابووری جیهانی و بازاڕە دارایییەکان)
7. **AI & Technology** (زیرەکی دەستکرد و تەکنەلۆژیا)
8. **Science & Research** (زانست و توێژینەوە)
9. **Health & Medical** (تەندروستی و پەرەسەندنی پزیشکی)
10. **Climate & Environment** (کەشوهەوا و ژینگە)
11. **Natural Disasters & Emergencies** (کارەساتی سروشتی و فریاگوزاری)
12. **Energy, Oil & Gas** (وزە، نەوت و گاز)
13. **Cybersecurity & Digital Incidents** (ئاسایشی سایبەری و ڕووداوە دیجیتاڵییەکان)
14. **International Organizations** (ڕێکخراوە نێودەوڵەتییەکان - UN, NATO, WHO, IAEA)
15. **Human Rights & Humanitarian** (مافی مرۆڤ و ڕووداوە مرۆیییەکان)
16. **Criminal Investigations & Security** (لێکۆڵینەوەی تاوان و ئاسایشی نێودەوڵەتی)
17. **Space & Astronomy** (بۆشایی ئاسمان و گەردوونناسی)
18. **International Sports** (وەرزشی نێودەوڵەتی)
19. **Business & Corporate** (بازرگانی و کۆمپانیا گەورەکان)
20. **Other Significant Global Events** (ڕووداوە گرنگەکانی تر)

---

## 3. Architecture & Data Flow

```
[ DuckDuckGo News ]   [ RSS/Atom Feeds ]   [ GDELT DOC API ]   [ Official (UN/WHO/NATO) ]
         │                    │                    │                      │
         └────────────────────┴──────────┬─────────┴──────────────────────┘
                                         ▼
                            [ 60-Second Async Discovery ]
                                         │
                                         ▼
                             [ Metadata & URL Extractor ]
                                         │
                                         ▼
                               [ Security & Quality Validator ]
                                         │
                                         ▼
                           [ Importance & Credibility Ranker ]
                                         │
                                         ▼
                        [ Multi-Layer Deduplication Engine ]
                           ├── Exact Canonical URL Match
                           ├── Content SHA256 Hash
                           ├── RapidFuzz Title Similarity
                           └── Event Fingerprint Analysis
                                         │
                                         ▼
                         [ Sorani Kurdish Translation Engine ]
                           ├── Google Translator (deep-translator)
                           ├── Terminology & Entity Preservation
                           └── Quality & Ratio Validation
                                         │
                                         ▼
                          [ Telegram Immediate Publisher ]
                           ├── High-Res Image Validation & Download
                           ├── Kurdish RTL HTML Formatting
                           ├── RetryAfter / Rate-Limit Backoff
                           └── Delivery Confirmation -> SQLite
```

---

## 4. System Requirements

* **Operating System:** Windows 10 (64-bit) or Windows 11 (64-bit) / Windows Server 2019+
* **Python Version:** Python 3.10, 3.11, or 3.12 (Tested with 3.12.10)
* **Network:** Stable internet connection with HTTPS access to Telegram API and news feeds.
* **Storage:** 500 MB free disk space for SQLite database, WAL logs, and automatic backups.

---

## 5. Step-by-Step Windows Installation & Deployment

### Step 1: Open PowerShell as Administrator
Open PowerShell or Windows Terminal in your project directory:
```powershell
Set-Location -Path "d:\guevara hawal\kurdish_global_news_bot"
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Required Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-asyncio
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in Notepad:
```powershell
notepad .env
```
Enter your Telegram credentials:
```env
TELEGRAM_BOT_TOKEN=1234567890:ABCDefGhIJKlmNoPQRsTUVwxyZ_1234567
TELEGRAM_CHANNEL_ID=@YourKurdishNewsChannel
NEWS_CHECK_INTERVAL_SECONDS=60
AUTO_PUBLISH=True
MIN_IMPORTANCE_SCORE=0.4
ENABLE_IMAGES=True
```

### Step 5: Telegram Channel Setup
1. Create a public or private Telegram Channel.
2. Open Telegram and search for `@BotFather`. Create a new bot with `/newbot` and copy the API Token.
3. Add your bot as an **Administrator** to your channel with **Post Messages** permission.
4. Verify your channel username (e.g., `@YourChannel`) or numeric chat ID (e.g., `-1001234567890`).

### Step 6: Verify and Run Automated Tests
Execute the full test suite (68 tests) to verify system integrity:
```powershell
python -m pytest tests -v
```

### Step 7: Launch the Application
```powershell
python main.py
```

---

## 6. Continuous 24/7 Windows Service Setup

To keep the bot running permanently 24/7 without terminal windows and ensure automatic startup on Windows boot:

### Option A: Windows Task Scheduler (Recommended Native Method)
1. Open PowerShell as Administrator.
2. Register a task to start the bot at system startup and restart on failure:
```powershell
$action = New-ScheduledTaskAction -Execute "d:\guevara hawal\kurdish_global_news_bot\venv\Scripts\python.exe" -Argument "main.py" -WorkingDirectory "d:\guevara hawal\kurdish_global_news_bot"
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 365)
Register-ScheduledTask -TaskName "KurdishGlobalNewsBot" -Action $action -Trigger $trigger -Settings $settings -User "SYSTEM"
```

### Option B: Keep PC Awake & Network Persistent
If running on a local desktop or laptop:
1. **Prevent Sleep Mode:**
   - Go to Windows Settings > **Power & battery** > **Screen and sleep**.
   - Set **"When plugged in, put my device to sleep after"** to **Never**.
2. **Prevent Network Adapter Power Off:**
   - Open Device Manager (`devmgmt.msc`).
   - Expand **Network adapters** > right-click your active adapter > **Properties** > **Power Management**.
   - Uncheck **"Allow the computer to turn off this device to save power"**.
3. *Note:* A local computer must remain powered on with electrical power to execute jobs; it cannot run while physically turned off.

---

## 7. Database, Logging & Backup Strategy

* **Database Engine:** SQLite 3 in **WAL (Write-Ahead Logging)** mode with `aiosqlite`.
  - Fast concurrent reads while writing.
  - Automatic checkpointing and transactions.
  - Zero loss of publication history after restart.
* **Automatic Backups:** Scheduled automated backups into `backups/news_bot_backup_YYYYMMDD_HHMMSS.db`.
* **Logging:**
  - `logs/newsbot.log`: General operational logs with 10 MB rotation (5 files).
  - `logs/errors.log`: Error isolation log with 5 MB rotation.
  - In-database audit trail in `system_logs` table.

---

## 8. Verification & Acceptance Criteria Audit

| Criteria | Description | Status | Verification Method |
|---|---|---|---|
| **TEST 1** | Discovery cycle executes every 60 seconds | **PASS** | `test_scheduler.py::test_default_interval` |
| **TEST 2** | Breaking articles published immediately without waiting | **PASS** | `test_acceptance.py::test_acceptance_criteria_2` |
| **TEST 3** | Normal eligible articles published immediately | **PASS** | `test_acceptance.py::test_acceptance_criteria_3` |
| **TEST 4** | No hourly scheduler or delayed batching exists | **PASS** | `test_publishing.py::test_no_hourly_scheduler_exists` |
| **TEST 5** | Multiple articles processed independently | **PASS** | `test_duplicates.py::test_different_stories_not_duplicate` |
| **TEST 6** | Cross-source duplicate stories prevented | **PASS** | `test_duplicates.py::test_exact_url_duplicate`, `test_fuzzy_title_duplicate` |
| **TEST 7** | Genuinely new developments published as updates | **PASS** | `test_duplicates.py::test_meaningful_update_allowed` |
| **TEST 8** | Source failures isolated without stopping engine | **PASS** | `test_sources.py::test_source_failure_isolation` |
| **TEST 9** | Database survives restarts without data loss | **PASS** | `test_database.py::test_database_survives_restart` |
| **TEST 10** | Interrupted jobs recovered automatically | **PASS** | `test_database.py::test_interrupted_articles_recovery` |
| **TEST 11** | Natural, professional Sorani Kurdish output | **PASS** | `test_translation.py::test_translate_title` |
| **TEST 12** | Telegram posts include source and original link | **PASS** | `test_publishing.py::test_format_telegram_message` |
| **TEST 13** | Image failure falls back gracefully to text | **PASS** | `test_publishing.py::test_text_only_when_no_image` |
| **TEST 14** | Telegram RetryAfter rate limits handled | **PASS** | `test_acceptance.py::test_acceptance_criteria_14` |
| **TEST 15** | System survives long run without leaks/dups | **PASS** | `test_acceptance.py::test_acceptance_criteria_15` |
