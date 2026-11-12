from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.v4.new_notebook()
notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}, 'colab': {'name': 'PeakGuard_AI_CPE312.ipynb', 'provenance': []}}
cells = []


def markdown(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip()))


markdown('''
# PeakGuard AI — CPE312
## ระบบพยากรณ์และแจ้งเตือนช่วงเวลาที่มีความต้องการใช้ไฟฟ้าสูงด้วยการเรียนรู้ของเครื่อง
**Peak Electricity Demand Prediction and Early Warning Using Machine Learning**

SDG 7 — Affordable and Clean Energy | Python | CPU Runtime | CRISP-DM

นี่คือการทดลองย้อนหลัง ไม่ใช่ระบบควบคุมโครงข่ายไฟฟ้า ผลประโยชน์ด้านพลังงานและต้นทุนยังไม่ได้พิสูจน์

### วิธีใช้ Google Colab
1. เปิด Colab → File → Upload notebook → เลือกไฟล์นี้ เลือก Python 3 CPU Runtime
2. Colab มี libraries หลักโดยทั่วไป หาก import ไม่ผ่าน รันเซลล์ติดตั้งแบบเลือกใช้ แล้ว Restart session
3. เลือก MODE และเส้นทางในเซลล์ตั้งค่า ก่อน Runtime → Run all
4. วิธี upload: MODE='upload' แล้วเลือก PJME_hourly.csv จากเครื่อง (ข้อมูลจะถูกส่งเข้า Google Colab ตามที่ผู้ใช้เลือก)
5. วิธี Drive: MODE='drive' ตั้ง DRIVE_PATH เช่น /content/drive/MyDrive/PeakGuard-AI/data/PJME_hourly.csv และอนุมัติ mount ด้วยตัวเอง
6. วิธี Kaggle: ติดตั้ง package kaggle เพิ่ม ตั้ง MODE='kaggle' และตั้ง credentials ผ่าน Colab Secrets/กลไกส่วนตัว ห้ามใส่ค่า key ในเซลล์หรือส่งเข้าระบบควบคุมเวอร์ชัน
7. วิธี local: MODE='local' และ CSV_PATH ชี้ไฟล์จริง รันจากโฟลเดอร์ PeakGuard-AI

ผลลัพธ์: outputs/figures/*.png (300 DPI), metrics CSV, prediction/error CSV, run_manifest.json, model artifact และ POSTER_SUMMARY.md
Notebook มีโค้ดทั้งหมดในตัว ไม่ต้องมี src/peakguard.py บน Colab

**ข้อจำกัดการตรวจรับ:** ไม่พบเอกสาร DS Capstone Project Requirements 1.pdf ใน Workspace จึงเทียบ R1–R7 จากโจทย์ที่ได้รับ ทีมต้องตรวจ PDF เพิ่มเมื่อได้รับ และต้องยืนยันการรันบน Colab จริง
''')
markdown('''
## R1 — Business Understanding
**Problem:** ผู้วางแผนจัดการความต้องการไฟฟ้าต้องการเตรียมรับชั่วโมงที่โหลดสูง โดยใช้ข้อมูลย้อนหลังที่ทราบแล้ว

**ผู้ใช้และการตัดสินใจ:** ในสถานการณ์จำลอง ผู้วางแผนใช้คำเตือนล่วงหน้า 1 ชั่วโมงเพื่อเตรียมตารางเลื่อนโหลดที่ยืดหยุ่น ประสานงาน และให้คนพิจารณามาตรการก่อนใช้งาน ไม่มีการสั่งตัดไฟอัตโนมัติ

**Prediction objective:** ณ t หลังมีค่าของชั่วโมง t แล้ว ทำนายว่า demand ณ t+1 >= P90 จาก Train หรือไม่ (0=Normal, 1=Peak)

**Business objective:** สนับสนุนการเตรียมการ ไม่รับรองลดค่าไฟหรือป้องกันไฟดับ ค่า P90 เป็นเกณฑ์สถิติ ไม่ใช่ขีดจำกัดของระบบจริง

**Inputs:** current demand, lags 1/24/168 ชั่วโมง, rolling mean/std 24/168 ชั่วโมง, trend และปฏิทินชั่วโมงเป้าหมายซึ่งรู้ล่วงหน้า

**False Positive:** เตือนทั้งที่ไม่ Peak อาจเสียเวลาเตรียมการหรือเลื่อนโหลดเกินจำเป็น

**False Negative:** ไม่เตือนทั้งที่ Peak อาจเตรียมมาตรการไม่ทัน สมมติให้มีน้ำหนักสูงกว่า แต่ยังไม่ทราบต้นทุนเงินจริง

**Metrics:** Recall และ F2 เป็นหลัก (F2 ให้น้ำหนัก Recall มากกว่า Precision); Accuracy, Precision, F1, Average Precision และ confusion matrix เป็นตัวประกอบ

**Success criteria:** pipeline รันจบด้วยข้อมูลจริง >= ประมาณ 1,000 แถว, ทุก Features เป็น causal, ไม่มี Test leakage, เปรียบเทียบ Baseline อย่างเป็นธรรม และรายงานผลจริงแม้ ML ไม่ชนะ ไม่ตั้งเป้า Accuracy โดยไม่มีหลักฐาน

**CRISP-DM:** R1 Business → R2 Data Understanding/Preparation → R3 EDA → R4 Modeling → R5 Evaluation → R6 Deployment considerations → R7 กระบวนการใช้ AI
''')
markdown('''
## Setup — การติดตั้งแบบเลือกใช้
เซลล์นี้ไม่ติดตั้งอะไรโดยอัตโนมัติ เลือกติดตั้งเฉพาะเมื่อ environment ขาด package; การติดตั้งต้องใช้ network ไป PyPI
หากใช้ Kaggle mode ต้องติดตั้ง kaggle เพิ่มด้วย ควรบันทึกเวอร์ชันจริงจาก run manifest
''')
code('''
INSTALL_DEPENDENCIES = False
INSTALL_KAGGLE = False
if INSTALL_DEPENDENCIES or INSTALL_KAGGLE:
    import subprocess
    import sys
    packages = ["numpy>=1.26,<3", "pandas>=2.2,<3", "matplotlib>=3.8,<4", "seaborn>=0.13,<0.14", "scikit-learn>=1.5,<2", "joblib>=1.4,<2"] if INSTALL_DEPENDENCIES else []
    if INSTALL_KAGGLE:
        packages.append("kaggle")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
''')
markdown('''
## ฟังก์ชันหลักของ pipeline
ฟังก์ชันต่อไปนี้แยกหน้าที่ loading, cleaning, features, splitting, training, evaluation และ export เพื่อตรวจสอบได้
ยังไม่มีการ train หรืออ่าน Test เพื่อเลือกโมเดลในเซลล์ประกาศฟังก์ชัน
''')
source = (ROOT / 'src' / 'peakguard.py').read_text(encoding='utf-8')
blocks = source.split('\n\ndef ')
code(blocks[0])
for block in blocks[1:]:
    code('def ' + block)
markdown('''
## R2 — Data Understanding & Preparation
### Dataset และ provenance
แหล่งข้อมูล: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

ตรวจ metadata สาธารณะจาก Kaggle เมื่อ 12 พฤศจิกายน 2026: licenseName = **CC0: Public Domain**; source description ระบุว่าข้อมูลมาจาก PJM และมีหน่วย MW
นี่คือ metadata ของผู้เผยแพร่ ไม่ใช่การรับรองสิทธิ upstream ทางกฎหมายอย่างอิสระ

**Data dictionary ที่ตรวจจาก CSV จริง:**
- Datetime: timestamp label แบบไม่มี timezone ในไฟล์; ยังไม่ยืนยันว่าเป็น hour-start/hour-end หรือวิธี DST
- PJME_MW: ค่าโหลดของ PJME หน่วย MW (กำลังไฟฟ้า ไม่ใช่พลังงาน MWh)

เวอร์ชันที่ดาวน์โหลดเพื่อตรวจมี 145,366 แถว 2 คอลัมน์ ช่วง 2002-01-01 01:00:00 ถึง 2018-08-03 00:00:00
แต่ละการรันจะคำนวณจำนวนแถว ช่วงเวลา คุณภาพ และ SHA-256 ใหม่ ไม่สมมติว่าทุกสำเนาเหมือนกัน

### นโยบายคุณภาพข้อมูล
- แปลง Datetime ตามรูปแบบที่ตรวจแล้ว; วันที่ไม่ถูกต้องถูกตัดและนับใน report
- ตัวเลขหาย/ไม่ใช่ตัวเลข/ไม่ finite/<=0 ถือว่าใช้ไม่ได้และคงเป็น missing
- ชั่วโมงซ้ำค่าตรงกันยุบได้; ชั่วโมงซ้ำขัดแย้งถือว่า unknown ไม่เลือกแถวสุดท้าย
- สร้าง hourly grid และเว้นช่องว่าง ไม่ interpolate, backfill หรือ shift ข้ามช่องว่าง
- ค่าที่สูงหรือต่ำเชิงสถิติไม่ถูกลบทิ้งอัตโนมัติ เพราะ Peak เป็นสิ่งที่ต้องการศึกษา
- DST อาจทำให้ timestamp ซ้ำ/หาย; ไม่อ้างว่านโยบายนี้กู้คืน elapsed UTC ได้ ถูกจำกัดเป็นการทดลองตาม source hour labels
''')
code('''
MODE = "local"
CSV_PATH = "data/PJME_hourly.csv"
DRIVE_PATH = None
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
raw, provenance = load_data(MODE, CSV_PATH, DRIVE_PATH)
cleaned, quality = clean_data(raw)
from IPython.display import display, Markdown
print("Dataset provenance:", provenance)
display(raw.head())
display(pd.DataFrame({"Before / After cleaning": {key: value for key, value in quality.items() if not isinstance(value, dict)}}))
print("Raw dtypes:", quality["raw_dtypes"])
print("Raw missing values:", quality["raw_missing_by_column"])
print("Observed time intervals (hours: count):", quality["observed_interval_counts_hours"])
if len(raw) < 1000:
    print("ข้อจำกัด: Dataset ต่ำกว่า 1,000 แถว ต้องอธิบายความไม่แน่นอนและอาจไม่พอสำหรับ lag 168 ชั่วโมง")
''')
markdown('''
### Causal features และ chronological split
Prediction time t ใช้ค่าชั่วโมง t ที่จบแล้ว รวม rolling window ที่สิ้นสุด t; ไม่ต้อง shift rolling เพิ่มเพราะ Target คือ t+1

lag_24h = D(t-24); seasonal baseline สำหรับชั่วโมงเป้าหมายวันก่อน = D(t+1-24) = D(t-23) ซึ่งต่างกัน 1 ชั่วโมง

Target = D(t+1) บน grid รายชั่วโมง ไม่ใช่แถวถัดไปหลังลบ missing; calendar ใช้ปฏิทิน t+1 ที่ทราบได้ ณ t

แบ่ง **raw grid 70/15/15 ก่อน drop missing features** แล้วตัด sample ที่ target_time ข้ามเขตชุดข้อมูล
ประวัติจาก Train ใช้เป็น Features ของ Validation/Test ได้ เพราะทราบอยู่แล้ว แต่อนาคตห้ามย้อนเข้า Train

Peak threshold = P90 ของ demand ที่สังเกตได้ใน raw Train เท่านั้น ใช้ค่าเดียวกับทุกชุดและทุกโมเดล
เลือก P90 เพื่อศึกษา tail สูงประมาณ 10% ของ Train; ties อาจทำให้อัตราไม่เท่ากับ 10% และ drift ทำให้ Test มีสัดส่วนต่างกัน
ห้ามเปลี่ยนนิยามจากผล Test; เกณฑ์วิศวกรรมอื่นเป็นการทดลองแยกในอนาคต
''')
code('''
feature_frame, feature_columns = make_features(cleaned)
partitions, peak_threshold, split_audit = split_data(cleaned, feature_frame, feature_columns)
print(f"Training-only P90: {peak_threshold:,.2f} MW")
display(pd.DataFrame(split_audit).T)
print("Input features:", feature_columns)
train_end = split_audit["train"]["raw_end_exclusive"]
train_history = cleaned.loc[cleaned.index < pd.Timestamp(train_end), "demand_mw"].dropna()
quartile1, quartile3 = train_history.quantile([.25, .75])
iqr = quartile3 - quartile1
outlier_limits = (float(quartile1 - 1.5 * iqr), float(quartile3 + 1.5 * iqr))
print("Train-only IQR audit bounds (MW):", outlier_limits)
print("Statistical extremes retained:", int(((train_history < outlier_limits[0]) | (train_history > outlier_limits[1])).sum()))
assert cleaned.index.is_monotonic_increasing and cleaned.index.is_unique
assert (cleaned.index.to_series().diff().dropna() == np.timedelta64(1, "h")).all()
for name, subset in partitions.items():
    assert (subset["target_time"] - subset.index == np.timedelta64(1, "h")).all()
    assert subset[feature_columns].notna().all().all()
    assert (subset["is_peak"] == (subset["target_mw"] >= peak_threshold).astype(int)).all()
print("PASS: hourly grid, exact next-hour targets, valid features, frozen labels and boundary purging")
''')
markdown('''
## R3 — EDA & Insights
ทุกกราฟต่อไปนี้ใช้ Train เท่านั้น เพื่อไม่ใช้ Test ชี้นำการออกแบบ คำอธิบายและ insight ภาษาไทยสร้างจากค่าจริงหลังรัน
กราฟใช้ข้อความอังกฤษเพื่อหลีกเลี่ยงปัญหา font ไทยใน Colab; มีชื่อแกนและหน่วยครบ คำอธิบายไทยอยู่ใต้กราฟ
PNG บันทึก 300 DPI ใน outputs/figures/ แนะนำ heatmap และ time plot สำหรับโปสเตอร์
''')
code('''
eda_captions = exploratory_plots(cleaned, partitions["train"], feature_columns, peak_threshold, train_end, OUTPUT_DIR)
for figure_name, caption in eda_captions.items():
    display(Markdown(f"**{figure_name}:** {caption}"))
''')
markdown('''
## R4 — Modeling & Baseline Comparison
1. Persistence: คาด D(t+1)=D(t), เตือนเมื่อ >= Train P90
2. Previous Day: คาด D(t+1)=D(t-23), ใช้ P90 เดียวกัน
3. Logistic Regression: StandardScaler fit เฉพาะ Train, class_weight='balanced'
4. Random Forest: 160 trees, depth 16, leaf >=5, balanced_subsample, seed=312, CPU สอง threads

Hyperparameters กำหนดไว้ก่อน Test ไม่ค้นจำนวนมากเพื่อลด overfitting ของ Validation และให้เหมาะกับ Colab CPU
แต่ละ ML ใช้ Validation เลือก decision cutoff จาก 0.05–0.95 ขั้น 0.025; ค่าจุดตัดนี้แยกจาก Peak threshold หน่วย MW
เลือกผู้ชนะจาก Validation F2 → Recall → Precision → ชื่อโมเดล กติกาเสมอแน่นอน รวม Baselines เป็นผู้ชนะได้
ไม่ refit บน Train+Validation เพื่อรักษาโมเดล/จุดตัดที่เลือกแล้ว ต้นทุนคือไม่ได้ใช้ Validation เพิ่มในการ fit
''')
code('''
candidates, validation_metrics, selected_name = train_candidates(partitions, feature_columns, peak_threshold)
display(validation_metrics)
print("Frozen winner selected on validation:", selected_name)
selection_snapshot = {"selected_name": selected_name, "peak_threshold_mw": peak_threshold, "decision_cutoffs": {name: candidate["cutoff"] for name, candidate in candidates.items()}, "selection_rule": "validation F2, Recall, Precision, model name"}
(OUTPUT_DIR / "selection_before_test.json").write_text(json.dumps(selection_snapshot, indent=2), encoding="utf-8")
''')
markdown('''
## R5 — Final Evaluation & Error Analysis
เปิด Test หลังล็อกการเลือกแล้ว เปรียบเทียบทุกโมเดลบน sample ชุดเดียวกัน ไม่ปรับโมเดล/Features/P90/cutoff จากผลนี้

**PR-AUC convention:** ใช้ Average Precision (AP) แบบ step-weighted จาก sklearn.average_precision_score ไม่ใช่ trapezoidal area
Baseline scores เป็นค่าคาดการณ์ MW ส่วน ML scores เป็น model probabilities ที่ยังไม่ได้ calibration; PR curves ใช้ ranking ของแต่ละโมเดล ไม่ใช้ MW เป็น probability

หาก Test มีคลาสเดียว AP รายงาน undefined; Precision/Recall/F-score ที่ denominator เป็นศูนย์ใช้ 0 และอ่านควบคู่ support
Feature importance ใช้ Random Forest training impurity importance แม้ไม่ใช่ผู้ชนะ; อาจลำเอียงต่อ feature ที่แบ่งค่าได้มากและ correlated lags จึงไม่ใช่ causal importance
''')
code('''
test_metrics, test_predictions = evaluate_test(candidates, partitions["test"], feature_columns, selected_name)
display(test_metrics)
print(classification_report(test_predictions["actual_peak"], test_predictions["selected_prediction"], labels=[0, 1], target_names=["Normal", "Peak"], zero_division=0))
error_tables = evaluation_plots(candidates, partitions["test"], feature_columns, test_metrics, test_predictions, selected_name, OUTPUT_DIR)
for grouping, error_table in error_tables.items():
    display(Markdown(f"### ข้อผิดพลาดแยกตาม {grouping} ของเวลาเป้าหมาย"))
    display(error_table)
    worst = error_table.loc[error_table["FN"].idxmax()]
    display(Markdown(f"กลุ่ม {int(worst['group'])} มี FN มากที่สุด {int(worst['FN'])} จาก actual peaks {int(worst['actual_peaks'])}; ต้องอ่านจำนวนตัวอย่างและ Recall ร่วมกัน จำนวนผิดมากไม่ได้แปลว่าอัตราผิดสูงสุด"))
false_positive = int((test_predictions["error_type"] == "FP").sum())
false_negative = int((test_predictions["error_type"] == "FN").sum())
display(Markdown(f"**{selected_name}:** FP={false_positive:,}, FN={false_negative:,}; FP เสี่ยงเตรียมการเกินจำเป็น ส่วน FN เสี่ยงเตรียมรับ Peak ไม่ทัน ทั้งสองกรณียังไม่แปลงเป็นต้นทุนเงินจริง"))
display(test_predictions.loc[test_predictions["error_type"] != "Correct"].head(10))
selected_test = test_metrics.loc[test_metrics["model"] == selected_name].iloc[0]
persistence_test = test_metrics.loc[test_metrics["model"] == "Persistence"].iloc[0]
display(Markdown(f"F2 ต่างจาก Persistence = {selected_test['f2'] - persistence_test['f2']:+.4f}; รายงานตามจริง ไม่เปลี่ยนผู้ชนะจาก Test"))
''')
markdown('''
### คำอธิบายกราฟประเมิน
- Model comparison: สเกล 0–1 เปรียบเทียบ Precision/Recall/F2/AP ร่วมกัน Accuracy สูงเพียงอย่างเดียวอาจเกิดจาก Normal มีมาก
- Confusion matrix: แถวคือค่าจริง คอลัมน์คือค่าทำนาย ช่อง Peak→Normal คือ FN หน่วยเป็นจำนวนตัวอย่าง
- PR curve: แสดง trade-off เมื่อเลื่อน cutoff ใช้วิเคราะห์ท้ายการทดลอง ไม่ใช้เลือก cutoff ใหม่จาก Test
- Feature importance: แสดงพฤติกรรมของ Random Forest ที่ fit บน Train เท่านั้น ไม่ใช่เหตุผลเชิงสาเหตุของ Peak
- Error plots: ชั่วโมง/เดือนที่ผิดมากต้องพิจารณา support และ actual_peaks ร่วม ไม่สรุปจากจำนวนอย่างเดียว

Class imbalance: baseline ทาย Normal ทั้งหมดอาจได้ Accuracy สูง แต่ Recall=0 จึงไม่ใช่ความสำเร็จของการเตือน Peak
''')
markdown('''
## R6 — Recommendations, Limitations & Deployment Considerations
ผู้ใช้เป้าหมายคือผู้วางแผน Demand Management ในสถานการณ์จำลอง เตรียมมาตรการหรือประสานงานก่อนชั่วโมงที่คาดว่า Peak และให้คนตรวจทาน

สิ่งที่ข้อมูลพิสูจน์ได้: metrics/ข้อผิดพลาดบน holdout ช่วงเวลานี้เท่านั้น ไม่ยืนยันประหยัดพลังงาน ลดค่าไฟ ลด CO2 หรือป้องกัน blackout

ก่อนใช้งานจริงต้องกำหนด timezone/hour-ending/DST กับผู้ให้ข้อมูล, telemetry latency, เกณฑ์โหลดเชิงวิศวกรรม, cost ของ FP/FN, alert fatigue, data drift และกระบวนการอนุมัติ
ควรประเมิน rolling-origin หลายช่วงเวลา เพิ่ม weather/holiday ที่ทราบ ณ t แล้วศึกษา probability calibration ด้วยข้อมูลแยกเพิ่มเติม ห้ามใช้ Test ชุดนี้ปรับซ้ำ

ข้อจำกัด: ข้อมูลอดีตสหรัฐฯ ช่วงเวลาคงที่, ไม่มี weather, ไม่มีเครือข่ายจริง, ไม่ยืนยันผลประเทศไทย, หนึ่ง temporal split, การทิ้ง missing ทำให้ coverage ลดลง และ naive timestamps ไม่รับรอง elapsed UTC รอบ DST

Joblib artifact ใช้เพื่อทำซ้ำการทดลองเท่านั้น ห้ามโหลดไฟล์ joblib จากแหล่งที่ไม่เชื่อถือ เพราะอาจรันโค้ดได้
''')
code('''
run_manifest, selected_artifact = export_results(OUTPUT_DIR, provenance, quality, split_audit, peak_threshold, eda_captions, validation_metrics, test_metrics, test_predictions, selected_name, candidates, feature_columns)
alert = simulate_alert(cleaned, selected_artifact)
(OUTPUT_DIR / "simulated_alert.json").write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")
print("Historical simulation at latest available source hour (future outcome unknown):")
display(alert)
print("Poster content written to POSTER_SUMMARY.md")
display(Markdown(Path("POSTER_SUMMARY.md").read_text(encoding="utf-8")))
''')
markdown('''
## R7 — AI Usage Documentation
บันทึกอย่างโปร่งใส: AI ใน session นี้ได้รับโจทย์ให้วางแผน สร้าง Notebook ป้องกัน leakage และจัดทำเอกสาร
ยังไม่ทราบการตัดสินใจจริงของสมาชิก จึงไม่อ้างว่าทีมรับหรือปฏิเสธคำแนะนำแล้ว

| Question Asked | AI Recommendation | Team Decision | Reason |
|---|---|---|---|
| โจทย์นี้: สร้าง Notebook PeakGuard AI ครบ R1–R7 | ใช้ causal features, Train P90, validation-only selection และรายงานผลจริง | Pending Team Confirmation | ทีมต้องอ่านโค้ดและตรวจผลก่อนตัดสินใจ |
| ทีมต้องถาม/ทบทวนวิธีจัดการ missing และ duplicate ชั่วโมงจริง | ข้อเสนอ AI: เก็บชั่วโมงขัดแย้งเป็น missing ไม่เลือกแถวสุดท้าย | Pending Team Confirmation | บันทึกเหตุผลจริงหากรับ แก้ไข หรือปฏิเสธ |
| Pending Actual Team Question | Pending Actual AI Recommendation | Pending Actual Team Decision | Pending Actual Reason |

**ยังไม่ครบเงื่อนไขการตัดสินใจ AI ของทีม:** ต้องมีอย่างน้อย 1 เหตุการณ์จริงที่ทีมแก้ไขหรือปฏิเสธคำแนะนำ พร้อมคำถาม คำแนะนำ การตัดสินใจ เหตุผล และหลักฐาน ห้ามใช้แถวตัวอย่างเป็นประวัติจริง

### คำถามที่ควรอธิบายต่ออาจารย์ได้
- ทำไมใช้ Recall/F2: FN มีผลต่อการเตรียมการ และ F2 ให้น้ำหนัก Recall มากกว่า
- ทำไมไม่ shuffle: ต้องจำลองการพยากรณ์อนาคตตามเวลา
- ทำไม P90 มาจาก Train: ไม่ให้นิยาม Peak เห็นอนาคตและใช้เกณฑ์เดียวอย่างเป็นธรรม
- ทำไม seasonal baseline shift 23: เป้าหมาย t+1 ย้อนวันก่อนคือ t-23
- ทำไม current demand ไม่ leakage: สมมติว่ารู้ค่าชั่วโมง t แล้ว แต่ห้ามใช้ demand t+1 เป็น Input
- ถ้า ML ไม่ชนะ: รายงานตามจริงและศึกษาข้อจำกัดโดยไม่ tune Test ซ้ำ
- ทำไม MW ไม่ใช่ MWh: MW เป็นกำลังไฟฟ้า ขณะที่ MWh เป็นพลังงานสะสม

### Final acceptance checklist
- การรันนี้สร้าง metrics และกราฟจาก CSV จริง ตรวจ manifest และ SHA-256
- การผ่าน schema/syntax/local execution ไม่ใช่หลักฐานว่าทดลอง Colab จริงแล้ว
- Colab upload/Drive/Kaggle credentials ต้องทดสอบในบัญชีผู้ใช้; ไม่เข้าถึง authentication ของผู้ใช้โดยอัตโนมัติ
- Course PDF / สมาชิก / ประวัติการตัดสินใจ AI / independent review: ยืนยันตามหลักฐานจริงก่อนส่ง
''')
notebook.cells = cells
nbformat.validate(notebook)
nbformat.write(notebook, ROOT / 'PeakGuard_AI_CPE312.ipynb')
print(f'Wrote {len(cells)} cells; all production functions embedded.')
