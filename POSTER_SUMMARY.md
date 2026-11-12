# PeakGuard AI — A2 Poster Summary

**ชื่อไทย:** ระบบพยากรณ์และแจ้งเตือนช่วงเวลาที่มีความต้องการใช้ไฟฟ้าสูงด้วยการเรียนรู้ของเครื่อง
**English:** Peak Electricity Demand Prediction and Early Warning Using Machine Learning
**SDG:** Goal 7 — Affordable and Clean Energy

## R1 — Problem Statement
จำลองการเตือนล่วงหน้า 1 ชั่วโมงให้ผู้วางแผนจัดการความต้องการไฟฟ้าเตรียมมาตรการลดโหลดที่เลื่อนได้ โดยยังไม่มีการเชื่อมต่อระบบจริงหรือตรวจวัดผลประหยัด

## R2 — Dataset & Preparation
PJM / PJME_hourly.csv: 145,366 แถว; ช่วง 2002-01-01 01:00:00 ถึง 2018-08-03 00:00:00; หน่วย MW (กำลังไฟฟ้า ไม่ใช่ MWh)
แบ่งเวลา 70/15/15 ก่อนคัดตัวอย่าง; Train P90 = 40,851.00 MW; ลบตัวอย่างที่ Target ข้ามชุดข้อมูล
ชั่วโมงที่หายจาก grid 30; duplicate ขัดแย้ง 4; ไม่เติมจากอนาคต

## R3 — EDA & Insights
- ข้อมูล Train มีค่าเฉลี่ย 32,412.5 MW; เส้น P90 ของข้อมูลรายชั่วโมงเท่ากับ 40,851.0 MW ไม่ใช่ P90 ของค่าเฉลี่ยรายวัน
- กลุ่ม hour ที่มีค่าเฉลี่ย Train สูงสุดคือ 19 (36,776.0 MW); เป็นความสัมพันธ์ ไม่ใช่หลักฐานเชิงสาเหตุ
- กลุ่ม weekday ที่มีค่าเฉลี่ย Train สูงสุดคือ 2 (33,628.5 MW); เป็นความสัมพันธ์ ไม่ใช่หลักฐานเชิงสาเหตุ
- กลุ่ม month ที่มีค่าเฉลี่ย Train สูงสุดคือ 7 (38,409.0 MW); เป็นความสัมพันธ์ ไม่ใช่หลักฐานเชิงสาเหตุ
- นิยาม Peak คือ demand >= Train P90; อัตรา Peak ในชั่วโมงที่สังเกตได้ของ Train = 10.00%
- สัดส่วน Peak สูงสุดใน Train อยู่ที่ชั่วโมง 18 (23.66%)
- สีแสดงค่าเฉลี่ย MW ของ Train ตามวันและชั่วโมง ใช้ค้นหารูปแบบ ไม่ใช้ Test ออกแบบ Features
- Correlation คำนวณจาก Features ของ Train เท่านั้น; ค่าสูงไม่ได้ยืนยันเหตุและผล และ lags มีความสัมพันธ์กันได้

## R4 — Baseline vs Machine Learning
เลือก **Random Forest** จาก Validation F2 แล้วล็อกก่อนเปิด Test; ตารางต่อไปนี้เป็นผล Test ขั้นสุดท้าย ไม่ใช้เลือกโมเดล

| Model | Accuracy | Precision | Recall | F1 | F2 | AP (PR-AUC summary) |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.9775 | 0.8757 | 0.8757 | 0.8757 | 0.8757 | 0.9593 |
| Previous Day | 0.9485 | 0.7170 | 0.7125 | 0.7147 | 0.7134 | 0.7917 |
| Logistic Regression | 0.9885 | 0.8944 | 0.9900 | 0.9398 | 0.9693 | 0.9922 |
| Random Forest | 0.9898 | 0.9018 | 0.9958 | 0.9464 | 0.9754 | 0.9961 |

F2 ต่างจาก Persistence = +0.0997; ไม่รับรองว่าจะชนะ Baseline ในทุกช่วงเวลา

## R5 — Error Analysis
โมเดลที่เลือก: FP 206 ตัวอย่าง, FN 8 ตัวอย่าง; รายละเอียด outputs/error_cases.csv และ errors_by_hour/month.csv
FP อาจทำให้เตรียมมาตรการเกินจำเป็น; FN อาจทำให้ไม่เตรียมรับ Peak จึงใช้ Recall และ F2 เป็นหลัก ต้นทุนเงินจริงยังไม่ได้วัด

## R6 — Recommendations & Limitations
ใช้เป็นข้อมูลประกอบการวางแผนและให้คนอนุมัติมาตรการก่อนลงมือ ต้องทดสอบ telemetry latency, timezone/DST, drift, calibration และผลทางธุรกิจก่อนใช้งานจริง
P90 เป็นนิยามเชิงสถิติ ไม่ใช่เกณฑ์วิศวกรรมของโครงข่าย และชุดข้อมูลสหรัฐฯ ไม่ยืนยันผลสำหรับประเทศไทย

## R7 — AI Usage Log
ใช้ AI ช่วยออกแบบ causal pipeline, เขียนโค้ดและตรวจผล; การตรวจรับและกรณีที่ทีมแก้ไข/ปฏิเสธคำแนะนำ: Pending Team Confirmation ห้ามแต่งประวัติ

## Recommended 300-DPI figures
- outputs/figures/eda_heatmap.png
- outputs/figures/model_comparison.png
- outputs/figures/confusion_matrix.png
- outputs/figures/precision_recall.png
- outputs/figures/errors_by_hour.png

## Sources and License
https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
Kaggle public metadata checked 2026-11-12: licenseName = CC0: Public Domain. This is the distributor metadata, not independent legal verification of upstream PJM rights.
Data dictionary verified from actual CSV: Datetime = naive source timestamp label; PJME_MW = PJME demand in MW per source description. Timezone convention not established.
CSV SHA-256: 4eb2b16d42bf07ec41ab55cb842191594cb69452725a6d3c0991658a628fde84

## Team Must Complete
- ชื่อสมาชิก รหัสนิสิต อาจารย์และสถาบัน
- ตรวจข้อกำหนดเทียบ DS Capstone Project Requirements 1.pdf เมื่อได้รับไฟล์
- บันทึกการตัดสินใจ AI จริงอย่างน้อย 1 กรณีที่แก้ไขหรือปฏิเสธ พร้อมเหตุผล
- ตรวจบน Google Colab จริงและจัดวางโปสเตอร์ขนาด A2; ไฟล์นี้เป็นเนื้อหา ไม่ใช่โปสเตอร์สำเร็จ
- Independent review: see validation/INDEPENDENT_REVIEW.md for scope, findings and status
