# PeakGuard AI — CPE312

ระบบพยากรณ์และแจ้งเตือนช่วงเวลาที่มีความต้องการใช้ไฟฟ้าสูงด้วยการเรียนรู้ของเครื่อง

**Peak Electricity Demand Prediction and Early Warning Using Machine Learning**  
SDG 7 — Affordable and Clean Energy

## เป้าหมาย
ผู้วางแผนจัดการความต้องการไฟฟ้าใช้คำเตือนล่วงหน้า 1 ชั่วโมงเพื่อเตรียมมาตรการเลื่อนโหลดที่ยืดหยุ่นในสถานการณ์จำลอง ยังไม่มีการเชื่อมต่อหรือควบคุมโครงข่ายจริง และไม่มีหลักฐานรับรองผลประหยัดพลังงาน ค่าไฟ หรือ CO2

## เปิดบน Google Colab
1. เข้า Google Colab เลือก File → Upload notebook แล้วเปิด PeakGuard_AI_CPE312.ipynb
2. เลือก Python 3 CPU Runtime ไม่ต้องใช้ GPU
3. ตั้ง MODE ในเซลล์ R2 เป็น upload แล้ว Runtime → Run all และเลือก PJME_hourly.csv เมื่อปรากฏหน้าต่าง
4. หรือเลือก drive แล้วตั้ง DRIVE_PATH เป็นเส้นทางเต็ม เช่น /content/drive/MyDrive/PeakGuard-AI/data/PJME_hourly.csv อนุมัติ mount ด้วยบัญชีตนเอง
5. หรือเลือก kaggle ติดตั้ง package kaggle ผ่านเซลล์เลือกติดตั้ง แล้วกำหนด credentials ผ่านกลไกส่วนตัว ห้ามใส่ key ในเซลล์หรือไฟล์ที่ส่งงาน
6. หาก libraries ขาด ตั้ง INSTALL_DEPENDENCIES=True ในเซลล์ Setup รันแล้ว Restart session ก่อน Run all อีกครั้ง
7. ผลลัพธ์อยู่ outputs/ และ POSTER_SUMMARY.md ต้องดาวน์โหลดก่อนปิด Colab runtime เพราะ storage ชั่วคราว

Notebook มีฟังก์ชันทั้งหมดในตัว ไม่ต้องอัปโหลด src/peakguard.py ขึ้น Colab ด้วย ไม่ติดตั้ง dependencies หรือเชื่อมต่อ Kaggle/Drive อัตโนมัติเมื่อ MODE=local

Kaggle mode ใช้ environment variables KAGGLE_API_TOKEN หรือ KAGGLE_USERNAME/KAGGLE_KEY หรือ kaggle.json ตามกลไกของ Kaggle CLI; ตัว Notebook ไม่แสดงค่าและไม่ใส่ credential fallback หากใช้ Colab Secrets ให้ผู้ใช้กำหนดตัวแปรส่วนตัวเอง

## รันในเครื่อง
ต้องใช้ Python 3.12+ และสร้าง virtual environment ภายในเครื่องเอง เช่นจากโฟลเดอร์ PeakGuard-AI:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s validation -p "test_*.py" -v
.\.venv\Scripts\python.exe validation/execute_notebook.py
~~~

หรือเปิด Notebook ด้วย Jupyter และตั้ง MODE=local, CSV_PATH=data/PJME_hourly.csv ข้อมูลจริงต้องอยู่ตามเส้นทางนั้น
การติดตั้ง packages ใช้ network ไป PyPI; การรัน local ปกติไม่ส่งข้อมูล Dataset ไปภายนอก

requirements.txt ระบุช่วงเวอร์ชันที่รองรับ ส่วน validation/environment-freeze.txt บันทึกเวอร์ชันจริงที่ใช้ตรวจรอบนี้
requirements-dev.txt เพิ่ม Jupyter packages เฉพาะการพัฒนา/ตรวจรับ ไม่จำเป็นสำหรับผู้ใช้ Colab ที่มีพร้อมแล้ว

การทดสอบ discovery รวม pipeline และ loader portability: จำลองเฉพาะ Colab API และการดาวน์โหลด Kaggle ไม่เข้าบัญชีหรือใช้ credential จริง ทดสอบ generator ใน temporary project เพื่อไม่เขียนทับ Notebook ที่ผู้ใช้แก้ ผลปัจจุบันดู validation/followup_report.json

### ตรวจรับบน Colab จริงที่ยังต้องทำ
- Upload: เลือก MODE=upload และ PJME_hourly.csv แล้ว Run all; ตรวจ raw rows/hash ใน provenance, metrics และกราฟครบ รวมทั้งลองชื่อไฟล์ผิดเพื่ออ่าน error
- Drive: ตั้ง DRIVE_PATH ที่มีไฟล์จริง อนุมัติ mount ด้วยบัญชีของตนเอง แล้ว Run all; เส้นทางไม่พบไฟล์ต้องแสดงข้อผิดพลาด ไม่อ่านไฟล์อื่นแทน
- Kaggle: ติดตั้ง CLI และตั้ง credentials ผ่านช่องทางส่วนตัว แล้วเลือก MODE=kaggle; ตรวจว่าไฟล์และ hash ที่อ่านมาจากการดาวน์โหลดรอบนั้น และ error ไม่แสดง credentials
บันทึกผลแต่ละโหมดตามที่รันจริง พร้อม package versions จาก run_manifest.json; local mocks ไม่ยืนยัน authentication หรือความเข้ากันได้กับผู้ให้บริการ ห้ามส่ง credentials มาในรายงาน

## วิธีทดลองและป้องกัน Leakage
- แบ่ง hourly grid ตามเวลา 70% Train, 15% Validation, 15% Test ก่อนคัดแถวที่มี Features ครบ ไม่มี shuffle
- Target คือชั่วโมง t+1 จริงบน grid; ตัด sample ที่เวลา Target ข้าม partition ไม่ใช้แถวถัดไปข้าม missing
- P90 คำนวณจาก demand ที่สังเกตได้ใน Train เท่านั้น ใช้ค่าเดียวทุกชุด
- ณ t สมมติว่าค่าชั่วโมง t จบแล้วและพร้อมใช้; current demand กับ rolling สิ้นสุด t จึงใช้ได้ ห้ามใช้ demand ณ t+1 เป็น Input
- เปรียบเทียบ Persistence, Previous Day, Logistic Regression และ Random Forest ใช้ Test ชุดเดียวกัน
- Seasonal baseline ใช้ D(t-23) เพื่อพยากรณ์ D(t+1) จากเวลาเดียวกันวันก่อน ต่างจาก input lag_24h=D(t-24)
- Scaling และโมเดล fit เฉพาะ Train; Validation เลือก decision cutoff/โมเดลตาม F2, Recall, Precision แล้วล็อกก่อน Test ไม่ refit บน Validation
- เปิด Test ขั้นสุดท้ายเพื่อรายงานทุกโมเดล ไม่เปลี่ยนผู้ชนะจาก Test AP คือ Average Precision ไม่ใช่ trapezoidal PR area
- ตัด incomplete samples ไม่เติมจากอนาคต และไม่ลบค่าที่สูงเพียงเพราะเป็น statistical outlier

### เวลาและข้อมูลหาย
CSV ไม่ใส่ timezone จึงคง naive source labels ไว้ ชั่วโมงซ้ำค่าตรงกันยุบ ชั่วโมงซ้ำขัดแย้งเก็บเป็น missing แล้ว reindex รายชั่วโมง
วิธีนี้ไม่สามารถกู้ elapsed UTC รอบ DST หรือยืนยัน hour-ending ได้ จึงเป็นข้อจำกัดการทดลอง ต้องตกลง convention กับเจ้าของข้อมูลก่อนใช้งานจริง

## สิ่งที่ได้
- PeakGuard_AI_CPE312.ipynb: R1–R7 คำอธิบายไทย โค้ดและผลรันจริง
- POSTER_SUMMARY.md: เนื้อหาสำหรับโปสเตอร์ A2 สร้างจากผลรัน ไม่ใช่ layout A2 สำเร็จ
- outputs/figures/: กราฟ PNG 300 DPI
- outputs/validation_metrics.csv และ test_metrics.csv: แยกหลักฐานเลือกโมเดลกับการประเมินสุดท้าย
- outputs/selection_before_test.json: snapshot การเลือกก่อน Test
- outputs/test_predictions.csv และ error_cases.csv: ตรวจ FP/FN รายชั่วโมง
- outputs/errors_by_hour.csv และ errors_by_month.csv: support, actual peaks, FP/FN และ Recall
- outputs/run_manifest.json: SHA-256 ของ CSV, คุณภาพข้อมูล, partitions, seed, packages และ metrics
- outputs/peakguard_model.joblib และ simulated_alert.json: artifact และตัวอย่าง inference ย้อนหลังที่ไม่มี actual future
- validation/: checks, execution evidence และ generator สำหรับผู้พัฒนา; synthetic data ใช้เฉพาะ tests ไม่ถูกอ้างเป็นผลทดลอง

ห้ามโหลด joblib จากแหล่งไม่เชื่อถือ เพราะ deserialization อาจรันโค้ดได้
หากแก้ src/peakguard.py ต้องรัน validation/build_notebook.py เพื่อฝังโค้ดใหม่ แล้วทดสอบและรัน Notebook ใหม่ generator จะล้างผลเดิมใน Notebook จึงควรทำเฉพาะการพัฒนา

## การอ่านผลและข้อจำกัด
อ่านผลจริงจาก POSTER_SUMMARY.md และ outputs/run_manifest.json ไม่รับรองว่า ML จะชนะ baseline ในทุกช่วงเวลา
Recall/F2 ให้น้ำหนักการไม่พลาด Peak แต่ FP ทำให้เตรียมการเกินจำเป็น ค่า P90 เป็นนิยามสถิติ ไม่ใช่ขีดจำกัดโครงข่าย
โมเดลใช้ข้อมูลสหรัฐฯ ในอดีต ไม่ยืนยันผลประเทศไทย ไม่มี weather/holidays และใช้ temporal split ครั้งเดียว

## สถานะตรวจรับ
ดู VALIDATION.md สำหรับผลตรวจจริงและสิ่งที่ยังไม่ตรวจ ห้ามเรียก schema/local test ว่าเป็นการรันบน Colab จริง
Independent core-flow review ผ่านตามขอบเขตใน validation/INDEPENDENT_REVIEW.md; ยังต้องตรวจ PDF อาจารย์, โหมด upload/Drive/Kaggle ในบัญชีจริง และบันทึก AI ที่สมาชิกตัดสินใจจริงก่อนปิดงาน

## R7 — ทีมต้องยืนยันเอง
Notebook มีตาราง Question Asked | AI Recommendation | Team Decision | Reason โดยระบุ Pending Team Confirmation
ต้องเพิ่มอย่างน้อย 1 เหตุการณ์จริงที่ทีมแก้ไขหรือปฏิเสธคำแนะนำ AI พร้อมเหตุผล ไม่มีการแต่งว่าทีมตัดสินใจแล้ว
