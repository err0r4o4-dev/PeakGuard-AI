# PeakGuard AI — Validation Record

วันที่ตรวจ: 12 พฤศจิกายน 2026 (Asia/Bangkok)

## สถานะรวม: PARTIAL — ยังไม่ปิดการตรวจรับทั้งหมด
Notebook และผลทดลองสร้างแล้วและผ่านการรัน local Python/Jupyter CPU ผ่าน independent review เฉพาะ core flow และข้อแก้ไขที่ส่งตรวจแล้ว แต่ยังขาดการยืนยันโหมด Colab ที่ใช้บัญชีจริง, PDF อาจารย์ และประวัติการตัดสินใจ AI จากทีม
ผู้ใช้อนุญาต independent read-only reviewer แล้ว; การตรวจเริ่มต้นติดข้อจำกัดเครื่องมือ จึงส่งหลักฐานตรงในบริบทและได้ scoped PASS หลังแก้ finding ไม่มีการ Push หรือ Deploy

## หลักฐานปัจจุบัน
- PASS: ข้อมูลจริงจาก public Kaggle, 145,366 แถว Datetime/PJME_MW; source_metadata.json ระบุ CC0: Public Domain
- PASS ปัจจุบัน: unittest discovery ครบ 33 tests ใน 5.485 วินาที; validation/followup-tests.txt และ validation/followup_report.json (ผล 19 tests เดิมอยู่ pipeline-tests.txt)
- PASS: Notebook schema และ compile code cells; ทดสอบการฝังโค้ดตรงกับ src/peakguard.py
- PASS จากรอบก่อน: Run all ผ่าน nbclient บน local Jupyter CPU, 22 code cells ไม่มี cell error; validation/execution_report.json รอบปัจจุบันไม่ Run all หรือ regenerate ทับ Notebook ของผู้ใช้
- PASS: ตรวจ artifacts 61 project-only checks; validation/artifact_report.json รวมการคำนวณ metrics ซ้ำจาก predictions, reload artifact, ตรวจ Target +1 ชั่วโมง และกราฟ 14 ภาพ 300 DPI
- PASS: มี 14 PNG outputs ฝังใน Notebook และ export ใน outputs/figures/
- PASS: ทดสอบเปลี่ยนข้อมูลอนาคตไม่เปลี่ยน Features อดีต และเปลี่ยน Validation/Test ไม่เปลี่ยน Train/P90
- PASS: ข้อมูลหายไม่กลายเป็น Target ข้ามเวลา, conflicting duplicate ไม่เลือกค่าท้าย, Baseline วันก่อนใช้ t-23
- ผู้ใช้ยืนยันว่าลบ CPE312.ipynb เอง; ไม่กู้คืน และ verifier ไม่อ่านไฟล์นอกโครงการ
- Main ตรวจกราฟ model_comparison และ confusion_matrix ด้วยภาพจริง; ไม่ใช่ independent review

## คำสั่งที่ใช้จาก Workspace root
~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s PeakGuard-AI/validation -p "test_*.py" -v
.\.venv\Scripts\python.exe PeakGuard-AI/validation/verify_artifacts.py
~~~

build_notebook.py สร้าง Notebook ใหม่และล้างผลรันเก่า ต้อง execute และ verify ใหม่หลังใช้ generator
requirements-dev.txt ถูกตรวจว่าเป็น dependency list สำหรับ packages เท่านั้น ติดตั้งลง virtual environment ของ Workspace หลังผ่าน network approval
ไม่มี test script ส่ง project content ไปภายนอก ไม่มีการเปลี่ยนฐานข้อมูลหรือ permissions

## ผลทดลองจริง
Train P90 = 40,851 MW; usable Train 97,525, Validation 20,958, Test 20,958 ตัวอย่าง

Validation เลือก Random Forest, decision cutoff=0.375 ก่อนเปิด Test; Logistic Regression cutoff=0.65; Baselines ใช้ 40,851 MW

| Model | Test Recall | Test F2 | Test Average Precision |
|---|---:|---:|---:|
| Persistence | 0.875724 | 0.875724 | 0.959297 |
| Previous Day | 0.712480 | 0.713382 | 0.791728 |
| Logistic Regression | 0.989995 | 0.969272 | 0.992198 |
| Random Forest | 0.995787 | 0.975446 | 0.996095 |

Random Forest: Accuracy 0.989789, Precision 0.901764, F1 0.946446, TN 18,853, FP 206, FN 8, TP 1,891
ผลนี้ใช้ holdout ช่วงเดียว ไม่ใช่คำรับรอง generalization, ผลประหยัด หรือการควบคุมไฟฟ้าจริง

## คุณภาพและข้อจำกัดข้อมูล
- Input ไม่เรียงเวลา; จัดเรียงก่อนประมวลผล
- พบ duplicated timestamp rows 8 แถว ครอบคลุม 4 ชั่วโมงขัดแย้ง; เก็บชั่วโมงเหล่านั้นเป็น missing
- พบ source grid ขาด 30 ชั่วโมง รวม missing หลัง cleaning 34 ชั่วโมง
- ไม่เติมค่าจากอนาคต; ใช้ rolling 168 ชั่วโมงทำให้ตัดแถวเพิ่ม รายละเอียด coverage อยู่ split_audit
- Source ไม่มี timezone/UTC offset; ไม่ยืนยันชั่วโมงจริงรอบ DST หรือ hour-ending ต้องตรวจ upstream convention ก่อนใช้งานจริง
- CC0 มาจาก metadata ผู้เผยแพร่ ไม่ใช่การตรวจสิทธิ upstream อย่างอิสระ

## ปัญหาที่พบและการแก้
- Python ไม่อยู่ใน PATH: พบ Python 3.12.15 ที่ติดตั้งผ่าน uv และสร้าง environment ภายใน Workspace
- Network ใน sandbox ถูกปฏิเสธ: ใช้ supported escalation สำหรับ PyPI และ public Kaggle ไม่เปลี่ยน sandbox/ACL
- apply_patch ผ่าน PowerShell ส่ง multiline argument ไม่ครบ: ใช้ apply_patch executable เดิมผ่าน child_process โดยส่ง argument UTF-8 เป็นก้อนเดียว ไม่เปลี่ยนเครื่องมือหรือสิทธิ
- NumPy/Pandas timedelta warning: ทำ focused failing regression แล้วแก้ใช้ explicit NumPy time units และ duration seconds; ไม่ซ่อน warnings
- Legend บังแท่งกราฟ: ย้าย legend ออกนอก axes แล้วรัน export ใหม่ โดยไม่เปลี่ยน model settings หรือ selection
- Local Jupyter เตือน Windows event-loop fallback และ TCP transport ไม่มี encryption: เป็นคำเตือน runtime ที่บันทึกไว้ใน execution-log.txt; ไม่ใช่ cell failure ไม่เปลี่ยนการตั้งค่า network/security เพื่อหลบคำเตือน
- หลังผู้ใช้ลบ Notebook เดิม: regression test ทำซ้ำ FileNotFoundError ได้ก่อนแก้ แล้วจำกัด verifier ให้อ่านเฉพาะโครงการ; รอบนั้นผ่าน 18 tests, 61 artifact checks และ Notebook 22 code cells (16.45 วินาที)

## Follow-up: รักษาการแก้ไขของผู้ใช้และตรวจ loaders ในเครื่อง
- ผู้ใช้ยืนยันว่าแก้ Notebook เอง โดยลบ Phase labels จากหัวข้อ R1-R6; ปรับ generator ให้ตรงกัน ทดสอบก่อนแก้พบหัวข้อไม่ตรง แล้วผ่านหลังแก้
- ทดสอบ generation ใน temporary project เท่านั้น; SHA-256 ของ Notebook ก่อนและหลังตรงกัน: 3ad47c7a643f03d27d96c6538d97a3d21ab73b59506f89e32087afcc41332925
- เพิ่ม 14 tests: upload เลือก CSV/ปฏิเสธชื่อผิดหรือกำกวม, Drive path/mount, การใช้โหมด Colab นอก Colab, Kaggle direct CSV/ข้อผิดพลาด/ไม่มี output/ไม่มี CLI หรือ config/สมาชิก ZIP ที่ไม่คาดหมาย และ generator headings
- ใช้ CSV จริงใน temporary test directory และจำลองเฉพาะ Colab API กับ subprocess ของ Kaggle; ไม่เข้าบัญชี ไม่ใช้ credential จริง ไม่ดาวน์โหลด และไม่อ้างว่า Colab จริงผ่าน
- การรันทดสอบครั้งแรกพบ fixture ล้าง home environment ของ Windows จึงเข้า branch ผิด; เพิ่ม HOME/USERPROFILE ให้ชี้ temporary test directory แล้วรันทั้งชุดผ่าน ไม่แก้ production loader จากอาการของ fixture
- PASS: 33 tests และ 61 artifact checks จากรอบปัจจุบัน; ไม่เปลี่ยน src/peakguard.py โมเดลหรือผลทดลอง การตรวจ independent core-flow เดิมไม่ครอบคลุม test file ใหม่

## เงื่อนไขที่ยังไม่ผ่านการตรวจรับ
- PASS เฉพาะขอบเขต: independent correctness review ของ core flow, loader และ patch ที่ส่งตรวจ; ดู validation/INDEPENDENT_REVIEW.md ไม่อ้างว่าตรวจทุกไฟล์/กราฟ/Colab จริง
- BLOCKED: ตรวจเทียบ DS Capstone Project Requirements 1.pdf เพราะไม่ได้รับไฟล์
- BLOCKED: R7 ต้องให้ทีมยืนยันประวัติจริง; พบการแก้หัวข้อโดยผู้ใช้แล้ว แต่ยังไม่ทราบเหตุผลและการยืนยันในนามทีม จึงยังไม่เติมแทนตัวอย่าง Pending ใน Notebook
- NOT RUN: Google Colab จริง และ interactive upload/Drive mount/Kaggle credentials ในบัญชีผู้ใช้ ใช้ local Jupyter เป็น equivalent environment ตามโจทย์ แต่ไม่อ้างว่า interactive modes ผ่านแล้ว
- NOT APPLICABLE: Frontend/Backend/Database/Deployment เพราะอยู่นอกขอบเขต

## Review และ usage
Main พัฒนาและรันทดสอบเอง; reviewer แยกตรวจ source ที่ส่งในบริบท พบ Kaggle stale-CSV bug และให้ scoped PASS หลังแก้และตรวจซ้ำ ไม่ได้รันทดสอบเอง
Token usage, upstream inference identity และ monetary cost ที่ยืนยันไม่ได้: unknown ไม่ใช่ศูนย์
