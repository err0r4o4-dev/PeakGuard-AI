# ข้อมูล PJME

แหล่งที่โจทย์กำหนด: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
ไฟล์: PJME_hourly.csv

## การตรวจแหล่งข้อมูล
ตรวจ public Kaggle metadata เมื่อ 12 พฤศจิกายน 2026 เก็บ response ใน source_metadata.json:
- title: Hourly Energy Consumption
- licenseName: CC0: Public Domain
- description: ข้อมูลจากเว็บไซต์ PJM มีหน่วย MW
- lastUpdated: 2018-08-30T14:17:03.317Z

License นี้เป็นข้อความ metadata ของผู้เผยแพร่ Kaggle ไม่ใช่การตรวจสิทธิ upstream PJM ทางกฎหมายอย่างอิสระ

## โครงสร้างที่ตรวจจาก CSV จริง
- 145,366 แถว, 2 คอลัมน์
- Datetime: string รูปแบบ YYYY-MM-DD HH:MM:SS ไม่มี timezone
- PJME_MW: numeric ความต้องการกำลังไฟฟ้า PJME หน่วย MW
- ช่วง source labels: 2002-01-01 01:00:00 ถึง 2018-08-03 00:00:00
- ไม่มีหลักฐานใน CSV ยืนยัน timezone, UTC offset, DST หรือ hour-start/hour-ending จึงห้ามสมมติ

ไม่ใช่ข้อมูล MWh และไม่ได้บันทึกสภาพอากาศในไฟล์นี้
รายละเอียด SHA-256, missing, duplicate และช่วงเวลาจริงของแต่ละการรันอยู่ outputs/run_manifest.json

## วิธีนำเข้า
ดาวน์โหลดจาก Kaggle แล้ววาง PJME_hourly.csv ที่โฟลเดอร์นี้ หรือใช้ Colab upload/Drive/Kaggle ตาม README หลัก
ไม่ต้องส่ง credentials มากับงาน ไม่ใส่ kaggle.json หรือ key ใน repository/notebook
Notebook ตรวจชื่อคอลัมน์จริงทุกครั้งและหยุดพร้อมคำอธิบายหากไม่ตรง schema

สำเนาข้อมูลจริงนี้ดาวน์โหลดจาก public Kaggle endpoint ตามที่ระบุในโจทย์ ไม่ใช่ข้อมูลจำลอง
