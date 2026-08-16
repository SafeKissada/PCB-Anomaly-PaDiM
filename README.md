# PCB-Anomaly-PaDiM

Implement [PaDiM](https://arxiv.org/abs/2011.08785) (Defard et al., ICPR
2021) สำหรับรันบน dataset PCB defect/false-call เดียวกับ
[`Anomaly-Detection-THESIS`](https://github.com/SafeKissada/Anomaly-Detection-THESIS)
เพื่อเทียบกับ EXPERIMENT 0 (ConvNeXt+AE) ในเล่ม thesis — ดู
`thesis_ai_context.md` หัวข้อ "Baseline สามระดับ" สำหรับตำแหน่งของ repo
นี้ในภาพรวม

**สถานะ**: implement เสร็จ, smoke test ผ่าน (offline, random-init
backbone), **ยังไม่เคยรันกับข้อมูลจริง**

## จุดต้องระวังก่อนใช้งานจริง (สำคัญที่สุดในไฟล์นี้)

### 1. PaDiM ผูกกับ "ตำแหน่ง" (position-aware) — ต้องเช็ค alignment ของ crop ก่อน

ต่างจาก PatchCore ที่ patch feature ไม่สนใจตำแหน่ง (memory bank รวมทุก
ตำแหน่งเข้าด้วยกัน) **PaDiM สร้าง Gaussian distribution แยกต่อตำแหน่ง
(h, w) ตายตัว** — สมมติฐานคือทุกภาพต้อง align กันเป๊ะ (component เดิม
อยู่ตำแหน่งเดิมในทุกภาพ)

Dataset นี้เป็นภาพ crop จาก AOI ที่ตำแหน่งเดิมของ SMD component ในทาง
ทฤษฎีน่าจะ align กันดีอยู่แล้ว **แต่ควรตรวจสอบก่อนเชื่อผล**:
- Crop จาก AOI แต่ละครั้งตำแหน่ง/ขนาด crop เป๊ะเท่ากันทุกครั้งจริงไหม
  (ไม่มี jitter จากการ calibrate กล้อง/การเคลื่อนของสายพาน)?
- ถ้ามี misalignment แม้เพียงไม่กี่ pixel PaDiM จะเสียหายมากกว่า
  PatchCore/SimpleNet/RD4AD เพราะ Gaussian ที่ตำแหน่งนั้นจะไม่ตรงกับ
  ของจริง — ถ้าสงสัยเรื่องนี้ **ควรรัน PatchCore ควบคู่ไปด้วยเพื่อเทียบ**
  ว่า PaDiM แพ้ PatchCore เยอะผิดปกติไหม (สัญญาณของ misalignment)

### 2. `COV_REG_EPSILON` ต้องเช็คสำหรับ group ที่ data น้อย

Covariance matrix ต่อตำแหน่งต้อง invert ได้ — ถ้าจำนวนภาพ train (normal)
น้อยกว่าหรือใกล้เคียงกับ `NUM_SELECTED_CHANNELS` matrix จะเกือบ singular
(ill-conditioned) ทำให้ Mahalanobis distance ไม่เสถียร

Group ที่เสี่ยงที่สุด: **group 5 (159 normal), group 3 (155 normal)** —
ถ้า `NUM_SELECTED_CHANNELS=100` (default) จำนวนภาพ train (70% ของ
normal ≈ 108-111 ภาพ) **ใกล้เคียงหรือน้อยกว่า** จำนวน channel ที่เลือก
มาก — ควรลด `NUM_SELECTED_CHANNELS` ลง (เช่น 50) หรือเพิ่ม
`COV_REG_EPSILON` ให้สูงขึ้นสำหรับ group เล็กพวกนี้โดยเฉพาะ ไม่ใช้ค่า
default เดียวกันทุก group

### 3. ⚠️ RAM (ไม่ใช่ GPU memory) เป็นคอขวดจริงบน dataset ขนาดจริง — เคยทำ Colab session crash มาแล้ว

**พบจริงจากการใช้งาน**: รันบน group 1 (1,373 ภาพ train) บน Colab ที่มี
RAM 52GB แล้ว **session crash ด้วย "used all available RAM"** ทั้งที่
GPU memory ว่างเหลือเฟือ (L4 23GB) — สาเหตุ: เวอร์ชันแรกของโค้ดเก็บ
feature **ทุก channel** (concat จาก layer1+layer2+layer3 ≈ 1,792
channel สำหรับ wide_resnet50_2) ที่ spatial resolution ละเอียดสุด
(56×56 สำหรับ input 224×224) ไว้ในหน่วยความจำ **ก่อน** ค่อยสุ่มเลือก
100 channel ทีหลัง — คำนวณคร่าวๆ ใช้ RAM ประมาณ 30GB สำหรับแค่
1,373 ภาพ (ยังไม่รวม overhead อื่น)

**แก้แล้ว** (commit ล่าสุด): โค้ดตอนนี้เลือก channel **ทันที** หลัง
extract แต่ละ batch ก่อนเก็บเข้า list ลด memory footprint เหลือ
~`NUM_SELECTED_CHANNELS`/1792 ของเดิม (เช่น 100/1792 ≈ 5.6%)

**ยังต้องระวังอยู่ดีถ้า**:
- รัน group ที่มีภาพ normal เยอะกว่า group 1 มาก (เช่น group 2:
  3,659 ภาพ) — memory ยังโตเชิงเส้นตาม N ภาพอยู่ดี แค่ลด constant
  factor ลงมาก ถ้ายัง OOM ให้ลด `NUM_SELECTED_CHANNELS` เพิ่มอีก หรือ
  ลด `IMAGE_SIZE`/ใช้ `FEATURE_LAYERS` ที่ตื้นกว่า (spatial หยาบกว่า)
- Colab ฟรี tier มักมี RAM จำกัดกว่า (ประมาณ 12GB) — ถ้าใช้ tier ฟรี
  ความเสี่ยง OOM สูงกว่า tier ที่จ่ายเงิน (ที่ทดสอบจริงมี 52GB) มาก

### 4. ไม่มี coreset subsampling เหมือน PatchCore — inference เร็วกว่าแต่ fit ช้ากว่า

การ invert covariance matrix ต่อตำแหน่ง (`torch.linalg.inv`) เป็นจุดคอขวด
ตอน `fit()` — ยิ่ง spatial resolution สูง (ภาพใหญ่/layer ตื้น) ยิ่งช้า
ถ้า fit ช้าเกินไป ให้ลอง reduce `IMAGE_SIZE` หรือใช้ `FEATURE_LAYERS`
ที่ตื้นกว่า (resolution หยาบกว่า) แทน

## วิธีรัน

```bash
pip install -r requirements.txt
python tests/smoke_test.py   # เช็ค pipeline ก่อนเสมอ
```

แก้ `RUN.py`:
```python
OVERRIDES = dict(
    DATA_ROOT="/path/to/your/dataset",
    SPLIT_CACHE_PATH="/path/to/Anomaly-Detection-THESIS/splits/split_assignment.csv",
    NUM_SELECTED_CHANNELS=100,   # ลดลงถ้ารัน group 3/5
    COV_REG_EPSILON=0.01,        # เพิ่มขึ้นถ้ารัน group 3/5
)
```
```bash
python RUN.py
```

## Reference

Defard, T., Setkov, A., Loesch, A., & Audigier, R. (2021). *PaDiM: a Patch
Distribution Modeling Framework for Anomaly Detection and Localization.*
ICPR 2021 Workshops. https://arxiv.org/abs/2011.08785
