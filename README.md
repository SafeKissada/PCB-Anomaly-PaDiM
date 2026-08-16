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

### 3. ไม่มี coreset subsampling เหมือน PatchCore — inference เร็วกว่าแต่ fit ช้ากว่า

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
