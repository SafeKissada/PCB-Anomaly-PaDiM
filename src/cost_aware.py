"""Cost-aware threshold selection — เลือก threshold จาก total cost แทน
percentile/F1 ทั่วไป

Total Cost(t, r) = r * FN(t) + 1 * FP(t)
Convention เดียวกับ src/evaluate.py: y=1 คือ anomaly, y=0 คือ normal
pred = 1 ถ้า score >= threshold

Cost-aware threshold selection — picks a threshold by total cost instead
of the usual percentile/F1.

Total Cost(t, r) = r * FN(t) + 1 * FP(t)
Same convention as src/evaluate.py: y=1 is anomaly, y=0 is normal.
pred = 1 if score >= threshold.
"""
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix

from src.evaluate import compute_metrics  # reuse ของเดิม ไม่เขียนซ้ำ / reuse the existing one, don't duplicate


def cost_at_threshold(scores: np.ndarray, y_true: np.ndarray,
                       threshold: float, r: float) -> Dict:
    """คำนวณ FN, FP, total cost ที่ threshold หนึ่งๆ

    ใช้ labels=[0, 1] ตรงๆ เสมอ (เหมือน src/evaluate.py) กัน
    confusion_matrix คืน shape ผิดถ้า pred หรือ y_true ที่ส่งเข้ามาดัน
    มีแค่ class เดียวในช่วงนั้นพอดี (เช่น threshold สูงมากจน pred เป็น 0
    ทั้งหมด) — ถ้าไม่ล็อก labels ไว้ .ravel() จะ unpack ผิดตำแหน่งเงียบๆ

    Compute FN, FP, and total cost at a given threshold.

    Always uses labels=[0, 1] explicitly (same as src/evaluate.py) to
    prevent confusion_matrix from returning the wrong shape if the given
    pred or y_true happens to contain only one class in that range (e.g.
    threshold so high that pred is all 0) — without pinning labels,
    .ravel() would silently unpack into the wrong positions.
    """
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_cost = r * fn + 1 * fp
    return dict(threshold=float(threshold), r=float(r),
                fn=int(fn), fp=int(fp), tp=int(tp), tn=int(tn),
                total_cost=float(total_cost))


def sweep_thresholds(scores: np.ndarray, y_true: np.ndarray,
                      r: float, n_points: int = 500) -> List[Dict]:
    """ไล่ threshold ทุกจุดที่มีความหมาย — ใช้ unique score values เป็น
    จุดตัด แทน linspace เพื่อไม่พลาด threshold ที่ทำให้ FN/FP เปลี่ยนค่าจริง

    ข้อจำกัดที่รู้อยู่แล้ว (ตรงกับ oracle_threshold_diagnostic() เดิมใน
    evaluate.py ที่ตัด endpoint สุดท้ายทิ้งด้วย [:-1] เหมือนกัน): เพราะ
    candidate มากสุดคือ max(scores) ซึ่งยัง flag ภาพที่ score เท่ากับ
    max อยู่ดี การ sweep นี้จึง**ไม่มีทาง**ได้ threshold ที่ "ไม่ flag
    เลย" (pred=0 ทุกภาพ, เทียบเท่า threshold > max(scores)) เข้ามาเป็น
    ตัวเลือก — ในทางปฏิบัติแทบไม่กระทบเพราะ r ที่ใช้จริง (1-100) ทำให้
    optimal threshold มักจะต่ำ (flag มากกว่า) อยู่แล้ว แต่ควรระบุเป็น
    known limitation ในเล่ม ไม่ใช่เงียบไว้

    Sweep every meaningful threshold — uses unique score values as cut
    points instead of linspace, so no threshold that actually changes
    FN/FP gets skipped.

    Known limitation (shared with the existing oracle_threshold_
    diagnostic() in evaluate.py, which also drops the last endpoint via
    [:-1]): since the largest candidate is max(scores), which still
    flags the image(s) whose score equals that max, this sweep can
    never produce the "flag nothing" threshold (pred=0 for every image,
    equivalent to threshold > max(scores)) as an option. In practice
    this rarely matters since the r values actually used (1-100) tend
    to push the optimal threshold lower (more flagging) anyway — but it
    should be stated as a known limitation in the thesis, not left
    unmentioned.
    """
    candidates = np.unique(scores)
    if len(candidates) > n_points:
        idx = np.linspace(0, len(candidates) - 1, n_points).astype(int)
        candidates = candidates[idx]
    return [cost_at_threshold(scores, y_true, t, r) for t in candidates]


def select_cost_optimal_threshold(val_scores: np.ndarray, val_y: np.ndarray,
                                   r: float) -> Dict:
    """หา threshold cost ต่ำสุด บน Validation set เท่านั้น (ห้ามใช้ Test
    ตรงนี้ — เลือก threshold ต้องทำบน val เหมือนหลักการเดียวกับ
    select_percentile_threshold เดิม)

    หมายเหตุเรื่อง tie-breaking: candidates เรียงจากน้อยไปมาก (จาก
    np.unique) และ min(..., key=...) ของ Python คืนตัวแรกที่เจอเมื่อ
    total_cost เท่ากันหลายจุด — เท่ากับว่าถ้ามีหลาย threshold ให้ cost
    เท่ากัน จะเลือก**ตัวต่ำสุด** (flag เยอะสุด) เป็น default โดยปริยาย
    ซึ่งเข้าทาง QC (เอนเอียงไป escape น้อยกว่า) แต่เป็นพฤติกรรมที่มาจาก
    ลำดับของ candidates ไม่ใช่การตัดสินใจ explicit — ควรระบุไว้ในเล่ม
    ถ้าอ้างอิงพฤติกรรมนี้

    Find the lowest-cost threshold, Validation set only (never use Test
    here — threshold selection must happen on val, same principle as
    the existing select_percentile_threshold).

    Note on tie-breaking: candidates are ascending (from np.unique), and
    Python's min(..., key=...) returns the first one encountered when
    several thresholds tie on total_cost — meaning that among tied
    options, the **lowest** threshold (flags the most) is picked by
    default. This happens to favor QC (fewer escapes) but it's a side
    effect of candidate ordering, not an explicit decision — worth
    stating in the thesis if this behavior is relied upon.
    """
    results = sweep_thresholds(val_scores, val_y, r)
    return min(results, key=lambda d: d['total_cost'])


def cost_sweep_report(val_scores: np.ndarray, val_y: np.ndarray,
                       test_scores: np.ndarray, test_y: np.ndarray,
                       r_values: List[float]) -> List[Dict]:
    """สำหรับแต่ละค่า r: หา threshold ที่ดีที่สุดบน val แล้ววัดผลเต็มชุด
    ทั้ง val และ test ด้วย threshold เดียวกัน — คืนเป็น list พร้อม metric
    ครบทุกตัวของทั้งสอง split ในแถวเดียวกัน (prefix val_/test_) ให้เขียน
    ลง CSV เดียวได้ตรงๆ ไม่ต้องแยกไฟล์

    ทุก field ของ val_*/test_* มาจาก compute_metrics() ตัวเดียวกับที่
    evaluate.py ใช้ทั้ง repo (reuse ไม่เขียน metric ซ้ำ) ใช้ naming
    convention เดียวกันทั้งโปรเจกต์: tt/tf/ft/ff = (actual, predicted)
    ตาม (anomaly,anomaly)/(anomaly,normal)/(normal,anomaly)/(normal,normal)
    — เหมือนกับที่ใช้ใน naive_baselines และ results ของ final_results.json
    ทุกประการ ไม่ใช่ fn/fp แบบที่เคยใช้ในเวอร์ชันก่อนหน้า (breaking change
    จาก schema เดิม — ถ้ามี cost_aware_sweep.csv เก่าอยู่ ต้องรันใหม่)

    val_cost คือค่าที่ใช้ตัดสินใจเลือก threshold จริง (r·FN + FP ณ
    threshold ที่เลือก) แยกต่างหากจาก metric อื่นเพราะเป็นค่าเฉพาะของ
    cost-aware framework ไม่ได้มาจาก compute_metrics()

    auc/ap ของ val_*/test_* จะเหมือนกันทุกแถว (ไม่ขึ้นกับ threshold ที่
    ต่างกัน เพราะเป็น ranking-based metric) แต่ยังคงไว้ในทุกแถวเพื่อให้
    CSV มีทุกค่าครบในตัวเอง ไม่ต้องเปิดไฟล์อื่นมาเทียบ

    For each r: find the best threshold on val, then measure the full
    metric set on BOTH val and test using that same threshold — returns
    a list with every metric from both splits in the same row
    (val_/test_ prefixed), ready to write straight to a single CSV, no
    need for separate files.

    Every val_*/test_* field comes from the same compute_metrics() used
    throughout evaluate.py (reused, not duplicated), using the same
    naming convention as the rest of the project: tt/tf/ft/ff =
    (actual, predicted) — identical to what naive_baselines and
    final_results.json's results use. NOT fn/fp as in the previous
    version (breaking change from the old schema — re-run if you have
    an old cost_aware_sweep.csv).

    val_cost is the value actually used to pick the threshold (r·FN + FP
    at the chosen threshold), kept separate from the other metrics since
    it's specific to the cost-aware framework, not from compute_metrics().

    val_*/test_* auc/ap will be identical across every row (they don't
    depend on the threshold, being ranking-based metrics), but are kept
    in every row so the CSV is self-contained without needing to open
    another file to cross-reference.
    """
    # หมายเหตุ (แก้เพิ่มระหว่าง sync baseline repos, ไม่ได้อยู่ใน cost_aware.py
    # ต้นฉบับของ repo หลัก): evaluate.py เวอร์ชันของ repo หลักตอนนี้คืน
    # tt/tf/ft/ff ตรงจาก compute_metrics() แล้ว แต่ evaluate.py ของ
    # PCB-Anomaly-Baselines-PatchCore (ต้นแบบที่ 4 baseline นี้ต้อง reuse
    # ตาม BASELINE_IMPLEMENTATION_GUIDE.md) ยังเป็นเวอร์ชันเก่าที่คืน
    # 'cm' (confusion matrix ndarray) แทน — สอง repo ไม่ sync กันเอง
    # (นอกเหนือ scope ของงาน sync 4 baseline นี้) จึงแปลง cm -> tt/tf/ft/ff
    # ที่นี่แทนการแก้ evaluate.py/io_utils.py/visual.py ที่ guide สั่งห้ามแก้
    #
    # Note (added while syncing the baselines, not present in the main
    # repo's original cost_aware.py): the main repo's evaluate.py now
    # returns tt/tf/ft/ff directly from compute_metrics(), but
    # PCB-Anomaly-Baselines-PatchCore's evaluate.py (the template these 4
    # baselines must reuse per BASELINE_IMPLEMENTATION_GUIDE.md) is an
    # older version that returns 'cm' (confusion matrix ndarray) instead —
    # the two repos are out of sync with each other (outside this sync
    # task's scope). We convert cm -> tt/tf/ft/ff here instead of touching
    # evaluate.py/io_utils.py/visual.py, which the guide says not to edit.
    def _tt_tf_ft_ff(m: Dict) -> Dict:
        tn, fp, fn, tp = m['cm'].ravel()
        return dict(tt=int(tp), tf=int(fn), ft=int(fp), ff=int(tn))

    metric_fields = ['auc', 'ap', 'acc', 'precision', 'recall', 'f1',
                     'tt', 'tf', 'ft', 'ff',
                     'auto_clear_rate', 'escape_rate', 'residual_fcr']

    report = []
    for r in r_values:
        best = select_cost_optimal_threshold(val_scores, val_y, r)
        t_star = best['threshold']

        val_metrics  = compute_metrics(val_scores,  val_y,  t_star)
        test_metrics = compute_metrics(test_scores, test_y, t_star)
        val_metrics  = {**val_metrics,  **_tt_tf_ft_ff(val_metrics)}
        test_metrics = {**test_metrics, **_tt_tf_ft_ff(test_metrics)}

        row = dict(r=r, threshold=t_star, val_cost=best['total_cost'])
        for field in metric_fields:
            row[f'val_{field}']  = val_metrics[field]
            row[f'test_{field}'] = test_metrics[field]
        report.append(row)

    return report


def find_elbow_r(report: List[Dict]) -> Dict:
    """หาแถวใน report (ผลจาก cost_sweep_report) ที่เป็นจุด "elbow" ของ
    val FN-FP trade-off curve — ใช้เป็น**ตัวอย่างประกอบ**ในเล่มเวลาต้อง
    โชว์เลขตัวแทนสักตัว ไม่ใช่ deployment number จริง (ต้องเขียนกำกับไว้
    เสมอว่าเป็นตัวอย่าง ไม่ใช่ threshold สุดท้าย)

    วิธี: max-distance-from-chord (เส้นตรงที่ลากจากจุดแรกไปจุดสุดท้าย
    ของ curve) — จุดที่ห่างจากเส้นตรงนี้มากที่สุดคือจุดที่โค้งงอมากที่สุด
    เป็นวิธีมาตรฐานสำหรับหา knee/elbow point (เทียบเท่าหลักการเดียวกับ
    Kneedle algorithm แบบย่อ) ใช้ **val_tf (escape) / val_ft (false
    alarm) เท่านั้น** (ไม่ใช้
    test) เพราะการตัดสินใจ (เลือก r ตัวแทน) ต้องทำบน val — เหมือนหลักการ
    เดียวกับ select_cost_optimal_threshold ที่ห้ามใช้ Test ตัดสินใจ

    Normalize ทั้งสองแกน (fn, fp) เข้าช่วง [0,1] ก่อนวัดระยะ เพราะ fn/fp
    มักคนละ scale กัน (เช่น fn อยู่หลักสิบ, fp อยู่หลักร้อย) ถ้าไม่
    normalize ก่อน แกนที่ scale ใหญ่กว่าจะครอบงำการวัดระยะทั้งหมด

    ต้องมีอย่างน้อย 3 ค่า r ถึงจะหา elbow ได้อย่างมีความหมาย (น้อยกว่านั้น
    ไม่มี "โค้ง" ให้วัด)

    Find the row in report (from cost_sweep_report) that sits at the
    "elbow" of the val FN-FP trade-off curve — meant as an **illustrative
    example** in the thesis when a single representative number is
    needed, NOT the real deployment number (must always be labeled as an
    example, not the final threshold).

    Method: max-distance-from-chord (a straight line from the curve's
    first point to its last point) — the point farthest from this line
    is where the curve bends the most. This is a standard way to find a
    knee/elbow point (equivalent in spirit to a simplified Kneedle
    algorithm). Uses **val_tf (escape/FN) / val_ft (false alarm/FP) only**
    (never test), since this
    decision (picking a representative r) must be made on val — same
    principle as select_cost_optimal_threshold, which never uses Test to
    decide.

    Both axes (fn, fp) are normalized to [0,1] before measuring distance,
    since fn and fp are often on different scales (e.g. fn in the tens,
    fp in the hundreds) — without normalizing first, the larger-scale
    axis would dominate the distance measurement entirely.

    Requires at least 3 r values for the elbow to be meaningful (fewer
    than that, there's no "bend" to measure).
    """
    if len(report) < 3:
        raise ValueError('ต้องมีอย่างน้อย 3 ค่า r ถึงจะหา elbow ได้อย่างมีความหมาย '
                         '/ need at least 3 r values for a meaningful elbow')

    fn = np.array([row['val_tf'] for row in report], dtype=float)
    fp = np.array([row['val_ft'] for row in report], dtype=float)

    def _normalize(a: np.ndarray) -> np.ndarray:
        span = a.max() - a.min()
        return (a - a.min()) / span if span > 0 else np.zeros_like(a)

    fn_n, fp_n = _normalize(fn), _normalize(fp)

    p_first = np.array([fn_n[0], fp_n[0]])
    p_last  = np.array([fn_n[-1], fp_n[-1]])
    chord = p_last - p_first
    chord_len = np.linalg.norm(chord)

    if chord_len == 0:
        # ทุกค่า r ให้ fn/fp เหมือนกันหมด (เช่น score แยกคลาสสมบูรณ์แบบ
        # จน threshold ไม่ขยับเลยไม่ว่า r จะเป็นเท่าไหร่) — ไม่มี elbow
        # ให้หาจริงๆ คืนค่ากลางของ list ไปเป็น fallback ที่สมเหตุสมผล
        #
        # Every r gives identical fn/fp (e.g. perfectly separable scores,
        # so the threshold never moves regardless of r) — there's no real
        # elbow to find. Fall back to the middle of the list as a
        # reasonable default.
        return report[len(report) // 2]

    chord_unit = chord / chord_len
    distances = []
    for i in range(len(report)):
        p = np.array([fn_n[i], fp_n[i]]) - p_first
        proj = np.dot(p, chord_unit) * chord_unit
        perp = p - proj
        distances.append(float(np.linalg.norm(perp)))

    elbow_idx = int(np.argmax(distances))
    result = dict(report[elbow_idx])
    result['_elbow_distance'] = distances[elbow_idx]  # ไว้ debug/plot เพิ่มเติมถ้าต้องการ
    return result


def select_recall_constrained_threshold(val_scores: np.ndarray, val_y: np.ndarray,
                                        max_escape_rate: float) -> Dict:
    """หา threshold บน Validation set ที่ทำให้ escape_rate ไม่เกิน
    max_escape_rate ที่กำหนด โดยในบรรดา threshold ที่ผ่านเงื่อนไข เลือก
    ตัวที่ auto_clear_rate สูงสุด (เข้มงวดน้อยที่สุดเท่าที่จำเป็น — ไม่
    flag เกินความจำเป็นเพื่อลด false alarm ให้น้อยที่สุดเท่าที่ยัง
    การันตี escape_rate ตามเป้าได้)

    วิธีนี้ผูกกับเป้าหมายทางธุรกิจ (เช่น spec "escape ต้องไม่เกิน 5%")
    โดยตรง ไม่ต้องรู้ cost ratio (r) เป็นตัวเลขเงินเลย ต่างจาก
    cost_sweep_report ที่ต้องมี r มาก่อน

    ⚠️ ข้อจำกัดเดียวกับ sweep_thresholds(): candidate ที่ใช้ค้นหาไม่รวม
    threshold ที่ "ไม่ flag เลย" (pred=0 ทุกภาพ) เพราะงั้นถ้า
    max_escape_rate ที่ตั้งไว้หลวมมาก (ใกล้ 1.0) ผลที่ได้อาจไม่ใช่
    threshold ที่หลวมที่สุดจริงๆ ที่เป็นไปได้ในทางทฤษฎี — ในทางปฏิบัติแทบ
    ไม่กระทบเพราะ QC มักตั้ง max_escape_rate ต่ำ (เข้มงวด) อยู่แล้ว

    Raises:
      ValueError: ถ้าไม่มี threshold ไหนใน val เลยที่ทำให้ escape_rate
        <= max_escape_rate ได้ (พบได้ยากมาก เพราะ threshold ต่ำสุดที่มี
        ในข้อมูลมักให้ escape_rate ใกล้ 0 อยู่แล้ว แต่เป็นไปได้ถ้า
        max_escape_rate ตั้งไว้เข้มกว่าที่ข้อมูลรองรับได้จริง)

    Find the Validation-set threshold that keeps escape_rate at or below
    the given max_escape_rate. Among thresholds that satisfy this, picks
    the one with the highest auto_clear_rate (as lenient as necessary —
    don't over-flag beyond what's needed to still guarantee the target
    escape_rate).

    This ties the decision directly to a business goal (e.g. "escape
    must stay under 5%") without ever needing a cost ratio (r) in
    monetary terms — unlike cost_sweep_report, which requires r upfront.

    ⚠️ Same limitation as sweep_thresholds(): the search space excludes
    the "flag nothing" threshold (pred=0 for every image), so if
    max_escape_rate is set very loose (close to 1.0), the result may not
    be the theoretically most lenient threshold possible. In practice
    this rarely matters since QC settings usually set a low (strict)
    max_escape_rate anyway.

    Raises:
      ValueError: if no threshold in val achieves escape_rate <=
        max_escape_rate at all (rare — the lowest available threshold
        usually gives escape_rate near 0 already — but possible if
        max_escape_rate is stricter than the data can actually support).
    """
    candidates = np.unique(val_scores)
    best = None
    for t in candidates:
        m = compute_metrics(val_scores, val_y, float(t))
        if np.isnan(m['escape_rate']) or m['escape_rate'] > max_escape_rate:
            continue
        if best is None or m['auto_clear_rate'] > best['auto_clear_rate']:
            best = dict(
                threshold=float(t),
                escape_rate=m['escape_rate'],
                auto_clear_rate=m['auto_clear_rate'],
                precision=m['precision'], recall=m['recall'], f1=m['f1'],
            )
    if best is None:
        raise ValueError(
            f'ไม่มี threshold ไหนใน val ที่ทำให้ escape_rate <= {max_escape_rate} ได้เลย '
            f'/ no threshold in val achieves escape_rate <= {max_escape_rate}')
    return best