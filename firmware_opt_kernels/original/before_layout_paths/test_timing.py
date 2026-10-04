"""Checks the GUI's timing model against the firmware it is modelling.

The cycle counts themselves are estimates and cannot be unit-tested, but
everything they are built on is a hard number: the op counts have to be the
ones CNN_Inference_Edge-AI_with_CLA actually performs, and the MFCC working
set has to be the one its .map file reports. Those are what this file pins
down, so a change to the model that quietly stops describing the firmware
fails here instead of in a paper.

    python tools/test_timing.py

Reads CNN_Inference_Edge-AI_with_CLA/definitions.h when the submodule is
checked out, and falls back to the values recorded below when it is not.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edge_ai_train_model_gui_quantization_according_succes_rate as gui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRMWARE = os.path.join(ROOT, "CNN_Inference_Edge-AI_with_CLA")

# What definitions.h says at the commit this test was written against, used
# when the submodule is not checked out.
FALLBACK = {"IMAGE_H": 65, "IMAGE_W": 12, "NUM_FILTERS": 9, "NUM_CLASSES": 2,
            "DENSE_LAYER_INPUT": 1008, "DENSE_LAYER_WEIGHTS": 2016,
            "POOL1_OUT_H": 31, "POOL1_OUT_W": 10,
            "POOL2_OUT_H": 14, "POOL2_OUT_W": 8}
# mfcc_extract.c's static buffers, as the linker counted them:
# Debug/dense_layer_cla.map, mfcc_extract.obj (.ebss): log_mel 0x1450 +
# z 0x404 + tw4 0x400 + tw 0x400 + rev4 0x40 + 1 = 8389 words = 16.4 KB.
# (The indexed version kept a float copy of the slice and came to 85.4 KB.)
MAP_MFCC_KB = 16.4

fails = []


def check(name, got, want, tol=0):
    ok = abs(got - want) <= tol
    if not ok:
        fails.append(f"{name}: got {got}, expected {want} (tol {tol})")
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:34s} {got}"
          + ("" if ok else f"   != {want}"))


def firmware_defines():
    """The simple `#define NAME <int>` lines of definitions.h. The derived
    ones (CONV1_OUT_H and friends) are expressions, so they are recomputed
    here from the literals rather than parsed."""
    path = os.path.join(FIRMWARE, "definitions.h")
    if not os.path.isfile(path):
        print(f"definitions.h not found ({path}) -- using recorded values")
        return dict(FALLBACK)
    text = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"^#define\s+(\w+)\s+(\d+)\s*(?://.*)?$",
                         text, re.M):
        out[m.group(1)] = int(m.group(2))
    ih, iw, f = out["IMAGE_H"], out["IMAGE_W"], out["NUM_FILTERS"]
    c1h, c1w = ih - 2, iw - 2
    p1h, p1w = c1h // 2, c1w              # 2x1 pooling: height only
    c2h, c2w = p1h - 2, p1w - 2
    p2h, p2w = c2h // 2, c2w
    out.update({"POOL1_OUT_H": p1h, "POOL1_OUT_W": p1w,
                "POOL2_OUT_H": p2h, "POOL2_OUT_W": p2w,
                "DENSE_LAYER_INPUT": f * p2h * p2w,
                "DENSE_LAYER_WEIGHTS": f * p2h * p2w * out["NUM_CLASSES"]})
    return out


d = firmware_defines()
ih, iw = d["IMAGE_H"], d["IMAGE_W"]
ch, ncls = d["NUM_FILTERS"], d["NUM_CLASSES"]

print("op counts vs the firmware's own dimensions")
counts = gui.stage_op_counts(ch, 2, ih, iw, ncls, pool_w=False)
# conv1 computes every row a 2x1 pool consumes: 2 * POOL1_OUT_H rows of
# CONV1_OUT_W columns, one 3x3 window per output channel
check("conv1 MACs", counts[0]["macs"],
      ch * 2 * d["POOL1_OUT_H"] * d["POOL1_OUT_W"] * 1 * 9)
check("conv2 MACs", counts[1]["macs"],
      ch * 2 * d["POOL2_OUT_H"] * d["POOL2_OUT_W"] * ch * 9)
check("dense MACs == DENSE_LAYER_WEIGHTS", counts[2]["macs"],
      d["DENSE_LAYER_WEIGHTS"])

print("\nfeature plan")
plan = gui.feature_plan(ih, iw, {"sr": 16000, "n_fft": 512, "hop": 256,
                                 "n_mels": 40})
# FEAT_SLICE_LEN in feature_tables.h: hop*(frames-1) + n_fft, center=False
check("derived slice_len == FEAT_SLICE_LEN", plan["slice_len"], 16896)
check("window length (ms)", round(plan["window_ms"], 1), 1056.0, tol=0.1)
mf = gui.mfcc_cost(plan)
check("MFCC working set (KB) vs the .map", round(mf["ram_bytes"] / 1024, 1),
      MAP_MFCC_KB, tol=MAP_MFCC_KB * 0.10)
# One pass of mfcc_extract() on this 65-frame configuration was measured at
# 32.25 ms on a 200 MHz F28379D (-O0, code running from RAM, second window
# onwards; from flash, where it runs now, about 35 ms -- to be re-measured). CYC_FLOP is solved from it, so the model has to reproduce it.
# (With the anchor's own plan: it carries the header's real FEAT_MEL_NNZ,
# 490, where the plan above falls back to 2 * bins = 514.)
mf_anchor = gui.mfcc_cost(gui.feature_plan(gui.MFCC_ANCHOR["input_h"],
                                           gui.MFCC_ANCHOR["input_w"],
                                           gui.MFCC_ANCHOR))
check("MFCC reproduces the measurement (ms)",
      round(1000.0 * mf_anchor["cycles"] / 200e6, 2), gui.MFCC_ANCHOR["ms"],
      tol=0.01)
check("cycles per float op on the C28x", round(gui.CYC_FLOP, 2), 7.05,
      tol=0.3)

print("\nthe fitted measurements, reproduced")
# The conv model is two prices per core, a MAC and an accumulation window,
# SOLVED from the measurements marked "fit". It therefore has to give those
# back; this only checks the algebra and the counts under it.
K = gui.KERNEL_SETUP
k_plan = gui.feature_plan(K["input_h"], K["input_w"],
                          {"sr": 16000, "n_fft": 512, "hop": 256,
                           "n_mels": 40})
ALL = {"cla": {"stages": [True, True], "dense": True},
       "cpu": {"stages": [False, False], "dense": False}}
EXPECTED_OPS = {4: (55472, 3378), 9: (215532, 7598), 12: (359952, 10130)}


def model_ms(channels, core):
    r = gui.pipeline_timing(k_plan, channels, K["layers"], K["input_h"],
                            K["input_w"], K["classes"], K["pool_w"],
                            ALL[core], K["mhz"], K["mhz"])
    return r


for m in gui.KERNEL_MEASUREMENTS:
    c = gui.stage_op_counts(m["channels"], K["layers"], K["input_h"],
                            K["input_w"], K["classes"], K["pool_w"])
    macs, wins = EXPECTED_OPS[m["channels"]]
    check(f"{m['channels']:2d} channels: MACs", sum(x["macs"] for x in c), macs)
    check(f"{m['channels']:2d} channels: accumulations",
          sum(x["windows"] for x in c), wins)
    if m["fit"]:
        for core in ("cpu", "cla"):
            check(f"{m['channels']:2d} channels, {core}: model == measured (ms)",
                  round(model_ms(m["channels"], core)["infer_ms"], 4),
                  m[core + "_ms"], tol=0.0005)

print("\nwhat the fit says")
# Not assertions about the silicon -- a record of what these measurements
# gave, so that changing a measurement or a count shows up as a diff here.
check("C28x cycles per MAC", round(gui.CPU_LEVELS["taps"], 2), 6.46, tol=0.1)
check("C28x cycles per accumulation", round(gui.CPU_LEVELS["windows"], 1),
      57.2, tol=1.5)
check("CLA cycles per MAC", round(gui.CLA_LEVELS["taps"], 2), 8.10, tol=0.1)
check("CLA cycles per accumulation", round(gui.CLA_LEVELS["windows"], 1),
      48.0, tol=1.5)

print("\nthe held-out measurement")
# The fitted points prove nothing; this one the model had no chance to fit,
# and it is the only number in this file that can genuinely fail. Widening
# the tolerance to make a change pass would be deleting the evidence, not
# fixing the model.
for _v in gui.VALIDATIONS:
    check(f"held-out {_v['channels']}-channel, {_v['core']} (ms)",
          round(_v["predicted_ms"], 3), _v["measured_ms"],
          tol=_v["measured_ms"] * 0.02)
# The band across the held-out points IS the accuracy claim. It is not
# allowed to be quietly widened.
_lo, _hi = gui.VALIDATION_BAND
check("held-out error band, low end (%)", round(_lo, 2), 0.0, tol=1.0)
check("held-out error band, high end (%)", round(_hi, 2), 0.0, tol=1.0)

# The CLA is still the dearer place to run the network, by much less than
# it was (1.84x with the indexed kernels).
_a = model_ms(9, "cla")
check("CLA / C28x, effective cycles per MAC at 9 channels",
      round(_a["mac_cost"]["cla"] / _a["mac_cost"]["cpu"], 3), 1.162,
      tol=0.02)

print("\npipelining rules")
cla_all = {"stages": [True, True], "dense": True}
t = gui.pipeline_timing(plan, ch, 2, ih, iw, ncls, False, cla_all, 200, 200)
check("all on CLA: period == max(MFCC, inference)",
      round(t["period_ms"], 6), round(max(t["fe_ms"], t["infer_ms"]), 6),
      tol=1e-6)
check("all on CLA: latency == MFCC + inference",
      round(t["latency_ms"], 6), round(t["fe_ms"] + t["infer_ms"], 6),
      tol=1e-6)

split = {"stages": [True, False], "dense": True}
ts = gui.pipeline_timing(plan, ch, 2, ih, iw, ncls, False, split, 200, 200)
# only the trailing run of CLA stages can overlap the next window
check("split chain: overlap is the CLA tail only",
      round(ts["overlap_ms"], 6), round(ts["stages"][2]["ms"], 6), tol=1e-6)
if not ts["period_ms"] > t["period_ms"]:
    fails.append("a CPU stage mid-chain did not cost throughput")
print(f"  {'ok  ' if ts['period_ms'] > t['period_ms'] else 'FAIL'}  "
      f"{'mid-chain CPU stage costs throughput':34s} "
      f"{ts['period_ms']:.2f} ms > {t['period_ms']:.2f} ms")

cpu_all = {"stages": [False, False], "dense": False}
tc = gui.pipeline_timing(plan, ch, 2, ih, iw, ncls, False, cpu_all, 200, 200)
check("all on CPU: nothing overlaps", round(tc["overlap_ms"], 6), 0.0, tol=1e-9)
check("all on CPU: period == latency",
      round(tc["period_ms"], 6), round(tc["latency_ms"], 6), tol=1e-6)

# halving the clock doubles every time
th = gui.pipeline_timing(plan, ch, 2, ih, iw, ncls, False, cla_all, 100, 100)
check("half the clock, twice the latency", round(th["latency_ms"], 4),
      round(t["latency_ms"] * 2, 4), tol=1e-3)

# A measured feature-extraction time replaces the estimate and touches
# NOTHING else. The front end is the user's own component, so the ratio
# between their time and this project's MFCC says nothing about their conv
# kernels -- it is reported, not applied.
tm = gui.pipeline_timing(plan, ch, 2, ih, iw, ncls, False, cpu_all, 200, 200,
                         measured_fe_ms=t["fe_ms"] * 2.0)
check("measured feature time is used as given", round(tm["fe_ms"], 4),
      round(t["fe_ms"] * 2.0, 4), tol=1e-3)
check("its distance from the estimate is reported",
      round(tm["calib"], 4), 2.0, tol=1e-3)
check("but the stages are left alone", round(tm["stages"][0]["ms"], 6),
      round(tc["stages"][0]["ms"], 6), tol=1e-6)
check("and so is the per-MAC cost", round(tm["mac_cost"]["cpu"], 4),
      round(tc["mac_cost"]["cpu"], 4), tol=1e-6)

# a measured inference time re-anchors the per-MAC cost for every stage
ti = gui.pipeline_timing(plan, ch, 2, ih, iw, ncls, False, cla_all, 200, 200,
                         measured_infer_ms=t["infer_ms"] * 3.0,
                         measured_fe_ms=t["fe_ms"] * 2.0)
check("measured inference is used as given", round(ti["infer_ms"], 4),
      round(t["infer_ms"] * 3.0, 4), tol=1e-3)
# the per-MAC figure the GUI quotes is effective -- the whole stage over its
# MACs -- so tripling the measurement has to triple it
check("and re-anchors cycles per MAC", round(ti["mac_cost"]["cla"], 1),
      round(t["mac_cost"]["cla"] * 3.0, 1), tol=0.1)

print("\nonly tail plans can be picked")
# The GUI lets the CLA stages be the END of the chain and nothing else.
for bits, want in (("000", True), ("001", True), ("011", True), ("111", True),
                   ("100", False), ("010", False), ("110", False),
                   ("101", False)):
    got = gui.is_tail_plan([b == "1" for b in bits])
    if got != want:
        fails.append(f"is_tail_plan({bits}) = {got}")
    print(f"  {'ok  ' if got == want else 'FAIL'}  S1S2S3={bits} "
          f"{'allowed' if got else 'refused'}")
# every single click, from every allowed plan, has to land on an allowed one
bad_moves = []
for start in ("000", "001", "011", "111"):
    for idx in range(3):
        for to_cla in (True, False):
            new, _ = gui.assign_keeping_tail([b == "1" for b in start],
                                             idx, to_cla)
            if not gui.is_tail_plan(new) or new[idx] != to_cla:
                bad_moves.append((start, idx, to_cla, new))
if bad_moves:
    fails.append(f"assign_keeping_tail left a non-tail plan: {bad_moves}")
print(f"  {'ok  ' if not bad_moves else 'FAIL'}  every click from an allowed "
      f"plan gives an allowed plan")
check("S1 to the CLA carries S2 and S3",
      int(gui.assign_keeping_tail([False, False, False], 0, True)[0]
          == [True, True, True]), 1)
check("S3 to the C28x carries S1 and S2",
      int(gui.assign_keeping_tail([True, True, True], 2, False)[0]
          == [False, False, False]), 1)
check("S2 to the C28x from all-CLA leaves S3 on the CLA",
      int(gui.assign_keeping_tail([True, True, True], 1, False)[0]
          == [False, False, True]), 1)

print("\ndevice catalogue")
keys = [x["key"] for x in gui.C2000_DEVICES]
if len(keys) != len(set(keys)):
    fails.append("duplicate device keys")
for dev in gui.C2000_DEVICES:
    b = gui.device_budgets(dev)
    bad = []
    if b["flash_kb"] <= 0 or b["cpu_kb"] <= 0 or b["cla_kb"] <= 0:
        bad.append("non-positive budget")
    if b["flash_kb"] > dev["flash_kb"]:
        bad.append("flash budget exceeds the device's flash")
    if dev["pool_kb"] and dev["cla_data_kb"] + dev["cla_prog_kb"] > dev["pool_kb"]:
        bad.append("CLA data + program exceed the shared pool")
    if dev["cla_mhz"] > dev["cpu_mhz"]:
        bad.append("CLA faster than the C28x")
    if bad:
        fails.append(f"{dev['key']}: {', '.join(bad)}")
    print(f"  {'ok  ' if not bad else 'FAIL'}  {dev['key']:12s} "
          f"{b['flash_kb']:6.0f} KB flash  {b['cpu_kb']:5.0f} KB RAM  "
          f"{b['cla_kb']:4.0f} KB CLA  @{dev['cpu_mhz']:3d} MHz"
          + ("" if not bad else "   " + "; ".join(bad)))

print()
if fails:
    print(f"{len(fails)} failure(s):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all timing-model checks passed")
