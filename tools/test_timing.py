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
FALLBACK = {"IMAGE_H": 65, "IMAGE_W": 12, "NUM_FILTERS": 6, "NUM_CLASSES": 3,
            "DENSE_LAYER_INPUT": 672, "DENSE_LAYER_WEIGHTS": 2016,
            "POOL1_OUT_H": 31, "POOL1_OUT_W": 10,
            "POOL2_OUT_H": 14, "POOL2_OUT_W": 8}
# mfcc_extract.c's static buffers, as the linker counted them:
# Debug/dense_layer_cla.map, RAMGS_TOTAL_CNN used = 0xa6c0 words = 85.4 KB.
MAP_MFCC_KB = 85.4

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
# Second anchor: one pass of mfcc_extract() on this 65-frame configuration
# was measured at 202 ms on a 200 MHz F28379D (Debug build). CYC_FLOP is
# set from it, so the model has to reproduce it.
check("MFCC reproduces the measurement (ms)",
      round(1000.0 * mf["cycles"] / 200e6, 1), gui.MFCC_ANCHOR["ms"], tol=1.0)

print("\nthe anchors, reproduced")
# The cycle model is SOLVED from the anchors, so it has to give the measured
# numbers back. Everything in the conv model is hand-counted off
# dense_layer.cla except one fitted correction, CLA_TUNE.
#
# The inference anchor is the firmware as it stands: 6 channels, which
# definitions.h states and the .map confirms three ways over (conv1_weights
# 54 words = 9*1*F, conv2_weights 324 = 9*F*F, workspace 2536 =
# F*(31*10 + 14*8)). Measured 2026-09-26 at 73 ms.
a_cfg = gui.CLA_ANCHOR
a_plan = gui.feature_plan(a_cfg["input_h"], a_cfg["input_w"],
                          {"sr": 16000, "n_fft": 512, "hop": 256,
                           "n_mels": 40})
a = gui.pipeline_timing(a_plan, a_cfg["channels"], a_cfg["layers"],
                        a_cfg["input_h"], a_cfg["input_w"], a_cfg["classes"],
                        a_cfg["pool_w"],
                        {"stages": [True, True], "dense": True},
                        a_cfg["mhz"], a_cfg["mhz"])
counts_a = gui.stage_op_counts(a_cfg["channels"], a_cfg["layers"],
                               a_cfg["input_h"], a_cfg["input_w"],
                               a_cfg["classes"], a_cfg["pool_w"])
check("anchor MACs (6 channels)", sum(c["macs"] for c in counts_a), 108072)
check("anchor accumulations", sum(c["windows"] for c in counts_a), 5067)
check("CLA inference reproduces the measurement (ms)",
      round(a["infer_ms"], 2), a_cfg["ms"], tol=0.05)
check("effective cycles per MAC on the CLA",
      round(a["mac_cost"]["cla"]), 135, tol=2)
check("cycles per float op on the C28x", round(gui.CYC_FLOP, 1), 14.6,
      tol=0.2)
# The correction on the hand count, and the strongest evidence the model
# describes the real code: the CLA measurement asks for 4.48 and the C28x
# measurement -- different core, different kernel, taken independently --
# asks for 4.88. One unoptimised build, one factor. If these ever part
# company, the primitive table has stopped matching one of them.
check("the fitted correction", round(gui.CLA_TUNE, 2), 4.48, tol=0.3)
_P = gui.CPU_PRIMITIVES
_bf = (10 * _P["fmul"] + 6 * _P["fadd"] + 4 * _P["load"] + 4 * _P["store"]
       + 6 * _P["iadd"] + _P["loop"])          # one fft512 butterfly, by hand
check("and the C28x measurement agrees with it",
      round(16 * gui.CYC_FLOP / _bf, 2), round(gui.CLA_TUNE, 2), tol=0.6)
# The CLA is the DEARER place to run this kernel -- index-heavy int16 code
# on a narrow integer unit with no hardware loop. That falls out of the
# primitive table, not out of a fitted difference: both cores carry the
# same tune factor.
# The held-out point. The anchor is fitted, so reproducing it proves
# nothing; this one the model had no chance to fit, and it is the only
# number in this file that can genuinely fail. Widening the tolerance to
# make a change pass would be deleting the evidence, not fixing the model.
check("held-out 4-channel measurement",
      round(gui.VALIDATION["predicted_ms"], 1),
      gui.CLA_VALIDATION["ms"], tol=gui.CLA_VALIDATION["ms"] * 0.20)
check("...and its error stays where it was measured",
      round(gui.VALIDATION["error_pct"]), 13, tol=4)

check("CLA costs more per MAC than the C28x",
      round(a["mac_cost"]["cla"] / a["mac_cost"]["cpu"], 2), 1.84, tol=0.2)

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
