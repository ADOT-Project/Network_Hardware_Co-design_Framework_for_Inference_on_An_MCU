#!/usr/bin/env bash
# PC check: baseline MFCC (original/) vs. drop_in/mfcc_extract.c on the golden
# slices, and what the difference does to the network's input and decision.
# Extra arguments go to the compiler for the new file, e.g.
#   ./run_mfcc.sh -DMFCC_FAST_LOG=0      library log10f instead of power_db()
# PROJECT= overrides the firmware folder.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
proj="${PROJECT:-$here/../../CNN_Inference_Edge-AI_with_CLA}"
drop="$here/../drop_in"; ref="$here/../original"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
CC="${CC:-gcc}"
fl="-std=gnu99 -O0 -ffp-contract=off -Wno-unknown-pragmas -Wno-overflow -I$tmp"
cp "$proj"/definitions.h "$proj"/weights.h "$proj"/convolution.h "$proj"/feature_tables.h \
   "$proj"/cla_pipeline_config.h "$proj"/golden_vector*.h "$tmp"/
$CC $fl -Dmfcc_extract=mfcc_ref -c "$ref/mfcc_extract.c"  -o "$tmp/mref.o" || exit 4
$CC $fl -Dmfcc_extract=mfcc_new -Wall -Wextra "$@" -c "$drop/mfcc_extract.c" -o "$tmp/mnew.o" || exit 4
$CC $fl -Dconv1_pool_fused_cpu=ref_conv1 -Dconv2_pool_fused_cpu=ref_conv2 -Ddense_fc_cpu=ref_dense \
    -c "$ref/convolution.c" -o "$tmp/conv.o" || exit 4
$CC $fl -c "$here/compare_mfcc.c" -o "$tmp/cmp.o" || exit 4
$CC "$tmp/mref.o" "$tmp/mnew.o" "$tmp/conv.o" "$tmp/cmp.o" -lm -o "$tmp/t" || exit 4
"$tmp/t"
