#!/usr/bin/env bash
# PC check: indexed kernels (original/) vs. drop_in kernels, all 8 CPU/CLA splits.
#
#   ./run.sh                              default switches
#   ./run.sh -DKERNEL_ADDR_WALK=0         constant-offset addressing
#   ./run.sh -DKERNEL_POOL_FLOAT=0        pool on int16, as the indexed kernels
#   ./run.sh -DKERNEL_USE_RECIPROCAL=1    reports how far the results move
#
# N_IMG= sets the number of inputs per set (default 300).
#
# PROJECT= overrides the firmware folder (default: ../../CNN_Inference_Edge-AI_with_CLA).
set -u
here="$(cd "$(dirname "$0")" && pwd)"
proj="${PROJECT:-$here/../../CNN_Inference_Edge-AI_with_CLA}"
drop="$here/../drop_in"
ref="$here/../original"            # the indexed kernels, kept as the reference
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
CC="${CC:-gcc}"
fl="-std=gnu99 -O0 -ffp-contract=off -Wno-unknown-pragmas -Wno-overflow"
rc=0        # OR of every split's exit code: bit 0 runtime set, bit 1 overflow set
for cfg in 000 001 010 011 100 101 110 111; do
    d="$tmp/$cfg"; mkdir -p "$d"
    cat > "$d/cla_pipeline_config.h" <<H
#ifndef CLA_PIPELINE_CONFIG_H
#define CLA_PIPELINE_CONFIG_H
#define CLA_STAGE_CONV1_POOL1  ${cfg:0:1}
#define CLA_STAGE_CONV2_POOL2  ${cfg:1:1}
#define CLA_STAGE_DENSE        ${cfg:2:1}
#define CLA_SINGLE_TASK  (CLA_STAGE_CONV1_POOL1 && CLA_STAGE_CONV2_POOL2 && CLA_STAGE_DENSE)
#define CLA_ANY_STAGE    (CLA_STAGE_CONV1_POOL1 || CLA_STAGE_CONV2_POOL2 || CLA_STAGE_DENSE)
#define CLA_TOUCHES_WORKSPACE  CLA_ANY_STAGE
#endif
H
    # Project headers are COPIED next to the generated config instead of being
    # reached with -I: weights.h and cla_dense_layer_shared.h include
    # "cla_pipeline_config.h" with quotes, which looks in their own folder
    # first -- from the project folder that is the project's split, not $cfg.
    cp "$proj"/definitions.h "$proj"/weights.h "$proj"/convolution.h \
       "$proj"/cla_dense_layer_shared.h "$d"/
    inc="-I$d -I$here/stubs"
    # reference: original/convolution.c, fused functions renamed
    $CC $fl $inc \
        -Dconv1_pool_fused_cpu=ref_conv1 -Dconv2_pool_fused_cpu=ref_conv2 -Ddense_fc_cpu=ref_dense \
        -c "$ref/convolution.c" -o "$d/ref.o" || { rc=$(( rc | 4 )); continue; }
    # new: drop_in sources; headers they do not carry come from the project
    $CC $fl $inc -I"$drop" "$@" -c "$drop/convolution.c" -o "$d/new_cpu.o" || { rc=$(( rc | 4 )); continue; }
    $CC $fl $inc -I"$drop" "$@" -D__interrupt= -x c -c "$drop/dense_layer.cla" -o "$d/new_cla.o" || { rc=$(( rc | 4 )); continue; }
    $CC $fl $inc -c "$here/compare.c" -o "$d/cmp.o" || { rc=$(( rc | 4 )); continue; }
    $CC -Wl,--allow-multiple-definition "$d/ref.o" "$d/new_cpu.o" "$d/new_cla.o" "$d/cmp.o" -lm -o "$d/t" || { rc=$(( rc | 4 )); continue; }
    "$d/t" ${N_IMG:-300}; rc=$(( rc | $? ))
done
echo
if   [ $(( rc & 4 )) -ne 0 ]; then echo "RESULT: build failed in at least one split"
elif [ $rc -eq 0 ];            then echo "RESULT: bit-identical in all 8 splits, both input sets"
elif [ $rc -eq 2 ];            then echo "RESULT: bit-identical on runtime inputs in all 8 splits; differs only where int16 overflows"
else                                echo "RESULT: differs on runtime inputs (see above)"
fi
exit $rc
