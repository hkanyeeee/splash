#pragma once
#include "metal/kernels/common/quant_formats.h"

// An expert pass decodes its tensors' formats alone, function constants 0 and 1 for the down pass (routed, shared) and
// 0-3 for the gate/up pass (ops/MoE.cpp: each once in ascending order, the last repeated past them): the GPU compiler
// builds the loops of every format a pipeline decodes, for each of its live-row matmuls, and a cold 32-row down
// pipeline took 68 s to build with all 44 on the M5 Max, 0.25 s with one. moe_formats<Count> runs body(F()) with the
// format type of `format`, one body per distinct format; an id the pipeline does not decode decodes as its first, so
// every id runs one body.
constant uint kMoeFormat0 [[function_constant(0)]];
constant uint kMoeFormat1 [[function_constant(1)]];
constant uint kMoeFormat2 [[function_constant(2)]];
constant uint kMoeFormat3 [[function_constant(3)]];
template <uint Count> struct moe_formats {
  template <class Body> void operator()(uint format, Body body) const {
    if constexpr (Count > 3)
      if (kMoeFormat3 != kMoeFormat2 && format == kMoeFormat3) return quant_format_switch(kMoeFormat3, body);
    if constexpr (Count > 2)
      if (kMoeFormat2 != kMoeFormat1 && format == kMoeFormat2) return quant_format_switch(kMoeFormat2, body);
    if (kMoeFormat1 != kMoeFormat0 && format == kMoeFormat1) return quant_format_switch(kMoeFormat1, body);
    quant_format_switch(kMoeFormat0, body);
  }
};
