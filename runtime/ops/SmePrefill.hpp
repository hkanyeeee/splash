#pragma once

#include "ops/Linear.hpp"

namespace splash::ops {

// For a 40-core M4 Max, the trailing columns of a segment the CPU can compute
// beside the GPU's leading columns. Zero when the arena cannot hold the tiles.
[[nodiscard]] uint32_t smePrefillColumns(const PrefillCpuScratch &scratch, const metal::MetalBuffer &input,
                                         const QuantizedSegment &segment, uint32_t rows, uint32_t columns);

// Writes a separate CPU result during the GPU dispatch, then joins it into
// the strided output. The caller submits the two parts as one graph.
void addSmePrefill(metal::CommandGraph &graph, const PrefillCpuScratch &scratch, metal::MetalBuffer input,
                   metal::MetalBuffer output, const QuantizedSegment &segment, uint32_t rows,
                   uint32_t stride, uint32_t columns, const std::function<void()> &gpu,
                   metal::MetalBuffer residual = {});

} // namespace splash::ops
