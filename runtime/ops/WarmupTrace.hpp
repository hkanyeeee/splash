#pragma once

#include "metal/CommandGraph.hpp"

#include <array>

namespace splash::ops {

inline void addWarmupTrace(metal::CommandGraph &graph, const metal::MetalBuffer &buffer,
                           const metal::MetalBuffer &results, const metal::MetalBuffer &laneRows,
                           const metal::MetalBuffer &routeRows, uint32_t rows, uint32_t width,
                           uint32_t routes, uint32_t bytes, uint32_t slot, uint32_t stride) {
  const uint64_t count = uint64_t{rows} * routes * width;
  const std::array<uint32_t, 8> params{width, routes, rows, bytes, uint32_t(bool(laneRows)),
                                       uint32_t(bool(routeRows)), slot, stride};
  graph.add("warmup_first_nonfinite", {buffer, results, laneRows ? laneRows : buffer,
                                       routeRows ? routeRows : buffer},
            params, {(count + 127) / 128, 1, 1}, {128, 1, 1});
}

} // namespace splash::ops
