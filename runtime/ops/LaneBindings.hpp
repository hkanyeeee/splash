#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/MetalBackend.hpp"

#include <array>
#include <span>
#include <stdexcept>
#include <vector>

namespace splash::ops {

// Batched kernels bind one buffer per lane for each per-lane state tensor and
// index them by lane, so every slot must be bound even when fewer lanes run.
inline void appendLaneBindings(std::vector<metal::MetalBuffer> &bindings,
                               std::span<const metal::MetalBuffer> first,
                               std::span<const metal::MetalBuffer> second) {
  constexpr size_t lanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  if (first.empty() || first.size() > lanes || second.size() != first.size())
    throw std::invalid_argument("lane bindings must name matching nonempty lane sets");
  // A runtime with fewer lanes still binds every slot of the four-lane ABI.
  // Unused slots alias lane zero; the dispatch grid never executes them.
  for (const auto buffers : {first, second})
    for (size_t lane = 0; lane < lanes; ++lane)
      bindings.push_back(buffers[lane < buffers.size() ? lane : 0]);
}

// A per-lane buffer of each lane of `lanes` (at least one), and lane 0's in
// every slot past them, as a padded lane binds.
inline std::array<metal::MetalBuffer, SPLASH_MAXIMUM_BATCH_WIDTH>
laneBindings(std::span<const metal::MetalBuffer> lanes) {
  if (lanes.empty() || lanes.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("lane bindings must name 1 to every lane");
  std::array<metal::MetalBuffer, SPLASH_MAXIMUM_BATCH_WIDTH> result;
  for (size_t lane = 0; lane < result.size(); ++lane) result[lane] = lanes[lane < lanes.size() ? lane : 0];
  return result;
}

} // namespace splash::ops
