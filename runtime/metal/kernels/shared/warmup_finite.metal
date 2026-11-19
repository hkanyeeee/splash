// Optional B1 warmup diagnostic. Integer exponent tests survive Metal fast math.
#include <metal_stdlib>
#include "metal/abi/VerifyTree.h"
using namespace metal;

struct Probe { uint width, routes, rows, bytes, masked, mapped, slot; };

kernel void warmup_first_nonfinite(device const ushort *data [[buffer(0)]],
                                   device atomic_uint *first [[buffer(1)]],
                                   device const SplashLaneRows *lane [[buffer(2)]],
                                   device const uint *route_rows [[buffer(3)]],
                                   constant Probe &p [[buffer(4)]],
                                   uint i [[thread_position_in_grid]]) {
  const uint width = p.width, routes = p.routes, row = i / width / routes;
  if (row >= p.rows || (p.masked && (row < lane->first || row - lane->first >= lane->rows))) return;
  const uint group = i / width;
  const uint mapped = p.mapped ? route_rows[group] : group;
  // Unverified routes carry a sentinel. Only the active lane's live routes are observed.
  if (mapped >= p.rows * routes) return;
  const uint element = mapped * width + i % width;
  const bool bad = p.bytes == 4
      ? (reinterpret_cast<device const uint *>(data)[element] & 0x7f800000u) == 0x7f800000u
      : (data[element] & 0x7f80u) == 0x7f80u;
  if (bad) atomic_fetch_min_explicit(first + p.slot, i, memory_order_relaxed);
}
