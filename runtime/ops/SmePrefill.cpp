#include "ops/SmePrefill.hpp"

#include "ops/RowCopy.hpp"

#include <arm_sme.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace splash::ops {
namespace {

// M4 streams 512-bit vectors. Four 16x16 outer products fill a 32x32 tile.
constexpr uint32_t kTile = 32;
__arm_locally_streaming __attribute__((target("sme"))) static uint32_t tileWidth() {
  return 2 * svcntw();
}

template <class T> T load(const uint8_t *bytes) {
  T value;
  std::memcpy(&value, bytes, sizeof(T));
  return value;
}

// Transpose pairs from the GGUF plane tiles directly into the SME right
// operand, [output tile][input][output lane]. Round weights to half exactly
// where the Metal prefill tile stages them, then widen them for the FP32 MMA.
template <uint32_t Format>
void packWeights(const QuantizedSegment &s, uint32_t columns, float *dest) {
  constexpr auto format = kQuantFormats[Format];
  const auto *low = static_cast<const uint8_t *>(s.plane0.contents());
  const auto *high = static_cast<const uint8_t *>(s.plane1.contents());
  const auto *meta = static_cast<const uint8_t *>(s.meta.contents());
  const uint32_t first = s.outputSize - columns, k = s.inputSize, groups = k / 32;
  dispatch_apply(columns / kTile, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t tile) {
    for (uint32_t group = 0; group < groups; ++group) {
      for (uint32_t lane = 0; lane < kTile; ++lane) {
        const uint32_t row = first + tile * kTile + lane;
        const size_t wi = quant_tile_index(row, group, groups);
        const uint32_t unit = group / format.meta_groups, j = group % format.meta_groups;
        const uint8_t *header = meta + quant_tile_index(row, unit, groups / format.meta_groups) * format.meta_bytes;
        float *packed = dest + (tile * k + group * 32) * kTile + lane;
        if constexpr (Format == GGUF_FMT_Q80) {
          const float scale = load<_Float16>(header);
          for (uint32_t i = 0; i < 32; ++i)
            packed[i * kTile] = float(_Float16(scale * load<int8_t>(low + wi * 32 + quant_slot(i))));
        } else {
          float scale[2], bias = 0;
          if constexpr (Format == GGUF_FMT_Q6K) {
            const float d = load<_Float16>(header + 16);
            for (uint32_t side = 0; side < 2; ++side)
              scale[side] = d * load<int8_t>(header + 2 * j + side);
          } else {
            const uint8_t *scales = header + 4;
            const uint32_t sc = j < 4 ? scales[j] & 63 : (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4);
            const uint32_t mn = j < 4 ? scales[j + 4] & 63 : (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            scale[0] = scale[1] = float(load<_Float16>(header)) * sc;
            bias = -float(load<_Float16>(header + 2)) * mn;
          }
          for (uint32_t c = 0; c < 4; ++c) {
            const uint32_t lowBits = load<uint32_t>(low + wi * 16 + c * 4);
            const uint32_t highBits = Format == GGUF_FMT_Q6K ? load<uint16_t>(high + wi * 8 + c * 2) : high[wi * 4 + c];
            for (uint32_t pair = 0; pair < 4; ++pair) {
              const uint32_t side = pair / 2, i = side * 16 + 4 * c + 2 * (pair % 2);
              const uint32_t l = lowBits >> (4 * pair);
              const uint32_t h = highBits >> ((Format == GGUF_FMT_Q6K ? 4 : 2) * pair);
              for (uint32_t e = 0; e < 2; ++e) {
                const uint32_t lo = (l >> (16 * e)) & 15;
                if constexpr (Format == GGUF_FMT_Q6K) {
                  const int code = int(lo | (((h >> (2 * e)) & 3) << 4)) - 32;
                  packed[(i + e) * kTile] = float(_Float16(scale[side] * float(code)));
                } else {
                  const int code = int(lo | (((h >> e) & 1) << 4));
                  packed[(i + e) * kTile] = float(_Float16(std::fma(scale[side], float(code), bias)));
                }
              }
            }
          }
        }
      }
    }
  });
}

// One FP32 multiply-add per K, in the order of the GPU accumulator. The
// two-products-per-step FP16 SME instruction rounds at different positions.
__arm_new("za") __arm_locally_streaming __attribute__((target("sme")))
static void multiply(const float *a, const float *b, uint32_t k, float *tile) {
  const auto pg = svptrue_b32();
  svzero_za();
  for (uint32_t i = 0; i < k; ++i) {
    const auto a0 = svld1_f32(pg, a), a1 = svld1_f32(pg, a + kTile / 2);
    const auto b0 = svld1_f32(pg, b), b1 = svld1_f32(pg, b + kTile / 2);
    svmopa_za32_f32_m(0, pg, pg, a0, b0);
    svmopa_za32_f32_m(1, pg, pg, a0, b1);
    svmopa_za32_f32_m(2, pg, pg, a1, b0);
    svmopa_za32_f32_m(3, pg, pg, a1, b1);
    a += kTile;
    b += kTile;
  }
  for (uint32_t row = 0; row < kTile / 2; ++row) {
    const auto zero = svdup_f32(0);
    float *upper = tile + row * kTile, *lower = upper + kTile * kTile / 2;
    svst1(pg, upper, svread_hor_za32_f32_m(zero, pg, 0, row));
    svst1(pg, upper + kTile / 2, svread_hor_za32_f32_m(zero, pg, 1, row));
    svst1(pg, lower, svread_hor_za32_f32_m(zero, pg, 2, row));
    svst1(pg, lower + kTile / 2, svread_hor_za32_f32_m(zero, pg, 3, row));
  }
}

void project(const PrefillCpuScratch &scratch, const metal::MetalBuffer &input, const QuantizedSegment &segment,
             uint32_t rows, uint32_t columns, const metal::MetalBuffer &residual, uint32_t stride) {
  auto *activations = static_cast<float *>(scratch.activations.contents());
  auto *weights = static_cast<float *>(scratch.weights.contents());
  auto *output = static_cast<__bf16 *>(scratch.result.contents());
  const auto *values = static_cast<const __bf16 *>(input.contents());
  const uint32_t k = segment.inputSize;
  const auto *skip = static_cast<const __bf16 *>(residual.contents());
  const uint32_t offset = segment.columnOffset + segment.outputSize - columns;
  switch (segment.formatId) {
  case GGUF_FMT_Q6K: packWeights<GGUF_FMT_Q6K>(segment, columns, weights); break;
  case GGUF_FMT_Q5K: packWeights<GGUF_FMT_Q5K>(segment, columns, weights); break;
  case GGUF_FMT_Q80: packWeights<GGUF_FMT_Q80>(segment, columns, weights); break;
  default: throw std::logic_error("unsupported SME prefill format");
  }
  dispatch_apply((rows + kTile - 1) / kTile, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
    const uint32_t start = block * kTile, live = std::min(kTile, rows - start);
    float *a = activations + size_t(start) * k;
    for (uint32_t i = 0; i < k; ++i) {
      float *packed = a + i * kTile;
      for (uint32_t row = 0; row < live; ++row) packed[row] = float(values[size_t(start + row) * k + i]);
      std::fill(packed + live, packed + kTile, 0.f);
    }
    alignas(64) float tile[kTile * kTile];
    for (uint32_t col = 0; col < columns; col += kTile) {
      multiply(a, weights + size_t(col) * k, k, tile);
      for (uint32_t row = 0; row < live; ++row) {
        const size_t destinationRow = start + row;
        __bf16 *out = output + destinationRow * columns + col;
        const float *sum = tile + row * kTile;
        if (skip) {
          const __bf16 *addend = skip + destinationRow * stride + offset + col;
          for (uint32_t j = 0; j < kTile; ++j) out[j] = __bf16(sum[j] + float(addend[j]));
        } else {
          for (uint32_t j = 0; j < kTile; ++j) out[j] = __bf16(sum[j]);
        }
      }
    }
  });
}
} // namespace

uint32_t smePrefillColumns(const PrefillCpuScratch &scratch, const metal::MetalBuffer &input,
                           const QuantizedSegment &s, uint32_t rows, uint32_t columns) {
  const uint32_t k = s.inputSize;
  if (rows < 1024 || (k != 5120 && k != 6144) ||
      (s.formatId != GGUF_FMT_Q6K && s.formatId != GGUF_FMT_Q5K && s.formatId != GGUF_FMT_Q80))
    return 0;
  static const bool supported = tileWidth() == kTile;
  // On the 40-core M4 Max this share includes the cost of packing both operands.
  const uint32_t part = uint64_t{columns} * 15 / 100 / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
  if (!supported || !part || part >= s.outputSize || !input.contents() || !s.plane0.contents() ||
      (s.formatId != GGUF_FMT_Q80 && !s.plane1.contents()) || !s.meta.contents() || !scratch.activations.contents() ||
      !scratch.weights.contents() || !scratch.result.contents())
    return 0;
  const uint64_t padded = (uint64_t{rows} + kTile - 1) / kTile * kTile;
  return scratch.activations.sizeBytes() >= padded * k * sizeof(float) &&
         scratch.weights.sizeBytes() >= uint64_t{part} * k * sizeof(float) &&
         scratch.result.sizeBytes() >= uint64_t{rows} * part * sizeof(uint16_t) ? part : 0;
}

void addSmePrefill(metal::CommandGraph &graph, const PrefillCpuScratch &scratch, metal::MetalBuffer input,
                   metal::MetalBuffer output, const QuantizedSegment &segment, uint32_t rows,
                   uint32_t stride, uint32_t columns, const std::function<void()> &gpu, metal::MetalBuffer residual) {
  graph.host([=] { project(scratch, input, segment, rows, columns, residual, stride); }, {},
             metal::HostStepGpu::Idle, {}, gpu, {scratch.activations, scratch.weights, scratch.result});
  RowCopy::add(graph, scratch.result, {0, columns, 0}, output,
               {0, stride, segment.columnOffset + segment.outputSize - columns}, rows, columns);
}
} // namespace splash::ops
