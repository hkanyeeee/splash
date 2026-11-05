#include <metal_stdlib>
using namespace metal;

// Optional exact-prefix checkpoint for the local Qwen3.8-27B layout.
// Runtime guards this fixed geometry and FP32 state before dispatch.
// Each group folds one value head's committed prefix of the native tape.
constant uint Heads = 48, KeyHeads = 16, D = 128, Conv = 10240, KV = 8192;
// GdnStateLayout aligns each layer's convolution carry to a 16 KiB host page.
constant ulong ConvLayerBytes = 65536ul;
constant ulong ConvBytes = 48ul * ConvLayerBytes;
constant ulong RecLayerBytes = ulong(Heads) * D * D * 4;
constant ulong TapeKV = 8ul * Conv * 2;
constant ulong TapeDecay = TapeKV + 8ul * KV * 2;
constant ulong TapeBeta = TapeDecay + 8ul * Heads * 4;
constant ulong TapeBytes = TapeBeta + 8ul * Heads * 2;

kernel void verify_gdn_prefix_f32(device const uchar *state [[buffer(0)]],
                              device const uchar *tape [[buffer(1)]],
                              device uchar *destination [[buffer(2)]],
                              constant uint &prefix [[buffer(3)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint head = group.x, layer = group.y, kh = head / 3;
  const auto source = reinterpret_cast<device const float *>(state + ConvBytes + layer * RecLayerBytes);
  const auto output = reinterpret_cast<device float *>(destination + ConvBytes + layer * RecLayerBytes);
  const auto kv = reinterpret_cast<device const bfloat *>(tape + layer * TapeBytes + TapeKV);
  const auto decay = reinterpret_cast<device const float *>(tape + layer * TapeBytes + TapeDecay);
  const auto beta = reinterpret_cast<device const bfloat *>(tape + layer * TapeBytes + TapeBeta);
  threadgroup bfloat keys[8 * D], values[8 * D];
  threadgroup float decays[8], betas[8];
  if (sg < prefix) {
    for (uint g = 0; g < 4; ++g) {
      keys[sg * D + 32 * g + lane] = kv[sg * KV + kh * D + 32 * g + lane];
      values[sg * D + 32 * g + lane] = kv[sg * KV + KeyHeads * D + head * D + 32 * g + lane];
    }
    if (!lane) {
      decays[sg] = decay[sg * Heads + head];
      betas[sg] = float(beta[sg * Heads + head]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Match the existing decode recurrence's two state rows in flight.
  for (uint batch = 0; batch < 16; batch += 2) {
    float rows[2][4];
    for (uint r = 0; r < 2; ++r)
      for (uint i = 0; i < 4; ++i)
        rows[r][i] = source[(head * D + (batch + r) * 8 + sg) * D + lane * 4 + i];
    for (uint t = 0; t < prefix; ++t) {
      float memory[2];
      for (uint r = 0; r < 2; ++r) {
        memory[r] = 0.f;
        for (uint i = 0; i < 4; ++i) {
          rows[r][i] *= decays[t];
          memory[r] += rows[r][i] * float(keys[t * D + lane * 4 + i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      for (uint r = 0; r < 2; ++r) {
        const float delta = (float(values[t * D + (batch + r) * 8 + sg]) - memory[r]) * betas[t];
        for (uint i = 0; i < 4; ++i)
          rows[r][i] += float(keys[t * D + lane * 4 + i]) * delta;
      }
    }
    for (uint r = 0; r < 2; ++r)
      for (uint i = 0; i < 4; ++i)
        output[(head * D + (batch + r) * 8 + sg) * D + lane * 4 + i] = rows[r][i];
  }
  // Carry the last three convolution inputs, including prefix=0/1/2.
  if (sg < 3) {
    const auto old = reinterpret_cast<device const bfloat *>(state + layer * ConvLayerBytes);
    const auto inputs = reinterpret_cast<device const bfloat *>(tape + layer * TapeBytes);
    const auto next = reinterpret_cast<device bfloat *>(destination + layer * ConvLayerBytes);
    const uint history = prefix + sg;
    for (uint g = 0; g < 4; ++g) {
      const uint vc = 2 * KeyHeads * D + head * D + 32 * g + lane;
      next[sg * Conv + vc] = history < 3 ? old[history * Conv + vc] : inputs[(history - 3) * Conv + vc];
      if (head % 3 == 0) {
        const uint qc = kh * D + 32 * g + lane, kc = KeyHeads * D + qc;
        next[sg * Conv + qc] = history < 3 ? old[history * Conv + qc] : inputs[(history - 3) * Conv + qc];
        next[sg * Conv + kc] = history < 3 ? old[history * Conv + kc] : inputs[(history - 3) * Conv + kc];
      }
    }
  }
}
