// GGUF decode on Apple9's register simdgroup_matrix path. The 8x8x8 MMA runs
// on the FP32 pipe there, so the kernel keeps every other FP32 operation to
// the minimum a group scale needs, and does the rest on the integer pipe:
// - weights enter the MMA as exact bf16: 128 + code for linear codes with a
//   min, 160 + code - zero for seeded linear codes (one add), the code itself
//   (converted) for 8-bit linear codes, or the codebook, int8 or grid value;
// - one MMA chain per 32 inputs (per 16 for 16-input scales), closed by the
//   fp32 epilogue s * chain; a format with a min adds b * sum once per
//   coefficient group, the F::Group inputs one scale and min cover (16, 32,
//   or 64 to 256 where the groups of a meta unit share them, as MLX affine's
//   groups of 64 and 128 do), but at least once per 64-input span of the
//   table, with the last chain of either (b = m - 128 s, or b = m for 8-bit
//   codes; sum those inputs, from the span's sums of 32), so that it reads
//   the current span alone; with a zero point, that chain starts from
//   -160 * sum instead (the operand being 160 + code - zero), the table's
//   seed of 16 inputs for 16-input scales, and Q2_K's min per 16 inputs adds
//   b * seed with b = m / -160;
// - coefficients decoded once per simdgroup and coefficient group into
//   threadgroup memory;
// - L request lanes per threadgroup share the weight operands and coefficients;
//   each lane's chains and epilogue run as with L = 1, so its result does not
//   depend on the batch width.
// Weights: the GGUF image (metal/abi/QuantFormat.h), read by chunk: lane c of
// a fragment reads chunk c of each group. Activations: the Table16 table and
// row sums (kernels/common/gguf_sgmatrix.h).
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/quant_formats.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"

namespace m4_parallel {

template <class F> struct Shape {
  enum : uint {
    Linear = F::Kind == QuantLinear,
    HasMin = Linear && F::Zero == 0,                          // Q4_K, Q5_K, Q4_1, Q5_1, Q2_K, MLX affine: s code + m
    // Linear codes past 127 (MLX affine 8-bit), which enter the MMA as the code itself: no bf16 128 + code holds them
    Wide = QuantWideCodes<F>::value,
    // Linear codes whose chains start from -160 times the sum of their inputs, so that 160 + code - zero enters the
    // MMA: s (code - zero) with a zero point, once per coefficient group and span from the table's sums of 32 where a
    // scale covers 32 inputs or more (Q4_0, Q5_0, IQ1_S, PQ2_0, Q2_0) and per 16-input chain from the table's seeds
    // (Q6_K, Q3_K, IQ1_M); and Q2_K's min per 16 inputs
    Seeded = Linear && (!HasMin || F::Group == 16),
    // bf16 bits of the operand of code 0: 128, or 160 - zero when seeded
    Operand = 0x4300 + (Seeded ? gguf_sg::kZeroPointOffset - 128 - F::Zero : 0),
    CG = F::Group == 16 ? 2 : 1,                              // MMA chains per 32 inputs
    CPG = CG * F::Group / 32,                                 // chains per coefficient group
    MinChains = CPG < 2 * CG ? CPG : 2 * CG,                  // chains per term: the group's, at most a span's
    // Spans decoded per coefficient unit: every span of eight groups (one meta unit of eight, or two of four), or
    // half of them where a span's coefficients take twice Q4_K's storage (Q2_K's (s, b) per 16 inputs)
    UnitSpans = F::MetaGroups == 1 ? 1 : CG * (HasMin ? 2 : 1) <= 2 ? 4 : 2,
    J = 2 * UnitSpans * CG / CPG,                             // coefficients per column and unit
  };
  // Each coefficient reads its own group's meta unit (coefficient_source), so a unit of two or four groups (MLX
  // affine's 64 or 128 elements, PQ2_0's 128, Q2_0's 64) splits a coefficient unit of four spans: such a format takes
  // rows of whole 256 inputs here (quant_register_tensor).
  static_assert(F::MetaGroups == 8 || F::MetaGroups == 4 || F::MetaGroups == 2 || F::MetaGroups == 1,
                "a meta unit is one, two, four or eight groups");
  static_assert(UnitSpans * GGUF_TABLE16_SPAN_INPUTS == quant_register_unit(F::Id),
                "the host splits K in the coefficient units of quant_register_unit");
  static_assert(!Wide || (HasMin && !Seeded), "wide codes have a min per group");
  static_assert(F::Group == 16 || (F::Group % 32 == 0 && F::Group <= 32 * F::MetaGroups),
                "a coefficient covers 16 inputs, or whole groups of one meta unit");
  static_assert(CPG >= 1 && 2 * UnitSpans * CG % CPG == 0, "a coefficient unit holds whole coefficient groups");
  // decode adds a min term with its last chain, or starts that chain from a zero point's, from the span's 32-input
  // sums of the term's chains: a chain covers one 32-input group, and a term's chains lie in one span, so it reads
  // no sum before the span.
  static_assert(!(HasMin && !Seeded) || CG == 1, "a min term's chains are 32-input groups");
  static_assert(2 * CG % MinChains == 0, "a min term's chains lie in one span");
};
template <class F> using Coef = metal::conditional_t<Shape<F>::HasMin != 0, float2, float>;
// The coefficients of a threadgroup's simdgroups for one unit; the run-time-format kernels hold every format's in
// storage of the largest size.
template <class F> constant constexpr uint kCoefs = GGUF_TILE_COLUMNS * Shape<F>::J;
static_assert(GGUF_REGISTER_COLUMNS == 2 * 8, "a simdgroup's columns are its two 8-column MMA fragments");
constant constexpr uint kCoefFloat2s = kCoefs<FmtQ4K>;
#define GGUF_SG_COEF_BYTES(F, f) static_assert(kCoefs<F> * sizeof(Coef<F>) <= kCoefFloat2s * sizeof(float2), #f " coefficients fit");
QUANT_REGISTER_FORMATS(GGUF_SG_COEF_BYTES)
#undef GGUF_SG_COEF_BYTES

// The exact bf16 operand of pair f of a chunk.
template <class F>
inline bfloat2 operand(typename F::Chunk ch, uint f, threadgroup const bfloat2 *lut) {
  typedef Shape<F> S;
  if constexpr (F::Kind == QuantLinear && S::Wide) {
    const uint pair = F::codes(ch)[f];
    return bfloat2(float2(pair & 0xFFFFu, pair >> 16));   // exact: bf16 holds every integer to 256
  } else if constexpr (F::Kind == QuantLinear) {
    return as_type<bfloat2>(F::codes(ch)[f] + S::Operand * 0x00010001u);   // Operand in both halves
  } else if constexpr (F::Kind == QuantCodebook) {
    return lut[(F::indices(ch) >> (8 * f)) & 0xFFu];
  } else if constexpr (F::Kind == QuantInt8) {
    const uint2 v = F::values(ch);
    return bfloat2(float2(as_type<char2>(ushort((f < 2 ? v.x : v.y) >> (16 * (f & 1))))));
  } else if constexpr (F::Kind == QuantFloat8) {   // exact: an E4M3 value has four significant bits
    const uint2 v = F::values(ch);
    const uint pair = ((f < 2 ? v.x : v.y) >> (16 * (f & 1))) & 0xFFFFu;
    return bfloat2(float2(quant_e4m3_pair(quant_byte_pairs(pair).x)));
  } else if constexpr (F::Kind == QuantFloat16) {   // the bf16 values themselves (quant_register_format)
    static_assert(is_same_v<typename F::Value, bfloat>, "the register tile takes bf16 values only");
    return as_type<bfloat2>(F::values(ch)[f]);
  } else {
    const uint2 g = F::grid(ch);
    const uchar4 m = as_type<uchar4>(f < 2 ? g.x : g.y);
    const uint s = F::signs(ch) >> (2 * f);
    const float2 v = float2(m[2 * (f & 1)], m[2 * (f & 1) + 1]);
    return bfloat2(select(v, -v, bool2(s & 1u, s & 2u)));
  }
}

// What coefficient j of a column in coefficient unit u is decoded from: its
// meta unit and, with ScaleInChunk, chunk 0 of its first group (which holds
// the scale).
template <class F> struct CoefSource {
  typename F::Meta meta;
  typename F::Chunk chunk;
};
template <class F>
inline CoefSource<F> coefficient_source(device uchar *w0, device uchar *w1, device uchar *meta, uint plane_tile,
                                        uint groups, uint column, uint u, uint j) {
  typedef Shape<F> S;
  const uint g = u * 2 * S::UnitSpans + j * S::CPG / S::CG;   // its first chain's 32-input group
  const uint units = groups / F::MetaGroups;
  CoefSource<F> src;
  src.meta = F::loadMeta(meta + ((ulong(plane_tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS + column) * F::MetaBytes);
  if constexpr (F::ScaleInChunk) {
    const ulong at = (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + column;
    src.chunk = F::loadChunk(w0 + at * F::P0, w1 + at * F::P1, 0);
  }
  return src;
}
// Coefficient j of a column in coefficient unit u, that of its first chain c =
// j CPG: 32-input group c / CG of the unit, and of 16-input coefficients half
// h = c % CG. Formats with a min return (s, m - 128 s), (s, m) with wide
// codes, or with a seeded min (s, m / -160), whose product with the seed is m
// times the sum of the inputs.
template <class F> inline Coef<F> coefficient(CoefSource<F> src, uint u, uint j) {
  typedef Shape<F> S;
  const uint g = u * 2 * S::UnitSpans + j * S::CPG / S::CG, h = j * S::CPG % S::CG;
  QuantCoef k;
  if constexpr (F::ScaleInChunk) k = F::coef(src.meta, src.chunk);
  else k = F::coef(src.meta, ushort(g % F::MetaGroups));
  const float s = h ? k.s.y : k.s.x, m = h ? k.m.y : k.m.x;
  if constexpr (S::HasMin && S::Seeded) return float2(s, m * (-1.0f / gguf_sg::kZeroPointOffset));
  else if constexpr (S::HasMin && S::Wide) return float2(s, m);
  else if constexpr (S::HasMin) return float2(s, fma(-128.0f, s, m));
  else return s;
}

// One threadgroup: 4 simdgroups x GGUF_REGISTER_COLUMNS = GGUF_TILE_COLUMNS columns of one segment,
// L request lanes of eight rows, one K partition (tg.y) of `splits`, into a destination of type Out.
template <class F, uint L, uint Columns>
inline void decode(device const bfloat *table,device const float *sums,device uchar *w0,device uchar *w1,device uchar *meta,
                   device coherent(device) float *partials,const GgufDecodeParams p,uint2 tg,uint sg,uint lane,
                   threadgroup const bfloat2 *lut,threadgroup Coef<F> *coefs) {
  typedef Shape<F> S;
  typedef Coef<F> C;
  constexpr uint FG = 4 / S::CG;  // fragments per chain
  const uint K = p.input_size, spans = K / GGUF_TABLE16_SPAN_INPUTS, groups = K / 32, splits = p.splits;
  const ulong tileSums = table16_sums_per_tile(K);
  const uint units = spans / S::UnitSpans;
  const uint u0 = tg.y * units / splits, u1 = (tg.y + 1) * units / splits;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * Columns + sg * GGUF_REGISTER_COLUMNS, plane_tile = base / QUANT_TILE_ROWS,
             plane_row = base % QUANT_TILE_ROWS;
  constexpr uint NC = GGUF_REGISTER_COLUMNS;
  threadgroup C *cu = coefs + sg * NC * S::J;

  // Weight streams: lane c reads chunk c of each group; a span's two groups
  // of one column are a plane tile of payloads apart.
  const uint first = u0 * S::UnitSpans;
  device uchar *p0[2], *p1[2];
#pragma unroll
  for (uint nf = 0; nf < 2; ++nf) {
    const ulong at = (ulong(plane_tile) * groups + 2 * first) * QUANT_TILE_ROWS + plane_row + nf * 8 + fm;
    p0[nf] = w0 + at * F::P0;
    p1[nf] = w1 + at * F::P1;
  }
  typename F::Chunk cur[2][2], nxt[2][2];
  const auto load = [&](thread typename F::Chunk (&ch)[2][2]) __attribute__((always_inline)) {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      ch[nf][0] = F::loadChunk(p0[nf], p1[nf], c);
      ch[nf][1] = F::loadChunk(p0[nf] + QUANT_TILE_ROWS * F::P0, p1[nf] + QUANT_TILE_ROWS * F::P1, c);
    }
  };
  device const vec<bfloat, 8> *xt =
      reinterpret_cast<device const vec<bfloat, 8> *>(table + ulong(first) * GGUF_TABLE16_SPAN_VALUES);
  device const float *seeds = sums + ulong(first) * GGUF_TABLE16_SPAN_SEEDS;
  device const float *s32 = sums + table16_sums32_offset(K) + ulong(first) * GGUF_TABLE16_SPAN_SUMS;

  float2 acc[L][2];
#pragma unroll
  for (uint r = 0; r < L; ++r) acc[r][0] = acc[r][1] = float2(0);
  load(cur);
  for (uint u = u0; u < u1; ++u) {
    // This unit's coefficients for the simdgroup's NC columns, each decoded
    // once. Every source load is issued before the first decode waits for one,
    // which takes 1-8% off the zero-point formats of 16-input scales (16
    // coefficients per column and unit) at one to four lanes on the 40-core
    // M3 Max.
    constexpr uint I = (NC * S::J + 31) / 32;
    CoefSource<F> src[I];
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * S::J) src[i] = coefficient_source<F>(w0, w1, meta, plane_tile, groups, plane_row + e % NC, u, e / NC);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * S::J) cu[e] = coefficient<F>(src[i], u, e / NC);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint us = 0; us < S::UnitSpans; ++us) {
      const bool more = us + 1 < S::UnitSpans || u + 1 < u1;
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        p0[nf] += 2 * QUANT_TILE_ROWS * F::P0;
        p1[nf] += 2 * QUANT_TILE_ROWS * F::P1;
      }
      if (more) load(nxt);
#pragma unroll
      for (uint q = 0; q < 2; ++q) {
        bfloat2 a[2][4];
        C cs[S::CG][2];
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) {
#pragma unroll
          for (uint f = 0; f < 4; ++f) a[nf][f] = operand<F>(cur[nf][q], f, lut);
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) cs[h][nf] = cu[((us * 2 + q) * S::CG + h) / S::CPG * NC + nf * 8 + fm];
        }
        // Whether this 32-input group of the table closes a term's inputs, a coefficient group or its span.
        const bool closes = ((us * 2 + q) * S::CG + S::CG) % S::MinChains == 0;
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          const vec<bfloat, 8> bq = xt[ulong(r) * K + (8 * q + fm) * 4 + c];
          float2 sum = float2(0), seed[S::CG];
          if ((S::HasMin || S::Seeded) && S::CG == 1 && closes) {
            // The sum of the term's inputs: this 32-input sum and, closing a term of 64 inputs, the span's other.
            sum = *(device const float2 *)(s32 + ulong(r) * tileSums + q * 8 + fn);
#pragma unroll
            for (uint t = 1; t < S::MinChains; ++t) {
              const int earlier = int(q) - int(t);   // an earlier 32-input group of the span
              sum += *(device const float2 *)(s32 + ulong(r) * tileSums + fn + earlier * 8);
            }
          }
#pragma unroll
          for (uint h = 0; h < S::CG; ++h)   // a 16-input chain's seed from the table, a term's once from its sum
            seed[h] = !S::Seeded ? float2(0)
                      : S::CG == 2 ? *(device const float2 *)(seeds + ulong(r) * tileSums + (2 * q + h) * 8 + fn)
                                   : -float(gguf_sg::kZeroPointOffset) * sum;
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) {
#pragma unroll
            for (uint nf = 0; nf < 2; ++nf) {
              float2 dot = seed[h];
#pragma unroll
              for (uint f = h * FG; f < (h + 1) * FG; ++f)
                sgmatrix::mma_acc<bfloat>(dot, a[nf][f], reinterpret_cast<thread const bfloat2 *>(&bq)[f]);
              if constexpr (S::HasMin) {
                acc[r][nf] = fma(dot, cs[h][nf].x, acc[r][nf]);
                if (S::Seeded || closes) acc[r][nf] = fma(S::Seeded ? seed[h] : sum, cs[h][nf].y, acc[r][nf]);
              } else {
                acc[r][nf] = fma(dot, float(cs[h][nf]), acc[r][nf]);
              }
            }
          }
        }
      }
      xt += GGUF_TABLE16_SPAN_VALUES / 8;
      seeds += GGUF_TABLE16_SPAN_SEEDS;
      s32 += GGUF_TABLE16_SPAN_SUMS;
      if (more) {
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) cur[nf][0] = nxt[nf][0], cur[nf][1] = nxt[nf][1];
      }
    }
  }

#pragma unroll
  for(uint r=0;r<L;++r)
#pragma unroll
    for(uint nf=0;nf<2;++nf){
      device coherent(device) float *slot=partials+((ulong(r)*splits+tg.y)*8+fn)*p.out_stride+p.out_offset+base+nf*8+fm;
      slot[0]=acc[r][nf].x;slot[p.out_stride]=acc[r][nf].y;
    }
}
template<class G,class U,uint L,uint Groups>
inline void pair(device const bfloat *table,device const float *sums,device uchar *u0,device uchar *u1,device uchar *um,device uchar *g0,device uchar *g1,device uchar *gm,device bfloat *out,device bfloat *gate_output,device coherent(device) float *up_partials,device atomic_uint *counters,device coherent(device) float *gate_partials,const GgufDecodeParams p,uint2 tg,uint tid,uint sg,uint lane,threadgroup bfloat2 *glut,threadgroup bfloat2 *ulut,threadgroup Coef<G> *gcoefs,threadgroup Coef<U> *ucoefs,threadgroup uint *arrival){
  constexpr uint Columns=Groups*GGUF_REGISTER_COLUMNS;
  quant_pair_table<G>(glut,tid,Groups*64);quant_pair_table<U>(ulut,tid,Groups*64);
  // Each branch contains only SIMD barriers; no threadgroup barrier is divergent.
  if(sg<Groups)decode<G,L,Columns>(table,sums,g0,g1,gm,gate_partials,p,tg,sg,lane,glut,gcoefs);
  else decode<U,L,Columns>(table,sums,u0,u1,um,up_partials,p,tg,sg-Groups,lane,ulut,ucoefs);
  device atomic_uint *counter=counters+(p.out_offset+tg.x*Columns)/Columns;
  if(p.splits>1){if(!split_arrive_last(counter,p.splits,tid,arrival))return;split_release(counter,tid);}
  else threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
  if(sg>=Groups)return;
  const auto l=sgmatrix::lane_map(lane);const uint base=p.out_offset+tg.x*Columns+sg*GGUF_REGISTER_COLUMNS;
#pragma unroll
  for(uint r=0;r<L;++r)
#pragma unroll
    for(uint nf=0;nf<2;++nf)
#pragma unroll
      for(uint i=0;i<2;++i){
        const uint column=base+nf*8+l.fm,row=r*8+l.fn+i;
        const auto partial=[&](uint s){const ulong at=((ulong(r)*p.splits+s)*8+l.fn+i)*p.out_stride+column;return float2(gate_partials[at],up_partials[at]);};
        float2 value=partial(0);
        if(p.splits>1){
#pragma clang fp reassociate(off)
          value=float2(0);for(uint s=0;s<p.splits;++s)value+=partial(s);
        }
        const bfloat gate=bfloat(value.x);const ulong at=ulong(row)*p.out_stride+column;
        gate_output[at]=gate;out[at]=bfloat(float(bfloat(value.y))*splash_silu(float(gate)));
      }
}
} // namespace m4_parallel

kernel void m4_borrow_pair_iq4xs_q6k_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4XS::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4XS> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4XS>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4XS,FmtQ6K,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_iq4xs_q6k_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4XS::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4XS> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4XS>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4XS,FmtQ6K,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_iq4xs_q6k_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4XS::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4XS> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4XS>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4XS,FmtQ6K,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q5k_q5k_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ5K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ5K,FmtQ5K,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q5k_q5k_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ5K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ5K,FmtQ5K,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q5k_q5k_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ5K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ5K,FmtQ5K,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q80_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ80> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ80,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q80_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ80> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ80,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q80_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ80> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ80,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q5k_q6k_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ5K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ5K,FmtQ6K,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q5k_q6k_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ5K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ5K,FmtQ6K,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q5k_q6k_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ5K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ5K,FmtQ6K,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q5k_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ5K,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q5k_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ5K,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q5k_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ5K,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_iq4nl_q5k_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4NL::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4NL> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4NL>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4NL,FmtQ5K,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_iq4nl_q5k_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4NL::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4NL> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4NL>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4NL,FmtQ5K,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_iq4nl_q5k_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4NL::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4NL> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4NL>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4NL,FmtQ5K,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_iq4xs_q5k_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4XS::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4XS> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4XS>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4XS,FmtQ5K,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_iq4xs_q5k_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4XS::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4XS> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4XS>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4XS,FmtQ5K,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_iq4xs_q5k_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtIQ4XS::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ5K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtIQ4XS> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtIQ4XS>::J];
 threadgroup m4_parallel::Coef<FmtQ5K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ5K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtIQ4XS,FmtQ5K,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q6k_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ6K,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q6k_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ6K,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q6k_q6k_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ6K::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ6K> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];
 threadgroup m4_parallel::Coef<FmtQ6K> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ6K>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ6K,FmtQ6K,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q80_q80_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ80> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];
 threadgroup m4_parallel::Coef<FmtQ80> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ80,FmtQ80,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q80_q80_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ80> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];
 threadgroup m4_parallel::Coef<FmtQ80> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ80,FmtQ80,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_q80_q80_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtQ80::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtQ80> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];
 threadgroup m4_parallel::Coef<FmtQ80> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtQ80>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtQ80,FmtQ80,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_af4g64_af4g64_l1_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtAF4G64::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtAF4G64::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtAF4G64> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtAF4G64>::J];
 threadgroup m4_parallel::Coef<FmtAF4G64> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtAF4G64>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtAF4G64,FmtAF4G64,1,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(1)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_af4g64_af4g64_l2_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtAF4G64::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtAF4G64::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtAF4G64> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtAF4G64>::J];
 threadgroup m4_parallel::Coef<FmtAF4G64> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtAF4G64>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtAF4G64,FmtAF4G64,2,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(2)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}

kernel void m4_borrow_pair_af4g64_af4g64_l4_g1(device const bfloat *table [[buffer(0)]],device const float *sums [[buffer(1)]],device uchar *u0 [[buffer(2)]],device uchar *u1 [[buffer(3)]],device uchar *um [[buffer(4)]],device uchar *g0 [[buffer(5)]],device uchar *g1 [[buffer(6)]],device uchar *gm [[buffer(7)]],device bfloat *out [[buffer(8)]],device bfloat *gate_output [[buffer(9)]],device coherent(device) float *up_partials [[buffer(10)]],device atomic_uint *counters [[buffer(11)]],device coherent(device) float *gate_partials [[buffer(12)]],constant GgufDecodeParams &p [[buffer(13)]],uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup bfloat2 glut[FmtAF4G64::Kind==QuantCodebook?kQuantPairTableEntries:1],ulut[FmtAF4G64::Kind==QuantCodebook?kQuantPairTableEntries:1];
 threadgroup m4_parallel::Coef<FmtAF4G64> gcoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtAF4G64>::J];
 threadgroup m4_parallel::Coef<FmtAF4G64> ucoefs[1*GGUF_REGISTER_COLUMNS*m4_parallel::Shape<FmtAF4G64>::J];threadgroup uint arrival;
 m4_parallel::pair<FmtAF4G64,FmtAF4G64,4,1>(table,sums,u0,u1,um,g0,g1,gm,out,gate_output,up_partials,counters,gate_partials+ulong(4)*8*p.splits*p.out_stride,p,tg,tid,sg,lane,glut,ulut,gcoefs,ucoefs,&arrival);
}
