#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/local_attention_flags.h"

kernel void m4_flags_verify_control(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],
 device float *statistics [[buffer(2)]],device const SplashKvPage *table [[buffer(3)]],
 device const SplashKvPage *unused1 [[buffer(4)]],
device const SplashKvPage *unused2 [[buffer(5)]],
device const SplashKvPage *unused3 [[buffer(6)]],
constant SplashVerifyAttentionParams *params [[buffer(7)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 (void)unused1;(void)unused2;(void)unused3;if(group.z!=0)return;
 const auto tile=splash_verify_attention_tile<4,6,256>(params[0],group.x,0);
 if(!tile.active || group.y>=tile.splits)return;
 alignas(16) threadgroup float scores[48*32];
 alignas(16) threadgroup bfloat probabilities[48*32];
 threadgroup float maximum[48],sum[48],scale[48];
 threadgroup atomic_uint rescale[1];
 m4_flags_attention<4,6,8,256,0,bfloat,true,false>(
   queries+tile.offset,table,params[0].kv,group.x,tile.first_position,tile.active_rows,
   tile.splits,group.y,partials,statistics,tile.first_slot+group.y,scores,
   probabilities,maximum,sum,scale,rescale,tid);
}

kernel void m4_flags_verify_no_initial(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],
 device float *statistics [[buffer(2)]],device const SplashKvPage *table [[buffer(3)]],
 device const SplashKvPage *unused1 [[buffer(4)]],
device const SplashKvPage *unused2 [[buffer(5)]],
device const SplashKvPage *unused3 [[buffer(6)]],
constant SplashVerifyAttentionParams *params [[buffer(7)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 (void)unused1;(void)unused2;(void)unused3;if(group.z!=0)return;
 const auto tile=splash_verify_attention_tile<4,6,256>(params[0],group.x,0);
 if(!tile.active || group.y>=tile.splits)return;
 alignas(16) threadgroup float scores[48*32];
 alignas(16) threadgroup bfloat probabilities[48*32];
 threadgroup float maximum[48],sum[48],scale[48];
 threadgroup atomic_uint rescale[1];
 m4_flags_attention<4,6,8,256,0,bfloat,false,false>(
   queries+tile.offset,table,params[0].kv,group.x,tile.first_position,tile.active_rows,
   tile.splits,group.y,partials,statistics,tile.first_slot+group.y,scores,
   probabilities,maximum,sum,scale,rescale,tid);
}

kernel void m4_flags_verify_double(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],
 device float *statistics [[buffer(2)]],device const SplashKvPage *table [[buffer(3)]],
 device const SplashKvPage *unused1 [[buffer(4)]],
device const SplashKvPage *unused2 [[buffer(5)]],
device const SplashKvPage *unused3 [[buffer(6)]],
constant SplashVerifyAttentionParams *params [[buffer(7)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 (void)unused1;(void)unused2;(void)unused3;if(group.z!=0)return;
 const auto tile=splash_verify_attention_tile<4,6,256>(params[0],group.x,0);
 if(!tile.active || group.y>=tile.splits)return;
 alignas(16) threadgroup float scores[48*32];
 alignas(16) threadgroup bfloat probabilities[48*32];
 threadgroup float maximum[48],sum[48],scale[48];
 threadgroup atomic_uint rescale[2];
 m4_flags_attention<4,6,8,256,0,bfloat,true,true>(
   queries+tile.offset,table,params[0].kv,group.x,tile.first_position,tile.active_rows,
   tile.splits,group.y,partials,statistics,tile.first_slot+group.y,scores,
   probabilities,maximum,sum,scale,rescale,tid);
}

kernel void m4_flags_verify_double_no_initial(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],
 device float *statistics [[buffer(2)]],device const SplashKvPage *table [[buffer(3)]],
 device const SplashKvPage *unused1 [[buffer(4)]],
device const SplashKvPage *unused2 [[buffer(5)]],
device const SplashKvPage *unused3 [[buffer(6)]],
constant SplashVerifyAttentionParams *params [[buffer(7)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 (void)unused1;(void)unused2;(void)unused3;if(group.z!=0)return;
 const auto tile=splash_verify_attention_tile<4,6,256>(params[0],group.x,0);
 if(!tile.active || group.y>=tile.splits)return;
 alignas(16) threadgroup float scores[48*32];
 alignas(16) threadgroup bfloat probabilities[48*32];
 threadgroup float maximum[48],sum[48],scale[48];
 threadgroup atomic_uint rescale[2];
 m4_flags_attention<4,6,8,256,0,bfloat,false,true>(
   queries+tile.offset,table,params[0].kv,group.x,tile.first_position,tile.active_rows,
   tile.splits,group.y,partials,statistics,tile.first_slot+group.y,scores,
   probabilities,maximum,sum,scale,rescale,tid);
}

kernel void m4_flags_prefill_control(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],
 device float *statistics [[buffer(2)]],device const SplashKvPage *table [[buffer(3)]],
 constant SplashPrefillAttentionParams &params [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 
 const auto tile=splash_prefill_attention_tile<4,6,256>(params,group.x,group.y);
 if(!tile.active || group.z>=tile.splits)return;
 alignas(16) threadgroup float scores[48*32];
 alignas(16) threadgroup bfloat probabilities[48*32];
 threadgroup float maximum[48],sum[48],scale[48];
 threadgroup atomic_uint rescale[1];
 m4_flags_attention<4,6,8,256,0,bfloat,true,false>(
   queries+tile.offset,table,params.kv,group.x,tile.first_position,tile.active_rows,
   tile.splits,group.z,partials,statistics,tile.first_slot+group.z,scores,
   probabilities,maximum,sum,scale,rescale,tid);
}

kernel void m4_flags_prefill_no_initial(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],
 device float *statistics [[buffer(2)]],device const SplashKvPage *table [[buffer(3)]],
 constant SplashPrefillAttentionParams &params [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 
 const auto tile=splash_prefill_attention_tile<4,6,256>(params,group.x,group.y);
 if(!tile.active || group.z>=tile.splits)return;
 alignas(16) threadgroup float scores[48*32];
 alignas(16) threadgroup bfloat probabilities[48*32];
 threadgroup float maximum[48],sum[48],scale[48];
 threadgroup atomic_uint rescale[1];
 m4_flags_attention<4,6,8,256,0,bfloat,false,false>(
   queries+tile.offset,table,params.kv,group.x,tile.first_position,tile.active_rows,
   tile.splits,group.z,partials,statistics,tile.first_slot+group.z,scores,
   probabilities,maximum,sum,scale,rescale,tid);
}

kernel void m4_flags_prefill_double(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],
 device float *statistics [[buffer(2)]],device const SplashKvPage *table [[buffer(3)]],
 constant SplashPrefillAttentionParams &params [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 
 const auto tile=splash_prefill_attention_tile<4,6,256>(params,group.x,group.y);
 if(!tile.active || group.z>=tile.splits)return;
 alignas(16) threadgroup float scores[48*32];
 alignas(16) threadgroup bfloat probabilities[48*32];
 threadgroup float maximum[48],sum[48],scale[48];
 threadgroup atomic_uint rescale[2];
 m4_flags_attention<4,6,8,256,0,bfloat,true,true>(
   queries+tile.offset,table,params.kv,group.x,tile.first_position,tile.active_rows,
   tile.splits,group.z,partials,statistics,tile.first_slot+group.z,scores,
   probabilities,maximum,sum,scale,rescale,tid);
}

kernel void m4_flags_prefill_double_no_initial(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],
 device float *statistics [[buffer(2)]],device const SplashKvPage *table [[buffer(3)]],
 constant SplashPrefillAttentionParams &params [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 
 const auto tile=splash_prefill_attention_tile<4,6,256>(params,group.x,group.y);
 if(!tile.active || group.z>=tile.splits)return;
 alignas(16) threadgroup float scores[48*32];
 alignas(16) threadgroup bfloat probabilities[48*32];
 threadgroup float maximum[48],sum[48],scale[48];
 threadgroup atomic_uint rescale[2];
 m4_flags_attention<4,6,8,256,0,bfloat,false,true>(
   queries+tile.offset,table,params.kv,group.x,tile.first_position,tile.active_rows,
   tile.splits,group.z,partials,statistics,tile.first_slot+group.z,scores,
   probabilities,maximum,sum,scale,rescale,tid);
}
