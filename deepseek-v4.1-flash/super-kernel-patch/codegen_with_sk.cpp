#include <kernel_operator.h>
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>

// 原始code的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto ones = simd_inst::vdup<float>(float(0x1p+0f/*1.000000e+00*/), v, MODE_ZEROING);
  auto zeros = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
  for (int32_t vector = 0; vector < 36; ++vector) {
    vector_bool v_1 = asc_create_mask_b16(PAT_ALL);
    auto val_l = simd_inst::vcvt<float>(simd_inst::vlds_unpack<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64)) + 9216)])), 0), v_1, PART_EVEN, MODE_ZEROING);
    auto val_r = simd_inst::vcvt<float>(simd_inst::vlds_unpack<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64)) + 13824)])), 0), v_1, PART_EVEN, MODE_ZEROING);
    auto val_l_1 = simd_inst::vmins(val_l, float(0x1.4p+3f/*1.000000e+01*/), v, MODE_ZEROING);
    auto val_r_1 = simd_inst::vmaxs(simd_inst::vmins(val_r, float(0x1.4p+3f/*1.000000e+01*/), v, MODE_ZEROING), float(-0x1.4p+3f/*-1.000000e+01*/), v, MODE_ZEROING);
    auto val = simd_inst::vmul(simd_inst::vdiv_0ulp_ftz_true(val_l_1, simd_inst::vadd(simd_inst::vexpdif(zeros, val_l_1, v), ones, v, MODE_ZEROING), v, MODE_ZEROING), val_r_1, v, MODE_ZEROING);
    simd_inst::vsts_norm(val, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64))])), 0, v);
  }
}

// 原始code的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_1(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  simd_inst::vsts_norm(simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 128) + 10368)])), 0, v);
  vector_bool v_1 = asc_create_mask_b32(PAT_ALL);
  simd_inst::vsts_norm(simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v_1, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 128) + 10432)])), 0, v_1);
}

// 原始code的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_2(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  vector_bool mask_low = asc_create_mask_b32(PAT_VL32);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto mask_high = simd_inst::vcmps_gt(simd_inst::vci<int32_t>(0, INC_ORDER), 31, v);
  for (int32_t vector = 0; vector < 36; ++vector) {
    auto values = simd_inst::vabs(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64))])), 0), v, MODE_ZEROING);
    simd_inst::vsts_1st(simd_inst::vcmax(values, mask_low, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (vector * 2)) + 10368)])), 0, v);
    simd_inst::vsts_1st(simd_inst::vcmax(values, mask_high, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (vector * 2)) + 10369)])), 0, v);
  }
}

// 原始code的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_3(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  for (int32_t sf_batch = 0; sf_batch < 2; ++sf_batch) {
    vector_bool v = asc_create_mask_b32(PAT_ALL);
    auto amax = simd_inst::vmaxs(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (sf_batch * 64)) + 10368)])), 0), float(0x1.a36e2eb1c432dp-14f/*1.000000e-04*/), v, MODE_ZEROING);
    auto scale_raw = simd_inst::vmuls(amax, float(0x1.2492492492492p-9f/*2.232143e-03*/), v, MODE_ZEROING);
    auto scale_exp = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub((*(vector_u32 *)(&(scale_raw))), simd_inst::vdup<uint32_t>(1, v, MODE_ZEROING), v, MODE_ZEROING), 23, v, MODE_ZEROING), 1, v, MODE_ZEROING);
    auto inv_exp = simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v, MODE_ZEROING), scale_exp, v, MODE_ZEROING);
      vector_u32 v_ = simd_inst::vshls(inv_exp, 23, v, MODE_ZEROING);
    auto scale_inv = (*(vector_f32 *)(&(v_)));
    simd_inst::vsts_pack_quarter(scale_exp, (__ubuf__ uint32_t*)(&(buf_dyn_shmem[((((xl_ub_version_counter_1 & 1) * 128) + (sf_batch * 64)) + 43520)])), 0, v);
    simd_inst::vsts_norm(scale_inv, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (sf_batch * 64)) + 10624)])), 0, v);
  }
}

// 原始code的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_4(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  vector_bool mask_low = asc_create_mask_b32(PAT_VL32);
  for (int32_t vector = 0; vector < 36; ++vector) {
    auto inverse_low = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (vector * 2)) + 10624)])), 0);
    auto inverse_high = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (vector * 2)) + 10625)])), 0);
    auto inverse = simd_inst::vsel(inverse_low, inverse_high, mask_low);
    vector_bool v = asc_create_mask_b32(PAT_ALL);
    auto scaled = simd_inst::vmul(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64))])), 0), inverse, v, MODE_ZEROING);
    vector_bool v_1 = asc_create_mask_b8(PAT_ALL);
    simd_inst::vsts_pack_quarter(simd_inst::vcvt<fp8_e4_t>(scaled, v_1, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64)) + 36864)])), 0, v_1);
  }
}

// 原始code的simdvf代码
__simt_vf__ __launch_bounds__(72) inline void swiglu_forward_and_per_token_cast_kernel_kernel_simt_vf_0(int32_t token_id, int32_t num_expanded_tokens, __gm__ uint8_t* out_sf, __ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1, int32_t sf_stride) {
  if (0 <= token_id) {
    if (token_id < num_expanded_tokens) {
      out_sf[((((int64_t)token_id) * ((int64_t)sf_stride)) + ((int64_t)((int32_t)threadIdx.x)))] = buf_dyn_shmem[((((xl_ub_version_counter_1 & 1) * 128) + ((int32_t)threadIdx.x)) + 43520)];
    }
  }
}

// sk开启后新增适配sk的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto ones = simd_inst::vdup<float>(float(0x1p+0f/*1.000000e+00*/), v, MODE_ZEROING);
  auto zeros = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
  for (int32_t vector = 0; vector < 36; ++vector) {
    vector_bool v_1 = asc_create_mask_b16(PAT_ALL);
    auto val_l = simd_inst::vcvt<float>(simd_inst::vlds_unpack<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64)) + 9216)])), 0), v_1, PART_EVEN, MODE_ZEROING);
    auto val_r = simd_inst::vcvt<float>(simd_inst::vlds_unpack<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64)) + 13824)])), 0), v_1, PART_EVEN, MODE_ZEROING);
    auto val_l_1 = simd_inst::vmins(val_l, float(0x1.4p+3f/*1.000000e+01*/), v, MODE_ZEROING);
    auto val_r_1 = simd_inst::vmaxs(simd_inst::vmins(val_r, float(0x1.4p+3f/*1.000000e+01*/), v, MODE_ZEROING), float(-0x1.4p+3f/*-1.000000e+01*/), v, MODE_ZEROING);
    auto val = simd_inst::vmul(simd_inst::vdiv_0ulp_ftz_true(val_l_1, simd_inst::vadd(simd_inst::vexpdif(zeros, val_l_1, v), ones, v, MODE_ZEROING), v, MODE_ZEROING), val_r_1, v, MODE_ZEROING);
    simd_inst::vsts_norm(val, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64))])), 0, v);
  }
}

// sk开启后新增适配sk的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_1(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  simd_inst::vsts_norm(simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 128) + 10368)])), 0, v);
  vector_bool v_1 = asc_create_mask_b32(PAT_ALL);
  simd_inst::vsts_norm(simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v_1, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 128) + 10432)])), 0, v_1);
}

// sk开启后新增适配sk的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_2(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  vector_bool mask_low = asc_create_mask_b32(PAT_VL32);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto mask_high = simd_inst::vcmps_gt(simd_inst::vci<int32_t>(0, INC_ORDER), 31, v);
  for (int32_t vector = 0; vector < 36; ++vector) {
    auto values = simd_inst::vabs(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64))])), 0), v, MODE_ZEROING);
    simd_inst::vsts_1st(simd_inst::vcmax(values, mask_low, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (vector * 2)) + 10368)])), 0, v);
    simd_inst::vsts_1st(simd_inst::vcmax(values, mask_high, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (vector * 2)) + 10369)])), 0, v);
  }
}

// sk开启后新增适配sk的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_3(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  for (int32_t sf_batch = 0; sf_batch < 2; ++sf_batch) {
    vector_bool v = asc_create_mask_b32(PAT_ALL);
    auto amax = simd_inst::vmaxs(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (sf_batch * 64)) + 10368)])), 0), float(0x1.a36e2eb1c432dp-14f/*1.000000e-04*/), v, MODE_ZEROING);
    auto scale_raw = simd_inst::vmuls(amax, float(0x1.2492492492492p-9f/*2.232143e-03*/), v, MODE_ZEROING);
    auto scale_exp = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub((*(vector_u32 *)(&(scale_raw))), simd_inst::vdup<uint32_t>(1, v, MODE_ZEROING), v, MODE_ZEROING), 23, v, MODE_ZEROING), 1, v, MODE_ZEROING);
    auto inv_exp = simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v, MODE_ZEROING), scale_exp, v, MODE_ZEROING);
      vector_u32 v_ = simd_inst::vshls(inv_exp, 23, v, MODE_ZEROING);
    auto scale_inv = (*(vector_f32 *)(&(v_)));
    simd_inst::vsts_pack_quarter(scale_exp, (__ubuf__ uint32_t*)(&(buf_dyn_shmem[((((xl_ub_version_counter_1 & 1) * 128) + (sf_batch * 64)) + 43520)])), 0, v);
    simd_inst::vsts_norm(scale_inv, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (sf_batch * 64)) + 10624)])), 0, v);
  }
}

// sk开启后新增适配sk的simdvf代码
__simd_vf__ inline void swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_4(__ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1) {
  vector_bool mask_low = asc_create_mask_b32(PAT_VL32);
  for (int32_t vector = 0; vector < 36; ++vector) {
    auto inverse_low = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (vector * 2)) + 10624)])), 0);
    auto inverse_high = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 128) + (vector * 2)) + 10625)])), 0);
    auto inverse = simd_inst::vsel(inverse_low, inverse_high, mask_low);
    vector_bool v = asc_create_mask_b32(PAT_ALL);
    auto scaled = simd_inst::vmul(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64))])), 0), inverse, v, MODE_ZEROING);
    vector_bool v_1 = asc_create_mask_b8(PAT_ALL);
    simd_inst::vsts_pack_quarter(simd_inst::vcvt<fp8_e4_t>(scaled, v_1, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((((xl_ub_version_counter_1 & 1) * 2304) + (vector * 64)) + 36864)])), 0, v_1);
  }
}

// sk开启后新增适配sk的simdvf代码
__simt_vf__ __launch_bounds__(72) inline void swiglu_forward_and_per_token_cast_kernel_kernel_sk_simt_vf_0(int32_t token_id, int32_t num_expanded_tokens, __gm__ uint8_t* out_sf, __ubuf__ uint8_t* buf_dyn_shmem, int32_t xl_ub_version_counter_1, int32_t sf_stride) {
  if (0 <= token_id) {
    if (token_id < num_expanded_tokens) {
      out_sf[((((int64_t)token_id) * ((int64_t)sf_stride)) + ((int64_t)((int32_t)threadIdx.x)))] = buf_dyn_shmem[((((xl_ub_version_counter_1 & 1) * 128) + ((int32_t)threadIdx.x)) + 43520)];
    }
  }
}

// 原始codegen的global 入口
extern "C" __global__ __vector__ void swiglu_forward_and_per_token_cast_kernel_kernel(__gm__ fp8_e4_t* out, __gm__ uint8_t* out_sf, __gm__ int32_t* psum_num_tokens_per_expert, __gm__ bfloat16_t* x, int32_t num_expanded_tokens, int32_t sf_stride) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t expert_id = 0;
  int32_t token_id = 0;
  int8_t valid = (int8_t)0;
  float weight = float(0x0p+0f/*0.000000e+00*/);
  int32_t xl_ub_version_counter_1 = 0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  xl_ub_version_counter_1 = 0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ int32_t*)buf_dyn_shmem)[10944]))), (__gm__ uint8_t*)((&(psum_num_tokens_per_expert[0]))), 1, 192, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 192, 192);
  asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
  expert_id = 0;
  asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
  for (int32_t w = 0; w < ((num_expanded_tokens + 55) / 56); ++w) {
    if (((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) {
      token_id = ((w * 56) + ((int32_t)block_idx));
      while (1) {
        bool __cond_0 = ((48 <= expert_id) || (token_id < ((__ubuf__ int32_t*)buf_dyn_shmem)[(expert_id + 10944)]));
        if (__cond_0) {
          break;
        }
        expert_id = (expert_id + 1);
      }
      valid = (int8_t)1;
      valid = ((int8_t)((expert_id < 48) && (token_id < ((__ubuf__ int32_t*)buf_dyn_shmem)[(expert_id + 10944)])));
    }
    int8_t __cond_1 = ((int8_t)(bool)0);
    if (((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) {
      __cond_1 = ((int8_t)((bool)valid));
      if ((bool)__cond_1) {
        weight = float(0x1p+0f/*1.000000e+00*/);
      }
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        if (0 < (num_expanded_tokens - token_id)) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + 9216)]))), (__gm__ uint8_t*)((&(x[(((int64_t)token_id) * (int64_t)4608)]))), 1, 4608, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4608, 4608);
        }
      }
      if ((bool)__cond_1) {
        if (0 < (num_expanded_tokens - token_id)) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + 13824)]))), (__gm__ uint8_t*)((&(x[((((int64_t)token_id) * (int64_t)4608) + (int64_t)2304)]))), 1, 4608, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4608, 4608);
        }
      }
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_0(buf_dyn_shmem, xl_ub_version_counter_1);
      }
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_1(buf_dyn_shmem, xl_ub_version_counter_1);
      }
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_2(buf_dyn_shmem, xl_ub_version_counter_1);
      }
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_3(buf_dyn_shmem, xl_ub_version_counter_1);
      }
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_simd_vf_4(buf_dyn_shmem, xl_ub_version_counter_1);
      }
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        asc_vf_call<swiglu_forward_and_per_token_cast_kernel_kernel_simt_vf_0>(cce::dim3(72), token_id, num_expanded_tokens, out_sf, buf_dyn_shmem, xl_ub_version_counter_1, sf_stride);
      }
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_pipe(PIPE_MTE3);
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        if (0 < (num_expanded_tokens - token_id)) {
          asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out[(((int64_t)token_id) * (int64_t)2304)]))), (__ubuf__ uint8_t*)((&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + 36864)]))), 1, 2304, static_cast<asc_store_l2_cache_mode>(4), 2304, 2304);
        }
      }
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if ((((w * 56) + ((int32_t)block_idx)) < num_expanded_tokens) && ((bool)__cond_1)) {
      xl_ub_version_counter_1 = (xl_ub_version_counter_1 + 1);
    }
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
}

// 为sk入口准备的参数定义
struct swiglu_forward_and_per_token_cast_kernel_kernel_sk_args {
  __gm__ fp8_e4_t* arg0;
  __gm__ uint8_t* arg1;
  __gm__ int32_t* arg2;
  __gm__ bfloat16_t* arg3;
  int32_t arg4;
  int32_t arg5;
};

// sk对应的入口函数
template<uint32_t tl_sk_variant>
__sk__ __vector__ void swiglu_forward_and_per_token_cast_kernel_kernel_sk(const swiglu_forward_and_per_token_cast_kernel_kernel_sk_args* tl_sk_args, sk::SkSystemArgs* tl_sk_sys) {
  asc_init();
  tl_sk_sys->SkInit();
  const int32_t tl_sk_num_blocks = tl_sk_sys->SkGetNumBlocks();
  const int32_t tl_sk_block_idx = tl_sk_sys->SkGetBlockIdx();
  __gm__ fp8_e4_t* out = tl_sk_args->arg0;
  __gm__ uint8_t* out_sf = tl_sk_args->arg1;
  __gm__ int32_t* psum_num_tokens_per_expert = tl_sk_args->arg2;
  __gm__ bfloat16_t* x = tl_sk_args->arg3;
  int32_t num_expanded_tokens = tl_sk_args->arg4;
  int32_t sf_stride = tl_sk_args->arg5;
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t expert_id = 0;
  int32_t token_id = 0;
  int8_t valid = (int8_t)0;
  float weight = float(0x0p+0f/*0.000000e+00*/);
  int32_t xl_ub_version_counter_1 = 0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  xl_ub_version_counter_1 = 0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ int32_t*)buf_dyn_shmem)[10944]))), (__gm__ uint8_t*)((&(psum_num_tokens_per_expert[0]))), 1, 192, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 192, 192);
  asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
  expert_id = 0;
  asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
  for (int32_t w = 0; w < ((num_expanded_tokens + 55) / 56); ++w) {
    if (((w * 56) + tl_sk_block_idx) < num_expanded_tokens) {
      token_id = ((w * 56) + tl_sk_block_idx);
      while (1) {
        bool __cond_0 = ((48 <= expert_id) || (token_id < ((__ubuf__ int32_t*)buf_dyn_shmem)[(expert_id + 10944)]));
        if (__cond_0) {
          break;
        }
        expert_id = (expert_id + 1);
      }
      valid = (int8_t)1;
      valid = ((int8_t)((expert_id < 48) && (token_id < ((__ubuf__ int32_t*)buf_dyn_shmem)[(expert_id + 10944)])));
    }
    int8_t __cond_1 = ((int8_t)(bool)0);
    if (((w * 56) + tl_sk_block_idx) < num_expanded_tokens) {
      __cond_1 = ((int8_t)((bool)valid));
      if ((bool)__cond_1) {
        weight = float(0x1p+0f/*1.000000e+00*/);
      }
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + tl_sk_block_idx) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        if (0 < (num_expanded_tokens - token_id)) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + 9216)]))), (__gm__ uint8_t*)((&(x[(((int64_t)token_id) * (int64_t)4608)]))), 1, 4608, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4608, 4608);
        }
      }
      if ((bool)__cond_1) {
        if (0 < (num_expanded_tokens - token_id)) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + 13824)]))), (__gm__ uint8_t*)((&(x[((((int64_t)token_id) * (int64_t)4608) + (int64_t)2304)]))), 1, 4608, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4608, 4608);
        }
      }
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + tl_sk_block_idx) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_0(buf_dyn_shmem, xl_ub_version_counter_1);
      }
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + tl_sk_block_idx) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_1(buf_dyn_shmem, xl_ub_version_counter_1);
      }
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_2(buf_dyn_shmem, xl_ub_version_counter_1);
      }
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_3(buf_dyn_shmem, xl_ub_version_counter_1);
      }
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + tl_sk_block_idx) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        swiglu_forward_and_per_token_cast_kernel_kernel_sk_simd_vf_4(buf_dyn_shmem, xl_ub_version_counter_1);
      }
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + tl_sk_block_idx) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        asc_vf_call<swiglu_forward_and_per_token_cast_kernel_kernel_sk_simt_vf_0>(cce::dim3(72), token_id, num_expanded_tokens, out_sf, buf_dyn_shmem, xl_ub_version_counter_1, sf_stride);
      }
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_pipe(PIPE_MTE3);
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if (((w * 56) + tl_sk_block_idx) < num_expanded_tokens) {
      if ((bool)__cond_1) {
        if (0 < (num_expanded_tokens - token_id)) {
          asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out[(((int64_t)token_id) * (int64_t)2304)]))), (__ubuf__ uint8_t*)((&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((xl_ub_version_counter_1 & 1) * 2304) + 36864)]))), 1, 2304, static_cast<asc_store_l2_cache_mode>(4), 2304, 2304);
        }
      }
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((xl_ub_version_counter_1 & 1)));
    }
    if ((((w * 56) + tl_sk_block_idx) < num_expanded_tokens) && ((bool)__cond_1)) {
      xl_ub_version_counter_1 = (xl_ub_version_counter_1 + 1);
    }
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
}

// 将global入口与sk入口进行映射
SK_BIND(swiglu_forward_and_per_token_cast_kernel_kernel, 0ULL, swiglu_forward_and_per_token_cast_kernel_kernel_sk<0>, swiglu_forward_and_per_token_cast_kernel_kernel_sk<1>, swiglu_forward_and_per_token_cast_kernel_kernel_sk<2>, swiglu_forward_and_per_token_cast_kernel_kernel_sk<3>);

