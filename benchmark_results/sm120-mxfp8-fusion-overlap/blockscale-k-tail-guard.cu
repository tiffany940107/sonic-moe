// Logical-bounds regression for block-scale MXFP8 MoE K-tail scale stores.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cute_sm12x_gemm/sm120_blockscaled/quantize_mxfp8_for_moe.cuh"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error = (call);                                                  \
    if (error != cudaSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(error));                                   \
      return 2;                                                                  \
    }                                                                            \
  } while (0)

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s {old|new} SOURCE_COMMIT\n", argv[0]);
    return 2;
  }
  const std::string variant = argv[1];
  const std::string source_commit = argv[2];
  if (variant != "old" && variant != "new") {
    std::fprintf(stderr, "variant must be old or new\n");
    return 2;
  }

  constexpr int kGranK = 32;
  constexpr int kSizeK = 1280;
  constexpr int kNumExperts = 6;
  constexpr int kTokenNum = 902;
  constexpr int kWarpsPerBlock = 4;
  constexpr int kElementsPerWarp = 512;
  constexpr int kPackedScalesPerInt32 = 4;
  constexpr int kScalePackK = kGranK * kPackedScalesPerInt32;
  constexpr int kValidScaleRows = (kSizeK + kScalePackK - 1) / kScalePackK;
  constexpr int kInt32PerWarp = 4;
  constexpr int kLaunchedScaleRows =
      ((kSizeK + kElementsPerWarp - 1) / kElementsPerWarp) * kInt32PerWarp;
  static_assert(kValidScaleRows == 10);
  static_assert(kLaunchedScaleRows == 12);

  const int host_offsets[kNumExperts + 1] = {0, 0, 1, 128, 256, 385, 902};
  const int64_t scale_leading_dim =
      ((kTokenNum + kNumExperts * 3) / 4) * 4;
  constexpr uint32_t kCanary = 0x5a5a5a5aU;

  __nv_bfloat16* input = nullptr;
  __nv_fp8_e4m3* output = nullptr;
  int32_t* offsets = nullptr;
  int32_t* scales = nullptr;
  CUDA_CHECK(cudaMalloc(&input, sizeof(__nv_bfloat16) * kTokenNum * kSizeK));
  CUDA_CHECK(cudaMalloc(&output, sizeof(__nv_fp8_e4m3) * kTokenNum * kSizeK));
  CUDA_CHECK(cudaMalloc(&offsets, sizeof(host_offsets)));
  CUDA_CHECK(cudaMalloc(
      &scales, sizeof(int32_t) * kLaunchedScaleRows * scale_leading_dim));
  CUDA_CHECK(cudaMemset(input, 0, sizeof(__nv_bfloat16) * kTokenNum * kSizeK));
  CUDA_CHECK(cudaMemset(output, 0, sizeof(__nv_fp8_e4m3) * kTokenNum * kSizeK));
  CUDA_CHECK(cudaMemset(
      scales, 0x5a,
      sizeof(int32_t) * kLaunchedScaleRows * scale_leading_dim));
  CUDA_CHECK(cudaMemcpy(offsets, host_offsets, sizeof(host_offsets),
                        cudaMemcpyHostToDevice));

  using Kernel = decltype(
      cute_sm12x_gemm::sm120_blockscaled::
          quantize_mxfp8_for_moe_kernel_sm120<
              kGranK, __nv_bfloat16, __nv_fp8_e4m3, kWarpsPerBlock>);
  auto kernel =
      cute_sm12x_gemm::sm120_blockscaled::
          quantize_mxfp8_for_moe_kernel_sm120<
              kGranK, __nv_bfloat16, __nv_fp8_e4m3, kWarpsPerBlock>;
  (void)sizeof(Kernel*);
  const int num_k_blocks =
      (kSizeK + kElementsPerWarp - 1) / kElementsPerWarp;
  const int num_token_blocks =
      (kTokenNum + kWarpsPerBlock - 1) / kWarpsPerBlock;
  dim3 grid(num_k_blocks, num_token_blocks);
  dim3 block(kWarpsPerBlock * 32);
  const int smem_size = (kNumExperts + 1) * sizeof(int32_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  kernel<<<grid, block, smem_size>>>(output, scales, input, offsets,
                                     kNumExperts, kSizeK,
                                     scale_leading_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  const int guard_rows = kLaunchedScaleRows - kValidScaleRows;
  std::vector<uint32_t> guard(
      static_cast<size_t>(guard_rows) * scale_leading_dim);
  CUDA_CHECK(cudaMemcpy(
      guard.data(), scales + kValidScaleRows * scale_leading_dim,
      sizeof(uint32_t) * guard.size(), cudaMemcpyDeviceToHost));
  size_t guard_bad_int32 = 0;
  for (uint32_t value : guard) {
    guard_bad_int32 += value != kCanary;
  }

  const bool expected =
      (variant == "old" && guard_bad_int32 > 0) ||
      (variant == "new" && guard_bad_int32 == 0);
  const char* status = variant == "old" ? "regression_reproduced" : "fixed";
  std::printf(
      "{\"benchmark\":\"blockscale_mxfp8_moe_k1280_tail_guard\","
      "\"source_commit\":\"%s\",\"variant\":\"%s\","
      "\"k\":%d,\"gran_k\":%d,\"valid_scale_rows\":%d,"
      "\"launched_scale_rows\":%d,\"guard_bad_int32\":%zu,"
      "\"expected_observation\":%s,\"status\":\"%s\"}\n",
      source_commit.c_str(), variant.c_str(), kSizeK, kGranK,
      kValidScaleRows, kLaunchedScaleRows, guard_bad_int32,
      expected ? "true" : "false", status);

  cudaFree(scales);
  cudaFree(offsets);
  cudaFree(output);
  cudaFree(input);
  return expected ? 0 : 1;
}
