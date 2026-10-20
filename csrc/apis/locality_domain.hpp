#pragma once

#include <c10/util/accumulate.h>
#include <paddle/phi/api/include/api.h>
#include <torch/python.h>

#include "../jit_kernels/impls/sm100_locality_domain.hpp"
#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/layout.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::locality_domain {

/// Localized allocations

static int get_num_locality_domains() {
    return LocalityDomainAllocator::get_num_locality_domains();
}

static size_t get_granularity() {
    return LocalityDomainAllocator::get_granularity();
}

static bool is_localization_available() {
    return mlopart::is_available();
}

static void release(void* ptr) {
    locality_domain_allocator.release(ptr);
}

// NOTES: Paddle has no pybind11 caster for `at::IntArrayRef`, so Python-facing sizes are taken as `std::vector<int64_t>`
// `torch.empty(sizes, dtype=dtype)` in the locality domain `domain_idx`
static torch::Tensor empty(const std::vector<int64_t>& sizes, const torch::ScalarType& dtype, const int& domain_idx) {
    auto& allocator = locality_domain_allocator;
    const auto num_bytes = align<size_t>(c10::multiply_integers(sizes) * c10::elementSize(dtype), allocator.get_granularity());
    const auto ptr = allocator.reserve(num_bytes);
    allocator.map(ptr, num_bytes, domain_idx);
    return torch::from_blob(ptr, sizes, release, torch::dtype(dtype).device(torch::kCUDA));
}

// `torch.empty((num_domains, *sizes), dtype=dtype)`, with slice `d` in domain `d`
static torch::Tensor empty_per_domain(const std::vector<int64_t>& sizes, const torch::ScalarType& dtype) {
    auto& allocator = locality_domain_allocator;
    const int64_t elem_size = static_cast<int64_t>(c10::elementSize(dtype)), num_domains = get_num_locality_domains();
    const int64_t num_slice_bytes = align<int64_t>(c10::multiply_integers(sizes) * elem_size, static_cast<int64_t>(allocator.get_granularity()));
    const auto ptr = static_cast<char*>(allocator.reserve(num_slice_bytes * num_domains));
    for (int d = 0; d < num_domains; ++ d)
        allocator.map(ptr + d * num_slice_bytes, num_slice_bytes, d);

    std::vector<int64_t> full_sizes = {num_domains}, strides = {num_slice_bytes / elem_size};
    full_sizes.insert(full_sizes.end(), sizes.begin(), sizes.end());
    const auto slice_strides = get_contiguous_strides(sizes);
    strides.insert(strides.end(), slice_strides.begin(), slice_strides.end());
    return torch::from_blob(ptr, full_sizes, strides, release, torch::dtype(dtype).device(torch::kCUDA));
}

/// Locality domain of the SMs

static constexpr int kNumSMsPerTPC = 2;
static constexpr int kNumLinesPerChunk = kNumLocalityDomainProbeHops;
static constexpr int kNumWordsPerLine = kNumLocalityDomainProbeLineBytes / sizeof(int);
static constexpr int kNumProbeChunksPerSM = 8;
// Coprime with the line count, so that the chain visits every line
static constexpr int kChainLineStride = 7;
static constexpr float kMinFarNearRatio = 1.25f;
static constexpr int kNumMaxProbeAttempts = 5;

static void flush_l2() {
    torch::zeros({256 << 20}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
}

// One chunk of word indices: word 0 of line `l` points to line `(l + kChainLineStride) % kNumLinesPerChunk`
static torch::Tensor get_probe_chain() {
    const auto options = torch::dtype(torch::kInt).device(torch::kCUDA);
    const auto chain = torch::zeros({kNumLinesPerChunk, kNumWordsPerLine}, options);
    const auto lines = torch::arange(kNumLinesPerChunk, options);
    const auto shifted = paddle::experimental::scale(lines._PD_GetInner(), 1, kChainLineStride);
    const auto wrapped = paddle::experimental::remainder(
        shifted, paddle::experimental::full_like(shifted, kNumLinesPerChunk));
    chain.select(1, 0).copy_(torch::Tensor(paddle::experimental::scale(wrapped, kNumWordsPerLine)));
    return chain.view(-1);
}

// `(num_domains, num_sms, kNumProbeChunksPerSM, kNumLocalityDomainProbeChunkBytes)` -> `(num_domains, num_sms)`
static torch::Tensor probe(const torch::Tensor& buf) {
    buf.view(torch::kInt).view({buf.size(0), -1, kNumLocalityDomainProbeChunkBytes / 4}).copy_(get_probe_chain());
    const auto latency = torch::zeros({buf.size(0), buf.size(1), buf.size(2)}, buf.options().dtype(torch::kInt16));
    flush_l2();
    for (int d = 0; d < buf.size(0); ++ d)
        sm100_locality_domain_probe_chase(buf[d], latency[d]);
    const torch::Tensor positive = paddle::experimental::greater_than(
        latency._PD_GetInner(), paddle::experimental::full_like(latency._PD_GetInner(), 0));
    DG_HOST_ASSERT(positive.all().item<bool>());
    // PyTorch selects the lower middle value for an even number of samples.
    return std::get<0>(paddle::experimental::median(
        latency.to(torch::kFloat)._PD_GetInner(), {-1}, false, "min"));
}

// Make an attempt
static torch::Tensor try_probe_sm_locality_domains(const torch::Tensor& buf) {
    const auto sorted = paddle::experimental::argsort(probe(buf)._PD_GetInner(), 0);
    const torch::Tensor latency = std::get<0>(sorted), order = std::get<1>(sorted);
    const auto sm_domain = order[0].to(torch::kUInt8);
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});
    const torch::Tensor separated = paddle::experimental::greater_equal(
        latency[1]._PD_GetInner(), paddle::experimental::scale(latency[0]._PD_GetInner(), kMinFarNearRatio));
    const auto clear = separated.all().item<bool>();
    const auto consistent = torch::equal(tpcs, tpcs.select(1, 0).unsqueeze(1).expand_as(tpcs));
    return clear and consistent ? sm_domain : torch::Tensor();
}

// Without localization, an arbitrary SM mapping is functionally correct
static torch::Tensor get_even_sm_locality_domains() {
    const auto tpc_idx = torch::arange(jit->device.get_num_sms() / kNumSMsPerTPC, torch::dtype(torch::kInt).device(torch::kCUDA));
    const torch::Tensor domains = paddle::experimental::remainder(
        tpc_idx._PD_GetInner(), paddle::experimental::full_like(tpc_idx._PD_GetInner(), get_num_locality_domains()));
    const torch::Tensor sm_domains = paddle::experimental::repeat_interleave(domains._PD_GetInner(), kNumSMsPerTPC, 0);
    return sm_domains.to(torch::kUInt8);
}

// Probe multiple times; fallback when unstable
static torch::Tensor probe_sm_locality_domains() {
    const auto num_sms = jit->device.get_num_sms();
    const auto buf = empty_per_domain({num_sms, kNumProbeChunksPerSM, kNumLocalityDomainProbeChunkBytes}, torch::kUInt8);
    for (int attempt = 0; attempt < kNumMaxProbeAttempts; ++ attempt) {
        if (const auto sm_domain = try_probe_sm_locality_domains(buf); sm_domain.defined())
            return sm_domain;
    }
    pybind11::module_::import("warnings").attr("warn")("The SM locality domain probe found no stable mapping; using an even one");
    return get_even_sm_locality_domains();
}

static torch::Tensor get_sm_locality_domains() {
    if (not runtime->sm_locality_domains.defined())
        runtime->sm_locality_domains = is_localization_available() ? probe_sm_locality_domains() : get_even_sm_locality_domains();
    return runtime->sm_locality_domains;
}

static torch::Tensor balance_sm_locality_domains(const torch::Tensor& sm_domain) {
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});
    const auto tpc_domain = tpcs.select(1, 0);
    DG_HOST_ASSERT(tpcs.size(0) % 2 == 0);
    const int64_t num_tpcs = tpcs.size(0), num_domain_1 = tpc_domain.sum().item<int64_t>();
    const uint8_t larger = num_domain_1 * 2 > num_tpcs;
    const auto majority = paddle::experimental::equal(
        tpc_domain._PD_GetInner(), paddle::experimental::full_like(tpc_domain._PD_GetInner(), larger));
    const torch::Tensor indices = paddle::experimental::nonzero(majority);
    const auto surplus = indices.flatten(0, -1).slice(0, num_tpcs / 2, indices.numel());
    const auto balanced = sm_domain.clone();
    auto balanced_tpcs = balanced.view({-1, kNumSMsPerTPC});
    paddle::experimental::index_fill_(balanced_tpcs._PD_GetInner(), surplus._PD_GetInner(), 0, larger ^ 1);
    return balanced;
}

static torch::Tensor get_balanced_sm_locality_domains() {
    if (not runtime->balanced_sm_locality_domains.defined())
        runtime->balanced_sm_locality_domains = balance_sm_locality_domains(get_sm_locality_domains());
    return runtime->balanced_sm_locality_domains;
}

static void release_mlopart() {
    get_balanced_sm_locality_domains();
    mlopart::release();
}

static void register_apis(pybind11::module_& m) {
    auto submodule = m.def_submodule("locality_domain", "Memory homed in the GPU's locality domains");
    submodule.def("get_num_locality_domains", &get_num_locality_domains);
    submodule.def("get_granularity", &get_granularity);
    submodule.def("is_localization_available", &is_localization_available);
    submodule.def("empty", &empty, pybind11::arg("sizes"), pybind11::arg("dtype"), pybind11::arg("domain_idx"));
    submodule.def("empty_per_domain", &empty_per_domain, pybind11::arg("sizes"), pybind11::arg("dtype"));
    submodule.def("is_localized", &is_localized);
    submodule.def("get_sm_locality_domains", &get_sm_locality_domains);
    submodule.def("get_balanced_sm_locality_domains", &get_balanced_sm_locality_domains);
    submodule.def("release_mlopart", &release_mlopart);
}

}  // namespace deep_gemm::locality_domain
