#include <chrono>
#include <memory>
#include <random>
#include <arm_neon.h>
#include <sys/mman.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif

#define restrict __restrict__

using ub1 = uint8_t;
using ub2 = uint16_t;
using ub4 = uint32_t;
using ub8 = uint64_t;
using RowIndex_t = uint16_t;

template <ub8 N, typename F>
inline ALWAYS_INLINE void unroll(F&& f) {
    [&]<ub8... I>(std::index_sequence<I...>) ALWAYS_INLINE {
        (f(I), ...);
    }(std::make_index_sequence<N>{});
}

/**
 * Pack the dense 1-byte/row representation into a 1-bit/row bitset expected by bitsetToSelectionColset().
 * 
 * We **pad** the buffer to a mutiple of 16 bytes so that the SIMD path in a bitsetToSelectionColset can
 * safely load full 16-byte blocks even when rowcount < 128.
 * 
 * ultra slow;
 */
void denseBytesToBitVec(size_t size, const ub1* restrict denseBytes, ub1* restrict packedBitvec) {
    const size_t packedSize = (size + 7) / 8; // real size
    const size_t paddedPackedSize = ((packedSize + 15) / 16) * 16; // round-up to 16-byte blk
    // packedBitvec.resize(paddedPackedSize, 0u);
    
    for (size_t rowIdx = 0; rowIdx < size; ++rowIdx) {
        if (denseBytes[rowIdx] != 0) {
            packedBitvec[rowIdx >> 3] |= static_cast<uint8_t>(1u << (rowIdx & 7));
        }
    }
}

/**
 * Lookup table to convert a byte to its bit indices.
 */
alignas(uint8x8_t) static constexpr auto s_byteToBitIndices = []() {
    std::array<std::array<ub1, 8>, 256> ret{};
    for (ub4 i = 0; i < 256; ++i) {
        ub1 t = i;
        for (int j = 0; t != 0; ++j) {
            ret[i][j] = __builtin_ctz(t);
            t &= t - 1;
        }
    }
    return ret;
}();

/**
 * Decode a 64-bit integer into a selection vector using SIMD instructions.
 * 
 * @param in The 64-bit integer to decode.
 * @param selVector The selection vector to store the result.
 * @param countsSum The sum of the counts.
 * @param base The base value.
 */
template <ub8 n = 0, ub8 shift = 0>
static inline __attribute__((always_inline)) void decode64(
    const ub8& in,
    RowIndex_t* restrict const selVector,
    const ub8& countsPrefixSum,
    const uint16x8_t& base
) {
    if constexpr (n < 8) {
        const ub1 byte = static_cast<ub1>(in >> shift);
        uint8x8_t bitIndices = vld1_u8(s_byteToBitIndices[byte].data());
        uint16x8_t byteBase = vaddw_u8(base, vdup_n_u8(n * 8));
        uint16x8_t selVectorIndices = vaddw_u8(byteBase, bitIndices);
        // countsSum >> (shift - 8) cant be greater than 56 because we handle 64 bits at a time.
        const ub1 offset = n == 0 ? 0 : static_cast<ub1>(countsPrefixSum >> (shift - 8));
        // offset is already multiplied by 2 (line ub8 prefixSum = counts[i] * 0x0202020202020202;), therefore cast to ub1
        vst1q_u16(reinterpret_cast<RowIndex_t*>(reinterpret_cast<ub1*>(selVector) + offset), selVectorIndices);
        decode64<n + 1, shift + 8>(in, selVector, countsPrefixSum, base);
    }
}


/**
 * Convert dense bit-vector (one ) into sparse format (array of 16bit row indices). Caller is responsible for having
 a allocated the destination buffer large enough for <code>size</code> indices for subsequently calling setSelectedRowCount
 */
RowIndex_t bitsetToSelectionColset(size_t size, const ub1* restrict const bitset, ub2* restrict outSel) {
    uint16x8_t base = vdupq_n_u16(0);
    const auto* outSelStart = outSel;
    for (size_t j = 0; j < (size + 127) / 128; ++j) {
        uint8x16_t batch = vld1q_u8(bitset + j * sizeof(uint8x16_t));
        uint64x2_t popCount = vreinterpretq_u64_u8(vcntq_u8(batch));

        std::array<uint64_t, 2> counts = {vgetq_lane_u64(popCount, 0), vgetq_lane_u64(popCount, 1)};
        uint64x2_t batchU64x2 = vreinterpretq_u64_u8(batch);
        std::array<uint64_t, 2> batchU64 = {vgetq_lane_u64(batchU64x2, 0), vgetq_lane_u64(batchU64x2, 1)};
        for (ub8 i = 0; i < 2; ++i) {
            ub8 prefixSum = counts[i] * 0x0202020202020202;
            ub8 total = prefixSum >> 56;
            decode64(batchU64[i], outSel, prefixSum, base);
            outSel = reinterpret_cast<ub2*>(reinterpret_cast<ub1*>(outSel) + total);
            base = vaddw_u8(base, vdup_n_u8(64));
        }
    }
    return outSel - outSelStart;
}

template <ub8 stride, typename I, typename F>
inline ALWAYS_INLINE uint8x16_t compare16(I begin, F&& f) {
    static constexpr ub8 levels = __builtin_ctzll(16UL / stride);
    uint8x16_t intermediateResults[levels + 1];
    for (ub8 j = 0; j < 16 / stride; ++j) {
        ub8 offset = j * stride;
        auto cmp = reinterpret_cast<uint8x16_t>(f(begin + offset));
        ub8 level = __builtin_clzll(j + 1);
        for (ub8 k = 0; k < level; ++k) {
            cmp = vuzp1q_u8(intermediateResults[k], cmp);
        }
        intermediateResults[level] = cmp;
    }
    return intermediateResults[levels]; 
}


template <ub8 stride, bool intersect = false, typename I, typename F, typename G>
inline ALWAYS_INLINE void compareAll(I begin, I end, ub1* out, F&& f, G&& g) {
    static_assert(__builtin_popcountll(stride) == 1);
    static_assert(stride <= 16);
    if constexpr (stride > 1) {
        static constexpr ub1 maskArray[] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
        uint8x16_t mask = vld1q_u8(maskArray);
        ub8 len128 = ub8(end - begin) & ~127;
        const I end128 = begin + len128;
        uint8x16_t results[4];

        auto accumulate16 = [&](ub8 i) ALWAYS_INLINE {
            uint8x16_t intermediateResult = compare16<stride>(begin + i * 16, f);
            uint8x16_t result = vandq_u8(intermediateResult, mask);
            ub8 resultIndex = __builtin_ctzll(i + 1);
            for (ub8 j = 0; j < resultIndex; ++j) {
                result = vpaddq_u8(results[j], result);
            }
            results[resultIndex] = result;
        };

        auto store = [](ub1* out, uint8x16_t result) ALWAYS_INLINE {
            if constexpr(intersect) {
                uint8x16_t current = vld1q_u8(out);
                uint8x16_t intersection = vandq_u8(current, result);
                vst1q_u8(out, intersection);
            } else {
                vst1q_u8(out, result);
            }
        };

        //process 128 input elements at a time
        for (; begin < end128; begin += 128) {
            unroll<8>(accumulate16);
            store(out, results[3]);
            out += 16;
        }

        // process 16 input elements at a time
        // len16 is in the range [0-8)
        size_t len16 = ub8(end - end128) >> 4;
        if (len16 > 0) {
            for (ub8 i = 0; i < len16; ++i) {
                accumulate16(i);
            }

            // finish any remaining parallel adds and store last 16 bytes
            ub8 resultIndex = __builtin_ctzll(len16);
            uint8x16_t result = results[resultIndex];
            for (ub8 i = resultIndex; i < 3; ++i) {
                result = vpaddq_u8(result, vdupq_n_u8(0));
            }
            // size of out is required to be a multiple of 16 bytes, so using a 16 byte store is safe
            store(out, result);
            out += len16 * 2;
            begin += len16 * 16;
        }
    } else {
        ub8 len32 = ub8(end - begin) & ~31;
        const I end32 = begin + len32;
        for (; begin < end32; begin += 32) {
            ub4 current32 = 0;
            for (ub8 i = 0; i < 32; ++i) {
                bool cmp = g(begin +i);
                current32 |= ub4(cmp) << i;
            }
            if constexpr (intersect) {
                ub4 previous;
                std::memcpy(&previous, out, sizeof(previous));
                current32 &= previous;
            }
            std::memcpy(out, &current32, sizeof(current32));
            out += sizeof(current32);
        }
    }
    ub8 remaining = end - begin;
    if (remaining > 0) {
        ub4 last32 = 0;
        for (ub8 i = 0; i < remaining; ++i) {
            bool cmp = g(begin + i);
            last32 |= ub4(cmp) << i;
        }
        // size of out is required to be a multiple of 16 bytes, so uing a 4 byte store is safe
        if constexpr (intersect) {
            ub4 previous;
            std::memcpy(&previous, out, sizeof(previous));
            last32 &= previous;
        }
        std::memcpy(out, &last32, sizeof(last32));
    }
}

template <typename T, typename F>
inline ALWAYS_INLINE void constantComparison(const T* restrict in, ub1* restrict out, F&& f) {
    using V __attribute__((vector_size(16), aligned(alignof(T)))) = T;
    // using V = T __attribute__((vector_size(16), aligned(alignof(T))));

    compareAll<16 / sizeof(T)>(0UL, 4096UL, out, 
        // Vector path: load 16 elements from each input and apply predicate
        [&](auto i) { return f(reinterpret_cast<V>(vld1q_u8(reinterpret_cast<const ub1*>(in + i)))); },
        [&](auto i) { return f(in[i]); }
    );
}

template <typename T, typename F>
inline ALWAYS_INLINE  void binaryComparison(
    const T* restrict lhs,
    const T* restrict rhs,
    ub1* restrict out,
    F&& f
) {
    using V __attribute__((vector_size(16), aligned(alignof(T)))) = T;
    // using V = T __attribute__((vector_size(16), aligned(alignof(T))));

    compareAll<16 / sizeof(T)>(0UL, 4096UL, out,
        // Vector path: load 16 elements from each input and apply predicate
        [&](auto i) {
            auto v1 = reinterpret_cast<V>(vld1q_u8(reinterpret_cast<const ub1*>(lhs + i)));
            auto v2 = reinterpret_cast<V>(vld1q_u8(reinterpret_cast<const ub1*>(rhs + i)));
            return f(v1, v2);
        },
        [&](auto i) {
            return f(lhs[i], rhs[i]);
        }
    );
}

template <typename T>
inline ALWAYS_INLINE RowIndex_t binaryVecToSelectionColsetEqual(
    const T* restrict vec1,
    const T* restrict vec2,
    RowIndex_t size,
    RowIndex_t* restrict outSel,
    std::vector<ub1>& nullIndicatorBitset
) {
    std::vector<ub1> bitvec(size / 8);
    ub1* const out = bitvec.data();
    binaryComparison(vec1, vec2, out, [&](auto v1, auto v2) {
        return (v1 != v2) && (v2 != v2) || (v1 == v2);
    });
    const ub1* restrict nullBits = nullIndicatorBitset.data();
    const size_t bytes = static_cast<size_t>(size) / 8;
    const size_t blocks128 = bytes / 16;
    for (size_t i = 0; i < blocks128; ++i) {
        out[i] &= nullBits[i];
    }
    return bitsetToSelectionColset(size, out, outSel);
}

template <typename T>
inline ALWAYS_INLINE void boolVecToBitset(const T* restrict boolVec, ub1* restrict bitsetRepresenttation) {
    constantComparison(boolVec, bitsetRepresenttation, [&](auto v) {
        return v != 0;
    });
}

template <typename T>
inline ALWAYS_INLINE RowIndex_t boolVecToSelectionColset(
    const T* restrict boolVec,
    RowIndex_t* restrict outSel
) {
    std::vector<ub1> bitvec(4096 / 8);
    ub1* const out = bitvec.data();
    constantComparison(boolVec, out, [&](auto v) {
        return v != 0;
    });
    return bitsetToSelectionColset(4096, out, outSel);
}