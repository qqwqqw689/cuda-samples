/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mergeSort_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

// data: Pointer to an array of unsigned integers.
// N: The number of elements in the array.
static void checkOrder(uint *data, uint N, uint sortDir) {
  if (N <= 1) {
    return;
  }

  for (uint i = 0; i < N - 1; i++)
    if ((sortDir && (data[i] > data[i + 1])) ||
        (!sortDir && (data[i] < data[i + 1]))) {
      fprintf(stderr, "checkOrder() failed!!!\n");
      exit(EXIT_FAILURE);
    }
}

static uint umin(uint a, uint b) { return (a <= b) ? a : b; }

static uint getSampleCount(uint dividend) {
  return ((dividend % SAMPLE_STRIDE) != 0) ? (dividend / SAMPLE_STRIDE + 1)
                                           : (dividend / SAMPLE_STRIDE);
}

// see
// https://gist.github.com/qqwqqw689/88bb8dc964ba923de3cddb1e50fb7256
static uint nextPowerOfTwo(uint x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}


// val: The value to be inserted.
// data: The array in which the position is to be found. 
// L: The length of the array data.
// sortDir: The sorting direction (1 for ascending, 0 for descending).
static uint binarySearchInclusive(uint val, uint *data, uint L, uint sortDir) {
  if (L == 0) {
    return 0;
  }

  uint pos = 0;

  for (uint stride = nextPowerOfTwo(L); stride > 0; stride >>= 1) {
    uint newPos = umin(pos + stride, L);

    if ((sortDir && (data[newPos - 1] <= val)) ||
        (!sortDir && (data[newPos - 1] >= val))) {
      pos = newPos;
    }
  }

  return pos;
}

// find the position in an array where a given value val should
// be inserted to maintain the order, excluding elements equal to val

// val: The value to be inserted.
// data: The array in which the position is to be found.
// L: The length of the array.
// sortDir: The sorting direction (1 for ascending, 0 for descending).
static uint binarySearchExclusive(uint val, uint *data, uint L, uint sortDir) {
  if (L == 0) {
    return 0;
  }

  uint pos = 0;

  for (uint stride = nextPowerOfTwo(L); stride > 0; stride >>= 1) {
    uint newPos = umin(pos + stride, L);

    if ((sortDir && (data[newPos - 1] < val)) ||
        (!sortDir && (data[newPos - 1] > val))) {
      pos = newPos;
    }
  }

  return pos;
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 1: find sample ranks in each segment
////////////////////////////////////////////////////////////////////////////////

// ranksA - An array to store the ranks of samples in the first part of each segment.
// ranksB - An array to store the ranks of samples in the second part of each segment.
// srcKey - The array of keys (values) to be sorted.
// stride - The size of each segment being processed.
// N - The total number of elements in the array being sorted.
// sortDir - A flag indicating the sorting direction.
static void generateSampleRanks(uint *ranksA, uint *ranksB, uint *srcKey,
                                uint stride, uint N, uint sortDir) {
  uint lastSegmentElements = N % (2 * stride);
  uint sampleCount =
      (lastSegmentElements > stride)
          ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE)
          : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  for (uint pos = 0; pos < sampleCount; pos++) {
    const uint i = pos & ((stride / SAMPLE_STRIDE) - 1);
    // This line calculates the offset i within a segment.
    // The length of a segemnt is 2*stride, and 2*stride / 2*SAMPLE_STRIDE == 
    // stride / SAMPLE_STRIDE
    // SAMPLE_STRIDE - 128
    // SHARED_SIZE_LIMIT - 1024
    // stride is 1*SHARED_SIZE_LIMIT, 2*SHARED_SIZE_LIMIT, 4*SHARED_SIZE_LIMIT, 8* ......
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    // segmentBase points to the start of the pair of segments that
    // contain the current sample position.

    const uint lenA = stride;
    const uint lenB = umin(stride, N - segmentBase - stride);
    // lenA and lenB define the lengths of the two halves of the segment.
    const uint nA = stride / SAMPLE_STRIDE;
    const uint nB = getSampleCount(lenB);
    // nA and nB determine the number of samples in each half.

    if (i < nA) { //  the index i is within the range of the first half segment
      ranksA[(segmentBase + 0) / SAMPLE_STRIDE + i] = i * SAMPLE_STRIDE;
      ranksB[(segmentBase + 0) / SAMPLE_STRIDE + i] =
          binarySearchExclusive(srcKey[segmentBase + i * SAMPLE_STRIDE],
                                srcKey + segmentBase + stride, lenB, sortDir);
    }

    if (i < nB) {
      ranksB[(segmentBase + stride) / SAMPLE_STRIDE + i] = i * SAMPLE_STRIDE;
      ranksA[(segmentBase + stride) / SAMPLE_STRIDE + i] =
          binarySearchInclusive(
              srcKey[segmentBase + stride + i * SAMPLE_STRIDE],
              srcKey + segmentBase, lenA, sortDir);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 2: merge ranks and indices to derive elementary intervals
////////////////////////////////////////////////////////////////////////////////
static void mergeRanksAndIndices(uint *limits, uint *ranks, uint stride,
                                 uint N) {
  uint lastSegmentElements = N % (2 * stride);
  uint sampleCount =
      (lastSegmentElements > stride)
          ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE)
          : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  for (uint pos = 0; pos < sampleCount; pos++) {
    const uint i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);

    const uint lenA = stride;
    const uint lenB = umin(stride, N - segmentBase - stride);
    const uint nA = stride / SAMPLE_STRIDE;
    const uint nB = getSampleCount(lenB);

    if (i < nA) {
      uint dstPosA =
          binarySearchExclusive(ranks[(segmentBase + 0) / SAMPLE_STRIDE + i],
                                ranks + (segmentBase + stride) / SAMPLE_STRIDE,
                                nB, 1) +
          i;
      assert(dstPosA < nA + nB);
      limits[(segmentBase / SAMPLE_STRIDE) + dstPosA] =
          ranks[(segmentBase + 0) / SAMPLE_STRIDE + i];
    }

    if (i < nB) {
      uint dstPosA = binarySearchInclusive(
                         ranks[(segmentBase + stride) / SAMPLE_STRIDE + i],
                         ranks + (segmentBase + 0) / SAMPLE_STRIDE, nA, 1) +
                     i;
      assert(dstPosA < nA + nB);
      limits[(segmentBase / SAMPLE_STRIDE) + dstPosA] =
          ranks[(segmentBase + stride) / SAMPLE_STRIDE + i];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary intervals (each interval is <= SAMPLE_STRIDE)
////////////////////////////////////////////////////////////////////////////////

// This merge function merges two sorted arrays (srcAKey and srcBKey) into a single sorted array (dstKey)
static void merge(uint *dstKey, uint *dstVal, uint *srcAKey, uint *srcAVal,
                  uint *srcBKey, uint *srcBVal, uint lenA, uint lenB,
                  uint sortDir) {
  checkOrder(srcAKey, lenA, sortDir);
  checkOrder(srcBKey, lenB, sortDir);

  for (uint i = 0; i < lenA; i++) {
    uint dstPos = binarySearchExclusive(srcAKey[i], srcBKey, lenB, sortDir) + i;
    assert(dstPos < lenA + lenB);
    dstKey[dstPos] = srcAKey[i];
    dstVal[dstPos] = srcAVal[i];
  }

  for (uint i = 0; i < lenB; i++) {
    uint dstPos = binarySearchInclusive(srcBKey[i], srcAKey, lenA, sortDir) + i;
    assert(dstPos < lenA + lenB);
    dstKey[dstPos] = srcBKey[i];
    dstVal[dstPos] = srcBVal[i];
  }
}

static void mergeElementaryIntervals(uint *dstKey, uint *dstVal, uint *srcKey,
                                     uint *srcVal, uint *limitsA, uint *limitsB,
                                     uint stride, uint N, uint sortDir) {
  uint lastSegmentElements = N % (2 * stride);
  uint mergePairs = (lastSegmentElements > stride)
                        ? getSampleCount(N)
                        : (N - lastSegmentElements) / SAMPLE_STRIDE;

  for (uint pos = 0; pos < mergePairs; pos++) {
    uint i = pos & ((2 * stride) / SAMPLE_STRIDE - 1);
    uint segmentBase = (pos - i) * SAMPLE_STRIDE;

    const uint lenA = stride;
    const uint lenB = umin(stride, N - segmentBase - stride);
    const uint nA = stride / SAMPLE_STRIDE;
    const uint nB = getSampleCount(lenB);
    const uint n = nA + nB;

    const uint startPosA = limitsA[pos];
    const uint endPosA = (i + 1 < n) ? limitsA[pos + 1] : lenA;
    const uint startPosB = limitsB[pos];
    const uint endPosB = (i + 1 < n) ? limitsB[pos + 1] : lenB;
    const uint startPosDst = startPosA + startPosB;

    assert(startPosA <= endPosA && endPosA <= lenA);
    assert(startPosB <= endPosB && endPosB <= lenB);
    assert((endPosA - startPosA) <= SAMPLE_STRIDE);
    assert((endPosB - startPosB) <= SAMPLE_STRIDE);

    merge(dstKey + segmentBase + startPosDst,
          dstVal + segmentBase + startPosDst,
          (srcKey + segmentBase + 0) + startPosA,
          (srcVal + segmentBase + 0) + startPosA,
          (srcKey + segmentBase + stride) + startPosB,
          (srcVal + segmentBase + stride) + startPosB, endPosA - startPosA,
          endPosB - startPosB, sortDir);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Retarded bubble sort
////////////////////////////////////////////////////////////////////////////////

// key: Pointer to the array of keys to be sorted.
// val: Pointer to the array of values associated with the keys.
// N: The number of elements in the key and val arrays.
// sortDir: The direction of sorting. 
// non-zero - ascending.
static void bubbleSort(uint *key, uint *val, uint N, uint sortDir) {
  if (N <= 1) {
    return;
  }

  for (uint bottom = 0; bottom < N - 1; bottom++) {
    uint savePos = bottom;
    uint saveKey = key[bottom];

    for (uint i = bottom + 1; i < N; i++)
      if ((sortDir && (key[i] < saveKey)) || (!sortDir && (key[i] > saveKey))) {
        savePos = i;
        saveKey = key[i];
      }

    if (savePos != bottom) {
      uint t;
      t = key[savePos];
      key[savePos] = key[bottom];
      key[bottom] = t;
      t = val[savePos];
      val[savePos] = val[bottom];
      val[bottom] = t;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////

// dstKey: Pointer to the destination array where the sorted keys will be stored.
// dstVal: Pointer to the destination array where the associated values will be stored after sorting.
// bufKey: Pointer to a temporary buffer array used during sorting for keys.
// bufVal: Pointer to a temporary buffer array used during sorting for values.
// srcKey: Pointer to the source array containing the keys to be sorted.
// srcVal: Pointer to the source array containing the associated values to be sorted along with the keys.
// N: The number of elements in the source array srcKey and srcVal.
// sortDir: The sorting direction (1 for ascending, 0 for descending).
extern "C" void mergeSortHost(uint *dstKey, uint *dstVal, uint *bufKey,
                              uint *bufVal, uint *srcKey, uint *srcVal, uint N,
                              uint sortDir) {
  uint *ikey, *ival, *okey, *oval;
  uint stageCount = 0;

  for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1, stageCount++)
    ;

  if (stageCount & 1) {
    ikey = bufKey;
    ival = bufVal;
    okey = dstKey;
    oval = dstVal;
  } else {
    ikey = dstKey;
    ival = dstVal;
    okey = bufKey;
    oval = bufVal;
  }

  printf("Bottom-level sort...\n");
  memcpy(ikey, srcKey, N * sizeof(uint));
  memcpy(ival, srcVal, N * sizeof(uint));

  for (uint pos = 0; pos < N; pos += SHARED_SIZE_LIMIT) {
    bubbleSort(ikey + pos, ival + pos, umin(SHARED_SIZE_LIMIT, N - pos),
               sortDir);
  }

  printf("Merge...\n");
  uint *ranksA = (uint *)malloc(getSampleCount(N) * sizeof(uint));
  uint *ranksB = (uint *)malloc(getSampleCount(N) * sizeof(uint));
  uint *limitsA = (uint *)malloc(getSampleCount(N) * sizeof(uint));
  uint *limitsB = (uint *)malloc(getSampleCount(N) * sizeof(uint));
  memset(ranksA, 0xFF, getSampleCount(N) * sizeof(uint));
  memset(ranksB, 0xFF, getSampleCount(N) * sizeof(uint));
  memset(limitsA, 0xFF, getSampleCount(N) * sizeof(uint));
  memset(limitsB, 0xFF, getSampleCount(N) * sizeof(uint));

  for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1) {
    uint lastSegmentElements = N % (2 * stride);

    // Find sample ranks and prepare for limiters merge
    generateSampleRanks(ranksA, ranksB, ikey, stride, N, sortDir);

    // Merge ranks and indices
    mergeRanksAndIndices(limitsA, ranksA, stride, N);
    mergeRanksAndIndices(limitsB, ranksB, stride, N);

    // Merge elementary intervals
    mergeElementaryIntervals(okey, oval, ikey, ival, limitsA, limitsB, stride,
                             N, sortDir);

    if (lastSegmentElements <= stride) {
      // Last merge segment consists of a single array which just needs to be
      // passed through
      memcpy(okey + (N - lastSegmentElements), ikey + (N - lastSegmentElements),
             lastSegmentElements * sizeof(uint));
      memcpy(oval + (N - lastSegmentElements), ival + (N - lastSegmentElements),
             lastSegmentElements * sizeof(uint));
    }

    uint *t;
    t = ikey;
    ikey = okey;
    okey = t;
    t = ival;
    ival = oval;
    oval = t;
  }

  free(limitsB);
  free(limitsA);
  free(ranksB);
  free(ranksA);
}
