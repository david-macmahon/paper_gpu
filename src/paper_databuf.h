#ifndef _PAPER_DATABUF_H
#define _PAPER_DATABUF_H

#include <stdint.h>
#include "hashpipe_databuf.h"
#include "config.h"

// Determined by F engine
#define N_CHAN_TOTAL_GENERATED (8192)
#define FENG_SAMPLE_RATE (500000000)
#define N_CHAN_TOTAL 6144
//#define N_FENGINES   192 # isn't this always the same as the number of antennas?
#define N_CHAN_PER_F N_CHAN_TOTAL

// Number of separate X-engines which deal with
// alternate time chunks
#define TIME_DEMUX 2

// Determined by F engine packetizer
#define N_INPUTS_PER_PACKET  6
#define N_CHAN_PER_PACKET    384
#define N_TIME_PER_PACKET    2
// N_BYTES_PER_PACKET excludes header!
#define N_BYTES_PER_PACKET  (N_INPUTS_PER_PACKET*N_CHAN_PER_PACKET*N_TIME_PER_PACKET)

// X engine sizing (from xGPU)
#define N_ANTS               XGPU_NSTATION
#define N_INPUTS          (2*XGPU_NSTATION)
#define N_TIME_PER_BLOCK     XGPU_NTIME
#define N_CHAN_PER_X         XGPU_NFREQUENCY

// Derived from above quantities
#define N_FENGINES		((N_INPUTS / N_INPUTS_PER_PACKET) * N_INPUTS_PER_PACKET)
//#define N_SUB_BLOCKS_PER_INPUT_BLOCK (N_TIME_PER_BLOCK / N_TIME_PER_PACKET)
//#define N_SUB_BLOCKS_PER_INPUT_BLOCK N_TIME_PER_BLOCK
//#define N_SUB_BLOCKS_PER_INPUT_BLOCK (N_TIME_PER_BLOCK / 2048)
#define N_BYTES_PER_BLOCK            (N_TIME_PER_BLOCK * N_CHAN_PER_X * N_INPUTS)
// Number of bytes per block if we assume an integral number of SNAP boards,
// with all other antennas being fake padding required for XGPU sizing compliance.
#define N_BYTES_PER_BLOCK_NET        (N_TIME_PER_BLOCK * N_CHAN_PER_X * N_FENGINES)
#define N_PACKETS_PER_BLOCK          (N_BYTES_PER_BLOCK_NET / N_BYTES_PER_PACKET)
#define N_PACKETS_PER_BLOCK_PER_F    (N_INPUTS_PER_PACKET * N_PACKETS_PER_BLOCK / N_FENGINES)
// Number of X-engines per time slice. E.g. for HERA: 16.
#define N_XENGINES_PER_TIME (N_CHAN_TOTAL / N_CHAN_PER_X)
#define N_XENGINES          (N_XENGINES_PER_TIME*TIME_DEMUX)

// Validate packet dimensions
#if    N_BYTES_PER_PACKET != (N_TIME_PER_PACKET*N_CHAN_PER_PACKET*N_INPUTS_PER_PACKET)
#error N_BYTES_PER_PACKET != (N_TIME_PER_PACKET*N_CHAN_PER_PACKET*N_INPUTS_PER_PACKET)
#endif

#define N_FLUFFED_BYTES_PER_BLOCK  (N_BYTES_PER_BLOCK * 2)
#define N_FLUFFED_WORDS_PER_BLOCK (N_FLUFFED_BYTES_PER_BLOCK / 8) 

// Number of floats in xGPU's "register tile order" output matrix.
#define N_OUTPUT_MATRIX (2 * N_CHAN_PER_X * (N_INPUTS/2 + 2) * N_INPUTS)

#define PAGE_SIZE (4096)
#define CACHE_ALIGNMENT (128)

// The HERA correlator is based largely on the PAPER correlator.  The main 
// difference will be in the F engines.  The ROACH2 based F engines are being
// replaced by SNAP based F engines.  Because SNAPs have a different number of
// inputs, the F engine packet format must change. 
// Even though the X engines are the same, the network thread and the
// fluffing threads for HERA are somewhat different.  The data buffer format
// for the GPU thread's input buffer is exactly the same.  The fluffing
// thread's input buffer is therefore sized the same (i.e. half the size of the
// GPU input buffer), but the format is different since the packets stored by
// the network thread will be different size/format.  Also changed is the
// number of time samples per packet i(and therefore per mcount).

/*
 * INPUT BUFFER STRUCTURES
 */

#define N_INPUT_BLOCKS 16
#ifndef N_DEBUG_INPUT_BLOCKS
#define N_DEBUG_INPUT_BLOCKS 0
#endif

typedef struct paper_input_header {
  int64_t good_data; // functions as a boolean, 64 bit to maintain word alignment
  uint64_t mcnt;     // mcount of first packet
} paper_input_header_t;

typedef uint8_t paper_input_header_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(paper_input_header_t)%CACHE_ALIGNMENT)
];

// For HERA:
//
// * HERA net thread output
// * HERA fluffer thread input
// * Ordered sequence of packets in "data" field:
//
//   +-- mcount
//   |       +-- Antenna
//   |       |       |<-packets->|
//   V       V       |           |
//   m       a       |c        t |
//   ==      ==      |==       ==|
//   m0 }--> a0 }--> |c0 }---> t0|\.
//   m1      a1      |c1       t1| \.
//   m2      a2      |c2       t2|  \ chan 0
//   :       :       |:        : |  / packet
//   :       :       |:        : | /.
//   :       :       |:        : |/.
//   :       :       +---- ------+
//   :       :       |Nc/2 }-> t0|\.
//   :       :       |:        t1| \.
//   :       :       |:        t2|  \ chan Nc/2
//   :       :       |:        : |  / packet
//   :       :       |:        : | /.
//   :       :       |:        : |/.
//   ==      ==       ===      ==
//   Nm      Na       Nc       Nt
//
//   Each time sample is 2 bytes comprising two 4b+4b complex values.
//   Note that the bits may not all be contiguous to facilitate fluffing.
//
// m = packet's mcount - block's first mcount
// a = packet's antenna number
// c = channel within within X engine
// t = time sample within channel (0 to Nt-1)

#define Nm (N_TIME_PER_BLOCK/N_TIME_PER_PACKET)
#define Na N_ANTS
#define Nc N_CHAN_PER_X
#define Nt N_TIME_PER_PACKET
// Number of pols = 2
#define Np 2
// Number of complex elements = 2
#define Nx 2

// HERA mcount index = m * Na * Nc * Nt * HERA_N_INPUTS_PER_PACKET / sizeof(uint64_t)
// HERA ant index = a * Nc * Nt * HERA_N_INPUTS_PER_PACKET / sizeof(uint64_t)
// HERA chan index = c * Nt * HERA_N_INPUTS_PER_PACKET / sizeof(uint64_t)
//
// Computes paper_input_databuf_t data word (uint64_t) offset for complex data
// word corresponding to the given parameters for HERA F engine packets.

#define paper_input_databuf_data_idx(m,a,c,t) \
  ((((m) * Na*Nc*Nt*Np) + ((a) * Nc*Nt*Np) + ((c)*Nt*Np) + ((t)*Np)) / sizeof(uint64_t))

#define paper_input_databuf_data_idx8(m,a,p,c,t) \
  ((((((m) * Na) + (a))*Nc + (c))*Nt + (t))*Np + p)

#define paper_input_databuf_data_idx256(m,a,c,t) \
  ((((((m) * Na) + (a))*Nc + (c))*Nt + (t))*Np / sizeof(__m256i))

typedef struct paper_input_block {
  paper_input_header_t header;
  paper_input_header_cache_alignment padding; // Maintain cache alignment
  uint64_t data[N_BYTES_PER_BLOCK/sizeof(uint64_t)];
} paper_input_block_t;

// Used to pad after hashpipe_databuf_t to maintain cache alignment
typedef uint8_t hashpipe_databuf_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(hashpipe_databuf_t)%CACHE_ALIGNMENT)
];

typedef struct paper_input_databuf {
  hashpipe_databuf_t header;
  hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
  paper_input_block_t block[N_INPUT_BLOCKS+N_DEBUG_INPUT_BLOCKS];
} paper_input_databuf_t;

/*
 * GPU INPUT BUFFER STRUCTURES
 */

#define N_GPU_INPUT_BLOCKS 4

// == paper_gpu_input_databuf_t ==
//
// * Fluffer thread output
// * GPU thread input
// GPU input buffer is input[time/4][channel][station][pol][complexity][time%4]
// * Multidimensional array in "data" field:
//
//   +--time   +--chan +--fid  +--complexity +--time%4
//   |         |       |       |             |
//   V         V       V       V             V
//   m         c       f       r             i
//   ==        ==      ==
//   m0 }----> c0 }--> f0 }--> real }------> i0
//   m1        c1      f1      imag          i1
//   m2        c2      f2                    i2
//   :         :       :                     i3
//   ==        ==      ==      ==            ==
//   Nm*Nt/4   Nc      Na      2             t%4
//
//   Each group of eight bytes contains four timesamples for
//   the real and imaginary part of a single ant-pol-chan

// Returns word (uint64_t) offset for real input data word
// corresponding to the given parameters.  Corresponding imaginary data
// word is 4 bytes later. Only valid for t a multiple of 4

//#define paper_gpu_input_databuf_data_idx(m,a,t,c) \ //foo
//  (((m)*Nt*Nc*Na) + (((t)/4)*Nc*Na*4) + ((c)*Na*4) + (a)*4)

#define paper_gpu_input_databuf_data_idx(t, c, a, p, x) \
  (((((t)>>2)*4*Nx*Np*Na*Nc) + ((c)*4*Nx*Np*Na) + ((a)*4*Np*Nx) + ((p)*4*Nx) + ((x)*4) + ((t)%4)) / sizeof(uint64_t))

#define paper_gpu_input_databuf_data_idx8(t, c, a, p, x) \
  ((((t)>>2)*4*Nx*Np*Na*Nc) + ((c)*4*Nx*Np*Na) + ((a)*4*Np*Nx) + ((p)*4*Nx) + ((x)*4) + ((t)%4))

#define paper_gpu_input_databuf_data_idx256(t, c, a, p, x) \
  (((((t)>>2)*4*Nx*Np*Na*Nc) + ((c)*4*Nx*Np*Na) + ((a)*4*Np*Nx) + ((p)*4*Nx) + ((x)*4) + ((t)%4)) / sizeof(__m256i))

#define paper_gpu_input_databuf_data_idx_tca_256(t, c, a) \
  (((((((t)*Nc) + 4*(c))*Na + 4*(a))*Nx*Np) + (t)%4) / sizeof(__m256i))
  //(((((((t)>>2)*Nc) + (c))*Na + (a))*Nx*4*Np + ((t)%4)) / sizeof(__m256i))

//#define paper_gpu_input_databuf_data_idx256(m,a,t,c) \ //foo
//  (4*((m*Nt*Nc*Na) + (t*Nc*Na) + (c*Na) + a) / sizeof(__m256i))

typedef struct paper_gpu_input_block {
  paper_input_header_t header;
  paper_input_header_cache_alignment padding; // Maintain cache alignment
  uint64_t data[(2*N_BYTES_PER_BLOCK/sizeof(uint64_t))];
} paper_gpu_input_block_t;

typedef struct paper_gpu_input_databuf {
  hashpipe_databuf_t header;
  hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
  paper_gpu_input_block_t block[N_GPU_INPUT_BLOCKS];
} paper_gpu_input_databuf_t;

/*
 * OUTPUT BUFFER STRUCTURES
 */

#define N_OUTPUT_BLOCKS 2

typedef struct paper_output_header {
  uint64_t mcnt;
  uint64_t flags[(N_CHAN_PER_X+63) /64];
} paper_output_header_t;

typedef uint8_t paper_output_header_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(paper_output_header_t)%CACHE_ALIGNMENT)
];

typedef struct paper_output_block {
  paper_output_header_t header;
  paper_output_header_cache_alignment padding; // Maintain cache alignment
  int32_t data[N_OUTPUT_MATRIX];
} paper_output_block_t;

typedef struct paper_output_databuf {
  hashpipe_databuf_t header;
  hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
  paper_output_block_t block[N_OUTPUT_BLOCKS];
} paper_output_databuf_t;

/* 
 * BASELINE DEPENDENT AVERAGING STRUCTURES
 */

#define N_BASELINES               (N_ANTS * (N_ANTS + 1)/2)
#define N_COMPLEX_PER_BASELINE    (N_STOKES * N_CHAN_PER_X)

#define N_BDABUF_BLOCKS 2
#define N_BDABUF_BINS   4
#define N_MAX_INTTIME   8  // The longest baselines are collected for 8 time samples

// integration bin indexing
#define hera_bda_buf_data_idx(o, c, p) \
  ((((o)*N_CHAN_PER_X*N_STOKES)+((c)*N_STOKES)+(p))*2)


typedef struct hera_bda_header{
  uint64_t *mcnt;               // mcnt of the first time sample in the data
  uint64_t datsize;               // size of buffer (from no. baselines)
  uint64_t baselines;
  uint16_t *ant_pair_0;
  uint16_t *ant_pair_1;
  uint32_t *bcnt; 
} hera_bda_header_t;

typedef uint8_t hera_bda_header_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(hera_bda_header_t)%CACHE_ALIGNMENT)
];

typedef struct hera_bda_block{
  hera_bda_header_t header[N_BDABUF_BINS];
  hera_bda_header_cache_alignment padding;
  int32_t *data[N_BDABUF_BINS];
} hera_bda_block_t;

typedef struct hera_bda_databuf{
  hashpipe_databuf_t header;
  hashpipe_databuf_cache_alignment padding;
  hera_bda_block_t block[N_BDABUF_BLOCKS];
} hera_bda_databuf_t;


/*
 * CATCHER BUFFER STRUCTURES     
 */

#define CATCHER_PORT            10000
#define OUTPUT_BYTES_PER_PACKET (4096)
#define CATCHER_N_BLOCKS        4
#define XENG_CHAN_SUM           4
#define CATCHER_CHAN_SUM        1
#define VIS_MATRIX_ENTRIES (N_CHAN_TOTAL/XENG_CHAN_SUM * (N_INPUTS * ((N_INPUTS>>1) + 1)))
#define VIS_MATRIX_ENTRIES_PER_CHAN (N_INPUTS * ((N_INPUTS>>1) + 1))
#define PACKETS_PER_VIS_MATRIX ((8L*TIME_DEMUX*VIS_MATRIX_ENTRIES) / OUTPUT_BYTES_PER_PACKET)
#define PACKETS_PER_XENG_ID    (PACKETS_PER_VIS_MATRIX/N_XENGINES)
#define N_STOKES                4

typedef struct hera_catcher_input_header{
  uint64_t mcnt;
  uint64_t good_data;
  //uint8_t flags[VIS_MATRIX_ENTRIES*TIME_DEMUX];
} hera_catcher_input_header_t;

typedef uint8_t hera_catcher_input_header_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(hera_catcher_input_header_t)%CACHE_ALIGNMENT)
];

typedef struct hera_catcher_input_block {
  hera_catcher_input_header_t header;
  hera_catcher_input_header_cache_alignment padding; // Maintain cache alignment
  uint64_t data[VIS_MATRIX_ENTRIES*TIME_DEMUX];
} hera_catcher_input_block_t;

typedef struct hera_catcher_input_databuf {
  hashpipe_databuf_t header;
  hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
  hera_catcher_input_block_t block[CATCHER_N_BLOCKS];
} hera_catcher_input_databuf_t;

// Catcher input buffer has dimensions:
// n-xengines x n-baselines x nchans-per-x x time-demux x stokes x real/imag
#define hera_catcher_input_databuf_idx32(t, x, o) \
  (2L*TIME_DEMUX*(VIS_MATRIX_ENTRIES_PER_CHAN * (N_CHAN_PER_X/XENG_CHAN_SUM)*(x)) + (TIME_DEMUX*((o)>>2)) + (2*N_STOKES*(t)))
#define hera_catcher_input_databuf_by_bl_idx32(x, b) \
  (2L*TIME_DEMUX*(N_CHAN_PER_X/XENG_CHAN_SUM)*((VIS_MATRIX_ENTRIES_PER_CHAN * (x)) + (N_STOKES*(b))))

// channels * vis_matrix * stokes * 4 bytes per re/imag
#define MAX_HERA_CATCHER_IDX32 (((int64_t)N_XENGINES)*(N_CHAN_PER_X/XENG_CHAN_SUM)*VIS_MATRIX_ENTRIES_PER_CHAN*N_STOKES*4)


/* 
 * CATCHER -- BDA
 * Structures and parameters
 */

#define BASELINES_PER_BLOCK     256 //8192
#define CATCHER_CHAN_SUM_BDA     4
#define CHAN_PER_CATCHER_PKT   (OUTPUT_BYTES_PER_PACKET/(N_STOKES * 8L))                    // 128
#define PACKETS_PER_BASELINE   (N_CHAN_TOTAL/CHAN_PER_CATCHER_PKT)                          //  48
#define PACKETS_PER_BL_PER_X   (PACKETS_PER_BASELINE/N_XENGINES_PER_TIME)                   //   3
#define PACKETS_PER_X          (BASELINES_PER_BLOCK*PACKETS_PER_BL_PER_X)                   // 768
#define PACKETS_PER_BLOCK      (BASELINES_PER_BLOCK * TIME_DEMUX * PACKETS_PER_BASELINE)    // 1572864
#define BYTES_PER_BLOCK        (PACKETS_PER_BLOCK * OUTPUT_BYTES_PER_PACKET)                // 6GB

// == hera_catcher_bda_input_databuf_t ==
//
// * catcher net thread output
// * catcher disk thread input
// Offset determined by baseline_id, time_demux, xend_id, offset
// * Data field is structured as:
//
//  +-- baseline  +-- (even/odd)  +-- freq          +-- stokes  
//  |             |               |                 |
//  V             V               V                 V
//  baseline_id   time_demux     (xeng_id+offset)   
//  -----------   ----------     ----------------
//  b0 }--------> even }-------> f0 }-------------> s0
//                odd            f1                 s1
//                                                  s2
//                                                  s3


#define hera_bda_buf_data_offset(l, s, b, o) \
  ((((l)*(b)*N_CHAN_PER_X*N_STOKES)+((s)*N_CHAN_PER_X*N_STOKES)+((o)*CHAN_PER_CATCHER_PKT*N_STOKES))*2)

// b-- bcnt; t-- time_demux ; x-- xeng_id; o-- freq offset(0,1,2)
#define  hera_catcher_bda_input_databuf_pkt_offset(b, t, x, o) \
     (((b)*TIME_DEMUX*PACKETS_PER_BASELINE) + ((t)*PACKETS_PER_BASELINE) + ((x)*PACKETS_PER_BL_PER_X) + (o))

#define hera_catcher_bda_input_databuf_by_bcnt_idx32(b,p) \
      (((b)*TIME_DEMUX*N_CHAN_TOTAL*N_STOKES*2) + ((p)*N_CHAN_TOTAL*N_STOKES*2))

// b- bcnt; p- parity (even=0/odd=1); f- freq; s- stokes
#define hera_catcher_bda_input_databuf_idx32(b,p,f,s) \
      (((b)*TIME_DEMUX*N_CHAN_TOTAL*N_STOKES) + ((p)*N_CHAN_PROCESSED*N_STOKES) + ((f)*N_STOKES) + (s))

typedef struct hera_catcher_bda_input_header{
  uint64_t good_data;
  uint32_t bcnt[BASELINES_PER_BLOCK];        // starting value of baseline_id for this block
  uint64_t mcnt[BASELINES_PER_BLOCK];        // times are diff for each baseline 
  uint16_t ant_pair_0[BASELINES_PER_BLOCK];  // list of antennas in this block
  uint16_t ant_pair_1[BASELINES_PER_BLOCK]; 
  //uint8_t flags[BASELINES_PER_BLOCK];
} hera_catcher_bda_input_header_t;

typedef uint8_t hera_catcher_bda_input_header_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(hera_catcher_bda_input_header_t)%CACHE_ALIGNMENT)
];

typedef struct hera_catcher_bda_input_block {
  hera_catcher_bda_input_header_t header;
  hera_catcher_bda_input_header_cache_alignment padding; // Maintain cache alignment
  uint32_t data[BYTES_PER_BLOCK/sizeof(uint32_t)];
} hera_catcher_bda_input_block_t;

typedef struct hera_catcher_bda_input_databuf {
  hashpipe_databuf_t header;
  hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
  hera_catcher_bda_input_block_t block[CATCHER_N_BLOCKS];
} hera_catcher_bda_input_databuf_t;

/* 
 * CATCHER - autocorr buffers
 */

#define BYTES_AUTOCORR_BLK  (N_CHAN_TOTAL * N_ANTS * N_STOKES * 8L)
#define AUTOCORR_N_BLOCKS    2

#define hera_catcher_autocorr_databuf_idx32(a) \
      ((a)*N_CHAN_TOTAL*N_STOKES*2L)

typedef struct hera_catcher_autocorr_header{
  uint64_t num_ants;
  double julian_time;
  uint8_t ant[N_ANTS]; //flag to show if this has been updated
} hera_catcher_autocorr_header_t;

typedef uint8_t hera_catcher_autocorr_header_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(hera_catcher_autocorr_header_t) % CACHE_ALIGNMENT)
];

typedef struct hera_catcher_autocorr_block {
  hera_catcher_autocorr_header_t header;
  hera_catcher_autocorr_header_cache_alignment padding;
  uint32_t data[BYTES_AUTOCORR_BLK/sizeof(uint32_t)];
} hera_catcher_autocorr_block_t;

typedef struct hera_catcher_autocorr_databuf {
  hashpipe_databuf_t header;
  hashpipe_databuf_cache_alignment padding;
  hera_catcher_autocorr_block_t block[AUTOCORR_N_BLOCKS];
} hera_catcher_autocorr_databuf_t;


/*
 * INPUT BUFFER FUNCTIONS
 */

hashpipe_databuf_t *paper_input_databuf_create(int instance_id, int databuf_id);

static inline paper_input_databuf_t *paper_input_databuf_attach(int instance_id, int databuf_id)
{
    return (paper_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int paper_input_databuf_detach(paper_input_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline void paper_input_databuf_clear(paper_input_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline int paper_input_databuf_block_status(paper_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_input_databuf_total_status(paper_input_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}


int paper_input_databuf_wait_free(paper_input_databuf_t *d, int block_id);

int paper_input_databuf_busywait_free(paper_input_databuf_t *d, int block_id);

int paper_input_databuf_wait_filled(paper_input_databuf_t *d, int block_id);

int paper_input_databuf_busywait_filled(paper_input_databuf_t *d, int block_id);

int paper_input_databuf_set_free(paper_input_databuf_t *d, int block_id);

int paper_input_databuf_set_filled(paper_input_databuf_t *d, int block_id);

/*
 * CATCHER BUFFER FUNCTIONS
 */

hashpipe_databuf_t *hera_catcher_input_databuf_create(int instance_id, int databuf_id);

static inline hera_catcher_input_databuf_t *hera_catcher_input_databuf_attach(int instance_id, int databuf_id)
{
    return (hera_catcher_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int hera_catcher_input_databuf_detach(hera_catcher_input_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline void hera_catcher_input_databuf_clear(hera_catcher_input_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline int hera_catcher_input_databuf_block_status(hera_catcher_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_catcher_input_databuf_total_status(hera_catcher_input_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

int hera_catcher_input_databuf_wait_free(hera_catcher_input_databuf_t *d, int block_id);

int hera_catcher_input_databuf_busywait_free(hera_catcher_input_databuf_t *d, int block_id);

int hera_catcher_input_databuf_wait_filled(hera_catcher_input_databuf_t *d, int block_id);

int hera_catcher_input_databuf_busywait_filled(hera_catcher_input_databuf_t *d, int block_id);

int hera_catcher_input_databuf_set_free(hera_catcher_input_databuf_t *d, int block_id);

int hera_catcher_input_databuf_set_filled(hera_catcher_input_databuf_t *d, int block_id);

/* 
 * CATCHER BDA BUFFER FUNCTIONS
 */

hashpipe_databuf_t *hera_catcher_bda_input_databuf_create(int instance_id, int databuf_id);

static inline hera_catcher_bda_input_databuf_t *hera_catcher_bda_input_databuf_attach(int instance_id, int databuf_id)
{
    return (hera_catcher_bda_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int hera_catcher_bda_input_databuf_detach(hera_catcher_bda_input_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline void hera_catcher_bda_input_databuf_clear(hera_catcher_bda_input_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline int hera_catcher_bda_input_databuf_block_status(hera_catcher_bda_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_catcher_bda_input_databuf_total_status(hera_catcher_bda_input_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

int hera_catcher_bda_input_databuf_wait_free(hera_catcher_bda_input_databuf_t *d, int block_id);

int hera_catcher_bda_input_databuf_busywait_free(hera_catcher_bda_input_databuf_t *d, int block_id);

int hera_catcher_bda_input_databuf_wait_filled(hera_catcher_bda_input_databuf_t *d, int block_id);

int hera_catcher_bda_input_databuf_busywait_filled(hera_catcher_bda_input_databuf_t *d, int block_id);

int hera_catcher_bda_input_databuf_set_free(hera_catcher_bda_input_databuf_t *d, int block_id);

int hera_catcher_bda_input_databuf_set_filled(hera_catcher_bda_input_databuf_t *d, int block_id);


/* 
 * CATCHER AUTOCORR BUFFER FUNCTIONS
 */

hashpipe_databuf_t *hera_catcher_autocorr_databuf_create(int instance_id, int databuf_id);

static inline hera_catcher_autocorr_databuf_t *hera_catcher_autocorr_databuf_attach(int instance_id, int databuf_id)
{
    return (hera_catcher_autocorr_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int hera_catcher_autocorr_databuf_detach(hera_catcher_autocorr_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline void hera_catcher_autocorr_databuf_clear(hera_catcher_autocorr_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline int hera_catcher_autocorr_databuf_block_status(hera_catcher_autocorr_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_catcher_autocorr_databuf_total_status(hera_catcher_autocorr_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

int hera_catcher_autocorr_databuf_wait_free(hera_catcher_autocorr_databuf_t *d, int block_id);

int hera_catcher_autocorr_databuf_busywait_free(hera_catcher_autocorr_databuf_t *d, int block_id);

int hera_catcher_autocorr_databuf_wait_filled(hera_catcher_autocorr_databuf_t *d, int block_id);

int hera_catcher_autocorr_databuf_busywait_filled(hera_catcher_autocorr_databuf_t *d, int block_id);

int hera_catcher_autocorr_databuf_set_free(hera_catcher_autocorr_databuf_t *d, int block_id);

int hera_catcher_autocorr_databuf_set_filled(hera_catcher_autocorr_databuf_t *d, int block_id);

/*
 * GPU INPUT BUFFER FUNCTIONS
 */

hashpipe_databuf_t *paper_gpu_input_databuf_create(int instance_id, int databuf_id);

static inline void paper_gpu_input_databuf_clear(paper_gpu_input_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline paper_gpu_input_databuf_t *paper_gpu_input_databuf_attach(int instance_id, int databuf_id)
{
    return (paper_gpu_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int paper_gpu_input_databuf_detach(paper_gpu_input_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline int paper_gpu_input_databuf_block_status(paper_gpu_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_gpu_input_databuf_total_status(paper_gpu_input_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int paper_gpu_input_databuf_wait_free(paper_gpu_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_gpu_input_databuf_busywait_free(paper_gpu_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_gpu_input_databuf_wait_filled(paper_gpu_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_gpu_input_databuf_busywait_filled(paper_gpu_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_gpu_input_databuf_set_free(paper_gpu_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_gpu_input_databuf_set_filled(paper_gpu_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

/*
 * BASELINE DEPENDENT AVERAGING BUFFER FUNCTIONS
 */

hashpipe_databuf_t *hera_bda_databuf_create(int instance_id, int databuf_id);

static inline void hera_bda_databuf_clear(hera_bda_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline hera_bda_databuf_t *hera_bda_databuf_attach(int instance_id, int databuf_id)
{
    return (hera_bda_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int hera_bda_databuf_detach(hera_bda_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline int hera_bda_databuf_block_status(hera_bda_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_bda_databuf_total_status(hera_bda_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int hera_bda_databuf_wait_free(hera_bda_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_bda_databuf_busywait_free(hera_bda_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_bda_databuf_wait_filled(hera_bda_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_bda_databuf_busywait_filled(hera_bda_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_bda_databuf_set_free(hera_bda_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int hera_bda_databuf_set_filled(hera_bda_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

/*
 * OUTPUT BUFFER FUNCTIONS
 */

hashpipe_databuf_t *paper_output_databuf_create(int instance_id, int databuf_id);

static inline void paper_output_databuf_clear(paper_output_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline paper_output_databuf_t *paper_output_databuf_attach(int instance_id, int databuf_id)
{
    return (paper_output_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int paper_output_databuf_detach(paper_output_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline int paper_output_databuf_block_status(paper_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_output_databuf_total_status(paper_output_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int paper_output_databuf_wait_free(paper_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_output_databuf_busywait_free(paper_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_output_databuf_wait_filled(paper_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_output_databuf_busywait_filled(paper_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_output_databuf_set_free(paper_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int paper_output_databuf_set_filled(paper_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

#endif // _PAPER_DATABUF_H
