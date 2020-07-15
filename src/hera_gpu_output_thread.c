// hera_gpu_output_thread.c
//
// Sends integrated GPU output to "catcher" machine for assimilation into a
// dataset. Unlike the original PAPER output thread, this code integrates
// over XENG_CHAN_SUM channels, and performs a channel x baseline transpose

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <endian.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <xgpu.h>

#include "hashpipe.h"
#include "paper_databuf.h"

//   Correlator data are sent to a data catcher using a simple UDP packetization
//   format:

//   uint64_t TIMESTAMP (set to be the MCNT provided by the databuf delivered by the upstream processor)
//   uint32_t OFFSET (offset in bytes where this packet should be placed in memory to build the complete output)
//   uint16_t X-ENGINE ID
//   uint16_t PAYLOAD LENGTH (length of data payload in this packet in bytes)

// Structure for packet header

typedef struct pkthdr {
  uint64_t timestamp;
  uint32_t offset;
  uint16_t xeng_id;
  uint16_t payload_len;
} pkthdr_t;

// Macros for generating values for the pkthdr_t fields
#define TIMESTAMP(x) (htobe64((uint64_t)x))
#define OFFSET(x)    (htobe32((uint32_t)x))
#define XENG_ID(x)    (htobe16((uint16_t)x))
#define PAYLOAD_LEN(x)    (htobe16((uint16_t)x))

#define CONVERT(x) (htobe32(x))

typedef int32_t pktdata_t;

// Structure of a packet
typedef struct pkt {
  pkthdr_t hdr;
  pktdata_t data[OUTPUT_BYTES_PER_PACKET/sizeof(pktdata_t)];
} pkt_t;

static XGPUInfo xgpu_info;

// PACKET_DELAY_NS is number of nanoseconds to delay between packets.  This is
// to prevent overflowing the network interface's TX queue.
// Keep in mind the total throughput of the network, and also the number
// of x-engine instances which will be running on each host.

// Note that the below calculations account only for the packet's payload (i.e.
// it considers the size of the packet header to be negligible).
//
// 1000 megabit per second =   1 nanosecond per bit
//  100 megabit per second =  10 nanosecond per bit
//   10 megabit per second = 100 nanosecond per bit

// Set to 200 Mbps -- OK for two instances per node.
// With 16 nodes, amounts to 6.4 Gbps of data

// Original
//#define PACKET_DELAY_NS (OUTPUT_BYTES_PER_PACKET>>2)

// delivers packets for ~6 sec with 96 chan
//#define PACKET_DELAY_NS (430 * 1000)

// packets for ~6sec with 384 chan
//#define PACKET_DELAY_NS (430 * 1000/4)

// packet for ~7.5 sec with 384 chan
#define PACKET_DELAY_NS (536 * 1000/4)

// bytes_per_dump depends on xgpu_info.triLength
static uint64_t bytes_per_dump = 0;
// packets_per_dump is bytes_per_dump / OUTPUT_BYTES_PER_PACKET
static unsigned int packets_per_dump = 0;

// Open and connect a UDP socket to the given host and port.  Note that port is
// a string and can be a stringified integer (e.g. "7148") or a service name
// (e.g. "ntp").  Returns -1 on error, otherwise a valid descriptor for an open
// and connected socket.
static int
open_udp_socket(const char *host, const char *port)
{
    struct addrinfo hints;
    struct addrinfo *result, *rp;
    int s, sfd=-1;

    // Obtain address(es) matching host/port

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = 0;
    hints.ai_protocol = 0;

    s = getaddrinfo(host, port, &hints, &result);
    if (s != 0) {
        hashpipe_error("getaddrinfo", gai_strerror(s));
        return -1;
    }

    // getaddrinfo() returns a list of address structures.
    // Try each address until we successfully connect(2).
    // If socket(2) (or connect(2)) fails, we (close the socket
    // and) try the next address.

    for (rp = result; rp != NULL; rp = rp->ai_next) {
        sfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);

        if (sfd == -1) {
            continue;
        }

        if (connect(sfd, rp->ai_addr, rp->ai_addrlen) != -1) {
            break; // Success
        }

        close(sfd);
        sfd = -1;
    }

    freeaddrinfo(result);

#if 0
    // Print send buffer size
    int bufsize;
    unsigned int bufsizesize = sizeof(bufsize);
    getsockopt(sfd, SOL_SOCKET, SO_SNDBUF, &bufsize, &bufsizesize);
    printf("send buffer size is %d\n", bufsize);
#endif

    return sfd;
}


// Computes the triangular index of an (i,j) pair as shown here...
// NB: Output is valid only if i >= j.
//
//      i=0  1  2  3  4..
//     +---------------
// j=0 | 00 01 03 06 10
//   1 |    02 04 07 11
//   2 |       05 08 12
//   3 |          09 13
//   4 |             14
//   :
static inline off_t tri_index(const int i, const int j)
{
  return (i * (i+1))/2 + j;
}

// regtile_chan_length is the number of complex cross products per channel for
// the xGPU register tile order correlator output format with N_INPUTS.
// NB: N_INPUTS = N_STATION * N_POL
#define REGTILE_CHAN_LENGTH (4 * 4 * N_INPUTS/4 * (N_INPUTS/4+1) / 2)
static inline size_t regtile_chan_length()
{
  // Four cross products for each quadrant of 4 input x 4 input tile
  return REGTILE_CHAN_LENGTH;
}

// Returns index into the GPU's register tile ordered output buffer for the
// real component of the cross product of inputs in0 and in1.  Note that in0
// and in1 are input indexes (i.e. 0 based) and often represent antenna and
// polarization by passing (2*ant_idx+pol_idx) as the input number (NB: ant_idx
// and pol_idx are also 0 based).  Return value is valid if in1 >= in0.  The
// corresponding imaginary component is located xgpu_info.matLength words after
// the real component.
static off_t regtile_index(const int in0, const int in1)
{
  const int a0 = in0 >> 1;
  const int a1 = in1 >> 1;
  const int p0 = in0 & 1;
  const int p1 = in1 & 1;
  const int num_words_per_cell = 4;

  // Index within a quadrant
  const int quadrant_index = tri_index(a1/2, a0/2);
  // Quadrant for this input pair
  const int quadrant = 2*(a0&1) + (a1&1);
  // Size of quadrant
  const int quadrant_size = (xgpu_info.nstation/2 + 1) * xgpu_info.nstation/4;
  // Index of cell (in units of cells)
  const int cell_index = quadrant*quadrant_size + quadrant_index;
  //printf("%s: in0=%d, in1=%d, a0=%d, a1=%d, cell_index=%d\n", __FUNCTION__, in0, in1, a0, a1, cell_index);
  // Pol offset
  const int pol_offset = 2*p1 + p0;
  // Word index (in units of words (i.e. floats) of real component
  const int index = (cell_index * num_words_per_cell) + pol_offset;
  return index;
}

// Returns index into a CASPER ordered buffer for the real component of the
// cross product of inputs in0 and in1.  Note that in0 and in1 are input
// indexes (i.e. 0 based) and often represent antenna and polarization by
// passing (2*ant_idx+pol_idx) as the input number (NB: ant_idx ad pol_idx are
// also 0 based).  Return value is valid if in1 >= in0.  The corresponding
// imaginary component is located in the word immediately following the real
// component.  A casper ordered buffer consists of four complex values for each
// pair of input pairs.  Thus, the number of complex values in a casper ordered
// buffer are: 4 * (N/2 * (N/2 + 1)) / 2 = N * (N/2 + 1)
static off_t casper_index(const int in0, const int in1, const int n)
{
  const int a0 = in0 >> 1;
  const int a1 = in1 >> 1;
  const int p0 = in0 & 1;
  const int p1 = in1 & 1;
  const int delta = a1-a0;
  const int num_words_per_cell = 8;
  const int nant_2 = (n/2) / 2;

  // Three cases: top triangle, middle rectangle, bottom triangle
  const int triangle_size = ((nant_2 + 1) * nant_2)/2;
  const int middle_rect_offset = triangle_size;
  const int last_cell_offset = 4*middle_rect_offset - nant_2 - 1;
  int cell_index;

  if(delta > nant_2) {
    // bottom triangle
    cell_index = last_cell_offset - tri_index(nant_2-2-a0, (n/2)-1-a1);
  } else if (a1 < (n/2)/2) {
    // top triangle
    cell_index = tri_index(a1, a0);
  } else {
    // middle rectangle
    cell_index = middle_rect_offset + (a1-nant_2)*(nant_2+1) + (nant_2-delta);
  }
  //printf("%s: a0=%d, a1=%d, delta=%d, cell_index=%d\n", __FUNCTION__, a0, a1, delta, cell_index);
  // Pol offset
  const int pol_offset = 2*(2*(p0^p1) + p0);
  // Word index (in units of words (i.e. floats) of real component
  const int index = (cell_index * num_words_per_cell) + pol_offset;
  return index;
}

// For each channel, a casper ordered buffer contains four complex values for
// each pair of input pairs.  Thus, the number of complex values in a casper
// ordered buffer are: 4 * (N/2 * (N/2 + 1)) / 2 = N * (N/2 + 1)
#define N_CASPER_COMPLEX_PER_CHAN (N_INPUTS * (N_INPUTS/2 + 1))

// Lookup table mapping casper_idx to regtile_idx
static off_t *idx_map;

static int init_idx_map()
{
  int a0, a1, p0, p1, i, j;
  idx_map = malloc(N_CASPER_COMPLEX_PER_CHAN * sizeof(off_t));
  if(!idx_map) {
    return -1;
  }

  for(a1=0; a1<N_INPUTS/2; a1++) {
    for(a0=0; a0<=a1; a0++) {
      for(p0=0; p0<2; p0++) {
        for(p1=0; p1<2; p1++) {
          i = 2*a0 + p0;
          j = 2*a1 + p1;
          idx_map[casper_index(i,j,N_INPUTS)/2] = regtile_index(i,j);
        }
      }
    }
  }
  return 0;
}

static int init(struct hashpipe_thread_args *args)
{
    // Get sizing parameters
    xgpuInfo(&xgpu_info);
    bytes_per_dump = xgpu_info.triLength * sizeof(Complex);
    packets_per_dump = bytes_per_dump / OUTPUT_BYTES_PER_PACKET;
    printf("bytes_per_dump = %lu\n", bytes_per_dump);

    if(init_idx_map()) {
      return -1;
    }

    // Success!
    return 0;
}

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

static void *run(hashpipe_thread_args_t * args)
{
    // Local aliases to shorten access to args fields
    // Our input buffer happens to be a paper_ouput_databuf
    paper_output_databuf_t *db = (paper_output_databuf_t *)args->ibuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    // Setup socket and message structures
    int sockfd;
    unsigned int xengine_id = 0;
    struct timespec packet_delay = {
      .tv_sec = 0,
      .tv_nsec = PACKET_DELAY_NS
    };

    hashpipe_status_lock_safe(&st);
    hgetu4(st.buf, "XID", &xengine_id); // No change if not found
    hputu4(st.buf, "XID", xengine_id);
    hputu4(st.buf, "OUTDUMPS", 0);
    hashpipe_status_unlock_safe(&st);

    pkt_t pkt;
    pkt.hdr.xeng_id = XENG_ID(xengine_id);
    pkt.hdr.payload_len = PAYLOAD_LEN(OUTPUT_BYTES_PER_PACKET);

    // TODO Get catcher hostname and port from somewhere

#define stringify2(x) #x
#define stringify(x) stringify2(x)

    // Open socket
    sockfd = open_udp_socket("catcher", stringify(CATCHER_PORT));
    if(sockfd == -1) {
        hashpipe_error(__FUNCTION__, "error opening socket");
        pthread_exit(NULL);
    }

#ifdef TEST_INDEX_CALCS
    int i, j;
    off_t o;
    for(i=0; i<32; i++) {
      for(j=i; j<32; j++) {
        regtile_index(2*i, 2*j);
      }
    }
    for(i=0; i<32; i++) {
      for(j=i; j<32; j++) {
        o = casper_index(2*i, 2*j, N_INPUTS);
        fprintf(stdout, "%d, %d, %d\n", i, j, (int) o);
      }
    }
#endif

    /* Main loop */
    int rv;
    int casper_chan, gpu_chan, sum_chan;
    int baseline, pol;
    unsigned int dumps = 0;
    int block_idx = 0;
    struct timespec start, stop;
    struct timespec pkt_start, pkt_stop;
    pktdata_t re, im;
    while (run_threads()) {

        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "waiting");
        hashpipe_status_unlock_safe(&st);

        // Wait for new block to be filled
        while ((rv=paper_output_databuf_wait_filled(db, block_idx))
                != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
                pthread_exit(NULL);
                break;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &start);

        // Note processing status, current input block
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "processing");
        hputi4(st.buf, "OUTBLKIN", block_idx);
        hashpipe_status_unlock_safe(&st);

        // Update header's timestamp for this dump.
        pkt.hdr.timestamp = TIMESTAMP(db->block[block_idx].header.mcnt);
        // Reset packet/byte counters to 0
        pkt.hdr.offset = OFFSET(0);
        uint32_t nbytes = 0;

        // Unpack and convert in packet sized chunks
        // output data in order: baseline x chan x stokes (slowest to fastest varying)
        pktdata_t * pf_re  = db->block[block_idx].data;
        pktdata_t * pf_im  = db->block[block_idx].data + xgpu_info.matLength;
        pktdata_t * p_out = pkt.data;
        clock_gettime(CLOCK_MONOTONIC, &pkt_start);
        // Iterate over blocks of N_STOKES baselines. All stokes are sent in adjacent words
        for(baseline=0; baseline<N_CASPER_COMPLEX_PER_CHAN; baseline+=N_STOKES) {
          // Iterate over blocks of XENG_CHAN_SUM channels. An inner loop will sum these
          for(casper_chan=0; casper_chan<N_CHAN_PER_X; casper_chan=casper_chan+XENG_CHAN_SUM) {
            for(pol=0; pol<N_STOKES; pol++) {
              // This used to de-interleave channels.  De-interleaving is no longer
              // needed, but we choose to continue maintaining the distinction
              // between casper_chan and gpu_chan.
              gpu_chan = casper_chan;
              off_t idx_regtile = idx_map[baseline + pol];
              for(sum_chan=0; sum_chan<XENG_CHAN_SUM; sum_chan++) {
                if (sum_chan == 0) {
                  re = pf_re[gpu_chan*REGTILE_CHAN_LENGTH+idx_regtile];
                  im = pf_im[gpu_chan*REGTILE_CHAN_LENGTH+idx_regtile];
                } else {
                  re += pf_re[(gpu_chan+sum_chan)*REGTILE_CHAN_LENGTH+idx_regtile];
                  im += pf_im[(gpu_chan+sum_chan)*REGTILE_CHAN_LENGTH+idx_regtile];
                }
              }
              *p_out++ = CONVERT(re);
              *p_out++ = CONVERT(-im); // Conjugate to match downstream expectations.
              nbytes += 2*sizeof(pktdata_t);
              if(nbytes % OUTPUT_BYTES_PER_PACKET == 0) {
                int bytes_sent = send(sockfd, &pkt, sizeof(pkt.hdr)+OUTPUT_BYTES_PER_PACKET, 0);
                if(bytes_sent == -1) {
                  // Send all packets even if cactcher is not listening (i.e. we
                  // we get a connection refused error), but abort sending this
                  // dump if we get any other error.
                  if(errno != ECONNREFUSED) {
                    perror("send");
                    // Update stats
                    hashpipe_status_lock_safe(&st);
                    hgetu4(st.buf, "OUTDUMPS", &dumps);
                    hputu4(st.buf, "OUTDUMPS", ++dumps);
                    hputr4(st.buf, "OUTSECS", 0.0);
                    hputr4(st.buf, "OUTMBPS", 0.0);
                    hashpipe_status_unlock_safe(&st);
                    // Break out of both for loops
                    goto done_sending;
                  }
                } else if(bytes_sent != sizeof(pkt.hdr)+OUTPUT_BYTES_PER_PACKET) {
                  printf("only sent %d of %lu bytes!!!\n", bytes_sent, sizeof(pkt.hdr)+OUTPUT_BYTES_PER_PACKET);
                }

                // Delay to prevent overflowing network TX queue
                clock_gettime(CLOCK_MONOTONIC, &pkt_stop);
                packet_delay.tv_nsec = PACKET_DELAY_NS - ELAPSED_NS(pkt_start, pkt_stop);
                if(packet_delay.tv_nsec > 0 && packet_delay.tv_nsec < 1000*1000*1000) {
                  nanosleep(&packet_delay, NULL);
                }

                // Setup for next packet
                p_out = pkt.data;
                pkt_start = pkt_stop;
                // Update header's byte_offset for this chunk
                pkt.hdr.offset = OFFSET(nbytes);
              }
            }
          }
        }

        clock_gettime(CLOCK_MONOTONIC, &stop);

        hashpipe_status_lock_safe(&st);
        hgetu4(st.buf, "OUTDUMPS", &dumps);
        hputu4(st.buf, "OUTDUMPS", ++dumps);
        hputu4(st.buf, "OUTBYTES", nbytes);
        hputr4(st.buf, "OUTSECS", (float)ELAPSED_NS(start,stop)/1e9);
        hputr4(st.buf, "OUTMBPS", (1e3*8*nbytes)/ELAPSED_NS(start,stop));
        hashpipe_status_unlock_safe(&st);

done_sending:

        // Mark block as free
        paper_output_databuf_set_free(db, block_idx);

        // Setup for next block
        block_idx = (block_idx + 1) % db->header.n_block;

        /* Will exit if thread has been cancelled */
        pthread_testcancel();
    }

    // Thread success!
    return NULL;
}

static hashpipe_thread_desc_t gpu_output_thread = {
    name: "hera_gpu_output_thread",
    skey: "OUTSTAT",
    init: init,
    run:  run,
    ibuf_desc: {paper_output_databuf_create},
    obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&gpu_output_thread);
}
