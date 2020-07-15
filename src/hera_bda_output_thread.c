/* Output baseline dependent averaged 
 * data to the network for the catcher
 * thread to accumulate.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <errno.h>
#include <endian.h>
#include <arpa/inet.h>

#include <xgpu.h>

#include "hashpipe.h"
#include "paper_databuf.h"

// Macros for generating values for the pkthdr_t fields
#define TIMESTAMP(x)      (htobe64((uint64_t)(x)))
#define BASELINE_ID(x)    (htobe32((uint32_t)(x)))
#define OFFSET(x)         (htobe32((uint32_t)(x)))
#define XENG_ID(x)        (htobe16((uint16_t)(x)))
#define PAYLOAD_LEN(x)    (htobe16((uint16_t)(x)))
#define ANTENNA(x)        (htobe16((uint16_t)(x)))

#define CONVERT(x)        (htobe32((x)))

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))


typedef int32_t pktdata_t;

// Structure for packet header
typedef struct pkthdr {
  uint64_t timestamp;
  uint32_t baseline_id;
  uint32_t offset;
  uint16_t ant0;
  uint16_t ant1;
  uint16_t xeng_id;
  uint16_t payload_len;
} pkthdr_t;

// Structure of a packet
typedef struct struct_pkt {
  pkthdr_t hdr;
  pktdata_t data[OUTPUT_BYTES_PER_PACKET/sizeof(pktdata_t)];
} pkt_t;

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

// 8 * OUTPUT_BYTES_PER_PACKET == 1 Gbps
// 4 * 8 * OUTPUT_BYTES_PER_PACKET == 0.25 Gbps

// For full BDA, 350 ants, 384 chan per pipeline
//#define PACKET_DELAY_NS (2 * 8 * OUTPUT_BYTES_PER_PACKET)

// For no-BDA, 192 ants, 384 chan per pipeline -- 125 usec
#define PACKET_DELAY_NS (125 * 1000)

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

#ifdef PRINT_TEST
    // Print send buffer size
    int bufsize;
    unsigned int bufsizesize = sizeof(bufsize);
    getsockopt(sfd, SOL_SOCKET, SO_SNDBUF, &bufsize, &bufsizesize);
    printf("send buffer size is %d\n", bufsize);
#endif

    return sfd;
}

static void *run(hashpipe_thread_args_t * args)
{
   // Local aliases to shorten access to args fields
   // Our input buffer happens to be a paper_ouput_databuf
   hera_bda_databuf_t *db = (hera_bda_databuf_t *)args->ibuf;
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
        o = baseline_index(2*i, 2*j, N_INPUTS);
        fprintf(stdout, "%d, %d, %d, %ld\n", i, j, (int) o, 
                                    (long int)regtile_idx_map[o]);
      }
    }
#endif

   /* Main loop */
   int rv;
   unsigned int dumps = 0;
   int block_idx = 0;
   struct timespec start, stop;
   uint32_t nbytes = 0;
   int chan;
   unsigned long bl;
   int i,j,p;
   struct timespec pkt_start, pkt_stop;
   int offset = 0;
   uint16_t ant0, ant1;
   hera_bda_block_t *buf;
   unsigned int n_samples;
   uint64_t datoffset;

   while (run_threads()) {

     hashpipe_status_lock_safe(&st);
     hputs(st.buf, status_key, "waiting");
     hashpipe_status_unlock_safe(&st);

     // Wait for new block to be filled
     while ((rv = hera_bda_databuf_wait_filled(db, block_idx))
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
     nbytes = 0;

     // Note processing status, current input block
     hashpipe_status_lock_safe(&st);
     hputs(st.buf, status_key, "processing");
     hputi4(st.buf, "OUTBLKIN", block_idx);
     hputu4(st.buf, "OUTBCNT",  db->block[block_idx].header[0].bcnt[0]);
     hashpipe_status_unlock_safe(&st);
     
     buf = &(db->block[block_idx]); 
     pktdata_t *p_out = pkt.data;

     // Loop through baselines and send the packets
     // Send all packets of one baseline and then the next 
     // (e.g. all 8 samples of 2s integrations) 

     for(j=0; j<N_BDABUF_BINS; j++){ //intbuf loop
   
         n_samples = N_MAX_INTTIME/(1<<j); // l in offset
         offset = 0;
         clock_gettime(CLOCK_MONOTONIC, &pkt_start);
 
         for(bl=0; bl< buf->header[j].baselines; bl++){
           ant0 = buf->header[j].ant_pair_0[bl];    // ordering set by config file 
           ant1 = buf->header[j].ant_pair_1[bl]; 
           pkt.hdr.ant1 = ANTENNA(ant1);
           pkt.hdr.ant0 = ANTENNA(ant0);

           for(i=0; i<n_samples; i++){ // Number of time samples of one baseline

             pkt.hdr.baseline_id = BASELINE_ID(buf->header[j].bcnt[bl*n_samples + i]);
             pkt.hdr.timestamp = TIMESTAMP(buf->header[j].mcnt[bl*n_samples + i]);
             offset = 0;
           
             for(chan=0; chan<N_CHAN_PER_X; chan+= CHAN_PER_CATCHER_PKT){

               pkt.hdr.offset = OFFSET(offset);
               // hera_bda_buf_data_idx(l,s,b,c,p)
               datoffset = hera_bda_buf_data_idx((bl*n_samples + i), chan, 0);
               for(p=0; p<OUTPUT_BYTES_PER_PACKET/sizeof(pktdata_t); p++){
                 *p_out++ = CONVERT(buf->data[j][datoffset+p]);
               }

               int bytes_sent = send(sockfd, &pkt, sizeof(pkt.hdr)+OUTPUT_BYTES_PER_PACKET, 0); 
               nbytes += bytes_sent;

               if(bytes_sent == -1){
                 // Send all packets even if catcher is not listening (i.e. we
                 // we get a connection refused error), but abort sending this
                 // dump if we get any other error.
                 if(errno != ECONNREFUSED){
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
               }else if(bytes_sent != sizeof(pkt.hdr)+OUTPUT_BYTES_PER_PACKET) {
                 printf("only sent %d of %lu bytes!!!\n", bytes_sent, 
                         sizeof(pkt.hdr)+OUTPUT_BYTES_PER_PACKET);
               }

               // Delay to prevent overflowing network TX queue
               clock_gettime(CLOCK_MONOTONIC, &pkt_stop);
               packet_delay.tv_nsec = PACKET_DELAY_NS - ELAPSED_NS(pkt_start, pkt_stop);
               if(packet_delay.tv_nsec > 0 && packet_delay.tv_nsec < 1000*1000*1000){
                 nanosleep(&packet_delay, NULL);
               }
               
               // Setup for next packet
               p_out = pkt.data;
               pkt_start = pkt_stop;
               offset++;
             } // chan
           } // samples
         } // baseline
     } // end intbuf loop

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
     hera_bda_databuf_set_free(db, block_idx);

     // Setup for next block
     block_idx = (block_idx + 1) % db->header.n_block;

     /* Will exit if thread has been cancelled */
     pthread_testcancel();
  }

    // Thread success!
    return NULL;
}

static hashpipe_thread_desc_t bda_output_thread = {
    name: "hera_bda_output_thread",
    skey: "OUTSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {hera_bda_databuf_create},
    obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&bda_output_thread);
}
