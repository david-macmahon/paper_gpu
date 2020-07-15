/* hera_pktsock_thread.c
 *
 * Routine to read packets from network and put them
 * into shared memory blocks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <errno.h>

#include <xgpu.h>

#include "hashpipe.h"
#include "paper_databuf.h"


#define DEBUG_NET

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

//#define PKTSOCK_BYTES_PER_FRAME (16384)
//#define PKTSOCK_FRAMES_PER_BLOCK (8)
//#define PKTSOCK_NBLOCKS (800)
//#define PKTSOCK_NFRAMES (PKTSOCK_FRAMES_PER_BLOCK * PKTSOCK_NBLOCKS)
#define PKTSOCK_BYTES_PER_FRAME (4864)
#define PKTSOCK_FRAMES_PER_BLOCK (128)
#define PKTSOCK_NBLOCKS (5000)
#define PKTSOCK_NFRAMES (PKTSOCK_FRAMES_PER_BLOCK * PKTSOCK_NBLOCKS)

typedef struct {
    uint64_t mcnt;         // MCNT of the first block in this integrations
    uint32_t offset;       // offset in bytes where this packet belongs in the correlation
    uint16_t xeng_id;	   // First channel in a packet
    uint16_t payload_len;  // Antenna in a packet
} packet_header_t;

// The fields of a block_info_t structure hold meta-data about the contents of
// each block
typedef struct {
    int initialized;
    int block_i;
    int block_packet_counter[CATCHER_N_BLOCKS];
    int xeng_pkt_counter[CATCHER_N_BLOCKS][N_XENGINES];
} block_info_t;


static hashpipe_status_t *st_p;
static const char * status_key;

// This sets the "current" block to be marked as filled.  The current block is
// the block corresponding to binfo->block_i.  Returns mcnt of the block
// being marked filled.
static uint64_t set_block_filled(hera_catcher_input_databuf_t *hera_catcher_input_databuf_p, block_info_t *binfo)
{
    static int last_filled = -1;

    uint64_t block_missed_pkt_cnt;
    uint32_t missed_pkt_cnt;
    uint32_t block_i = binfo->block_i;
    int i;

    // Validate that we're filling blocks in the proper sequence
    last_filled = (last_filled+1) % CATCHER_N_BLOCKS;
    if(last_filled != block_i) {
	printf("block %d being marked filled, but expected block %d!\n", block_i, last_filled);

    }

    // Validate that block_i matches binfo->block_i
    if(block_i != binfo->block_i) {
	hashpipe_warn(__FUNCTION__,
		"block_i for binfo's mcnt (%d) != binfo's block_i (%d)",
		block_i, binfo->block_i);
    }

    // If all packets are accounted for, mark this block as good
    if(binfo->block_packet_counter[block_i] == PACKETS_PER_VIS_MATRIX) {
	hera_catcher_input_databuf_p->block[block_i].header.good_data = 1;
    }

    // Set the block as filled
    if(hera_catcher_input_databuf_set_filled(hera_catcher_input_databuf_p, block_i) != HASHPIPE_OK) {
	hashpipe_error(__FUNCTION__, "error waiting for databuf filled call");
	pthread_exit(NULL);
    }

    // Calculate missing packets.
    block_missed_pkt_cnt = PACKETS_PER_VIS_MATRIX - binfo->block_packet_counter[block_i];

    // Update status buffer
    hashpipe_status_lock_busywait_safe(st_p);
    hputu4(st_p->buf, "NETBKOUT", block_i);
    if(block_missed_pkt_cnt) {
        fprintf(stdout, "Total expected %lu packets, Got %d\n", 
                PACKETS_PER_VIS_MATRIX, 
                binfo->block_packet_counter[block_i]);
        fprintf(stdout, "Percentage pkts per xeng:\n");
        // Dump per-xeng stats
        for (i=0; i<N_XENGINES; i++){
            fprintf(stdout, "Xeng_id %2d: %.2f\n", i, 
                    (float)binfo->xeng_pkt_counter[block_i][i]/PACKETS_PER_XENG_ID);
        }
	// Increment MISSEDPK by number of missed packets for this block
	hgetu4(st_p->buf, "MISSEDPK", &missed_pkt_cnt);
	missed_pkt_cnt += block_missed_pkt_cnt;
	hputu4(st_p->buf, "MISSEDPK", missed_pkt_cnt);
    }
    hashpipe_status_unlock_safe(st_p);

    return hera_catcher_input_databuf_p->block[block_i].header.mcnt;
}


// Initialize a block by clearing its "good data" flag and initializing its mcnt header.
static inline void initialize_block(hera_catcher_input_databuf_t * hera_catcher_input_databuf_p, uint64_t mcnt, uint64_t block_id)
{

    hera_catcher_input_databuf_p->block[block_id].header.good_data = 0;
    hera_catcher_input_databuf_p->block[block_id].header.mcnt = mcnt;
}

// This function must be called once and only once per block_info structure!
// Subsequent calls are no-ops.
static inline void initialize_block_info(block_info_t * binfo)
{
    int i,j;

    // If this block_info structure has already been initialized
    if(binfo->initialized) {
	return;
    }

    for(i = 0; i < CATCHER_N_BLOCKS; i++) {
	binfo->block_packet_counter[i] = 0;
        for(j = 0; j < N_XENGINES; j++){
            binfo->xeng_pkt_counter[i][j] = 0;
        }
    }

    // On startup mcnt_start will be zero and mcnt_log_late will be Nm.
    binfo->block_i = 0;

    binfo->initialized = 1;
}

// This function returns -1 unless the given packet causes a block to be marked
// as filled in which case this function returns the marked block's first mcnt.
// Any return value other than -1 will be stored in the status memory as
// NETMCNT, so it is important that values other than -1 are returned rarely
// (i.e. when marking a block as filled)!!!
static inline uint64_t process_packet(
	hera_catcher_input_databuf_t *hera_catcher_input_databuf_p, unsigned char*p_frame)
{

    int i, j;
    int rv;
    static block_info_t binfo;
    packet_header_t *packet_header_p;
    const uint32_t *payload_p;
    uint32_t *dest_p;
    int64_t pkt_mcnt_dist;
    uint64_t pkt_mcnt;
    int time_demux_block;
    uint64_t cur_mcnt = -1;
    uint64_t netmcnt = -1; // Value to return (!=-1 is stored in status memory)
    // These are just the header entries, after we've network->host endianed them
    uint64_t mcnt;
    uint32_t offset;
    uint16_t xeng_id;
    uint16_t payload_len;


    // Parse packet header
    packet_header_p = (packet_header_t *)PKT_UDP_DATA(p_frame);
    mcnt        = be64toh(packet_header_p->mcnt);
    offset      = be32toh(packet_header_p->offset);
    xeng_id     = be16toh(packet_header_p->xeng_id);
    payload_len = be16toh(packet_header_p->payload_len);

    // Split the mcnt into a "pkt_mcnt" which is the same for all even/odd samples,
    // and "time_demux_block", which indicates which even/odd block this packet came from
    time_demux_block = (mcnt / Nt) % TIME_DEMUX;
    pkt_mcnt = mcnt - (Nt*time_demux_block);

    // Holdoff until the start of an integration
    if(!binfo.initialized && (offset!=0)) {
        return -1;
    }

    // Lazy init binfo
    if(!binfo.initialized) {
	initialize_block_info(&binfo);
        // If this is the first block, then set it's mcnt (which was dummy init-ed to 0)`
        initialize_block(hera_catcher_input_databuf_p, pkt_mcnt, binfo.block_i);
    }

    cur_mcnt = hera_catcher_input_databuf_p->block[binfo.block_i].header.mcnt;

    pkt_mcnt_dist = pkt_mcnt - cur_mcnt;

    // pkt_mcnt values (which have the time_demux offsets removed, should always
    // be the same in consecutive packets, until an integration is complete, at which
    // point they should increase (in a step determined by the accumulation length)
    // If the MCNT has changed, we need to start a new buffer.
    // Mark the current block as filled
    if (0 != pkt_mcnt_dist) {
        netmcnt = set_block_filled(hera_catcher_input_databuf_p, &binfo);
        // Reset binfo's packet counter for this packet's block
        binfo.block_packet_counter[binfo.block_i] = 0;
        for(i=0; i<N_XENGINES; i++) binfo.xeng_pkt_counter[binfo.block_i][i] = 0;
        // Increment the current block number
        binfo.block_i = (binfo.block_i +1) % CATCHER_N_BLOCKS;
        // Wait (hopefully not long!) to acquire the block after next (i.e.
        // the block that gets the current packet).
        hashpipe_status_lock_safe(st_p);
        hputs(st_p->buf, status_key, "waiting for outbuf");
        hashpipe_status_unlock_safe(st_p);
        while ((rv=hera_catcher_input_databuf_wait_free(hera_catcher_input_databuf_p, binfo.block_i)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(st_p);
                hputs(st_p->buf, status_key, "blocked output");
                hashpipe_status_unlock_safe(st_p);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                pthread_exit(NULL);
                break;
            }
        }

        hashpipe_status_lock_safe(st_p);
        hputs(st_p->buf, status_key, "running");
        hashpipe_status_unlock_safe(st_p);
        // Initialize the newly acquired block
        initialize_block(hera_catcher_input_databuf_p, pkt_mcnt, binfo.block_i);
    
        if (0 > pkt_mcnt_dist) {
            // If the MCNT went down, then we are relocking. Flag this as a warning.
    	    hashpipe_warn("hera_catcher_net_thread", "Locking on to new MCNT: %lu", pkt_mcnt);
        }
    }
    
    // Increment packet count for block
    binfo.block_packet_counter[binfo.block_i]++;
    // Increment packet count for xeng_id
    binfo.xeng_pkt_counter[binfo.block_i][xeng_id]++;

    // Copy data into buffer
    dest_p = (uint32_t *)(hera_catcher_input_databuf_p->block[binfo.block_i].data) + (hera_catcher_input_databuf_idx32(time_demux_block, (xeng_id % N_XENGINES_PER_TIME), offset));
    payload_p = (uint32_t *)(PKT_UDP_DATA(p_frame) + sizeof(packet_header_t));
    if (hera_catcher_input_databuf_idx32(time_demux_block, xeng_id / TIME_DEMUX, offset) >= MAX_HERA_CATCHER_IDX32) {
        fprintf(stderr, "databuf offset outside allowed range!\n");
        fprintf(stderr, "offset is %lu\n", (hera_catcher_input_databuf_idx32(time_demux_block, xeng_id/TIME_DEMUX, offset)));
        fprintf(stderr, "mcnt: %lu, t-demux: %d, offset: %d, xeng: %d\n", mcnt, time_demux_block, offset, xeng_id/TIME_DEMUX);
    }
    //fprintf(stdout, "mcnt: %lu, t-demux: %d, offset: %d, xeng: %d pointer: %lu\n", mcnt, time_demux_block, offset, xeng_id, hera_catcher_input_databuf_idx32(time_demux_block, (xeng_id % N_XENGINES_PER_TIME), offset));
    //fprintf(stdout, "offset is %lu\n", (hera_catcher_input_databuf_idx32(time_demux_block, xeng_id/TIME_DEMUX, offset)));
    //fprintf(stdout, "offset: %d\n", hera_catcher_input_databuf_idx64(time_demux_block, packet_header_p->xeng_id, packet_header_p->offset));
    
    // Copy with byte swap
    for(i=0; i<(payload_len >> 2); i+=8){
        for (j=0; j<8; j++) {
            dest_p[TIME_DEMUX*i + j] = be32toh(payload_p[i + j]);
            //dest_p[i + j] = be32toh(payload_p[i + j]);
        }
    }
    
    return netmcnt;
}

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

static int init(hashpipe_thread_args_t *args)
{
    /* Read network params */
    char bindhost[80];
    int bindport = CATCHER_PORT;
    status_key = args->thread_desc->skey;

    strcpy(bindhost, "0.0.0.0");

    hashpipe_status_t st = args->st;

    hashpipe_status_lock_safe(&st);
    // Record version
    hputs(st.buf, "GIT_VER", GIT_VERSION);
    // Get info from status buffer if present (no change if not present)
    hgets(st.buf, "BINDHOST", 80, bindhost);
    hgeti4(st.buf, "BINDPORT", &bindport);
    // Store bind host/port info etc in status buffer
    hputs(st.buf, "BINDHOST", bindhost);
    hputi4(st.buf, "BINDPORT", bindport);
    hputu4(st.buf, "MISSEDFE", 0);
    hputu4(st.buf, "MISSEDPK", 0);
    hashpipe_status_unlock_safe(&st);

    fprintf(stdout, "Max offset allowed is set to: %ld\n", MAX_HERA_CATCHER_IDX32);

#ifndef TIMING_TEST
    /* Set up pktsock */
    struct hashpipe_pktsock *p_ps = (struct hashpipe_pktsock *)
	malloc(sizeof(struct hashpipe_pktsock));

    if(!p_ps) {
        perror(__FUNCTION__);
        return -1;
    }

    // Make frame_size be a divisor of block size so that frames will be
    // contiguous in mapped mempory.  block_size must also be a multiple of
    // page_size.  Easiest way is to oversize the frames to be 16384 bytes, which
    // is bigger than we need, but keeps things easy.
    p_ps->frame_size = PKTSOCK_BYTES_PER_FRAME;
    // total number of frames
    p_ps->nframes = PKTSOCK_NFRAMES;
    // number of blocks
    p_ps->nblocks = PKTSOCK_NBLOCKS;

    int rv = hashpipe_pktsock_open(p_ps, bindhost, PACKET_RX_RING);
    if (rv!=HASHPIPE_OK) {
        hashpipe_error("hera_pktsock_thread", "Error opening pktsock.");
        pthread_exit(NULL);
    }

    // Store packet socket pointer in args
    args->user_data = p_ps;
#endif

    // Success!
    return 0;
}

static void *run(hashpipe_thread_args_t * args)
{
    // Local aliases to shorten access to args fields
    // Our output buffer happens to be a paper_input_databuf
    hera_catcher_input_databuf_t *db = (hera_catcher_input_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;

    st_p = &st;	// allow global (this source file) access to the status buffer

    // Flag that holds off the net thread
    int holdoff = 1;

    // Force ourself into the hold off state
    fprintf(stdout, "Setting CNETHOLD state to 1. Waiting for someone to set it to 0\n");
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "CNETHOLD", 1);
    hputs(st.buf, status_key, "holding");
    hashpipe_status_unlock_safe(&st);

    while(holdoff) {
	// We're not in any hurry to startup
	sleep(1);
	hashpipe_status_lock_safe(&st);
	// Look for CNETHOLD value
	hgeti4(st.buf, "CNETHOLD", &holdoff);
	if(!holdoff) {
	    // Done holding, so delete the key
	    hdel(st.buf, "CNETHOLD");
	    hputs(st.buf, status_key, "starting");
            fprintf(stdout, "Starting...\n");
	}
	hashpipe_status_unlock_safe(&st);
    }

    // Acquire first block to start
    if(hera_catcher_input_databuf_busywait_free(db, 0) != HASHPIPE_OK) {
	if (errno == EINTR) {
	    // Interrupted by signal, return -1
	    hashpipe_error(__FUNCTION__, "interrupted by signal waiting for free databuf");
	    pthread_exit(NULL);
	} else {
	    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
	    pthread_exit(NULL);
	}
    }

    // Initialize the newly acquired block
    initialize_block(db, 0, 0);

    /* Read network params */
    int bindport = CATCHER_PORT;
    // (N_BYTES_PER_PACKET excludes header)
    size_t expected_packet_size = OUTPUT_BYTES_PER_PACKET + sizeof(packet_header_t);

#ifndef TIMING_TEST
    /* Get pktsock from args*/
    struct hashpipe_pktsock * p_ps = (struct hashpipe_pktsock*)args->user_data;
    pthread_cleanup_push(free, p_ps);
    pthread_cleanup_push((void (*)(void *))hashpipe_pktsock_close, p_ps);

    // Drop all packets to date
    unsigned char *p_frame;
    while((p_frame=hashpipe_pktsock_recv_frame_nonblock(p_ps))) {
	hashpipe_pktsock_release_frame(p_frame);
    }

    hashpipe_status_lock_safe(&st);
    // Get info from status buffer
    hgeti4(st.buf, "BINDPORT", &bindport);
    hputu4(st.buf, "MISSEDPK", 0);
    hputs(st.buf, status_key, "running");
    hashpipe_status_unlock_safe(&st);
#endif

    /* Main loop */
    uint64_t packet_count = 0;
    uint64_t wait_ns = 0; // ns for most recent wait
    uint64_t recv_ns = 0; // ns for most recent recv
    uint64_t proc_ns = 0; // ns for most recent proc
    uint64_t min_wait_ns = 99999; // min ns per single wait
    uint64_t min_recv_ns = 99999; // min ns per single recv
    uint64_t min_proc_ns = 99999; // min ns per single proc
    uint64_t max_wait_ns = 0;     // max ns per single wait
    uint64_t max_recv_ns = 0;     // max ns per single recv
    uint64_t max_proc_ns = 0;     // max ns per single proc
    uint64_t elapsed_wait_ns = 0; // cumulative wait time per block
    uint64_t elapsed_recv_ns = 0; // cumulative recv time per block
    uint64_t elapsed_proc_ns = 0; // cumulative proc time per block
    uint64_t status_ns = 0; // User to fetch ns values from status buffer
    float ns_per_wait = 0.0; // Average ns per wait over 1 block
    float ns_per_recv = 0.0; // Average ns per recv over 1 block
    float ns_per_proc = 0.0; // Average ns per proc over 1 block
    unsigned int pktsock_pkts = 0;  // Stats counter from socket packet
    unsigned int pktsock_drops = 0; // Stats counter from socket packet
    uint64_t pktsock_pkts_total = 0;  // Stats total for socket packet
    uint64_t pktsock_drops_total = 0; // Stats total for socket packet
    struct timespec start, stop;
    struct timespec recv_start, recv_stop;

    while (run_threads()) {
        /* Read packet */
	clock_gettime(CLOCK_MONOTONIC, &recv_start);
	do {
	    clock_gettime(CLOCK_MONOTONIC, &start);
	    //p.packet_size = recv(up.sock, p.data, HASHPIPE_MAX_PACKET_SIZE, 0);
	    p_frame = hashpipe_pktsock_recv_udp_frame_nonblock(p_ps, bindport);
	    clock_gettime(CLOCK_MONOTONIC, &recv_stop);
	} while (!p_frame && run_threads());

	if(!run_threads()) break;

	// Make sure received packet size matches expected packet size.
        int packet_size = PKT_UDP_SIZE(p_frame) - 8; // -8 for the UDP header (not the *App* header!)
	if (expected_packet_size != packet_size) {
	    // Log warning and ignore wrongly sized packet
	    #ifdef DEBUG_NET
	    hashpipe_warn("hera_pktsock_thread", "Invalid pkt size (%d)", packet_size);
	    #endif
	    hashpipe_pktsock_release_frame(p_frame);
	    continue;
	}

	packet_count++;

        // Copy packet into any blocks where it belongs.
        const uint64_t mcnt = process_packet((hera_catcher_input_databuf_t *)db, p_frame);
	// Release frame back to kernel
	hashpipe_pktsock_release_frame(p_frame);

	clock_gettime(CLOCK_MONOTONIC, &stop);
	wait_ns = ELAPSED_NS(recv_start, start);
	recv_ns = ELAPSED_NS(start, recv_stop);
	proc_ns = ELAPSED_NS(recv_stop, stop);
	elapsed_wait_ns += wait_ns;
	elapsed_recv_ns += recv_ns;
	elapsed_proc_ns += proc_ns;
	// Update min max values
	min_wait_ns = MIN(wait_ns, min_wait_ns);
	min_recv_ns = MIN(recv_ns, min_recv_ns);
	min_proc_ns = MIN(proc_ns, min_proc_ns);
	max_wait_ns = MAX(wait_ns, max_wait_ns);
	max_recv_ns = MAX(recv_ns, max_recv_ns);
	max_proc_ns = MAX(proc_ns, max_proc_ns);

        if(mcnt != -1) {
            // Update status
            ns_per_wait = (float)elapsed_wait_ns / packet_count;
            ns_per_recv = (float)elapsed_recv_ns / packet_count;
            ns_per_proc = (float)elapsed_proc_ns / packet_count;

	    // Get stats from packet socket
	    hashpipe_pktsock_stats(p_ps, &pktsock_pkts, &pktsock_drops);

            hashpipe_status_lock_busywait_safe(&st);

            hputu8(st.buf, "NETMCNT", mcnt);
	    // Gbps = bits_per_packet / ns_per_packet
	    // (N_BYTES_PER_PACKET excludes header, so +8 for the header)
            hputr4(st.buf, "NETGBPS", 8*(N_BYTES_PER_PACKET+8)/(ns_per_recv+ns_per_proc));
            hputr4(st.buf, "NETWATNS", ns_per_wait);
            hputr4(st.buf, "NETRECNS", ns_per_recv);
            hputr4(st.buf, "NETPRCNS", ns_per_proc);

	    // Get and put min and max values.  The "get-then-put" allows the
	    // user to reset the min max values in the status buffer.
	    hgeti8(st.buf, "NETWATMN", (long *)&status_ns);
	    status_ns = MIN(min_wait_ns, status_ns);
            hputi8(st.buf, "NETWATMN", status_ns);

            hgeti8(st.buf, "NETRECMN", (long *)&status_ns);
	    status_ns = MIN(min_recv_ns, status_ns);
            hputi8(st.buf, "NETRECMN", status_ns);

            hgeti8(st.buf, "NETPRCMN", (long *)&status_ns);
	    status_ns = MIN(min_proc_ns, status_ns);
            hputi8(st.buf, "NETPRCMN", status_ns);

            hgeti8(st.buf, "NETWATMX", (long *)&status_ns);
	    status_ns = MAX(max_wait_ns, status_ns);
            hputi8(st.buf, "NETWATMX", status_ns);

            hgeti8(st.buf, "NETRECMX", (long *)&status_ns);
	    status_ns = MAX(max_recv_ns, status_ns);
            hputi8(st.buf, "NETRECMX", status_ns);

            hgeti8(st.buf, "NETPRCMX", (long *)&status_ns);
	    status_ns = MAX(max_proc_ns, status_ns);
            hputi8(st.buf, "NETPRCMX", status_ns);

            hputu8(st.buf, "NETPKTS",  pktsock_pkts);
            hputu8(st.buf, "NETDROPS", pktsock_drops);

            hgetu8(st.buf, "NETPKTTL", (long unsigned int*)&pktsock_pkts_total);
            hgetu8(st.buf, "NETDRPTL", (long unsigned int*)&pktsock_drops_total);
            hputu8(st.buf, "NETPKTTL", pktsock_pkts_total + pktsock_pkts);
            hputu8(st.buf, "NETDRPTL", pktsock_drops_total + pktsock_drops);

            hashpipe_status_unlock_safe(&st);

	    // Start new average
	    elapsed_wait_ns = 0;
	    elapsed_recv_ns = 0;
	    elapsed_proc_ns = 0;
	    packet_count = 0;
        }

        /* Will exit if thread has been cancelled */
        pthread_testcancel();
    }

#ifndef TIMING_TEST
    /* Have to close all push's */
    pthread_cleanup_pop(1); /* Closes push(hashpipe_pktsock_close) */
    pthread_cleanup_pop(1); /* Closes push(hashpipe_udp_close) */
#endif
    return NULL;
}

static hashpipe_thread_desc_t hera_catcher_net_thread = {
    name: "hera_catcher_net_thread",
    skey: "CNETSTAT",
    init: init,
    run:  run,
    ibuf_desc: {NULL},
    obuf_desc: {hera_catcher_input_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&hera_catcher_net_thread);
}

// vi: set ts=8 sw=4 noet :
