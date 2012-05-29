/* vegas_net_thread.c
 *
 * Routine to read packets from network and put them
 * into shared memory blocks.
 */

#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>

#include <xgpu.h>

#include "fitshead.h"
#include "guppi_params.h"
#include "guppi_error.h"
#include "guppi_status.h"
#include "paper_databuf.h"
#include "guppi_udp.h"
#include "guppi_time.h"

#define STATUS_KEY "NETSTAT"  /* Define before guppi_threads.h */
#include "guppi_threads.h"
#include "guppi_defines.h"
#include "paper_thread.h"

#define DEBUG_NET

typedef struct {
    uint64_t mcnt;
    int      fid;	// Fengine ID
    int      xid;	// Xengine ID
} packet_header_t;

typedef struct {
    int initialized;
    uint64_t mcnt_start;
    uint64_t mcnt_offset;
    uint64_t mcnt_prior;
    int out_of_seq_cnt;
    int block_i;
    int sub_block_i;
    int block_active[N_INPUT_BLOCKS];
} block_info_t;

static struct guppi_status *st_p;

#if defined TIMING_TEST || defined NET_TIMING_TEST
static unsigned long fluffed_words = 0;
#endif

void print_pkt_header(packet_header_t * pkt_header) {

    printf("packet header : count %llu fid %d xid %d\n", (long long unsigned)pkt_header->mcnt, pkt_header->fid, pkt_header->xid);
}

void print_block_info(block_info_t * binfo) {
    printf("binfo : mcnt_start %llu mcnt_offset %llu block_i %d sub_block_i %d\n", 
           (long long unsigned)binfo->mcnt_start, (long long unsigned)binfo->mcnt_offset, binfo->block_i, binfo->sub_block_i);
}

void print_block_active(block_info_t * binfo) {
    int i;
    for(i=0;i<N_INPUT_BLOCKS;i++) { 
	if(i == binfo->block_i) {
		fprintf(stdout, "*%03d ", binfo->block_active[i]);	
	} else {
		fprintf(stdout, " %03d ", binfo->block_active[i]);	
	}
    }
    fprintf(stdout, "\n");
}

void print_ring_mcnts(paper_input_databuf_t *paper_input_databuf_p) {

    int i, j;

    for(i=0; i < N_INPUT_BLOCKS; i++) {
	for(j=0; j < N_SUB_BLOCKS_PER_INPUT_BLOCK; j++) {
		printf("block %d sub_block %d mcnt %lu\n", i, j, paper_input_databuf_p->block[i].header.mcnt[j]);
	}
    }
}

inline int inc_block_i(block_i) {
    return((block_i + 1) % N_INPUT_BLOCKS);
}

inline int dec_block_i(block_i) {
    return(block_i == 0 ? N_INPUT_BLOCKS - 1 : block_i - 1);
}

void get_header (struct guppi_udp_packet *p, packet_header_t * pkt_header) {

    uint64_t raw_header;
    raw_header = be64toh(*(unsigned long long *)p->data);
    pkt_header->mcnt        = raw_header >> 16;
    pkt_header->xid         = raw_header        & 0x000000000000000F;
    pkt_header->fid         = (raw_header >> 8) & 0x00000000000000FF;

#ifdef TIMING_TEST
    static int fake_mcnt=0;
    static int fake_fid=0;
    static int pkt_counter=0;

    if(pkt_counter == 8) {
	fake_mcnt++;
        fake_fid = 0;
	pkt_counter=0;
    } else if(pkt_counter % 8 == 0) {
	fake_fid += 4;
    }
    pkt_header->mcnt = fake_mcnt;
    pkt_counter++;
#endif
}

void set_block_filled(paper_input_databuf_t *paper_input_databuf_p, block_info_t *binfo, int block_i) { 

    static uint32_t missed_pkt_cnt;

    if(binfo->block_active[block_i]) {
	if(paper_input_databuf_set_filled(paper_input_databuf_p, block_i) != GUPPI_OK) {
	    guppi_error(__FUNCTION__, "error waiting for databuf filled call");
	    run_threads=0;
	    pthread_exit(NULL);
	    return;
	}

	missed_pkt_cnt += N_PACKETS_PER_BLOCK - binfo->block_active[block_i]; 
        guppi_status_lock_safe(st_p);
        hputu4(st_p->buf, "NETBKOUT", block_i);
        hputu4(st_p->buf, "MISSEDPK", missed_pkt_cnt);
        guppi_status_unlock_safe(st_p);

    	binfo->block_active[block_i] = 0;
    } 
}

void calc_block_indexes(block_info_t *binfo, uint64_t pkt_mcnt) {

    // calculate block and sub_block subscripts while taking care of count rollover
    if(pkt_mcnt >= binfo->mcnt_start) {
	binfo->mcnt_offset = pkt_mcnt - binfo->mcnt_start;
    } else {							 // we have a count rollover
	binfo->mcnt_offset = binfo->mcnt_offset + pkt_mcnt + 1;  // assumes that pkt_header.count is now very small, probably zero
    } 
    binfo->block_i     = (binfo->mcnt_offset) / N_SUB_BLOCKS_PER_INPUT_BLOCK % N_INPUT_BLOCKS; 
    binfo->sub_block_i = (binfo->mcnt_offset) % N_SUB_BLOCKS_PER_INPUT_BLOCK; 

    if(pkt_mcnt < binfo->mcnt_start && binfo->block_i == 0 && binfo->sub_block_i == 0) {
	binfo->mcnt_start = pkt_mcnt;	// on rollover, reset on block,sub 0
    }
} 

#define MAX_MCNT_DIFF 64 
int out_of_seq_mcnt(block_info_t * binfo, uint64_t pkt_mcnt) {

    if(abs(pkt_mcnt - binfo->mcnt_prior) <= MAX_MCNT_DIFF) {
        binfo->mcnt_prior = pkt_mcnt;
    	binfo->out_of_seq_cnt = 0;
	return 0;
    } else {
	printf("Out of seq : mcnt jumps from %llu to %llu\n", (long long unsigned)binfo->mcnt_prior, (long long unsigned)pkt_mcnt);
    	binfo->out_of_seq_cnt++;
	return 1;
    }
}

#define MAX_OUT_OF_SEQ 5
int handle_out_of_seq_mcnt(block_info_t * binfo) {

    if(binfo->out_of_seq_cnt > MAX_OUT_OF_SEQ) {
	printf("exceeded max (%d) out of sequence mcnts - restarting\n", MAX_OUT_OF_SEQ);
	binfo->initialized = 0;
    }
    return -1;
}

void initialize_block(paper_input_databuf_t * paper_input_databuf_p, block_info_t * binfo, uint64_t pkt_mcnt) {

    int i;

    paper_input_databuf_p->block[binfo->block_i].header.good_data = 0; 

    for(i=0; i<N_SUB_BLOCKS_PER_INPUT_BLOCK; i++) {
	paper_input_databuf_p->block[binfo->block_i].header.mcnt[i] = 0;
    }
}

void initialize_block_info(paper_input_databuf_t *paper_input_databuf_p, block_info_t * binfo, uint64_t pkt_mcnt) {

    int i;

    // We might be restarting so mark all currently active blocks, with the exception
    // of block_i, as filled. We will restart at block_i.  On program startup, this loop 
    // as no functional effect as no blocks are active and all block_active elements are 0. 
    for(i = 0; i < N_INPUT_BLOCKS; i++) {
	if(i == binfo->block_i) {
		binfo->block_active[i] = 0;	
	} else {
    		if(binfo->block_active[i]) {
			set_block_filled(paper_input_databuf_p, binfo, i);
		}
	}
    }		

    // On program startup block_i will be zero.  If we are restarting,  this will set 
    // us up to restart at the beginning of block_i.  TODO: mcnt rollover logic  
    binfo->mcnt_start = pkt_mcnt - binfo->block_i * N_SUB_BLOCKS_PER_INPUT_BLOCK;

    binfo->mcnt_prior = pkt_mcnt;
    binfo->out_of_seq_cnt = 0;
    binfo->initialized = 1;
}

uint64_t write_paper_packet_to_blocks(paper_input_databuf_t *paper_input_databuf_p, struct guppi_udp_packet *p) {

    static block_info_t binfo;
    packet_header_t pkt_header;
    const uint64_t *payload_p;
    int rv;
    int i, block_offset, sub_block_offset;
    uint64_t *dest_p;

    // housekeeping for each packet
    get_header(p, &pkt_header);
    if(! binfo.initialized) {
	// insist that we start on a multiple of sub_blocks/block
    	if(pkt_header.mcnt % N_SUB_BLOCKS_PER_INPUT_BLOCK != 0) {
		return -1;
    	}
	initialize_block_info(paper_input_databuf_p, &binfo, pkt_header.mcnt);
    }
    if(out_of_seq_mcnt(&binfo, pkt_header.mcnt)) {
    	return(handle_out_of_seq_mcnt(&binfo));
    }
    calc_block_indexes(&binfo, pkt_header.mcnt);
    if(! binfo.block_active[binfo.block_i]) {
	// new block
	if((rv = paper_input_databuf_busywait_free(paper_input_databuf_p, binfo.block_i)) != GUPPI_OK) {    
	    if (rv==GUPPI_TIMEOUT) {
		// run_threads is 0 (i.e. shutting down)
	        return -1;
	    } else {
	        guppi_error(__FUNCTION__, "error waiting for free databuf");
	        run_threads=0;
	        pthread_exit(NULL);
	        return -1;
	    }
	}

	initialize_block(paper_input_databuf_p, &binfo, pkt_header.mcnt); 
    }
    if(paper_input_databuf_p->block[binfo.block_i].header.mcnt[binfo.sub_block_i] == 0) {                
	// new sub_block
     	paper_input_databuf_p->block[binfo.block_i].header.mcnt[binfo.sub_block_i] = pkt_header.mcnt;  
    }
    binfo.block_active[binfo.block_i] += 1;	// increment packet count for block
    // end housekeeping

    // Calculate starting points for unpacking this packet into a sub_block.
    // One packet will never span more than one sub_block.
    block_offset     = binfo.block_i     * sizeof(paper_input_block_t);
    sub_block_offset = binfo.sub_block_i * sizeof(paper_input_sub_block_t);
    dest_p           = (uint64_t *)((uint8_t *)paper_input_databuf_p +
			sizeof(struct guppi_databuf)                 + 
			sizeof(guppi_databuf_cache_alignment)        +
			block_offset                                 + 
			sizeof(paper_input_header_t)                 + 
			sub_block_offset                             +
			pkt_header.fid*N_INPUTS_PER_FENGINE);
    payload_p        = (uint64_t *)(p->data+8);

    // unpack the packet, fluffing as we go
    for(i=0; i<(N_TIME*N_CHAN); i++) {
        uint64_t val = payload_p[i];
	// Using complex block size (cbs) of 32
	// 4 = cbs*sizeof(int8_t)/sizeof(uint64_t)
	dest_p[2*N_FENGINES*i] =  val & 0xf0f0f0f0f0f0f0f0LL;
	dest_p[2*N_FENGINES*i+4] = (val & 0x0f0f0f0f0f0f0f0fLL) << 4;
    }  // end upacking

#if defined TIMING_TEST || defined NET_TIMING_TEST
	fluffed_words += (N_TIME*N_CHAN);
#endif // TIMING_TEST || NET_TIMING_TEST

    // if all packets are accounted for, mark this block filled
    if(binfo.block_active[binfo.block_i] == N_PACKETS_PER_BLOCK) {
    	paper_input_databuf_p->block[binfo.block_i].header.good_data = 1; 
	// mark the block 2 back
	set_block_filled(paper_input_databuf_p, &binfo, dec_block_i(dec_block_i(binfo.block_i)));
#if defined PRINT_BLOCK_ACTIVE
	print_block_active(&binfo); 	
#endif // PRINT_BLOCK_ACTIVE
        return paper_input_databuf_p->block[binfo.block_i].header.mcnt[0];
    }

    return -1;
}

static int init(struct guppi_thread_args *args)
{
    /* Attach to status shared mem area */
    THREAD_INIT_STATUS(args->instance_id, STATUS_KEY);

    // Get sizing parameters
    XGPUInfo xgpu_info;
    xgpuInfo(&xgpu_info);

    /* Create paper_input_databuf for output buffer */
    THREAD_INIT_DATABUF(args->instance_id, paper_input_databuf, 4,
        xgpu_info.vecLength*sizeof(ComplexInput),
        args->output_buffer);

    // Success!
    return 0;
}

static void *run(void * _args)
{
    // Cast _args
    struct guppi_thread_args *args = (struct guppi_thread_args *)_args;

#ifdef DEBUG_SEMS
    fprintf(stderr, "s/tid %lu/NET/' <<.\n", pthread_self());
#endif

    THREAD_RUN_BEGIN(args);

    THREAD_RUN_SET_AFFINITY_PRIORITY(args);

    THREAD_RUN_ATTACH_STATUS(args->instance_id, st);
    st_p = &st;		// allow global (this source file) access to the status buffer

    /* Attach to paper_input_databuf */
    THREAD_RUN_ATTACH_DATABUF(args->instance_id, paper_input_databuf, db, args->output_buffer);

    /* Read in general parameters */
    struct guppi_params gp;
    struct sdfits pf;
    char status_buf[GUPPI_STATUS_SIZE];
    guppi_status_lock_busywait_safe(st_p);
    memcpy(status_buf, st_p->buf, GUPPI_STATUS_SIZE);
    guppi_status_unlock_safe(st_p);
    guppi_read_obs_params(status_buf, &gp, &pf);
    pthread_cleanup_push((void *)guppi_free_sdfits, &pf);

    /* Read network params */
    struct guppi_udp_params up;
    //guppi_read_net_params(status_buf, &up);
    paper_read_net_params(status_buf, &up);
    // Store bind host/port info in statsu buffer
    guppi_status_lock_busywait_safe(&st);
    hputs(st.buf, "BINDHOST", up.bindhost);
    hputi4(st.buf, "BINDPORT", up.bindport);
    guppi_status_unlock_safe(&st);

    struct guppi_udp_packet p;

    /* Give all the threads a chance to start before opening network socket */
    sleep(1);


#ifndef TIMING_TEST
    /* Set up UDP socket */
    int rv = guppi_udp_init(&up);
    if (rv!=GUPPI_OK) {
        guppi_error("guppi_net_thread",
                "Error opening UDP socket.");
        pthread_exit(NULL);
    }
    /* Set to non-blocking */
    fcntl(up.sock, F_SETFD, O_NONBLOCK);
    pthread_cleanup_push((void *)guppi_udp_close, &up);
#endif

    /* Main loop */
    unsigned waiting=-1;
    signal(SIGINT,cc);
    while (run_threads) {

#ifndef TIMING_TEST
        /* Wait for data */
#if 0
        rv = guppi_udp_wait(&up);
        if (rv!=GUPPI_OK) {
            if (rv==GUPPI_TIMEOUT) { 
                /* Set "waiting" flag */
                if (waiting!=1) {
                    guppi_status_lock_busywait_safe(st_p);
                    hputs(st_p->buf, STATUS_KEY, "waiting");
                    guppi_status_unlock_safe(st_p);
                    waiting=1;
                }
                continue; 
            } else {
                guppi_error("guppi_net_thread", 
                        "guppi_udp_wait returned error");
                perror("guppi_udp_wait");
                pthread_exit(NULL);
            }
        }
#endif
	
        /* Read packet */
	do {
	    p.packet_size = recv(up.sock, p.data, GUPPI_MAX_PACKET_SIZE, 0);
	} while (p.packet_size == -1 && (errno == EAGAIN || errno == EWOULDBLOCK) && run_threads);
	if(!run_threads) break;
        if (up.packet_size != p.packet_size) {
            if (p.packet_size != -1) {
                #ifdef DEBUG_NET
                guppi_warn("guppi_net_thread", "Incorrect pkt size");
                #endif
                continue; 
            } else {
                guppi_error("guppi_net_thread", 
                        "guppi_udp_recv returned error");
                perror("guppi_udp_recv");
                pthread_exit(NULL);
            }
        }
	
#endif
        /* Update status if needed */
        if (waiting!=0) {
            guppi_status_lock_busywait_safe(st_p);
            hputs(st_p->buf, STATUS_KEY, "receiving");
            guppi_status_unlock_safe(st_p);
            waiting=0;
        }

        // Copy packet into any blocks where it belongs.
        const uint64_t mcnt = write_paper_packet_to_blocks((paper_input_databuf_t *)db, &p);
        if(mcnt != -1) {
            guppi_status_lock_busywait_safe(&st);
            hputu8(st.buf, "NETMCNT", mcnt);
            guppi_status_unlock_safe(&st);
        }

#if defined TIMING_TEST || defined NET_TIMING_TEST
	static int loop_count=1;
	//if(loop_count == 1000000) run_threads = 0; 
	if(loop_count == 10*1000*1000) {
	    printf("fluffed %lu words\n", fluffed_words);
	    exit(0);
	}
	loop_count++;
#endif

        /* Will exit if thread has been cancelled */
        pthread_testcancel();
    }

    /* Have to close all push's */
#ifndef TIMING_TEST
    pthread_cleanup_pop(0); /* Closes push(guppi_udp_close) */
#endif
    pthread_cleanup_pop(0); /* Closes guppi_free_psrfits */
    THREAD_RUN_DETACH_DATAUF;
    THREAD_RUN_DETACH_STATUS;
    THREAD_RUN_END;

    return NULL;
}

static pipeline_thread_module_t module = {
    name: "paper_net_thread",
    type: PIPELINE_INPUT_THREAD,
    init: init,
    run:  run
};

static __attribute__((constructor)) void ctor()
{
  register_pipeline_thread_module(&module);
}

// vi: set ts=8 sw=4 noet :
