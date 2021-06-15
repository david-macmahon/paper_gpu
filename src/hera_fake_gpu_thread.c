/* Populates a GPU output buffer with fake data 
   to enable testing of the pipeline downstream.
   Mostly, baseline dependent integration
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

#include <xgpu.h>


#include "hashpipe.h"
#include "paper_databuf.h"

#define REGTILE_CHAN_LENGTH (4 * 4 * N_INPUTS/4 * (N_INPUTS/4+1) / 2)

static XGPUInfo xgpu_info;

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

static void *fake_gpu_thread_run(hashpipe_thread_args_t * args){
    paper_output_databuf_t *db = (paper_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char *status_key = args->thread_desc->skey;
   char config_fname[128] = "";

   // Flag that holds off the net thread
   int holdoff = 1;
 
   // Force this thread into holdoff until BDACONF is written
   fprintf(stdout, "Waiting for someone to supply BDACONF\n");
   hashpipe_status_lock_safe(&st);
   hputs(st.buf, "BDACONF", "");
   hputs(st.buf, status_key, "holding");
   hashpipe_status_unlock_safe(&st);
 
   while(holdoff) {
     sleep(1);
     hashpipe_status_lock_safe(&st);
     hgets(st.buf, "BDACONF", 128, config_fname);
     if (strlen(config_fname) > 1){
        holdoff = 0;
     }
     if(!holdoff) {
       // Done holding, so delete the key
       hputs(st.buf, status_key, "starting");
     }
     hashpipe_status_unlock_safe(&st);
   }


    /* Main loop */
    int rv;              // store return of buffer status calls
    uint64_t mcnt = 0;   // mcnt of each block
    int ant0, ant1, chan, pol;
    int block_idx = 0;
    off_t idx_regtile;
    int32_t fakereal = 0x00000001;
    int32_t fakeimag = 0x00000002;

    xgpuInfo(&xgpu_info);
    
    while (run_threads()) {

        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "waiting");
        hputi4(st.buf, "FGPUBKOUT", block_idx);
        hputi8(st.buf, "FGPUMCNT", mcnt);
        hashpipe_status_unlock_safe(&st);
        sleep(1);

        /* Wait for new block to be free, then clear it
         * if necessary and fill its header with new values.
         */
        while ((rv=paper_output_databuf_wait_free(db, block_idx))!= HASHPIPE_OK){
            if (rv==HASHPIPE_TIMEOUT){
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked");
                hashpipe_status_unlock_safe(&st);
                continue;
            }else{
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                pthread_exit(NULL);
                break;
            }
        }

        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "receiving");
        hashpipe_status_unlock_safe(&st);

        // Set block header
        db->block[block_idx].header.mcnt = mcnt;
        mcnt+=Nm;

        // Set all block data to zero
        int32_t *data_re = db->block[block_idx].data;
        int32_t *data_im = db->block[block_idx].data + xgpu_info.matLength;
        memset(data_re, 5, N_OUTPUT_MATRIX*sizeof(int32_t));

        // Populate block with fake data
        for(ant0 = 0; ant0 < N_INPUTS/2; ant0++){
           for(ant1 = ant0; ant1 < N_INPUTS/2; ant1++){
             idx_regtile = regtile_index(2*ant0, 2*ant1);
             //printf("%d\t %d\t %ld\n", ant0, ant1, idx_regtile);
             for(chan=0; chan<N_CHAN_PER_X; chan++){
               for(pol=0; pol<N_STOKES; pol++){
                 data_re[idx_regtile+(chan*REGTILE_CHAN_LENGTH)+pol] = fakereal+chan;
                 data_im[idx_regtile+(chan*REGTILE_CHAN_LENGTH)+pol] = fakeimag+chan;
               }
             }
           }
        } 
        // Mark block as full
        paper_output_databuf_set_filled(db, block_idx);
    
        // Setup for next block
        block_idx = (block_idx + 1)%db->header.n_block;

        /* Will exit if thread has been cancelled */
        pthread_testcancel();
    }

    // Thread success!
    return THREAD_OK;

}
 
static hashpipe_thread_desc_t hera_fake_gpu_thread = {
    name: "hera_fake_gpu_thread",
    skey: "FGPUSTAT",
    init: NULL,
    run:  fake_gpu_thread_run,
    ibuf_desc: {NULL},
    obuf_desc: {paper_output_databuf_create}
};

static __attribute__((constructor)) void ctor(){
  register_hashpipe_thread(&hera_fake_gpu_thread);
}
