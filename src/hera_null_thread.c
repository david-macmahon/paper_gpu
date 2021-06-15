/* Take the input databuf that becomes 
 * available and set it to free. This 
 * thread serves as the end thread for 
 * checking a pipeline until that point.
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

#include "hashpipe.h"
#include "paper_databuf.h"

static void *run(hashpipe_thread_args_t * args){
   // Local aliases to shorten access to args fields
   // Our input buffer is a hera_catcher_bda_input_databuf
   hera_catcher_bda_input_databuf_t *db_in = (hera_catcher_bda_input_databuf_t *)args->ibuf;
   hashpipe_status_t st = args->st;
   const char * status_key = args->thread_desc->skey;

   int block_idx = 0;
   uint32_t bcnt;
   int rv;

   while(run_threads()){
      hashpipe_status_lock_safe(&st);
      hputi4(st.buf, "NULLBLKIN", block_idx);
      hputs(st.buf, status_key, "waiting");
      hputi8(st.buf, "NULLBCNT", bcnt);
      hashpipe_status_unlock_safe(&st);

      /* Wait for new block to become available */
      while ((rv=hera_catcher_bda_input_databuf_busywait_filled(db_in, block_idx)) != HASHPIPE_OK) {
          if (rv==HASHPIPE_TIMEOUT) {
              hashpipe_status_lock_safe(&st);
              hputs(st.buf, status_key, "blocked_in");
              hashpipe_status_unlock_safe(&st);
              continue;
          } else {
              hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
              pthread_exit(NULL);
              break;
          }
      }

      bcnt = db_in->block[block_idx].header.bcnt[0];

      hashpipe_status_lock_safe(&st);
      hputi8(st.buf, "NULLBCNT", bcnt);
      hashpipe_status_unlock_safe(&st);

      // Mark input block as free and advance
      if(hera_catcher_bda_input_databuf_set_free(db_in, block_idx) != HASHPIPE_OK) {
          hashpipe_error(__FUNCTION__, "error marking databuf %d free", block_idx);
          pthread_exit(NULL);
      }
      block_idx = (block_idx + 1) % CATCHER_N_BLOCKS;
      sleep(1);

      /* Check for cancel */
      pthread_testcancel();       
   } 
   return NULL;
}

hashpipe_thread_desc_t hera_null_thread = {
   name: "hera_null_thread",
   skey: "NULLSTAT",
   init: NULL,
   run: run,
   ibuf_desc: {hera_catcher_bda_input_databuf_create},
   obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&hera_null_thread);
}
