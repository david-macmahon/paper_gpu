/* BDA thread that takes the auto correlations from
 * the disk thread and uploads them to redis
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <hdf5.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <hiredis.h>

#include "hashpipe.h"
#include "paper_databuf.h"
#include "lzf_filter.h"

static void *run(hashpipe_thread_args_t * args)
{
  // Local aliases to shorten access to args fields
  // Our input buffer is a paper_input_databuf
  // Our output buffer is a paper_gpu_input_databuf
  hera_catcher_autocorr_databuf_t *db_in = (hera_catcher_autocorr_databuf_t *)args->ibuf;
  hashpipe_status_t st = args->st;
  const char * status_key = args->thread_desc->skey;

  // A buffer for the real parts of a single auto-corr. Used for writing to redis
  float auto_corr_n[N_CHAN_TOTAL];
  float auto_corr_e[N_CHAN_TOTAL];

  // Redis connection
  redisContext *c;
  redisReply *reply;
  const char *hostname = "redishost";
  int redisport = 6379;
  int use_redis = 1;

  int rv;
  int ant;
  int blkin = 0;
  int chan;
  uint32_t acc_len;
  uint64_t offset;

  struct timeval redistimeout = {0, 100000 }; // 0.1 seconds
  c = redisConnectWithTimeout(hostname, redisport, redistimeout);
  if (c == NULL || c->err) {
      if (c) {
          fprintf(stderr, "Redis connection error: %s\n", c->errstr);
          redisFree(c);
          use_redis = 0;
      } else {
          fprintf(stderr, "Connection error: can't allocate redis context\n");
          use_redis = 0;
      }
  }

  while (run_threads()){
    hashpipe_status_lock_safe(&st);
    // Get the integration time reported by the correlator
    hgetu4(st.buf, "INTTIME", &acc_len);
    hashpipe_status_unlock_safe(&st);

    // Wait for new input block to be filled
    while((rv=hera_catcher_autocorr_databuf_busywait_filled(db_in, blkin))!= HASHPIPE_OK){
       if (rv==HASHPIPE_TIMEOUT){
          hashpipe_status_lock_safe(&st);
          hputs(st.buf, status_key, "blocked_in");
          hashpipe_status_unlock_safe(&st);
       } else {
          hashpipe_error(__FUNCTION__, "Error waiting for autocorr filled databuf");
          pthread_exit(NULL);
          break;
       }
    }

    hashpipe_status_lock_safe(&st);
    hputs(st.buf, status_key, "writing");
    hputi4(st.buf, "AUTOBKIN", blkin);
    hashpipe_status_unlock_safe(&st);

    // Write autocorrs to redis
    if(use_redis){
      for (ant=0; ant<N_ANTS; ant++) {
         if (db_in->block[blkin].header.ant[ant] == 1){
            for (chan=0; chan<N_CHAN_TOTAL; chan++){
              offset = hera_catcher_autocorr_databuf_idx32(ant);
              auto_corr_n[chan] = (float) db_in->block[blkin].data[offset + chan*N_STOKES*2]/ acc_len;
              auto_corr_e[chan] = (float) db_in->block[blkin].data[offset + chan*N_STOKES*2 + 2]/ acc_len;
              //auto_corr_n[chan] = db_in->block[blkin].data[offset + chan*N_STOKES*2];
              //auto_corr_e[chan] = db_in->block[blkin].data[offset + chan*N_STOKES*2 + 2];
            }

            //fprintf(stdout, "Reporting autocorrs for ant %d to redis\n", ant);
            reply = redisCommand(c, "SET auto:%dn %b", ant, auto_corr_n, (size_t) (sizeof(float) * N_CHAN_TOTAL));
            freeReplyObject(reply);
            reply = redisCommand(c, "SET auto:%de %b", ant, auto_corr_e, (size_t) (sizeof(float) * N_CHAN_TOTAL));
            freeReplyObject(reply);
         }else{
            reply = redisCommand(c, "DEL auto:%dn", ant);
            freeReplyObject(reply);
            reply = redisCommand(c, "DEL auto:%de", ant);
            freeReplyObject(reply);         
         }
      }

      //reply = redisCommand(c, "SET auto:timestamp %lf", julian_time);
      reply = redisCommand(c, "SET auto:timestamp  %b", &(db_in->block[blkin].header.julian_time), (size_t) (sizeof(double)));
      freeReplyObject(reply);
    }

    // Mark block as free and advance
    if (hera_catcher_autocorr_databuf_set_free(db_in, blkin) != HASHPIPE_OK) {
      hashpipe_error(__FUNCTION__, "error marking autocorr databuf %d free", blkin);
      pthread_exit(NULL);
    }

    blkin = (blkin + 1) % AUTOCORR_N_BLOCKS;

    /* Check for cancel */
    pthread_testcancel();
  }

  // Thread success!
  return NULL;    
}

static hashpipe_thread_desc_t hera_catcher_autocorr_thread = {
    name: "hera_catcher_autocorr_thread",
    skey: "AUTOSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {hera_catcher_autocorr_databuf_create},
    obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&hera_catcher_autocorr_thread);
}
