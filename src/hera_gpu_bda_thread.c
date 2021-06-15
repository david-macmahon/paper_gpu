/* Takes the output from the GPU and 
 * averages over time for the shorter 
 * baselines according to a config file.
 * Output is stored in CASPER ordered 
 * shared memory segments.
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

#define PRINT_TEST

#define LOG(x)           ((uint32_t)(log((x))/log(2))) // yields msb loc
#define CHECK_PWR2(x)    (!((x)&((x)-1)))

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

typedef struct {
    uint32_t baselines;   // Num baselines in bin
    uint16_t samp_in_bin; // Num samples per baseline
    uint16_t *ant_pair_0;
    uint16_t *ant_pair_1;
    uint32_t *bcnt;
} bda_info_t;

static XGPUInfo xgpu_info;
static bda_info_t binfo[N_BDABUF_BINS];

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

// Lookup table mapping baseline_idx to regtile_idx
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

static uint64_t get_sample_from_mcnt(uint64_t curr_mcnt, uint64_t start_bda_mcnt, 
                                     int int_count){
   uint64_t sample = (curr_mcnt - start_bda_mcnt) / (TIME_DEMUX * int_count);
   //fprintf(stderr, "curr: %ld, start: %ld, acclen:%d, sample: %ld\n", curr_mcnt, start_bda_mcnt, int_count, sample);
   return sample; 
}

static int init_bda_info(bda_info_t *binfo, char *config_fname){
   FILE *fp;
   int i,j,k, a0, a1, inttime, bin;
   uint32_t blctr[] = {0,0,0,0,0};
   uint32_t bctr = 0;

   if((fp=fopen(config_fname,"r")) == NULL){
      printf("Cannot open the configuration file.\n");
      exit(1);
   }
   while(fscanf(fp, "%d %d %d", &a0, &a1, &inttime)!=EOF){
      if(!CHECK_PWR2(inttime)){
        printf("(%d,%d): Samples to integrate not power of 2!\n",a0,a1);
        exit(1);
      }
      if(inttime == 0) continue;
      blctr[LOG(inttime)]+=1;
   }

   for(j=0; j<N_BDABUF_BINS; j++){ 
     binfo[j].baselines = blctr[j];
     binfo[j].samp_in_bin = N_MAX_INTTIME/(1<<j);
     blctr[j] = 0;
   }

   // malloc for storing ant pairs and bcnt values
   for(j=0; j<N_BDABUF_BINS; j++){
     binfo[j].ant_pair_0 = (uint16_t *)malloc(binfo[j].baselines * sizeof(uint16_t));
     binfo[j].ant_pair_1 = (uint16_t *)malloc(binfo[j].baselines * sizeof(uint16_t));
     binfo[j].bcnt       = (uint32_t *)malloc(binfo[j].baselines * binfo[j].samp_in_bin * sizeof(uint32_t));
   }
   rewind(fp); //re-read antpairs to store them
   while(fscanf(fp, "%d %d %d", &a0, &a1, &inttime)!=EOF){
     if (inttime == 0) continue;
     bin = LOG(inttime); 
     binfo[bin].ant_pair_0[blctr[bin]] = a0;
     binfo[bin].ant_pair_1[blctr[bin]++] = a1;
   }
   fclose(fp);

   // Init the bcnt values (just an incremental counter)
   for(j=0; j<N_BDABUF_BINS; j++){
     for(i=0; i<binfo[j].baselines; i++){
       for(k=0; k<binfo[j].samp_in_bin; k++){
         binfo[j].bcnt[i*binfo[j].samp_in_bin + k] = bctr++;
       }
     }
   }

   // Success!   
   return 0; 
} 

/* Initialize the header with config params in binfo */
static int init_bda_block_header(hera_bda_block_t *bdablk, int bcnt_offset){
   int i,j;

   for(j=0; j<N_BDABUF_BINS; j++){
     bdablk->header[j].baselines = binfo[j].baselines;
     memcpy(bdablk->header[j].ant_pair_0, binfo[j].ant_pair_0, binfo[j].baselines * sizeof(uint16_t));
     memcpy(bdablk->header[j].ant_pair_1, binfo[j].ant_pair_1, binfo[j].baselines * sizeof(uint16_t));
     memcpy(bdablk->header[j].bcnt, binfo[j].bcnt, binfo[j].baselines * binfo[j].samp_in_bin * sizeof(uint32_t));
     bdablk->header[j].datsize = binfo[j].samp_in_bin*binfo[j].baselines * N_COMPLEX_PER_BASELINE * 2 * sizeof(uint32_t);

     // offset all bcnts by the sample currently being processed. 
     // This ensures that bcnt is always increasing while the 
     // correlator is running. It also ties bcnt to mcnt.
     for(i=0; i< binfo[j].baselines*binfo[j].samp_in_bin; i++){
       bdablk->header[j].bcnt[i] += bcnt_offset;
     }
   }
 
   return 0; 
}

static int init(struct hashpipe_thread_args *args)
{ 
   // Get sizing parameters
   xgpuInfo(&xgpu_info);

   if(init_idx_map()) {
     return -1;
   }

   // Success!
   return 0;
}

static void *run(hashpipe_thread_args_t * args)
{
   // Local aliases to shorten access to args fields
   // Our input buffer happens to be a paper_ouput_databuf
   paper_output_databuf_t *idb = (paper_output_databuf_t *)args->ibuf;
   hera_bda_databuf_t *odb = (hera_bda_databuf_t *)args->obuf;
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
       fprintf(stdout, "Starting...\n");
     }
     hashpipe_status_unlock_safe(&st);
   }

   // Initialize binfo with config file params
   init_bda_info(binfo, config_fname); 

   int j;
   uint64_t total_baselines = 0;
   for(j=0; j<N_BDABUF_BINS;j++)
      total_baselines += binfo[j].samp_in_bin * binfo[j].baselines; 

#ifdef PRINT_TEST
   fprintf(stdout,"N_ANTS:%d\n", N_ANTS);
   fprintf(stdout,"Number of channels per X-Eng: %d\n",N_CHAN_PER_X);
   fprintf(stdout,"Number of BCNTS in a buffer: %ld\n", total_baselines);
#endif

   // Write the number of baselines per integration
   // bin to redis for downstream stuff
   hashpipe_status_lock_safe(&st);
   hputu8(st.buf, "NBL2SEC", binfo[0].baselines);
   hputu8(st.buf, "NBL4SEC", binfo[1].baselines);
   hputu8(st.buf, "NBL8SEC", binfo[2].baselines);
   hputu8(st.buf, "NBL16SEC",binfo[3].baselines);
   hputu8(st.buf, "NBDABLS", total_baselines);
   hputu4(st.buf, "BDASAMP", 0);
   hashpipe_status_unlock_safe(&st);

   // Allocate memory to the output data blocks based on the config file
   int blk_id;
   hera_bda_block_t *buf;   

   for(blk_id=0; blk_id < odb->header.n_block; blk_id++){
     for(j=0; j<N_BDABUF_BINS; j++){
       buf = &(odb->block[blk_id]);
       buf->header[j].ant_pair_0 = (uint16_t *) malloc(binfo[j].baselines * sizeof(uint16_t));
       buf->header[j].ant_pair_1 = (uint16_t *) malloc(binfo[j].baselines * sizeof(uint16_t));
       buf->header[j].bcnt       = (uint32_t *) malloc(binfo[j].baselines * binfo[j].samp_in_bin * sizeof(uint32_t));
       buf->header[j].mcnt       = (uint64_t *) malloc(binfo[j].baselines * binfo[j].samp_in_bin * sizeof(uint64_t));
     }
     
     init_bda_block_header(buf, 0);

     for(j=0; j<N_BDABUF_BINS; j++){
       buf->data[j] = malloc(buf->header[j].datsize);
       if(!buf->data[j]){
         hashpipe_error(__FUNCTION__, "Could not allocate BDA buffer");
         pthread_exit(NULL);
       }
       memset(buf->data[j], 0, buf->header[j].datsize);
     }
   }

   /* Main loop */
   int rv;
   int curblock_in = 0;
   int curblock_out = 0;
   struct timespec start, stop;
   int casper_chan, gpu_chan;
   uint64_t bl;
   int pol;
   unsigned long long datoffset = 0;
   uint16_t ant0, ant1;
   unsigned long long idx_baseline; 
   off_t idx_regtile;
   int32_t re, im;    //pktdata_t is 32bits
   char integ_status[17];
   int int_count; // number of mcnts integrated in GPU
   uint64_t start_bda_mcount; // mcount to start BD integration 
   uint64_t sample = 0;
   uint64_t prev_samp = -1;
   int sample_loc, bl_samp_loc;

   while (run_threads()) {

     hashpipe_status_lock_safe(&st);
     hputs(st.buf, status_key, "waiting");
     hashpipe_status_unlock_safe(&st);

     // Wait for new block to be filled
     while ((rv=paper_output_databuf_wait_filled(idb, curblock_in))
           != HASHPIPE_OK) {
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

     // Got a new block. Decide what to do with it.
     // Note waiting status,
     // query integrating status
     // and, if armed, start count
     hashpipe_status_lock_safe(&st);
     hputi4(st.buf, "BDABLKIN", curblock_in);
     hputu8(st.buf, "BDAMCNT", idb->block[curblock_in].header.mcnt);
     hgets(st.buf,  "INTSTAT", 16, integ_status);
     hashpipe_status_unlock_safe(&st);

     // If the GPU integration status is off, you 
     // can skip BDA as well. Do something only if
     // the status is start.
     if(!strcmp(integ_status, "off")) {
         // Mark input block as free and advance
         hashpipe_databuf_set_free((hashpipe_databuf_t *)idb, curblock_in);
         curblock_in = (curblock_in + 1) % idb->header.n_block;
         // Skip to next input buffer
         continue;
     }

     // Don't need to deal with the cases where integ_status is "start"
     // because this is handled by the gpu_thread already. Adding another
     // INTSTAT type variable is redundant and using the same INTSTAT
     // skips over these conditions directly to "on" because the gpu thread
     // changes it almost immediately.
 
     // If integ_status is "stop" or "on"

     hashpipe_status_lock_safe(&st);
     hputs(st.buf, status_key, "processing");
     hgeti8(st.buf, "INTSYNC", (long int *)&start_bda_mcount);
     hgeti4(st.buf, "INTCOUNT", &int_count);
     hashpipe_status_unlock_safe(&st);

     sample = get_sample_from_mcnt(idb->block[curblock_in].header.mcnt, 
                                   start_bda_mcount+int_count, int_count);
     if (sample != (prev_samp+1)){
        fprintf(stderr, "Missed sample %ld!!\n", prev_samp);
        prev_samp = sample-1;
     }
     prev_samp++;

     if (sample%N_MAX_INTTIME == 0){
        // Wait for new output block to be free
        while ((rv=hera_bda_databuf_wait_free(odb, curblock_out)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked_out");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                pthread_exit(NULL);
                break;
            }
        }
        // Init header of newly acquired block
        init_bda_block_header(&(odb->block[curblock_out]), (sample/N_MAX_INTTIME)*total_baselines);
        for(j=0; j<N_BDABUF_BINS; j++){ 
          memset(odb->block[curblock_out].data[j], 0, odb->block[curblock_out].header[j].datsize);
        }
        buf = &(odb->block[curblock_out]); 
     }

     // Note processing status, current input block
     hashpipe_status_lock_safe(&st);
     hputs(st.buf, status_key, "processing");
     hputi4(st.buf, "BDABLKIN", curblock_in);
     hputi4(st.buf, "BDABLKOUT", curblock_out);
     hputu4(st.buf, "BDASAMP", sample);
     hashpipe_status_unlock_safe(&st);
     
     clock_gettime(CLOCK_MONOTONIC, &start);

     /* Perform the baseline dependent averaging  */

     int32_t *pf_re  = idb->block[curblock_in].data;
     int32_t *pf_im  = idb->block[curblock_in].data + xgpu_info.matLength;
      
     for(j=0; j<N_BDABUF_BINS; j++){ //intbuf loop

       // Location of this sample within the bin: sample_loc
       sample_loc = (sample/(1<<j)) % binfo[j].samp_in_bin;

       for(bl=0; bl< buf->header[j].baselines; bl++){
         //fprintf(stderr,"Bin:%d, 2**j:%d\n",j,(1<<j));
         ant0 = buf->header[j].ant_pair_0[bl]; 
         ant1 = buf->header[j].ant_pair_1[bl];
         idx_baseline = casper_index(2*ant0, 2*ant1, N_INPUTS); 

         // baseline dependent offset of this sample
         bl_samp_loc = bl*binfo[j].samp_in_bin + sample_loc;
         buf->header[j].mcnt[bl_samp_loc] = idb->block[curblock_in].header.mcnt;

         //fprintf(stderr, "blk:%d\tant0:%d\tant1:%d\tbcnt:%d\n",curblock_out,ant0,ant1,buf->header[j].bcnt[bl_samp_loc]);

         for(pol=0; pol<N_STOKES; pol++){
           idx_regtile = idx_map[idx_baseline/2 + pol]; 

           for(casper_chan=0; casper_chan<N_CHAN_PER_X; casper_chan++){
             gpu_chan = casper_chan;
             re = pf_re[(gpu_chan*REGTILE_CHAN_LENGTH)+idx_regtile];
             im = pf_im[(gpu_chan*REGTILE_CHAN_LENGTH)+idx_regtile];
             datoffset = hera_bda_buf_data_idx(bl_samp_loc, gpu_chan, pol);
             buf->data[j][datoffset] += re;
             buf->data[j][datoffset+1] += -im;
           }
         }
       }
    
     } // intbuf

     clock_gettime(CLOCK_MONOTONIC, &stop);

     hashpipe_status_lock_safe(&st);
     hputr4(st.buf, "BDASECS", (float)ELAPSED_NS(start,stop)/1e9);
     hputr4(st.buf, "BDAMBPS", (float)(total_baselines*N_CHAN_PER_X*N_STOKES*8*1e3)/ELAPSED_NS(start,stop));
     hashpipe_status_unlock_safe(&st);

     // Mark input databuf as free and advance
     paper_output_databuf_set_free(idb, curblock_in);
     curblock_in = (curblock_in + 1) % idb->header.n_block;

     if(sample%N_MAX_INTTIME == N_MAX_INTTIME-1){
       // Mark output databuf as filled as advance
       hera_bda_databuf_set_filled(odb, curblock_out);
       curblock_out = (curblock_out + 1) % odb->header.n_block;
     }

     /* Will exit if thread has been cancelled */
     pthread_testcancel();
  }

    // Thread success!
    return NULL;
}

static hashpipe_thread_desc_t gpu_bda_thread = {
    name: "hera_gpu_bda_thread",
    skey: "BDASTAT",
    init: init,
    run:  run,
    ibuf_desc: {paper_output_databuf_create},
    obuf_desc: {hera_bda_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&gpu_bda_thread);
}

