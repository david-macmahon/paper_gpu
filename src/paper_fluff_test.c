#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "paper_fluff.h"

#define TEST_ITERATIONS (10)

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

#define N_WORDS_PER_BLOCK_IN (N_BYTES_PER_BLOCK/sizeof(uint64_t))
#define N_WORDS_PER_BLOCK_OUT (2*N_WORDS_PER_BLOCK_IN)
#define N_WORDS_PER_PACKET  (N_BYTES_PER_PACKET/sizeof(uint64_t))

void fluff_check_random()
{
  /*
  The in-buffer order is time[Nm] x antenna x channel x time[Nt] x pol x complexity
  Each real+imag sample is 8-bits
  Make a test input vector to test each of these elements:
  complexity-test : all real parts 1, all imag parts 2
  time-test : each 4+4 bit time sample increments by 1, starting at 0, wrapping at 256
  channel-test: each 4+4 but channel increments by 1, starting at 0, wrapping at 256
  antenna-test: each 4+4 bit ant-pol sample increments by 1, wrapping at 256
  random-test: write in random integers between +/- 7
  */
  uint64_t *in, *out;
  uint8_t tv_in;
  int8_t real, imag, real_in, imag_in;
  int m, a, p, c, t;
  if(posix_memalign((void **)&in,  CACHE_ALIGNMENT,   N_BYTES_PER_BLOCK)
  || posix_memalign((void **)&out, CACHE_ALIGNMENT, 2*N_BYTES_PER_BLOCK)) {
    printf("cannot allocate memory\n");
    return;
  }

  srand(0x1234);

  printf("Running random number test...");
  for (m=0; m<Nm; m++){
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (t=0; t<Nt; t++) {
            real = ((rand() % 16) - 8) & 0xf;
            imag = ((rand() % 16) - 8) & 0xf;
            *((uint8_t *) in + paper_input_databuf_data_idx8(m,a,p,c,t)) = (uint8_t)((real << 4) + imag);
          }
        }
      }
    }
  }


  paper_fluff(in, out);
  int errs = 0;
  for (t=0; t<Nm*Nt; t++) {
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          real = *((int8_t *)out + paper_gpu_input_databuf_data_idx8(t, c, a, p, 1));
          imag = *((int8_t *)out + paper_gpu_input_databuf_data_idx8(t, c, a, p, 0));
          // indexing in input buffer is by m and t, so compute these
          tv_in = *((uint8_t *)in + paper_input_databuf_data_idx8(t/Nt, a, p, c, t%Nt));
          real_in = (tv_in >> 4) & 0xf;
          imag_in = tv_in & 0xf;
          if (real_in > 7) {
            real_in = real_in - 16;
          }
          if (imag_in > 7) {
            imag_in = imag_in - 16;
          }
          if (real_in != real) {
             errs++;
             //printf("real in: %d. real out: %d, 0x%x\n", real_in, real, tv_in);
          }
          if (imag_in != imag) {
             //printf("imag in: %d. imag out: %d, 0x%x\n", imag_in, imag, tv_in);
             errs++;
          }
        }
      }
    }
  } 
  if (errs == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED with %d errors\n", errs);
  }
}

void fluff_check_complexity()
{
  /*
  The in-buffer order is time[Nm] x antenna x channel x time[Nt] x pol x complexity
  Each real+imag sample is 8-bits
  Make a test input vector to test each of these elements:
  complexity-test : all real parts 1, all imag parts 2
  time-test : each 4+4 bit time sample increments by 1, starting at 0, wrapping at 256
  channel-test: each 4+4 but channel increments by 1, starting at 0, wrapping at 256
  antenna-test: each 4+4 bit ant-pol sample increments by 1, wrapping at 256
  */
  uint64_t *in, *out;
  int m, a, p, c, t, x;
  if(posix_memalign((void **)&in,  CACHE_ALIGNMENT,   N_BYTES_PER_BLOCK)
  || posix_memalign((void **)&out, CACHE_ALIGNMENT, 2*N_BYTES_PER_BLOCK)) {
    printf("cannot allocate memory\n");
    return;
  }
  // Time test
  for (m=0; m<Nm; m++){
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (t=0; t<Nt; t++) {
            // real in MSBs
            *((uint8_t *) in + paper_input_databuf_data_idx8(m,a,p,c,t)) = ((1) << 4) + (0);
          }
        }
      }
    }
  }


  printf("Running complexity-ordering test...");
  paper_fluff(in, out);
  int errs = 0;
  for (t=0; t<Nm*Nt; t++) {
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (x=0; x<Nx; x++) {
            if (*((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)) != x%8) {
              printf("Fluff test failed. t=%d, c=%d, a=%d, p=%d, x=%d. Expected %d, got 0x%llx!\n", t, c, a, p, x, x%8, *(unsigned long long *)((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)));
              errs++;
            }
          }
        }
      }
    }
  } 
  if (errs == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }
}

void fluff_check_pol()
{
  /*
  The in-buffer order is time[Nm] x antenna x channel x time[Nt] x pol x complexity
  Each real+imag sample is 8-bits
  Make a test input vector to test each of these elements:
  complexity-test : all real parts 1, all imag parts 2
  time-test : each 4+4 bit time sample increments by 1, starting at 0, wrapping at 256
  channel-test: each 4+4 but channel increments by 1, starting at 0, wrapping at 256
  antenna-test: each 4+4 bit ant-pol sample increments by 1, wrapping at 256
  */
  uint64_t *in, *out;
  int m, a, p, c, t, x;
  if(posix_memalign((void **)&in,  CACHE_ALIGNMENT,   N_BYTES_PER_BLOCK)
  || posix_memalign((void **)&out, CACHE_ALIGNMENT, 2*N_BYTES_PER_BLOCK)) {
    printf("cannot allocate memory\n");
    return;
  }
  // Time test
  for (m=0; m<Nm; m++){
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (t=0; t<Nt; t++) {
            *((uint8_t *) in + paper_input_databuf_data_idx8(m,a,p,c,t)) = ((p % 8) << 4) + (p % 8);
          }
        }
      }
    }
  }


  printf("Running pol-ordering test...");
  paper_fluff(in, out);
  int errs = 0;
  for (t=0; t<Nm*Nt; t++) {
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (x=0; x<Nx; x++) {
            if (*((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)) != p%8) {
              printf("Fluff test failed. t=%d, c=%d, a=%d, p=%d, x=%d. Expected %d, got 0x%llx!\n", t, c, a, p, x, p%8, *(unsigned long long *)((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)));
              errs++;
            }
          }
        }
      }
    }
  } 
  if (errs == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }
}

void fluff_check_time()
{
  /*
  The in-buffer order is time[Nm] x antenna x channel x time[Nt] x pol x complexity
  Each real+imag sample is 8-bits
  Make a test input vector to test each of these elements:
  complexity-test : all real parts 1, all imag parts 2
  time-test : each 4+4 bit time sample increments by 1, starting at 0, wrapping at 256
  channel-test: each 4+4 but channel increments by 1, starting at 0, wrapping at 256
  antenna-test: each 4+4 bit ant-pol sample increments by 1, wrapping at 256
  */
  uint64_t *in, *out;
  int m, a, p, c, t, x;
  if(posix_memalign((void **)&in,  CACHE_ALIGNMENT,   N_BYTES_PER_BLOCK)
  || posix_memalign((void **)&out, CACHE_ALIGNMENT, 2*N_BYTES_PER_BLOCK)) {
    printf("cannot allocate memory\n");
    return;
  }
  // Time test
  for (m=0; m<Nm; m++){
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (t=0; t<Nt; t++) {
            *((uint8_t *) in + paper_input_databuf_data_idx8(m,a,p,c,t)) = (((m*Nt + t) % 8) << 4) + ((m*Nt + t) % 8);
          }
        }
      }
    }
  }


  printf("Running time-ordering test...");
  paper_fluff(in, out);
  int errs = 0;
  for (t=0; t<Nm*Nt; t++) {
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (x=0; x<Nx; x++) {
            if (*((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)) != t%8) {
              errs++;
            }
          }
        }
      }
    }
  } 
  if (errs == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }
}

void fluff_check_ant()
{
  /*
  The in-buffer order is time[Nm] x antenna x channel x time[Nt] x pol x complexity
  Each real+imag sample is 8-bits
  Make a test input vector to test each of these elements:
  complexity-test : all real parts 1, all imag parts 2
  time-test : each 4+4 bit time sample increments by 1, starting at 0, wrapping at 256
  channel-test: each 4+4 but channel increments by 1, starting at 0, wrapping at 256
  antenna-test: each 4+4 bit ant-pol sample increments by 1, wrapping at 256
  */
  uint64_t *in, *out;
  int m, a, p, c, t, x;
  if(posix_memalign((void **)&in,  CACHE_ALIGNMENT,   N_BYTES_PER_BLOCK)
  || posix_memalign((void **)&out, CACHE_ALIGNMENT, 2*N_BYTES_PER_BLOCK)) {
    printf("cannot allocate memory\n");
    return;
  }
  // Time test
  for (m=0; m<Nm; m++){
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (t=0; t<Nt; t++) {
            *((uint8_t *) in + paper_input_databuf_data_idx8(m,a,p,c,t)) = ((a % 8) << 4) + (a % 8);
          }
        }
      }
    }
  }

  for (m=0; m<1; m++){
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<1; c++) {
          for (t=0; t<1; t++) {
              //printf("Initializing. m=%d, a=%d, p=%d, c=%d, t=%d. Expected %d, got %llx\n", m,a,p,c,t,a%8, *(unsigned long long *)((uint8_t *)in + paper_input_databuf_data_idx8(m,a,p,c,t)));
          }
        }
      }
    }
  }


  printf("Running ant-ordering test...");
  paper_fluff(in, out);
  int errs = 0;
  for (t=0; t<Nm*Nt; t++) {
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (x=0; x<Nx; x++) {
            if (*((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)) != a%8) {
              printf("Fluff test failed. t=%d, c=%d, a=%d, p=%d, x=%d. Expected %d, got 0x%llx!\n", t, c, a, p, x, a%8, *(unsigned long long *)((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)));
              errs++;
            }
          }
        }
      }
    }
  } 
  if (errs == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }
}

void fluff_check_chan()
{
  /*
  The in-buffer order is time[Nm] x antenna x channel x time[Nt] x pol x complexity
  Each real+imag sample is 8-bits
  Make a test input vector to test each of these elements:
  complexity-test : all real parts 1, all imag parts 2
  time-test : each 4+4 bit time sample increments by 1, starting at 0, wrapping at 16
  channel-test: each 4+4 bit channel increments by 1, starting at 0, wrapping at 16 
  antenna-test: each 4+4 bit ant-pol sample increments by 1, wrapping at 256
  */
  uint64_t *in, *out;
  int m, a, p, c, t, x;
  if(posix_memalign((void **)&in,  CACHE_ALIGNMENT,   N_BYTES_PER_BLOCK)
  || posix_memalign((void **)&out, CACHE_ALIGNMENT, 2*N_BYTES_PER_BLOCK)) {
    printf("cannot allocate memory\n");
    return;
  }
  // Time test
  for (m=0; m<Nm; m++){
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (t=0; t<Nt; t++) {
            *((uint8_t *) in + paper_input_databuf_data_idx8(m,a,p,c,t)) = ((c % 8) << 4) + (c % 8);
          }
        }
      }
    }
  }

  // Time test
  for (m=0; m<Nm; m++){
    for (a=0; a<Na; a++) {
      for (p=0; p<1; p++) {
        for (c=0; c<Nc; c++) {
          for (t=0; t<1; t++) {
            //printf("Initializing. m=%d, a=%d, p=%d, c=%d, t=%d. 0x%llx\n", m,a,p,c,t, *(unsigned long long *)((uint8_t *) in + paper_input_databuf_data_idx8(m,a,p,c,t)));
          }
        }
      }
    }
  }


  printf("Running chan-ordering test...");
  paper_fluff(in, out);
  int errs = 0;
  for (t=0; t<Nm*Nt; t++) {
    for (a=0; a<Na; a++) {
      for (p=0; p<Np; p++) {
        for (c=0; c<Nc; c++) {
          for (x=0; x<Nx; x++) {
            if (*((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)) != c%8) {
              printf("Fluff test failed. t=%d, c=%d, a=%d, p=%d, x=%d. Expected %d, got %d!\n", t, c, a, p, x, c%8, *((uint8_t *) out + paper_gpu_input_databuf_data_idx8(t, c, a, p, x)));
              errs++;
            }
          }
        }
      }
    }
  } 
  if (errs == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }
}

void fluff_check()
{
  fluff_check_complexity();
  fluff_check_pol();
  fluff_check_ant();
  fluff_check_time();
  fluff_check_chan();
  fluff_check_random();
}

int main(int argc, char *argv[])
{
  int i, j;
  uint64_t *in, *out;
  struct timespec start, stop;
  int fluffed;

  setbuf(stdout, NULL); // send printf-s to screen immediately

  fprintf(stdout, "Built %s %s@%s\n", BUILD_DATE, BUILD_USER, BUILD_HOST);
  fprintf(stdout, "%s\n\n", GIT_VERSION);

  if(posix_memalign((void **)&in,  CACHE_ALIGNMENT,   N_BYTES_PER_BLOCK)
  || posix_memalign((void **)&out, CACHE_ALIGNMENT, 2*N_BYTES_PER_BLOCK)) {
    printf("cannot allocate memory\n");
    return 1;
  }

  //printf("in  = %p\n", in);
  //printf("out = %p\n", out);

#if 0
  in[0] = htobe64(0x0123456789abcdef);
  in[1] = htobe64(0xfedcba9876543210);

  uint8_t * cin = (uint8_t *)in;
  for(i=0; i<2; i++) {
    for(j=0; j<8; j++) {
      printf("%02x ", cin[8*i+j]);
    }
    printf("\n");
  }
  printf("\n");

  paper_fluff(in, out);

  uint8_t * cout = (uint8_t *)out;
  for(i=0; i<4; i++) {
    for(j=0; j<8; j++) {
      printf("%02x ", cout[8*i+j]);
    }
    printf("\n");
  }

  return 0;
}
#else


  printf("N_CHAN_PER_PACKET=%u\n", N_CHAN_PER_PACKET);
  printf("N_TIME_PER_PACKET=%u\n", N_TIME_PER_PACKET);
  printf("N_WORDS_PER_PACKET=%lu\n", N_WORDS_PER_PACKET);
  printf("N_PACKETS_PER_BLOCK=%u\n", N_PACKETS_PER_BLOCK);
  printf("N_BYTES_PER_BLOCK=%u\n", N_BYTES_PER_BLOCK);



#ifdef DEBUG_FLUFF
  fluffed = paper_fluff(in, out);
#else
  for(j=0; j<4; j++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(i=0; i<TEST_ITERATIONS; i++) {
      fluffed = paper_fluff(in, out);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    printf("fluffed %d words in %.6f ms (%.3f us per packet, %.3f Gbps)\n",
        fluffed, ELAPSED_NS(start, stop)/1e6/TEST_ITERATIONS,
        ELAPSED_NS(start, stop)/1e3/TEST_ITERATIONS/N_PACKETS_PER_BLOCK,
        ((float)fluffed*TEST_ITERATIONS*8*sizeof(uint64_t))/ELAPSED_NS(start, stop));
  }
#endif // DEBUG_FLUFF_GEN

  fluff_check();

  return 0;
}
#endif // 1
