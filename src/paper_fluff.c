#include <stdio.h>
#include <pthread.h>
//#include <stdlib.h>
//#include <unistd.h>
//#include <stdint.h>
//#include <time.h>

#include <smmintrin.h>
#include <immintrin.h>

#include "paper_fluff.h"

#define N_WORD128_PER_PACKET  (N_BYTES_PER_PACKET/sizeof(__m128i))

#if 0
typedef union {
  __m128i m128;
  __m64   m64[2];
} vec_t;
#define m128(v) (v.m128)
#define m64(v,i) (v.m64[i])
#endif // 0

/*
 * The xGPU input buffer is time x channel x antenna x pol x complexity x 4[samples]. Each real/imag
 * input component is a 1Byte + 1Byte complex value.
 *
 * The netthread output buffer has a shape dictated by the structure of a single
 * packet. It has dimensions time x antenna x channel x times-per-packet x pol.
 * Each pol is a 4bit + 4bit complex value.
 * Thus, the fluffing thread has to do some gymnastics to write the GPU input buffer
 * correctly. You'll probably want this to hand -- https://software.intel.com/sites/landingpage/IntrinsicsGuide/ (!)
 * 
 * This code isn't generic anymore, it assumes that Nt = 2. Sometimes the variable Nt is used to provide
 * hints about what might need changing for different Nt values.
*/

struct data {
    uint64_t * in;
    uint64_t * out;
    int n_threads;
    int thread_id;
};

void paper_fluff_threaded(void *args)
{
  //uint64_t v0, v1, v2, v3;
  //vec_t in0, in1, in2, in3;
  //vec_t v0, v1, v2, v3;
  struct data *d = args;
  uint64_t *in = d->in;
  uint64_t *out = d->out;
  int n_threads = d->n_threads;
  int thread_id = d->thread_id;

  int m, c, a, j, k, nn, mm, wa, wc, rn, i;

  __m256i int0, int1, int2, int3, outx, outy;

  __m256i *p_in = (__m256i *)in;
  __m256i *p_out;
  const __m256i maskhi = _mm256_set_epi64x(0xff00ff00ff00ff00ULL, 0xff00ff00ff00ff00ULL,
                                           0xff00ff00ff00ff00ULL, 0xff00ff00ff00ff00ULL);
  const __m256i masklo = _mm256_set_epi64x(0x00ff00ff00ff00ffULL, 0x00ff00ff00ff00ffULL,
                                           0x00ff00ff00ff00ffULL, 0x00ff00ff00ff00ffULL);
  const __m256i shuffle_map = _mm256_set_epi64x(0x0f0d0b090e0c0a08ULL, 0x0705030106040200ULL,
                                                0x0f0d0b090e0c0a08ULL, 0x0705030106040200ULL);
  //__m256i row[16], temp0[16], temp1[16], temp2[16];
  __m256i row[8], temp[8];

  p_out = (__m256i *)out;
  
  /*
   * Loop over 256-bit lines of the input buffer.
   * Input order is mcount x antenna x channel x time x polarization x complexity.
   * Data samples are 4+4 bits.
   * Thus, a 256-bit line consists of 1 mcount x 1 ant x 8 channels x 2 times x 2 pols x 1Byte
   * We need to gather data to be written to the output buffer in order:
   * time/4 x channel x antenna x polarization x complexity x t%4, with 8+8-bit data.
   * An 256-bit line of the output buffer is thus:
   * 1*[time/4] x 1 channel x 2 antennas x 2 pols x 2 complexities x 4 times
   *
   * So.....
   * Let's load 2 mcounts x 2 ants x 16 chans, 2 times, 2 pols into 16 256-bit registers and transpose.
   * Hopefully we'll end up with:
   * 1*[time/4] x 16 chans x 4 ants x 2 pols x 2 complexities x 4 times, ready for the GPU buffer.
   */

  // First loop is over time/4 (i.e., mcounts in steps of 2)
  for(m=thread_id*4/Nt; m<Nm; m=m+(n_threads*4/Nt)) { //dealing with 4 time samples, which come from two m-addresses
    // Second loop over antennas, loading 4 each iteration
    for(a=0; a<Na; a=a+4) {
      // Third loop over channels, loading 8 in each iteration
      for(c=0; c<Nc; c=c+8) {
        // Load a 16-chan-sample by 4 antenna matrix into 8 x 256 bit registers
        // We want:
        // Ant0C[0:7]T[0:1] // 256 bits
        // Ant0C[0:7]T[2:3]
        // Ant1C[0:7]T[0:1]
        // Ant1C[0:7]T[2:3]
        // Ant2C[0:7]T[0:1]
        // Ant2C[0:7]T[2:3]
        // Ant3C[0:7]T[0:1]
        // Ant3C[0:7]T[2:3]
        for(i=0; i<8; i++) {
          row[i]  = _mm256_load_si256(p_in + paper_input_databuf_data_idx256(m + i%Nt, a+i/Nt, c, 0));
        }

        /* TRANSPOSE in blocks of 32 bits*/
        for(j=0; j<4; j++) {
          temp[j]   = _mm256_unpacklo_epi32(row[2*j], row[2*j+1]);
          temp[j+4] = _mm256_unpackhi_epi32(row[2*j], row[2*j+1]);
        }
        for(k=0; k<4; k++) {
          row[k]   = _mm256_unpacklo_epi64(temp[2*k], temp[2*k+1]);
          row[k+4] = _mm256_unpackhi_epi64(temp[2*k], temp[2*k+1]);
        }
        /* END TRANSPOSE -- actually we have transposed 4 4x4 blocks of the matrix.
           For the full transpose we'd need an unpackX_epi128, which isn't available.
           The data order here is (for Nt=2):

           ANT[ 0: 1]C0T[0:3], ANT[ 0: 1]C4T[0:3] // 256 bits
           ANT[ 2: 3]C0T[0:3], ANT[ 2: 3]C4T[0:3]
           ANT[ 0: 1]C2T[0:3], ANT[ 0: 1]C6T[0:3]
           ANT[ 2: 3]C2T[0:3], ANT[ 2: 3]C6T[0:3]

           ANT[ 0: 1]C1T[0:3], ANT[ 0: 1]C5T[0:3]
           ANT[ 2: 3]C1T[0:3], ANT[ 2: 3]C5T[0:3]
           ANT[ 0: 1]C3T[0:3], ANT[ 0: 1]C7T[0:3]
           ANT[ 2: 3]C3T[0:3], ANT[ 2: 3]C7T[0:3]
        */


        /*
           We still need to swizzle the blocks of 32 bits.
           They are currently ant x chan x time x pol x complexity
           We require chan x ant x pol x complexity x time
           And we must fluff each real/imag component to 8 bits
        */

            //unsigned long long foo0[4];
            //unsigned long long foo1[4];
        // Loop over rows. For convenience separate out into top and bottom of matrix
        for(nn=0; nn<2; nn++) { // Loop over two blocks (top and bottom)
          for(mm=0; mm<4; mm++) { // Loop over 4 rows per block
            rn = 4*nn + mm;   // row number
            wa = (mm%2) * 2; // antenna offset. I.e., antenna being processed is a+wa
            wc = nn + 2*(mm/2);   // channel offset. I.e., channel being processed is c+wc
            // fluff and load
            // shift so that we have four bitvectors, where the top 4 bits
            // of each 16-bit word are consecutive values
            // eg, for n=0, p=0; int0 is:
            //    A0C0T0xr, A0C0T0xi, A0C0T0yr, A0C0T0yi // 16-bits
            //    A0C0T1xr, A0C0T1xi, A0C0T1yr, A0C0T1yi 
            //    A0C0T2xr, A0C0T2xi, A0C0T2yr, A0C0T2yi 
            //    A0C0T3xr, A0C0T3xi, A0C0T3yr, A0C0T3yi 
            //    A1C0T0xr, A1C0T0xi, A1C0T0yr, A1C0T0yi // 16-bits
            //    A1C0T1xr, A1C0T1xi, A1C0T1yr, A1C0T1yi 
            //    A1C0T2xr, A1C0T2xi, A1C0T2yr, A1C0T2yi 
            //    A1C0T3xr, A1C0T3xi, A1C0T3yr, A1C0T3yi 
            //    A0C4T0xr, A0C4T0xi, A0C4T0yr, A0C4T0yi // 16-bits
            //    A0C4T1xr, A0C4T1xi, A0C4T1yr, A0C4T1yi 
            //    A0C4T2xr, A0C4T2xi, A0C4T2yr, A0C4T2yi 
            //    A0C4T3xr, A0C4T3xi, A0C4T3yr, A0C4T3yi 
            //    A1C4T0xr, A1C4T0xi, A1C4T0yr, A1C4T0yi // 16-bits
            //    A1C4T1xr, A1C4T1xi, A1C4T1yr, A1C4T1yi 
            //    A1C4T2xr, A1C4T2xi, A1C4T2yr, A1C4T2yi 
            //    A1C4T3xr, A1C4T3xi, A1C4T3yr, A1C4T3yi 
            int0 = row[rn];
            // eg, for n=0, p=0; int1 is:
            //    A0C0T0xi, A0C0T0yr, A0C0T0yi, 00000000 // 16-bits
            //    A0C0T1xi, A0C0T1yr, A0C0T1yi, 00000000 
            //    A0C0T2xi, A0C0T2yr, A0C0T2yi, 00000000 
            //    A0C0T3xi, A0C0T3yr, A0C0T3yi, 00000000 
            //    A1C0T0xi, A1C0T0yr, A1C0T0yi, 00000000 // 16-bits
            //    A1C0T1xi, A1C0T1yr, A1C0T1yi, 00000000 
            //    A1C0T2xi, A1C0T2yr, A1C0T2yi, 00000000 
            //    A1C0T3xi, A1C0T3yr, A1C0T3yi, 00000000 
            //    A0C4T0xi, A0C4T0yr, A0C4T0yi, 00000000 // 16-bits
            //    A0C4T1xi, A0C4T1yr, A0C4T1yi, 00000000 
            //    A0C4T2xi, A0C4T2yr, A0C4T2yi, 00000000 
            //    A0C4T3xi, A0C4T3yr, A0C4T3yi, 00000000 
            //    A1C4T0xi, A1C4T0yr, A1C4T0yi, 00000000 // 16-bits
            //    A1C4T1xi, A1C4T1yr, A1C4T1yi, 00000000 
            //    A1C4T2xi, A1C4T2yr, A1C4T2yi, 00000000 
            //    A1C4T3xi, A1C4T3yr, A1C4T3yi, 00000000 
            int1 = _mm256_slli_epi16(row[rn], 4);
            // eg, for n=0, p=0; int2 is:
            //    A0C0T0yr, A0C0T0yi, 00000000, 00000000 // 16-bits
            //    A0C0T1yr, A0C0T1yi, 00000000, 00000000 
            //    A0C0T2yr, A0C0T2yi, 00000000, 00000000 
            //    A0C0T3yr, A0C0T3yi, 00000000, 00000000 
            //    A1C0T0yr, A1C0T0yi, 00000000, 00000000 // 16-bits
            //    A1C0T1yr, A1C0T1yi, 00000000, 00000000 
            //    A1C0T2yr, A1C0T2yi, 00000000, 00000000 
            //    A1C0T3yr, A1C0T3yi, 00000000, 00000000 
            //    A0C4T0yr, A0C4T0yi, 00000000, 00000000 // 16-bits
            //    A0C4T1yr, A0C4T1yi, 00000000, 00000000 
            //    A0C4T2yr, A0C4T2yi, 00000000, 00000000 
            //    A0C4T3yr, A0C4T3yi, 00000000, 00000000 
            //    A1C4T0yr, A1C4T0yi, 00000000, 00000000 // 16-bits
            //    A1C4T1yr, A1C4T1yi, 00000000, 00000000 
            //    A1C4T2yr, A1C4T2yi, 00000000, 00000000 
            //    A1C4T3yr, A1C4T3yi, 00000000, 00000000 
            int2 = _mm256_slli_epi16(row[rn], 8);
            // eg, for n=0, p=0; int3 is:
            //    A0C0T0yi, 00000000, 00000000, 00000000 // 16-bits
            //    A0C0T1yi, 00000000, 00000000, 00000000 
            //    A0C0T2yi, 00000000, 00000000, 00000000 
            //    A0C0T3yi, 00000000, 00000000, 00000000 
            //    A1C0T0yi, 00000000, 00000000, 00000000 // 16-bits
            //    A1C0T1yi, 00000000, 00000000, 00000000 
            //    A1C0T2yi, 00000000, 00000000, 00000000 
            //    A1C0T3yi, 00000000, 00000000, 00000000 
            //    A0C4T0yi, 00000000, 00000000, 00000000 // 16-bits
            //    A0C4T1yi, 00000000, 00000000, 00000000 
            //    A0C4T2yi, 00000000, 00000000, 00000000 
            //    A0C4T3yi, 00000000, 00000000, 00000000 
            //    A1C4T0yi, 00000000, 00000000, 00000000 // 16-bits
            //    A1C4T1yi, 00000000, 00000000, 00000000 
            //    A1C4T2yi, 00000000, 00000000, 00000000 
            //    A1C4T3yi, 00000000, 00000000, 00000000 
            int3 = _mm256_slli_epi16(row[rn], 12);

            // shift down with sign extension. AKA, "fluff"
            // This makes the top 8 bits of the int0 and int2 values valid 8-bit signed numbers
            // The shift by 12 makes the lower 8-bits of the int1 and in3 values 8-bit signed numbers
            // eg, for rn=0; int0 is: A0C0T[0:3]xr-8b, ..., A1C0T[0:3]xr-8b, ..., A0C4T[0:3]xr-8b, ..., A1C4T[0:3]xr-8b, ...,
            int0 = _mm256_srai_epi16(int0, 4);
            // eg, for rn=0; int1 is: A0C0T[0:3]xi-16b, ..., A1C0T[0:3]xi-16b, ..., A0C4T[0:3]xi-16b, ..., A1C4T[0:3]xi-16b, ...,
            int1 = _mm256_srai_epi16(int1, 12);
            // eg, for rn=0; int2 is: A0C0T[0:3]yr-8b, ..., A1C0T[0:3]yr-8b, ..., A0C4T[0:3]yr-8b, ..., A1C4T[0:3]yr-8b, ...,
            int2 = _mm256_srai_epi16(int2, 4);
            // eg, for rn=0; int3 is: A0C0T[0:3]yi-16b, ..., A1C0T[0:3]yi-16b, ..., A0C4T[0:3]yi-16b, ..., A1C4T[0:3]yi-16b, ...,
            int3 = _mm256_srai_epi16(int3, 12);

            // Mask all values to 8b signed ints, ready to OR together
            // eg, for n=0, p=0; int0 is: A0C0T0xr-8b, 8'b0 ...
            // eg, for rn=0; int0 is: A0C0T[0:3]xr-8b, 8'b0, A1C0T[0:3]xr-8b, 8'b0, A0C4T[0:3]xr-8b, 8'b0, A1C4T[0:3]xr-8b, 8'b0,
            int0 = _mm256_and_si256(int0, maskhi);
            // eg, for rn=0; int1 is: 8'b0, A0C0T[0:3]xi-8b, 8'b0, A1C0T[0:3]xi-8b, 8'b0, A0C4T[0:3]xi-8b, 8'b0, A1C4T[0:3]xi-8b,
            int1 = _mm256_and_si256(int1, masklo);
            // eg, for rn=0; int2 is: A0C0T[0:3]yr-8b, 8'b0, A1C0T[0:3]yr-8b, 8'b0, A0C4T[0:3]yr-8b, 8'b0, A1C4T[0:3]yr-8b, 8'b0,
            int2 = _mm256_and_si256(int2, maskhi);
            // eg, for rn=0; int1 is: 8'b0, A0C0T[0:3]yi-8b, 8'b0, A1C0T[0:3]yi-8b, 8'b0, A0C4T[0:3]yi-8b, 8'b0, A1C4T[0:3]yi-8b,
            int3 = _mm256_and_si256(int3, masklo);

            // OR together pairs of vectors
            // eg, for rn=0; outx is: 
            //    A0C0T0xr, A0C0T0xi // 16-bits
            //    A0C0T1xr, A0C0T1xi 
            //    A0C0T2xr, A0C0T2xi 
            //    A0C0T3xr, A0C0T3xi 
            //    A1C0T0xr, A1C0T0xi // 16-bits
            //    A1C0T1xr, A1C0T1xi 
            //    A1C0T2xr, A1C0T2xi 
            //    A1C0T3xr, A1C0T3xi 
            //    A0C4T0xr, A0C4T0xi // 16-bits
            //    A0C4T1xr, A0C4T1xi 
            //    A0C4T2xr, A0C4T2xi 
            //    A0C4T3xr, A0C4T3xi 
            //    A1C4T0xr, A1C4T0xi // 16-bits
            //    A1C4T1xr, A1C4T1xi 
            //    A1C4T2xr, A1C4T2xi 
            //    A1C4T3xr, A1C4T3xi 
            outx = _mm256_or_si256(int0, int1);
            // eg, for rn=0; outy is:
            //    A0C0T0yr, A0C0T0yi // 16-bits
            //    A0C0T1yr, A0C0T1yi 
            //    A0C0T2yr, A0C0T2yi 
            //    A0C0T3yr, A0C0T3yi 
            //    A1C0T0yr, A1C0T0yi // 16-bits
            //    A1C0T1yr, A1C0T1yi 
            //    A1C0T2yr, A1C0T2yi 
            //    A1C0T3yr, A1C0T3yi 
            //    A0C4T0yr, A0C4T0yi // 16-bits
            //    A0C4T1yr, A0C4T1yi 
            //    A0C4T2yr, A0C4T2yi 
            //    A0C4T3yr, A0C4T3yi 
            //    A1C4T0yr, A1C4T0yi // 16-bits
            //    A1C4T1yr, A1C4T1yi 
            //    A1C4T2yr, A1C4T2yi 
            //    A1C4T3yr, A1C4T3yi 
            outy = _mm256_or_si256(int2, int3);

            // Interleave these to gather X and Y pols
            // and separate out the two channels.
            // and can be fixed later with a bit shuffle
            // eg, for rn=0 int0 = 
            //    [x/y]A[0:1]C0T[0:3][r/i] // 256-bits
            int0 = _mm256_permute2x128_si256(outx, outy, 0x02); //channel 0
            //    and then make A[0:1][x/y]C0T[0:3][r/i] // 256-bits
            int0 = _mm256_permute4x64_epi64(int0, 0xd8); //channel 0
            // eg, for rn=0, int1 =
            //    [x/y]A[0:1]C4T[0:3][r/i] // 256-bits
            int1 = _mm256_permute2x128_si256(outx, outy, 0x13); //channel 4
            //    and then make A[0:1][x/y]C4T[0:3][r/i] // 256-bits
            int1 = _mm256_permute4x64_epi64(int1, 0xd8); //channel 0

            // Finally perform a bytewise transpose using a shuffle
            // Outputs are ordered 2-pol x 2-antenna x 1-chan x 4-time x 2-complexity
            // We want 1-chan x 2-antenna x 2-pol x complexity x 4-time ([ants 0..1][real/imag][time 0..3])
            int0 = _mm256_shuffle_epi8(int0, shuffle_map); // Channel 0
            int1 = _mm256_shuffle_epi8(int1, shuffle_map); // Channel 4

            // Store these two 256-bit words with _mm256_stream_si256
            _mm256_stream_si256(p_out + paper_gpu_input_databuf_data_idx_tca_256(Nt*m, c+wc, a+wa), int0);
            _mm256_stream_si256(p_out + paper_gpu_input_databuf_data_idx_tca_256(Nt*m, c+wc+4, a+wa), int1);
            //_mm256_stream_si256((__m256i *) foo0, int0);
            //_mm256_stream_si256((__m256i *) foo1, int1);
            //printf("Writing int0: t:%04d, c:%03d, a:%03d, Val: 0x%016llx_%016llx_%016llx_%016llx\n", Nt*m, c+wc, a+wa, foo0[0], foo0[1], foo0[2], foo0[3]);
            //printf("Writing int1: t:%04d, c:%03d, a:%03d, Val: 0x%016llx_%016llx_%016llx_%016llx\n", Nt*m, c+wc+4, a+wa, foo1[0], foo1[1], foo1[2], foo1[3]);
          }
        }
      }
    }
  }
  // Return number of 64 bit words fluffed
  //return Nm*Na*2*N_WORD128_PER_PACKET;
  //return N_BYTES_PER_BLOCK >> 3;
}


#define N_THREADS 2
int paper_fluff(uint64_t * in, uint64_t * out)
{
  int i;
  struct data args[N_THREADS];
  pthread_t threads[N_THREADS];

  for (i=0; i<N_THREADS; i++) {
    args[i].in = in;
    args[i].out = out;
    args[i].n_threads = N_THREADS;
    args[i].thread_id = i;
  }

  for(i=0; i<N_THREADS; i++){
    if (pthread_create(&threads[i], NULL, (void *)&paper_fluff_threaded, (void *)(&args[i]))) {
        fprintf(stderr, "Failed to create fluff thread %d\n", i);
    }
  }
  for(i=0; i<N_THREADS; i++){
    pthread_join(threads[i], NULL);
  }
    
  return N_BYTES_PER_BLOCK >> 3;
}


int paper_fluff_lut(const uint64_t const * const in, uint64_t * out)
{
  uint16_t * in16  = (uint16_t *) in;
  uint32_t * out32 = (uint32_t *) out;
  int i;

  for(i=0; i<N_BYTES_PER_BLOCK>>1; i++) {
    out32[i] = lut[in16[i]];
  }

  //uint64_t * in64  = (uint64_t *) in;
  //uint64_t * out64 = (uint64_t *) out;
  //int i;

  //for(i=0; i<N_BYTES_PER_BLOCK>>3; i++) {
  //  out64[2*i]     = (lut[(in64[i]>>48)] << 32) +        (lut[(in64[i]>>32) & 0xff]);
  //  out64[2*i + 1] = (lut[(in64[i]>>16) & 0xff] << 32) + (lut[(in64[i] & 0xff)]);
  //}

  // Return number of 64 bit words fluffed
  return N_BYTES_PER_BLOCK >> 3;
}
