/*
 * hera_catcher_bda_disk_thread.c
 *
 * Writes correlated data to disk as hdf5 files.
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

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#define N_DATA_DIMS (4)
#define N_CHAN_PROCESSED (N_CHAN_TOTAL / (CATCHER_CHAN_SUM_BDA))
#define N_CHAN_RECEIVED (N_CHAN_TOTAL)
#define N_BL_PER_WRITE (32)

#define CPTR(VAR,CONST) ((VAR)=(CONST),&(VAR))

static hid_t complex_id;
static hid_t boolenumtype;
//static hid_t boolean_id;
static uint64_t bcnts_per_file;

typedef enum {
    FALSE,
    TRUE
} bool_t;

typedef struct {
    double e;
    double n;
    double u;
} enu_t;

typedef struct {
    int a;
    int b;
} bl_t;

typedef struct {
    int ant0;
    int ant1;
    int tsamp;
} bl_bda_t;

typedef struct {
    hid_t file_id;
    hid_t header_gid;
    hid_t data_gid;
    hid_t extra_keywords_gid;
    hid_t visdata_did;
    hid_t flags_did;
    hid_t nsamples_did;
    hid_t time_array_did;
    hid_t integration_time_did;
    hid_t ant_1_array_did;
    hid_t ant_2_array_did;
    hid_t visdata_fs;
    hid_t flags_fs;
    hid_t nsamples_fs;
    hid_t time_array_fs;
    hid_t integration_time_fs;
    hid_t ant_1_array_fs;
    hid_t ant_2_array_fs;
} hdf5_id_t;

static hid_t open_hdf5_from_template(char * sourcename, char * destname)
{
    int read_fd, write_fd;
    struct stat stat_buf;
    off_t offset = 0;
    hid_t status;
    
    read_fd = open(sourcename, O_RDONLY);
    if (read_fd < 0) {
        hashpipe_error(__FUNCTION__, "error opening %s", sourcename);
        pthread_exit(NULL);
    }

    // Get the file size
    fstat(read_fd, &stat_buf);

    write_fd = open(destname, O_WRONLY | O_CREAT, stat_buf.st_mode);
    if (write_fd < 0) {
        hashpipe_error(__FUNCTION__, "error opening %s", destname);
        pthread_exit(NULL);
    }

    sendfile(write_fd, read_fd, &offset, stat_buf.st_size);
    
    close(read_fd);
    close(write_fd);

    status = H5Fopen(destname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (status < 0) {
        hashpipe_error(__FUNCTION__, "error opening %s as HDF5 file", destname);
        pthread_exit(NULL);
    }

    return status;
}

# define FILTER_H5_LZF 32000
# define FILTER_H5_BITSHUFFLE 32008
static void init_data_dataset(hdf5_id_t *id){

   hsize_t data_dims[N_DATA_DIMS] = {bcnts_per_file, 1, N_CHAN_PROCESSED, N_STOKES};
   hsize_t chunk_dims[N_DATA_DIMS] = {N_BL_PER_WRITE, 1, N_CHAN_PROCESSED, N_STOKES};

   hid_t file_space = H5Screate_simple(N_DATA_DIMS, data_dims, NULL);

   //make plist for LZF datasets
   int r;
   r = register_lzf();
   if (r<0) {
     hashpipe_error(__FUNCTION__, "Failed to register LZF filter");
     pthread_exit(NULL);
   }
   hid_t plist_lzf = H5Pcreate(H5P_DATASET_CREATE);
   H5Pset_chunk(plist_lzf, N_DATA_DIMS, chunk_dims);
   H5Pset_shuffle(plist_lzf);
   H5Pset_filter(plist_lzf, H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL, 0, NULL);

   // make plist for non-compressed datasets
   hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
   H5Pset_layout(plist, H5D_CHUNKED);
   H5Pset_chunk(plist, N_DATA_DIMS, chunk_dims);

   // Now we have the dataspace properties, create the datasets
   id->visdata_did = H5Dcreate(id->data_gid, "visdata", complex_id, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);
   if (id->visdata_did < 0) {
       hashpipe_error(__FUNCTION__, "Failed to create visdata dataset");
       pthread_exit(NULL);
   }

   id->nsamples_did = H5Dcreate(id->data_gid, "nsamples", H5T_IEEE_F32LE, file_space, H5P_DEFAULT, plist_lzf, H5P_DEFAULT);
   if (id->nsamples_did < 0) {
       hashpipe_error(__FUNCTION__, "Failed to create nsamples dataset");
       pthread_exit(NULL);
   }

   id->flags_did = H5Dcreate(id->data_gid, "flags", boolenumtype, file_space, H5P_DEFAULT, plist_lzf, H5P_DEFAULT);
   if (id->flags_did < 0) {
       hashpipe_error(__FUNCTION__, "Failed to create flags dataset");
       pthread_exit(NULL);
   }

   id->visdata_fs  = H5Dget_space(id->visdata_did);
   if (id->visdata_fs < 0) {
       hashpipe_error(__FUNCTION__, "Failed to get visdata filespace\n");
       pthread_exit(NULL);
   }
   id->flags_fs    = H5Dget_space(id->flags_did);
   if (id->flags_fs < 0) {
       hashpipe_error(__FUNCTION__, "Failed to get flags filespace\n");
       pthread_exit(NULL);
   }
   id->nsamples_fs = H5Dget_space(id->nsamples_did);
   if (id->nsamples_fs < 0) {
       hashpipe_error(__FUNCTION__, "Failed to get nsamples filespace\n");
       pthread_exit(NULL);
   }

   H5Pclose(plist);
   H5Pclose(plist_lzf);
   H5Sclose(file_space);
}

/* Create the extensible header entries which have dimensions ~Nblts.
 * These are:
 * Header/uvw_array (Nblts x 3)
 * Header/time_array (Nblts)
 * Header/integration_time (Nblts)
 * Header/ant_1_array (Nblts)
 * Header/ant_2_array (Nblts)
 */
#define DIM1 1
#define DIM2 2
static void init_headers_dataset(hdf5_id_t *id) {
   hsize_t dims1[DIM1] = {bcnts_per_file};
   hsize_t chunk_dims1[DIM1] = {1};

   hid_t file_space = H5Screate_simple(DIM1, dims1, NULL);
   hid_t plist = H5Pcreate(H5P_DATASET_CREATE);

   H5Pset_layout(plist, H5D_CHUNKED);
   H5Pset_chunk(plist, DIM1, chunk_dims1);

   // Now we have the dataspace properties, create the datasets
   id->time_array_did = H5Dcreate(id->header_gid, "time_array", H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);
   if (id->time_array_did < 0) {
       hashpipe_error(__FUNCTION__, "Failed to create time_array dataset");
       pthread_exit(NULL);
   }

   id->ant_1_array_did = H5Dcreate(id->header_gid, "ant_1_array", H5T_NATIVE_INT, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);
   if (id->ant_1_array_did < 0) {
       hashpipe_error(__FUNCTION__, "Failed to create ant_1_array dataset");
       pthread_exit(NULL);
   }

   id->ant_2_array_did = H5Dcreate(id->header_gid, "ant_2_array", H5T_NATIVE_INT, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);
   if (id->ant_2_array_did < 0) {
       hashpipe_error(__FUNCTION__, "Failed to create ant_2_array dataset");
       pthread_exit(NULL);
   }

    id->integration_time_did = H5Dcreate(id->header_gid, "integration_time", H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);
    if (id->integration_time_did < 0) {
        hashpipe_error(__FUNCTION__, "Failed to create integration_time dataset");
        pthread_exit(NULL);
    }

   id->time_array_fs = H5Dget_space(id->time_array_did);
   id->ant_1_array_fs = H5Dget_space(id->ant_1_array_did);
   id->ant_2_array_fs = H5Dget_space(id->ant_2_array_did);
   id->integration_time_fs = H5Dget_space(id->integration_time_did);

   H5Pclose(plist);
   H5Sclose(file_space);
}

#define VERSION_BYTES 32
static void start_file(hdf5_id_t *id, char *template_fname, char *hdf5_fname, uint64_t file_obs_id, double file_start_t, char* tag) 
{
    hid_t dataset_id;
    hid_t memtype;
    hid_t stat;
    char ver[VERSION_BYTES] = GIT_VERSION; // defined at compile time

    id->file_id = open_hdf5_from_template(template_fname, hdf5_fname);

    // Open HDF5 header groups and create data group
    id->header_gid = H5Gopen(id->file_id, "Header", H5P_DEFAULT);
    if (id->header_gid < 0) {
        hashpipe_error(__FUNCTION__, "Failed to open Header");
        pthread_exit(NULL);
    }
    
    id->extra_keywords_gid = H5Gopen(id->header_gid, "extra_keywords", H5P_DEFAULT);
    if (id->extra_keywords_gid < 0) {
        hashpipe_error(__FUNCTION__, "Failed to open Header/extra_keywords");
        pthread_exit(NULL);
    }

    id->data_gid = H5Gcreate(id->file_id, "Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (id->data_gid < 0) {
        hashpipe_error(__FUNCTION__, "Failed to create Data group");
        pthread_exit(NULL);
    }

    // Create the "Data" group datasets. This function
    // assigns all the dataset ids to the id struct
    init_data_dataset(id);

    // Create the "Header/*" group datasets. This function
    // assigns all the dataset ids to the id struct
    init_headers_dataset(id);

    // Write meta-data values we know at file-open

    // Write data tag
    memtype = H5Tcopy(H5T_C_S1);
    stat = H5Tset_size(memtype, 128);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to set size of tag memtype");
    }
    dataset_id = H5Dopen(id->extra_keywords_gid, "tag", H5P_DEFAULT);
    if (dataset_id < 0) {
        hashpipe_error(__FUNCTION__, "Failed to open Header/tag");
    }
    stat = H5Dwrite(dataset_id, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, tag);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to write Header/tag");
    }
    stat = H5Dclose(dataset_id);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close Header/tag");
    }

    // obs_id
    dataset_id = H5Dopen(id->extra_keywords_gid, "obs_id", H5P_DEFAULT);
    if (dataset_id < 0) {
        hashpipe_error(__FUNCTION__, "Failed to open Header/extra_keywords/obs_id");
    }
    stat = H5Dwrite(dataset_id, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &file_obs_id);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to write Header/extra_keywords/obs_id");
    }
    stat = H5Dclose(dataset_id);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close Header/extra_keywords/obs_id");
    }

    // version
    dataset_id = H5Dopen(id->extra_keywords_gid, "corr_ver", H5P_DEFAULT);
    if (dataset_id < 0) {
        hashpipe_error(__FUNCTION__, "Failed to open Header/extra_keywords/corr_ver");
    }
    memtype = H5Tcopy(H5T_C_S1);
    stat = H5Tset_size(memtype, VERSION_BYTES);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to set size of corr_ver memtype");
    }
    stat = H5Dwrite(dataset_id, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, ver);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to write Header/extra_keywords/corr_ver");
    }
    stat = H5Dclose(dataset_id);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close Header/extra_keywords/corr_ver");
    }

    // startt
    dataset_id = H5Dopen(id->extra_keywords_gid, "startt", H5P_DEFAULT);
    if (dataset_id < 0) {
        hashpipe_error(__FUNCTION__, "Failed to open Header/extra_keywords/startt");
    }
    stat = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &file_start_t);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to write Header/extra_keywords/startt");
    }
    stat = H5Dclose(dataset_id);
    if (stat < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close Header/extra_keywords/startt");
    }
}


static void close_file(hdf5_id_t *id, double file_stop_t, double file_duration, uint64_t file_nblts) {
    hid_t dataset_id;
    dataset_id = H5Dopen(id->extra_keywords_gid, "stopt", H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &file_stop_t);
    H5Dclose(dataset_id);
    dataset_id = H5Dopen(id->extra_keywords_gid, "duration", H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &file_duration);
    H5Dclose(dataset_id);
    dataset_id = H5Dopen(id->header_gid, "Nblts", H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &file_nblts);
    H5Dclose(dataset_id);

    // Close datasets
    if (H5Dclose(id->visdata_did) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close visdata dataset");
    }
    if (H5Dclose(id->flags_did) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close flags dataset");
    }
    if (H5Dclose(id->nsamples_did) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close nsamples dataset");
    }
    if (H5Dclose(id->time_array_did) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close nsamples dataset");
    }
    if (H5Dclose(id->ant_1_array_did) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close nsamples dataset");
    }
    if (H5Dclose(id->ant_2_array_did) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close nsamples dataset");
    }
    if (H5Dclose(id->integration_time_did) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close nsamples dataset");
    }

    // Close groups
    if (H5Gclose(id->extra_keywords_gid) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close extra_keywords group");
    }
    if (H5Gclose(id->header_gid) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close header group");
    }
    if (H5Gclose(id->data_gid) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close data group");
    }

    // Close file
    if (H5Fflush(id->file_id, H5F_SCOPE_GLOBAL) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to flush file");
    }
    if (H5Fclose(id->file_id) < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close file");
    }
}

static void close_filespaces(hdf5_id_t *id) 
{
    H5Sclose(id->visdata_fs);
    H5Sclose(id->flags_fs);
    H5Sclose(id->nsamples_fs);
    H5Sclose(id->time_array_fs);
    H5Sclose(id->ant_1_array_fs);
    H5Sclose(id->ant_2_array_fs);
    H5Sclose(id->integration_time_fs);
}

// The data in the files should be indexed in real antennas numbers, not 
// in correlator numbers. Get the corr_to_hera map from header to get right labelling.
/* Read the correlator to hera_antennas map from an HDF5 file via the 
   Header/corr_to_hera_map dataset */
static void get_corr_to_hera_map(hdf5_id_t *id, int *corr_to_hera_map) {
    hid_t dataset_id;
    herr_t status;
    dataset_id = H5Dopen(id->header_gid, "corr_to_hera_map", H5P_DEFAULT);
    if (dataset_id < 0) {
        hashpipe_error(__FUNCTION__, "Failed to open Header/corr_to_hera_map dataset");
        pthread_exit(NULL);
    }
    status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, corr_to_hera_map);
    if (status < 0) {
        hashpipe_error(__FUNCTION__, "Failed to read Header/corr_to_hera_map dataset");
        pthread_exit(NULL);
    }
    status = H5Dclose(dataset_id);
    if (status < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close Header/corr_to_hera_map dataset");
        pthread_exit(NULL);
    }
}


/* Get the integration time for each baseline from header (set by config file) */
static void get_integration_time(hdf5_id_t *id, double *integration_time_buf, uint32_t acc_len) {

    hid_t dataset_id;
    herr_t status;
    int i;
    dataset_id = H5Dopen(id->header_gid, "integration_bin", H5P_DEFAULT);
    if (dataset_id < 0) {
        hashpipe_error(__FUNCTION__, "Failed to open Header/integration_time dataset");
        pthread_exit(NULL);
    }
    status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, integration_time_buf);
    if (status < 0) {
       hashpipe_error(__FUNCTION__, "Failed to read Header/integration_time dataset");
       pthread_exit(NULL);
    }
    status = H5Dclose(dataset_id);
    if (status < 0) {
        hashpipe_error(__FUNCTION__, "Failed to close Header/integration_time dataset");
    }

    for(i=0; i< bcnts_per_file; i++){
       integration_time_buf[i] *= acc_len * TIME_DEMUX * 2L * N_CHAN_TOTAL_GENERATED/(double)FENG_SAMPLE_RATE;
    } 
}

/*
Turn an mcnt into a UNIX time in double-precision.
*/
static double mcnt2time(uint64_t mcnt, uint64_t sync_time_ms)
{
    return (sync_time_ms / 1000.) + (mcnt * (2L * N_CHAN_TOTAL_GENERATED / (double)FENG_SAMPLE_RATE));
}

/* 
 *  Compute JD for the given gps time
 
static double unix2julian(double unixtime)
{
    return (2440587.5 + (unixtime / (double)(86400.0)));
}
*/

static double compute_jd_from_mcnt(uint64_t mcnt, uint64_t sync_time_ms, double integration_time)
{
   double unix_time = (sync_time_ms / 1000.) + (mcnt * (2L * N_CHAN_TOTAL_GENERATED / (double)FENG_SAMPLE_RATE));
   unix_time = unix_time - integration_time/2;
   
   return (2440587.5 + (unix_time / (double)(86400.0)));
}

/* 
 * Write N_BL_PER_WRITE bcnts to the dataset, at the right offset
 */
static void write_baseline_index(hdf5_id_t *id, hsize_t bcnt, hsize_t nblts, hid_t mem_space, 
                                 uint64_t *visdata_buf, hbool_t *flags, float *nsamples)
{
    hsize_t start[N_DATA_DIMS] = {bcnt, 0, 0, 0};
    hsize_t count[N_DATA_DIMS] = {nblts, 1, N_CHAN_PROCESSED, N_STOKES};

    // Data
    if (H5Sselect_hyperslab(id->visdata_fs, H5S_SELECT_SET, start, NULL, count, NULL) <0){
       hashpipe_error(__FUNCTION__, "Error selecting data hyperslab from: %d to %d", bcnt, nblts);
    }
    if (H5Dwrite(id->visdata_did, complex_id, mem_space, id->visdata_fs, H5P_DEFAULT, visdata_buf) <0){
       hashpipe_error(__FUNCTION__, "Error writing data to file");
    }

    // nsamples
    if (H5Sselect_hyperslab(id->nsamples_fs, H5S_SELECT_SET, start, NULL, count, NULL) <0){
       hashpipe_error(__FUNCTION__, "Error selecting hyperslab of nsamples");
    }
    if (H5Dwrite(id->nsamples_did, H5T_IEEE_F32LE, mem_space, id->nsamples_fs, H5P_DEFAULT, nsamples) <0){
       hashpipe_error(__FUNCTION__, "Error writing nsamples to file");
    }

    // flags
    if (H5Sselect_hyperslab(id->flags_fs, H5S_SELECT_SET, start, NULL, count, NULL) <0){
       hashpipe_error(__FUNCTION__, "Error selecting flags hyperslab");
    }
    if (H5Dwrite(id->flags_did, boolenumtype, mem_space, id->flags_fs, H5P_DEFAULT, flags) <0){
       hashpipe_error(__FUNCTION__, "Error writing flags to file");
    }

}


/* 
 * Write information to the hdf5 header.
 * Write: time, ant0, ant1
 */

static void write_header(hdf5_id_t *id, double *time_array_buf, int *ant_0_array, int *ant_1_array, double *integration_time_buf)
{  
   // time stamp of integration
   if (H5Dwrite(id->time_array_did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, time_array_buf) < 0) {
       hashpipe_error(__FUNCTION__, "Error writing time_array");
   }

   // ant0
   if (H5Dwrite(id->ant_1_array_did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ant_0_array) < 0) {
       hashpipe_error(__FUNCTION__, "Error writing ant_1_array");
   }

   // ant1
   if (H5Dwrite(id->ant_2_array_did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ant_1_array) < 0) {
       hashpipe_error(__FUNCTION__, "Error writing ant_2_array");
   }

   // integration_time
   if (H5Dwrite(id->integration_time_did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, integration_time_buf) < 0) {
       hashpipe_error(__FUNCTION__, "Error writing integration time");
   }
}

// Get the even-sample / first-pol / first-complexity of the correlation buffer for chan `c` baseline `b`
static void compute_sum_diff(int32_t *in, int32_t *out_sum, int32_t *out_diff, uint32_t bl) {

    int xchan, chan, bcnt, offset;

    //Buffers for a single full-stokes baseline
    // 256 bits = 4 stokes * 2 real/imag * 32 bits == 1 channel

    #if CATCHER_CHAN_SUM_BDA != 1
      int c;
      __m256i sum_even = _mm256_set_epi64x(0ULL,0ULL,0ULL,0ULL);
      __m256i sum_odd  = _mm256_set_epi64x(0ULL,0ULL,0ULL,0ULL);
    #endif

    __m256i val_even = _mm256_set_epi64x(0ULL,0ULL,0ULL,0ULL);
    __m256i val_odd  = _mm256_set_epi64x(0ULL,0ULL,0ULL,0ULL);

    __m256i *in_even256, *in_odd256;
    __m256i *out_sum256  = (__m256i *)out_sum;
    __m256i *out_diff256 = (__m256i *)out_diff;

    for(bcnt=0; bcnt<N_BL_PER_WRITE; bcnt++){
       offset = hera_catcher_bda_input_databuf_by_bcnt_idx32(bcnt+bl, 0);
       in_even256 = (__m256i *)(in + offset);
       offset = hera_catcher_bda_input_databuf_by_bcnt_idx32(bcnt+bl, 1);
       in_odd256  = (__m256i *)(in + offset);

       for(xchan=0; xchan< N_CHAN_TOTAL; xchan+= CATCHER_CHAN_SUM_BDA){
          chan = xchan/CATCHER_CHAN_SUM_BDA;

          #if CATCHER_CHAN_SUM_BDA != 1
             // Add channels
             for(c=0; c<CATCHER_CHAN_SUM_BDA; c++){
                val_even = _mm256_load_si256(in_even256 + xchan + c);
                val_odd  = _mm256_load_si256(in_odd256  + xchan + c);
                if (c==0){
                   sum_even = val_even;
                   sum_odd  = val_odd;
                }
                else{
                   sum_even = _mm256_add_epi32(sum_even, val_even);
                   sum_odd  = _mm256_add_epi32(sum_odd,  val_odd);
                }
             }
             // Write to output
             _mm256_store_si256((out_sum256  + (bcnt*N_CHAN_PROCESSED + chan)), _mm256_add_epi32(sum_even, sum_odd));
             _mm256_store_si256((out_diff256 + (bcnt*N_CHAN_PROCESSED + chan)), _mm256_sub_epi32(sum_even, sum_odd));

          #else
             // Load and write sum/diff
             val_even = _mm256_load_si256(in_even256 + xchan);
             val_odd  = _mm256_load_si256(in_odd256 + xchan);
             _mm256_store_si256((out_sum256  + (bcnt*N_CHAN_PROCESSED+ chan)), _mm256_add_epi32(val_even, val_odd));
             _mm256_store_si256((out_diff256 + (bcnt*N_CHAN_PROCESSED+ chan)), _mm256_sub_epi32(val_even, val_odd));
          #endif
       }
    }
return;
}

static void add_mc_obs(char *fname)
{
  char cmd[256];
  int err;
  fprintf(stdout, "Adding observation %s to M&C\n", fname);
  // Launch (hard-coded) python script in the background and pass in filename
  sprintf(cmd, "/home/hera/hera-venv/envs/hera/bin/mc_add_observation.py %s", fname);
  if (fork() == 0) {
    err = system(cmd);
    if (err != 0) {
      fprintf(stderr, "Error adding observation %s to M&C\n", fname);
    } else {
      // Add to rtp_launch_record table
      sprintf(cmd, "/home/hera/hera-venv/envs/hera/bin/mc_rtp_launch_record.py %s", fname);
      err = system(cmd);
      if (err != 0) {
	fprintf(stderr, "Error adding observation %s to RTP\n", fname);
      }
    }
    exit(0);
  }
}

static int init(hashpipe_thread_args_t *args)
{
    //hashpipe_status_t st = args->st;
    //if (register_lzf() < 0) {
    //    hashpipe_error(__FUNCTION__, "error registering LZF filter");
    //    pthread_exit(NULL);
    //}
    fprintf(stdout, "Initializing Catcher disk thread\n");

    // generate the complex data type
    complex_id = H5Tcreate(H5T_COMPOUND, 8);
    H5Tinsert(complex_id, "r", 0, H5T_STD_I32LE);
    H5Tinsert(complex_id, "i", 4, H5T_STD_I32LE);

    // generate the boolean data type
    bool_t val;
    boolenumtype = H5Tcreate(H5T_ENUM, sizeof(bool_t));
    H5Tenum_insert(boolenumtype, "FALSE", CPTR(val, FALSE ));
    H5Tenum_insert(boolenumtype, "TRUE",  CPTR(val, TRUE  ));
    //H5Tinsert(boolean_id, "FLAG", 0, boolenumtype);
    return 0;
}

static void *run(hashpipe_thread_args_t * args)
{
    // Local aliases to shorten access to args fields
    // Our input buffer is a hera_catcher_bda_input_databuf
    hera_catcher_bda_input_databuf_t *db_in = (hera_catcher_bda_input_databuf_t *)args->ibuf;
    hera_catcher_autocorr_databuf_t *db_out = (hera_catcher_autocorr_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    // Timers for performance monitoring
    struct timespec t_start, t_stop, w_start, w_stop;
    float gbps, min_gbps;

    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &finish);
    uint64_t t_ns;
    uint64_t w_ns;
    uint64_t min_t_ns = 999999999;
    uint64_t min_w_ns = 999999999;
    uint64_t max_t_ns = 0;
    uint64_t max_w_ns = 0;
    uint64_t elapsed_t_ns = 0;
    uint64_t elapsed_w_ns = 0;
    float bl_t_ns = 0.0;
    float bl_w_ns = 0.0;
     
    // Buffers for file name strings
    char template_fname[128];
    char hdf5_diff_fname[128];
    char hdf5_sum_fname[128];
    char data_directory[128];

    // Variables for sync time and computed gps time / JD
    uint64_t sync_time_ms = 0;
    double gps_time;
    double julian_time;
    int int_jd;

    // Variables for data collection parameters
    uint32_t acc_len;
    uint32_t nfiles = 1;
    uint32_t file_cnt = 0;
    uint32_t trigger = 0;
    char tag[128];
    uint64_t baseline_dist[N_BDABUF_BINS];
    uint64_t Nants;
    int corr_to_hera_map[N_ANTS];

    // Init status variables
    hashpipe_status_lock_safe(&st);
    hputi8(st.buf, "DISKMCNT", 0);
    hputu4(st.buf, "TRIGGER", trigger);
    hputu4(st.buf, "NDONEFIL", file_cnt);
    hashpipe_status_unlock_safe(&st);

    // Redis connection
    redisContext *c;
    const char *hostname = "redishost";
    int redisport = 6379;
    int use_redis = 1;
    int idle = 0;

    struct timeval redistimeout = { 0, 100000 }; // 0.1 seconds
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

    // Indicate via redis that we've started but not taking data
    redisCommand(c, "HMSET corr:is_taking_data state False time %d", (int)time(NULL));
    redisCommand(c, "EXPIRE corr:is_taking_data 60");

    // Reinitialize the list of files taken this session
    redisCommand(c, "DEL rtp:file_list");

    /* Loop(s) */
    int32_t *db_in32;
    hera_catcher_bda_input_header_t header;
    int rv;
    int curblock_in=0;
    int curblock_out = 0;
    double file_start_t, file_stop_t, file_duration; // time from bcnt
    int64_t file_obs_id, file_nblts=0;
    int32_t curr_file_bcnt = -1;
    uint32_t bctr, strt_bcnt, stop_bcnt, break_bcnt;        // bcnt variable
    int i,b;
    unsigned int nbls, block_offset;
    uint32_t file_offset;
    uint32_t offset_in, offset_out; // for autocorrs
    int auto_ants_filled = 0;
    uint16_t ant;
    herr_t status;

    hdf5_id_t sum_file;
    #ifndef SKIP_DIFF
    hdf5_id_t diff_file;
    #endif
    
    // aligned_alloc because we're going to use 256-bit AVX instructions
    int32_t *bl_buf_sum  = (int32_t *)aligned_alloc(32, N_BL_PER_WRITE * N_CHAN_PROCESSED * N_STOKES * 2 * sizeof(int32_t));
    int32_t *bl_buf_diff = (int32_t *)aligned_alloc(32, N_BL_PER_WRITE * N_CHAN_PROCESSED * N_STOKES * 2 * sizeof(int32_t));

    memset(bl_buf_sum,  0, N_BL_PER_WRITE * N_CHAN_PROCESSED * N_STOKES * 2 * sizeof(int32_t));
    memset(bl_buf_diff, 0, N_BL_PER_WRITE * N_CHAN_PROCESSED * N_STOKES * 2 * sizeof(int32_t));

    // Init here, realloc after reading baseline distribution from sharedmem
    double *integration_time_buf = (double *)malloc(1 * sizeof(double));
    double *time_array_buf       = (double *)malloc(1 * sizeof(double));
    int *ant_0_array             =    (int *)malloc(1 * sizeof(int));
    int *ant_1_array             =    (int *)malloc(1 * sizeof(int));

    // Allocate an array of bools for flags and n_samples
    hbool_t *flags     = (hbool_t *) malloc(N_BL_PER_WRITE * N_CHAN_PROCESSED * N_STOKES * sizeof(hbool_t));
    float *nsamples = (float *)malloc(N_BL_PER_WRITE * N_CHAN_PROCESSED * N_STOKES * sizeof(float));

    memset(flags, 0, N_BL_PER_WRITE * N_CHAN_PROCESSED * N_STOKES * sizeof(hbool_t));
    // PLP: we want nsamples=1 for valid data, and memset Does The Wrong Thing for non-integer data.
    // Someday we would like nsamples to also reflect the number of dropped packets.
    for (i=0; i<(N_BL_PER_WRITE * N_CHAN_PROCESSED * N_STOKES); i++) {
      nsamples[i] = 1.0;
    }

    // Define memory space of a block
    hsize_t dims[N_DATA_DIMS] = {N_BL_PER_WRITE, 1, N_CHAN_PROCESSED, N_STOKES};
    hid_t mem_space_bl_per_write = H5Screate_simple(N_DATA_DIMS, dims, NULL);

    while (run_threads()) {
        // Note waiting status,
        hashpipe_status_lock_safe(&st);
        if (idle) {
            hputs(st.buf, status_key, "idle");
        } else {
            hputs(st.buf, status_key, "waiting");
        }
        hashpipe_status_unlock_safe(&st);

        // Expire the "corr:is_taking_data" key after 60 seconds.
        // If this pipeline goes down, we will know because the key will disappear
        redisCommand(c, "EXPIRE corr:is_taking_data 60");

        // Wait for new input block to be filled
        while ((rv=hera_catcher_bda_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
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

        db_in32 = (int32_t *)db_in->block[curblock_in].data;
        header = db_in->block[curblock_in].header;

        // Got a new data block, update status
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "writing");
        hputi4(st.buf, "DISKBKIN", curblock_in);
        hputu8(st.buf, "DISKMCNT", header.mcnt[0]);
        hputu8(st.buf, "DISKBCNT", header.bcnt[0]);
        hashpipe_status_unlock_safe(&st);

        /* Copy auto correlations to autocorr buffer */

        if (auto_ants_filled == 0){
           // Wait for next buffer to get free
           while ((rv= hera_catcher_autocorr_databuf_busywait_free(db_out, curblock_out)) != HASHPIPE_OK) {
               if (rv==HASHPIPE_TIMEOUT) {
                   hashpipe_status_lock_safe(&st);
                   hputs(st.buf, status_key, "blocked redis thread");
                   hashpipe_status_unlock_safe(&st);
                   continue;
               } else {
                   hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                   pthread_exit(NULL);
                   break;
               }
           }
           for (i=0; i<N_ANTS; i++){
               db_out->block[curblock_out].header.ant[i] = 0;
           }
        }

        for (bctr=0; bctr < BASELINES_PER_BLOCK; bctr++){
            // Autocorr blocks are indexed by antennas numbers (not corr numbers)
            ant = corr_to_hera_map[header.ant_pair_0[bctr]];
            if((header.ant_pair_0[bctr] == header.ant_pair_1[bctr]) && (db_out->block[curblock_out].header.ant[ant]==0)){
               offset_in = hera_catcher_bda_input_databuf_by_bcnt_idx32(bctr, 0);
               offset_out = hera_catcher_autocorr_databuf_idx32(ant);
               memcpy((db_out->block[curblock_out].data + offset_out), (db_in32 + offset_in), N_CHAN_TOTAL*N_STOKES*2*sizeof(uint32_t));
               auto_ants_filled++;
               db_out->block[curblock_out].header.ant[ant] = 1;
            } 
        }

        // If you have autocorrs of all antennas
        // Mark output block as full and advance
        if (auto_ants_filled >= Nants){
           // Update databuf headers
           db_out->block[curblock_out].header.num_ants = Nants;
           db_out->block[curblock_out].header.julian_time = compute_jd_from_mcnt(header.mcnt[bctr-1], sync_time_ms, 2);
           if (hera_catcher_autocorr_databuf_set_filled(db_out, curblock_out) != HASHPIPE_OK) {
              hashpipe_error(__FUNCTION__, "error marking out databuf %d full", curblock_out);
              pthread_exit(NULL);
           }
           curblock_out = (curblock_out + 1) % AUTOCORR_N_BLOCKS;
           auto_ants_filled = 0; 
        }
        
        // reset elapsed time counters
        elapsed_w_ns = 0.0;
        elapsed_t_ns = 0.0;

        // Get template filename from sharedmem
        hashpipe_status_lock_safe(&st);
        hgets(st.buf, "HDF5TPLT", 128, template_fname);

        // Get time that F-engines were last sync'd
        hgetu8(st.buf, "SYNCTIME", &sync_time_ms);

        // Get the integration time reported by the correlator
        hgetu4(st.buf, "INTTIME", &acc_len);

        // Get the number of files to write
        hgetu4(st.buf, "NFILES", &nfiles);
        // Update the status with how many files we've already written
        hputu4(st.buf, "NDONEFIL", file_cnt);

        // Data tag
        hgets(st.buf, "TAG", 128, tag);

        // Wait for the trigger to write files
        hgetu4(st.buf, "TRIGGER", &trigger);
        hashpipe_status_unlock_safe(&st);

        // If we have written all the files we were commanded to
        // start marking blocks as done and idling until a new
        // trigger is received
        if (trigger) {
          fprintf(stdout, "Catcher got a new trigger and will write %d files\n", nfiles);
          file_cnt = 0;
          hashpipe_status_lock_safe(&st);
          hputu4(st.buf, "TRIGGER", 0);
          hputu4(st.buf, "NDONEFIL", file_cnt);
            
          // Get baseline distribution from sharedmem -- this has to be done here
          // to ensure that redis database is updated before reading.
          hgetu8(st.buf,"BDANANT", &Nants);
          hgetu8(st.buf,"NBL2SEC", &baseline_dist[0]);
          hgetu8(st.buf,"NBL4SEC", &baseline_dist[1]);
          hgetu8(st.buf,"NBL8SEC", &baseline_dist[2]);
          hgetu8(st.buf,"NBL16SEC",&baseline_dist[3]);
          hashpipe_status_unlock_safe(&st);

          bcnts_per_file = 8*baseline_dist[0] + 4*baseline_dist[1] + 2*baseline_dist[2] + baseline_dist[3];
          fprintf(stdout,"Baseline Distribution per file:\n");
          fprintf(stdout,"8 x %ld\t 4 x %ld\t 2 x %ld\t 1 x %ld\n",
                  baseline_dist[0],baseline_dist[1],baseline_dist[2],baseline_dist[3]);
          fprintf(stdout,"Total Baselines: %ld\n", bcnts_per_file);

          fprintf(stdout, "N_CHAN_PROCESSED: %d\n", N_CHAN_PROCESSED);
          fprintf(stdout, "CATCHER_CHAN_SUM_BDA: %d\n", CATCHER_CHAN_SUM_BDA);

          integration_time_buf = (double *)realloc(integration_time_buf, bcnts_per_file * sizeof(double));
          time_array_buf       = (double *)realloc(time_array_buf,       bcnts_per_file * sizeof(double));
          ant_0_array          =    (int *)realloc(ant_0_array,          bcnts_per_file * sizeof(int));
          ant_1_array          =    (int *)realloc(ant_1_array,          bcnts_per_file * sizeof(int));

    
          idle = 0;
          if (use_redis) {
              // Create the "corr:is_taking_data" hash. This will be set to 
              // state=False when data taking is complete. Or if this pipeline 
              // exits the key will expire.
              redisCommand(c, "HMSET corr:is_taking_data state True time %d", (int)time(NULL));
              redisCommand(c, "EXPIRE corr:is_taking_data 60");
          }

        } else if (file_cnt >= nfiles || idle) {
          // If we're transitioning to idle state
          // Indicate via redis that we're no longer taking data
          if (!idle) {
              redisCommand(c, "HMSET corr:is_taking_data state False time %d", (int)time(NULL));
              redisCommand(c, "EXPIRE corr:is_taking_data 60");
          }
          idle = 1;
          // Mark input block as free and advance
          if(hera_catcher_bda_input_databuf_set_free(db_in, curblock_in) != HASHPIPE_OK) {
              hashpipe_error(__FUNCTION__, "error marking databuf %d free", curblock_in);
              pthread_exit(NULL);
          }
          if (use_redis) {
            // Let RTP know we have a new session available
            redisCommand(c, "HMSET rtp:has_new_data state True");
          }
          curblock_in = (curblock_in + 1) % CATCHER_N_BLOCKS;
          continue;
        }

        // If we make it to here we're not idle any more.
        // Usually this would mean there has been another trigger but
        // it could be some weirdness where someone tried to take more
        // data by incrementing NFILES without retriggering.
        idle = 0;

        // Start writing files!
        // A file is defined as bcnts_per_file number of bcnts. If a bcnt belonging
        // to a new intergation arrives, close the old file and start a new file.

        clock_gettime(CLOCK_MONOTONIC, &start);
             
        for (bctr=0 ; bctr< BASELINES_PER_BLOCK; bctr += N_BL_PER_WRITE){

          // We write N_BL_PER_WRITE at a time.
          // these variables store the baseline numbers for the start and end of these blocks
          strt_bcnt = header.bcnt[bctr];
          stop_bcnt = header.bcnt[bctr+N_BL_PER_WRITE-1]; 

          clock_gettime(CLOCK_MONOTONIC, &t_start);
          compute_sum_diff(db_in32, bl_buf_sum, bl_buf_diff, bctr);
          clock_gettime(CLOCK_MONOTONIC, &t_stop);

          t_ns = ELAPSED_NS(t_start, t_stop);
          elapsed_t_ns += t_ns;
          min_t_ns = MIN(t_ns, min_t_ns);
          max_t_ns = MAX(t_ns, max_t_ns);

          // If the start and end of this block belong in the same file AND
          // The start of this block is not the start of a new file...
          if (((strt_bcnt / bcnts_per_file) == (stop_bcnt / bcnts_per_file)) &&
               (strt_bcnt % bcnts_per_file != 0)){

             // If there is a file already open...
             // Copy all contents
             if (curr_file_bcnt >= 0){

                file_offset = strt_bcnt - curr_file_bcnt;

                clock_gettime(CLOCK_MONOTONIC, &w_start);
                write_baseline_index(&sum_file, file_offset, N_BL_PER_WRITE, mem_space_bl_per_write, 
                                    (uint64_t *)bl_buf_sum, flags, nsamples);
                #ifndef SKIP_DIFF
                write_baseline_index(&diff_file, file_offset, N_BL_PER_WRITE, mem_space_bl_per_write, 
                                    (uint64_t *)bl_buf_diff, flags, nsamples);
                #endif

                clock_gettime(CLOCK_MONOTONIC, &w_stop);

                for(b=0; b< N_BL_PER_WRITE; b++){
                   ant_0_array[file_offset+b]    = corr_to_hera_map[header.ant_pair_0[bctr+b]];
                   ant_1_array[file_offset+b]    = corr_to_hera_map[header.ant_pair_1[bctr+b]];

                   time_array_buf[file_offset+b] = compute_jd_from_mcnt(header.mcnt[bctr+b], sync_time_ms,  
                                                   integration_time_buf[file_offset+b]);
                }
                
                file_nblts += N_BL_PER_WRITE;

                w_ns = ELAPSED_NS(w_start, w_stop);
                elapsed_w_ns += w_ns;
                min_w_ns = MIN(w_ns, min_w_ns);
                max_w_ns = MAX(w_ns, max_w_ns);
             }
          } else {
             // the block has a file boundary OR this block starts with a new file.

             
             // Calculate the bcnt where we need to start a new file.
             // This block might start at a new file. Otherwise
             // We need a new file at the next bcnts_per_file boundary.
             if (strt_bcnt % bcnts_per_file == 0){
                break_bcnt = strt_bcnt;
             } else {
                 break_bcnt = ((strt_bcnt / bcnts_per_file) + 1) * bcnts_per_file;
             }

             // If there is an open file, copy the relevant part of the block 
             // and close the file. Open a new file for the rest of the block.
             if (curr_file_bcnt >=0){

                 // copy data
                 nbls = break_bcnt - strt_bcnt;
                 
                 if (nbls > 0){
                    // select the hyperslab of the shared mem to write to file
                    hsize_t start[N_DATA_DIMS] = {0, 0, 0 ,0};
                    hsize_t count[N_DATA_DIMS] = {nbls, 1, N_CHAN_PROCESSED, N_STOKES};

                    status = H5Sselect_hyperslab(mem_space_bl_per_write, H5S_SELECT_SET, start, NULL, count, NULL);
                    if (status < 0){
                       hashpipe_error(__FUNCTION__, "Failed to select hyperslab of shared databuf\n");
                       pthread_exit(NULL);
                    }

                    file_offset = strt_bcnt - curr_file_bcnt;

                    clock_gettime(CLOCK_MONOTONIC, &w_start);
                    write_baseline_index(&sum_file, file_offset, nbls, mem_space_bl_per_write, 
                                        (uint64_t *)bl_buf_sum, flags, nsamples);
                    #ifndef SKIP_DIFF
                      write_baseline_index(&diff_file, file_offset, nbls, mem_space_bl_per_write, 
                                          (uint64_t *)bl_buf_diff, flags, nsamples);
                    #endif

                    clock_gettime(CLOCK_MONOTONIC, &w_stop);

                    for(b=0; b< nbls; b++){
                       ant_0_array[file_offset+b]    = corr_to_hera_map[header.ant_pair_0[bctr+b]];
                       ant_1_array[file_offset+b]    = corr_to_hera_map[header.ant_pair_1[bctr+b]];
                       time_array_buf[file_offset+b] = compute_jd_from_mcnt(header.mcnt[bctr+b], sync_time_ms, 
                                                       integration_time_buf[file_offset+b]);
                    }
                    file_nblts += nbls;

                    // reset selection
                    status = H5Sselect_all(mem_space_bl_per_write);
                    if (status < 0){
                       hashpipe_error(__FUNCTION__, "Failed to reset selection\n");
                       pthread_exit(NULL);
                    }

                    w_ns = ELAPSED_NS(w_start, w_stop);
                    elapsed_w_ns += w_ns;
                    min_w_ns = MIN(w_ns, min_w_ns);
                    max_w_ns = MAX(w_ns, max_w_ns);
                 }

                 // finish meta data and close the file
                 gps_time = mcnt2time(header.mcnt[bctr+nbls], sync_time_ms);
                 file_stop_t = gps_time;
                 file_duration = file_stop_t - file_start_t;

                 write_header(&sum_file, time_array_buf, ant_0_array, ant_1_array, integration_time_buf);
                 close_filespaces(&sum_file);
                 close_file(&sum_file, file_stop_t, file_duration, file_nblts);

                 #ifndef SKIP_DIFF 
                 write_header(&diff_file, time_array_buf, ant_0_array, ant_1_array, integration_time_buf);
                 close_filespaces(&diff_file);
                 close_file(&diff_file, file_stop_t, file_duration, file_nblts);
                 #endif

                 file_cnt += 1;

                 // add file to M&C
                 add_mc_obs(hdf5_sum_fname);

                 hashpipe_status_lock_safe(&st);
                 hputr4(st.buf, "FILESEC", file_duration);
                 hputi8(st.buf, "NDONEFIL", file_cnt);
                 hashpipe_status_unlock_safe(&st); 

                 // If this is the last file, mark this block done and get out of the loop
                 if (file_cnt >= nfiles) {
                     fprintf(stdout, "Catcher has written %d file and is going to sleep\n", file_cnt);
                     curr_file_bcnt = -1; //So the next trigger will start a new file
                     break;
                 }
             }

             // Open new sum and difference files
             // Init all counters to zero

             file_nblts = 0;
             memset(ant_0_array,          0, bcnts_per_file * sizeof(uint16_t));
             memset(ant_1_array,          0, bcnts_per_file * sizeof(uint16_t));
             memset(time_array_buf,       0, bcnts_per_file * sizeof(double));

             curr_file_bcnt = break_bcnt;
             block_offset = bctr + break_bcnt - strt_bcnt;
             fprintf(stdout, "Curr file bcnt: %d\n", curr_file_bcnt);
             fprintf(stdout, "Curr file mcnt: %ld\n", header.mcnt[block_offset]);
             gps_time = mcnt2time(header.mcnt[block_offset], sync_time_ms);
             julian_time = 2440587.5 + (gps_time / (double)(86400.0)); 
             file_start_t = gps_time;
             file_obs_id = (int64_t)gps_time;

             // Make a new folder for output
             if (file_cnt == 0) {
               int_jd = (int)julian_time;
               sprintf(data_directory, "%d", int_jd);
               fprintf(stdout, "Making directory %s\n", data_directory);
               mkdir(data_directory, 0777);
               chmod(data_directory, 0777);
             }

             sprintf(hdf5_sum_fname, "%d/zen.%7.5lf.sum.uvh5", int_jd, julian_time);
             fprintf(stdout, "Opening new file %s\n", hdf5_sum_fname);
             start_file(&sum_file, template_fname, hdf5_sum_fname, file_obs_id, file_start_t, tag);
             if (use_redis) {
               redisCommand(c, "RPUSH rtp:file_list %s", hdf5_sum_fname);
             }

             #ifndef SKIP_DIFF
               sprintf(hdf5_diff_fname, "%d/zen.%7.5lf.diff.uvh5", int_jd, julian_time);
               fprintf(stdout, "Opening new file %s\n", hdf5_diff_fname);
               start_file(&diff_file, template_fname, hdf5_diff_fname, file_obs_id, file_start_t, tag);
               if (use_redis) {
                 redisCommand(c, "RPUSH rtp:file_list %s", hdf5_diff_fname);
               }
             #endif

             // Get the antenna positions and baseline orders
             // These are needed for populating the ant_[1|2]_array and uvw_array
             //get_ant_pos(&sum_file, ant_pos);
             get_corr_to_hera_map(&sum_file, corr_to_hera_map);
             get_integration_time(&sum_file, integration_time_buf, acc_len);
       
             // Copy data to the right location
             nbls = stop_bcnt - break_bcnt + 1;

             if (nbls > 0){
                if (nbls % N_BL_PER_WRITE){
                   hsize_t start[N_DATA_DIMS] = {block_offset - bctr, 0, 0, 0};
                   hsize_t count[N_DATA_DIMS] = {nbls, 1, N_CHAN_PROCESSED, N_STOKES};

                   status = H5Sselect_hyperslab(mem_space_bl_per_write, H5S_SELECT_SET, start, NULL, count, NULL);
                   if (status < 0){
                      hashpipe_error(__FUNCTION__, "Failed to select hyperslab of shared databuf\n");
                      pthread_exit(NULL);
                   }
                }
                file_offset = break_bcnt - curr_file_bcnt;

                for(b=0; b< nbls; b++){
                   ant_0_array[file_offset+b]    = corr_to_hera_map[header.ant_pair_0[block_offset+b]];
                   ant_1_array[file_offset+b]    = corr_to_hera_map[header.ant_pair_1[block_offset+b]];
                   time_array_buf[file_offset+b] = compute_jd_from_mcnt(header.mcnt[block_offset+b], sync_time_ms, 
                                                   integration_time_buf[file_offset+b]);
                }
                
                clock_gettime(CLOCK_MONOTONIC, &w_start);
                write_baseline_index(&sum_file, file_offset, nbls, mem_space_bl_per_write, 
                                    (uint64_t *)bl_buf_sum, flags, nsamples);
                #ifndef SKIP_DIFF
                  write_baseline_index(&diff_file, file_offset, nbls, mem_space_bl_per_write, 
                                      (uint64_t *)bl_buf_diff, flags, nsamples);
                #endif
                clock_gettime(CLOCK_MONOTONIC, &w_stop); 

                file_nblts += nbls;

                //reset selection
                status = H5Sselect_all(mem_space_bl_per_write);
                if (status < 0){
                   hashpipe_error(__FUNCTION__, "Failed to reset selection\n");
                   pthread_exit(NULL);
                }

                w_ns = ELAPSED_NS(w_start, w_stop);
                elapsed_w_ns += w_ns;
                min_w_ns = MIN(w_ns, min_w_ns);
                max_w_ns = MAX(w_ns, max_w_ns);
             }
          }
        }

        clock_gettime(CLOCK_MONOTONIC, &finish);

        // Compute processing time for this block
        bl_t_ns = (float)elapsed_t_ns / BASELINES_PER_BLOCK;
        bl_w_ns = (float)elapsed_w_ns / BASELINES_PER_BLOCK;
 
        hashpipe_status_lock_safe(&st);
        hputr4(st.buf, "DISKTBNS", bl_t_ns);
        hputi8(st.buf, "DISKTMIN", min_t_ns);
        hputi8(st.buf, "DISKTMAX", max_t_ns);
        hputr4(st.buf, "DISKWBNS", bl_w_ns);
        hputi8(st.buf, "DISKWMIN", min_w_ns);
        hputi8(st.buf, "DISKWMAX", max_w_ns);
         
        hputi8(st.buf, "DISKWBL", w_ns/BASELINES_PER_BLOCK);

        hgetr4(st.buf, "DISKMING", &min_gbps);
        #ifndef SKIP_DIFF
          gbps = (float)(2 * BASELINES_PER_BLOCK*N_CHAN_PROCESSED*N_STOKES*64L)/ELAPSED_NS(start,finish);
        #else
          gbps = (float)(BASELINES_PER_BLOCK*N_CHAN_PROCESSED*N_STOKES*64L)/ELAPSED_NS(start,finish); 
        #endif
        hputr4(st.buf, "DISKGBPS", gbps);
        hputr4(st.buf, "DUMPMS", ELAPSED_NS(start,finish) / 1000000.0);
        if(min_gbps == 0 || gbps < min_gbps) {
          hputr4(st.buf, "DISKMING", gbps);
        }
        hashpipe_status_unlock_safe(&st);

        // Mark input block as free and advance
        if(hera_catcher_bda_input_databuf_set_free(db_in, curblock_in) != HASHPIPE_OK) {
            hashpipe_error(__FUNCTION__, "error marking databuf %d free", curblock_in);
            pthread_exit(NULL);
        }
        curblock_in = (curblock_in + 1) % CATCHER_N_BLOCKS;

        /* Check for cancel */
        pthread_testcancel();
    }

    // Thread success!
    return NULL;
}
 
static hashpipe_thread_desc_t hera_catcher_disk_thread_bda = {
    name: "hera_catcher_disk_thread_bda",
    skey: "DISKSTAT",
    init: init,
    run:  run,
    ibuf_desc: {hera_catcher_bda_input_databuf_create},
    obuf_desc: {hera_catcher_autocorr_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&hera_catcher_disk_thread_bda);
}
