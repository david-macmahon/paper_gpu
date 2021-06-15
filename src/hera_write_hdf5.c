#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <hdf5.h>

#include "hera_hdf5.h"

#define FILENAME "/tmp/foo.h5"

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

#define H5_WRITE_HEADER_I64(group_id, h5_name, val, dims) \
  dataspace_id = H5Screate_simple(1, dims, NULL); \
  dataset_id = H5Dcreate2(group_id, h5_name, H5T_STD_I64LE, dataspace_id,\
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(val)); \
  H5Sclose(dataspace_id); \
  H5Dclose(dataset_id)



// Write the header contents of hdf5 header struct to
// a new group "Header" of the HDF5 file referenced by `file_id`
void write_hdf5_header(struct hdf5_header *header, hid_t file_id)
{
  hid_t group_id, dataset_id, dataspace_id;
  hsize_t dims[4];

  group_id = H5Gcreate2(file_id, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  dims[0] = 1;
  H5_WRITE_HEADER_I64(group_id, "Nants_data", header->Nants_data, dims);
  H5_WRITE_HEADER_I64(group_id, "Nants_telescope", header->Nants_telescope, dims);
  H5_WRITE_HEADER_I64(group_id, "Nbls", header->Nbls, dims);
  H5_WRITE_HEADER_I64(group_id, "Nblts", header->Nblts, dims);
  H5_WRITE_HEADER_I64(group_id, "Nfreqs", header->Nfreqs, dims);
  H5_WRITE_HEADER_I64(group_id, "Npols", header->Npols, dims);
  H5_WRITE_HEADER_I64(group_id, "Nspws", header->Nspws, dims);
  H5_WRITE_HEADER_I64(group_id, "Ntimes", header->Ntimes, dims);
  H5Gclose(group_id);
}

struct hdf5_header * initialize_header()
{
  struct hdf5_header * header;
  header = calloc(1, sizeof(hdf5_header_t));

  header->Nants_data = N_ANTS;
  header->Nants_telescope = N_ANTS;
  header->Nbls = 0;//N_BLS;
  header->Nfreqs = N_CHAN_TOTAL;
  header->Npols = 4;
  return header;
}

void write_hdf5()
{
  struct hdf5_header * header;
  header =  initialize_header();

  hid_t file_id;
  herr_t status;
  
  file_id = H5Fcreate(FILENAME, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  write_hdf5_header(header, file_id);

  status = H5Fclose(file_id);
  fprintf(stderr, "%d", (int)status);
  
}

int main(int argc, char *argv[])
{
  write_hdf5();
  return 0;
}
