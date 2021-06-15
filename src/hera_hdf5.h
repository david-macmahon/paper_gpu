#ifndef _HERA_HDF5_H
#define _HERA_HDF5_H

#include "paper_databuf.h"

// HERA HDF5 header fields, based on PLP's sample file
// Comment descriptions are guesses
typedef struct hdf5_extra_keywords {
  uint64_t foo;
} hdf5_extra_keywords_t;

typedef struct hdf5_header {
  int64_t Nants_data;      // Number of antennas for which data are valid
  int64_t Nants_telescope; // Number of antennas in telescope
  int64_t Nbls;            // Number of visibilities in dataset
  int64_t Nblts;           // ???
  int64_t Nfreqs;          // Number of frequency channels in dataset
  int64_t Npols;           // Number of polarizations in dataset
  int64_t Nspws;           // ??
  int64_t Ntimes;          // Number of times in dataset
  double  altitude;        // Array altitude in some unit
  int64_t ant_1_array[N_BASELINES]; // Array of first antenna in each baseline 
  int64_t ant_2_array[N_BASELINES]; // Array of second antenna in each baseline 
  double  antenna_diameters[N_ANTS]; // Antenna diameters in ? units
  char    antenna_names[N_ANTS][32]; // Antenna names
  int64_t antenna_numbers[N_ANTS];   // HERA-spec antenna numbers
  double  antenna_positions[N_ANTS][3]; // Antenna locations in ? units
  double  channel_width; // Frequency channel width in ? units
  struct hdf5_extra_keywords extra_keywords; // Extra keywords(!)
  double  freq_array[N_CHAN_TOTAL]; // Frequency channel centers in ? units
  char    history[1024]; // history string
  char    instrument[32]; // Instrument name
  double  integration_time; // Integration time in ? units
  double  latitude; // Array latitude in ? units
  double  longitude; // Array longitude in ? units
  double  *lst_array; // Array of LST's for each time sample?
  char    object_name[32]; // Name of astronomical source
  char    phase_type[32]; // Type of phasing (zenith, none, to source) in this data set
  int64_t polarization_array; // Polarization of data in this data setby integer key
  int64_t *spw_array;
  char    telescope_name[32]; // How is this different to "instrument"?
  double  *time_array; // Times (UTC / Unix / MJD ?) of time samples
  double  *uvw_array[N_ANTS][3]; // UVW co-ords of antennas
  char    vis_units[32];
  double  zenith_dec;
  double  zenith_ra;
} hdf5_header_t;

  
#endif // _HERA_HDF5_H
