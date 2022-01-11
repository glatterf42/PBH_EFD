/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file pbh_particle_data.h
 *
 *  \brief defines the structure holding the extra data for a single PBH particle
 */

#ifndef PBHPARTDATA_H
#define PBHPARTDATA_H

#include "../data/constants.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/macros.h"
#include "../data/mymalloc.h"
#include "../mpi_utils/setcomm.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

/* in this structure, all PBH variables are put that are needed for passive
 * particles in the scatter calculation. Only this part will be sent
 * to other nodes if needed
 */
struct pbh_particle_data_core
{
  MyFloat Hsml;               /*!< current smoothing length */
  MyFloat DhsmlDensityFactor; /*!< correction factor needed in entropy formulation of SPH */
  MyFloat VelPred[3];         /*!< predicted SPH particle velocity at the current time, needed if particle is inactive */

  MyFloat DivVel;  /*!< local velocity divergence */
  MyFloat CurlVel; /*!< local velocity curl */
  MyFloat Csnd;

  MyFloat Density;  /*!< current baryonic mass density of particle */
  MyFloat Pressure; /*!< current pressure */

  MyFloat dist_over_time; /*!< auxiliary value to avoid zero relative velocities */
};

/** Holds data that is stored for each pbh particle in addition to
    the collisionless variables.
 */
struct pbh_particle_data : public pbh_particle_data_core
{
  MyFloat Entropy;     /*!< value of the entropic function */
  MyFloat EntropyPred; /*!< predicted entropy at current time, needed if the particle is inactive */

  MyFloat HydroAccel[3]; /*!< acceleration due to hydrodynamical forces */
#ifdef HIERARCHICAL_GRAVITY
  MyFloat FullGravAccel[3]; /*!< most recent full calculation of gravitational acceleration, used to advanced VelPred */
#endif
  MyFloat DtEntropy; /*!< rate of change of entropy */
  MyFloat DtDensity; /*!< rate of change of density, needed to predict densities for passive particles */
  MyFloat DtHsml;    /*!< rate of change of smoothing length, needed to predict hsml for passive particles */

  MyFloat NumNgb; /*!< effective number of neighbours used in density estimation loop (note: this could be changed to a temporary
                     variable in density) */

  MyFloat Rot[3]; /*!< local velocity curl */

  MyFloat CurrentMaxTiStep;

  bool scatter_occurrence; /*!< distinguishes particles that experience scattering in the current time step */
  MyFloat PbhVel[3]; /*!< velocity change due to pbh scatter interactions */


#ifdef IMPROVED_VELOCITY_GRADIENTS
  struct
  {
    MyFloat dx_dx;
    MyFloat dx_dy;
    MyFloat dx_dz;
    MyFloat dy_dy;
    MyFloat dy_dz;
    MyFloat dz_dz;
  } dpos; /* contains the matrix elements needed for the improved gradient estimate */

  MyFloat dvel[NUMDIMS][NUMDIMS]; /* contains the velocity gradients */
#endif

  inline MyFloat get_sound_speed(void)
  {
    MyFloat csnd;

    if(Density > 0)
      csnd = sqrt(static_cast<MyReal>(GAMMA) * Pressure / Density);
    else
      csnd = 0;

    return csnd;
  }

  /* compute the pressure of particle i */
  inline MyFloat get_pressure(void)
  {
    return EntropyPred * pow(Density, (MyFloat)GAMMA);
  }

  inline void set_thermodynamic_variables(void)
  {
    Pressure = get_pressure();

    if(Pressure < 0)
      Terminate("Pressure=%g  rho=%g  entr=%g entrpred=%g\n", Pressure, Density, Entropy, EntropyPred);

    Csnd = get_sound_speed();
  }

  inline MyFloat get_Hsml() { return Hsml; }

#ifdef IMPROVED_VELOCITY_GRADIENTS
  void set_velocity_gradients(void);
#endif
};

#endif
