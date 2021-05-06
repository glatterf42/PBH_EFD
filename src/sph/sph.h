/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  sph.h
 *
 *  \brief defines a class for the SPH computations
 */

#ifndef SPH_H
#define SPH_H

#include "../mpi_utils/shared_mem_handler.h"
#include "../ngbtree/ngbtree.h"

#define MAX_NGBS 500000

class sph : public ngbtree
{
 public:
  void compute_densities(void);
  void density(int *targetlist, int ntarget);
  void hydro_forces_determine(int ntarget, int *targetlist);
  void tree_based_timesteps(void);

#ifdef PRESSURE_ENTROPY_SPH
  void setup_entropy_to_invgamma(void);
#endif

  double fac_mu;

 private:
  int max_ncycles;

  double fac_vsic_fix;

  double MaxBoxDist;

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  bool skip_actual_force_computation;
#endif

  struct pinfo
  {
    int target;
    int numngb;

    MyIntPosType *searchcenter;
    MyIntPosType search_min[3], search_range[3];
    MyIntPosType inthsml;
    MyNgbTreeFloat hsml;
    MyNgbTreeFloat hsml2;
  };

  inline void get_pinfo(int i, pinfo &pdat)
  {
    pdat.target = i;

    pdat.searchcenter = Tp->P[i].IntPos;
    pdat.hsml         = Tp->SphP[i].Hsml;
    pdat.hsml2        = pdat.hsml * pdat.hsml;
    pdat.inthsml      = pdat.hsml * Tp->FacCoordToInt;

    for(int i = 0; i < 3; i++)
      {
        pdat.search_min[i]   = pdat.searchcenter[i] - pdat.inthsml;
        pdat.search_range[i] = pdat.inthsml + pdat.inthsml;
      }

    pdat.numngb = 0;
  }

  struct ngbdata_density
  {
    MyIntPosType *IntPos;
    MyFloat *VelPred;
    MyDouble Mass;
#ifdef PRESSURE_ENTROPY_SPH
    MyDouble EntropyToInvGammaPred;
#endif
#ifdef TIMEDEP_ART_VISC
    MyDouble Csnd;
#endif
  };

  ngbdata_density *Ngbdensdat;

  struct ngbdata_hydro
  {
    MyIntPosType *IntPos;
    sph_particle_data_hydrocore *SphCore;

    MyDouble Mass;
    signed char TimeBinHydro;
    unsigned int ID;
  };

  ngbdata_hydro *Ngbhydrodat;
    
  struct scatter_event
  {
    unsigned int scatter_partner_one;
    unsigned int scatter_partner_two;
    double scattering_probability;
  };
  scatter_event *scatter_list;
  
  int numberofparticles;
  int numberoflocalparticles;
  int numberofforeignparticles;
  int pairsconsidered;
  int nscatterevents;
  int n0vrelbefore;
  int n0vrelafter;
  double ti_step_to_phys;
  double scatter_prob_to_phys;

  inline foreign_sphpoint_data *get_foreignpointsp(int n, unsigned char shmrank)
  {
    return (foreign_sphpoint_data *)((char *)TreeSharedMemBaseAddr[shmrank] + TreeForeign_Points_offsets[shmrank]) + n;
  }
  void densities_determine(int ntarget, int *targetlist);
  void density_evaluate_kernel(pinfo &pdat);
  void sph_density_interact(pinfo &pdat, int no, char no_type, unsigned char shmrank, int mintopleafnode, int committed);
  inline void sph_density_open_node(pinfo &pdat, ngbnode *nop, int mintopleafnode, int committed);
  inline int sph_density_evaluate_particle_node_opening_criterion(pinfo &pdat, ngbnode *nop);
  inline void sph_density_check_particle_particle_interaction(pinfo &pdat, int p, int p_type, unsigned char shmrank);
  inline void clear_density_result(sph_particle_data *SphP);

  void hydro_evaluate_kernel(pinfo &pdat);
  void scatter_evaluate_kernel(pinfo &pdat);
  void scatter_list_evaluate(scatter_event *scatter_list, int nscatterevents);
  static bool by_scatter_prob(const scatter_event &s1, const scatter_event &s2)
  {
    return s1.scattering_probability > s2.scattering_probability;//should have descending order
  }
  inline int get_index_from_ID(MyIDType ID, int h);
  inline void sph_hydro_interact(pinfo &pdat, int no, char no_type, unsigned char shmrank, int mintopleafnode, int committed);
  inline void sph_hydro_open_node(pinfo &pdat, ngbnode *nop, int mintopleafnode, int committed);
  inline int sph_hydro_evaluate_particle_node_opening_criterion(pinfo &pdat, ngbnode *nop);
  inline void sph_hydro_check_particle_particle_interaction(pinfo &pdat, int p, int p_type, unsigned char shmrank);
  inline void clear_hydro_result(sph_particle_data *SphP);

  inline void sph_treetimestep_interact(pinfo &pdat, int no, char no_type, unsigned char shmrank, int mintopleafnode, int committed);
  inline void sph_treetimestep_open_node(pinfo &pdat, ngbnode *nop, int mintopleafnode, int committed);
  inline int sph_treetimestep_evaluate_particle_node_opening_criterion(pinfo &pdat, ngbnode *nop);
  inline void sph_treetimestep_check_particle_particle_interaction(pinfo &pdat, int p, int p_type, unsigned char shmrank);

#ifdef PRESSURE_ENTROPY_SPH
  void init_entropy(void);
#endif
};

#endif
