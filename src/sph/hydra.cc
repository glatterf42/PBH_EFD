/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  hydra.cc
 *
 *  \brief computation of SPH forces and rate of entropy generation
 */

#include <gsl/gsl_rng.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "../data/allvars.h"
#include "../data/constants.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../main/simulation.h"
#include "../mpi_utils/mpi_utils.h"
#include "../ngbtree/ngbtree.h"
#include "../sort/cxxsort.h"
#include "../sph/kernel.h"
#include "../sph/sph.h"
#include "../system/system.h"
#include "../time_integration/driftfac.h"
#include "gadgetconfig.h"

/*! This file contains the "second SPH loop", where the SPH forces are
 *  computed, and where the rate of change of entropy due to the shock heating
 *  (via artificial viscosity) is computed.
 */

inline int sph::sph_hydro_evaluate_particle_node_opening_criterion(pinfo &pdat, ngbnode *nop)
{
  if(nop->level <= LEVEL_ALWAYS_OPEN)  // always open the root node (note: full node length does not fit in the integer type)
    return NODE_OPEN;

  if(nop->Ti_Current != All.Ti_Current)
    nop->drift_node(All.Ti_Current, Tp);

  MyNgbTreeFloat dist = std::max<MyNgbTreeFloat>(nop->MaxHsml, pdat.hsml);

  MyIntPosType search_min[3], search_range[3];

  MyIntPosType inthsml = dist * Tp->FacCoordToInt;

  for(int i = 0; i < 3; i++)
    {
      search_min[i]   = pdat.searchcenter[i] - inthsml;
      search_range[i] = inthsml + inthsml;
    }

  MyIntPosType left[3], right[3];

  left[0]  = Tp->nearest_image_intpos_to_intpos_X(nop->center_offset_min[0] + nop->center[0], search_min[0]);
  right[0] = Tp->nearest_image_intpos_to_intpos_X(nop->center_offset_max[0] + nop->center[0], search_min[0]);

  /* check whether we can stop walking along this branch */
  if(left[0] > search_range[0] && right[0] > left[0])
    return NODE_DISCARD;

  left[1]  = Tp->nearest_image_intpos_to_intpos_Y(nop->center_offset_min[1] + nop->center[1], search_min[1]);
  right[1] = Tp->nearest_image_intpos_to_intpos_Y(nop->center_offset_max[1] + nop->center[1], search_min[1]);

  /* check whether we can stop walking along this branch */
  if(left[1] > search_range[1] && right[1] > left[1])
    return NODE_DISCARD;

  left[2]  = Tp->nearest_image_intpos_to_intpos_Z(nop->center_offset_min[2] + nop->center[2], search_min[2]);
  right[2] = Tp->nearest_image_intpos_to_intpos_Z(nop->center_offset_max[2] + nop->center[2], search_min[2]);

  /* check whether we can stop walking along this branch */
  if(left[2] > search_range[2] && right[2] > left[2])
    return NODE_DISCARD;

  return NODE_OPEN;
}

inline void sph::sph_hydro_check_particle_particle_interaction(pinfo &pdat, int p, int p_type, unsigned char shmrank)
{
#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  if(skip_actual_force_computation)
    return;
#endif

  if(p_type == NODE_TYPE_LOCAL_PARTICLE) /* local particle */
    {
      particle_data *P        = get_Pp(p, shmrank);
      sph_particle_data *SphP = get_SphPp(p, shmrank);

      if(P->getType() > 0)
        return;

      if(P->get_Ti_Current() != All.Ti_Current)
        Tp->drift_particle(P, SphP, All.Ti_Current);  // this function avoids race conditions

      MyNgbTreeFloat dist   = std::max<MyNgbTreeFloat>(SphP->Hsml, pdat.hsml);
      MyNgbTreeFloat distsq = dist * dist;

      double posdiff[3];
      Tp->nearest_image_intpos_to_pos(P->IntPos, pdat.searchcenter, posdiff); /* converts the integer distance to floating point */

      double rad2 = posdiff[0] * posdiff[0] + posdiff[1] * posdiff[1] + posdiff[2] * posdiff[2];
      if(rad2 > distsq || rad2 == 0)
        return;

      if(pdat.numngb >= MAX_NGBS)
        Terminate("pdat.numngb >= MAX_NGBS");

      int n = pdat.numngb++;

      Ngbhydrodat[n].SphCore = SphP;
      Ngbhydrodat[n].IntPos  = P->IntPos;
      Ngbhydrodat[n].Mass    = P->getMass();
      Ngbhydrodat[n].ID      = P->ID.get();
#ifndef LEAN
      Ngbhydrodat[n].TimeBinHydro = P->TimeBinHydro;
#endif
      numberofparticles++;
      numberoflocalparticles++;
    }
  else if(p_type == NODE_TYPE_FETCHED_PARTICLE)
    {
      foreign_sphpoint_data *foreignpoint = get_foreignpointsp(p - EndOfForeignNodes, shmrank);

      MyNgbTreeFloat dist   = std::max<MyNgbTreeFloat>(foreignpoint->SphCore.Hsml, pdat.hsml);
      MyNgbTreeFloat distsq = dist * dist;

      /* converts the integer distance to floating point */
      double posdiff[3];
      Tp->nearest_image_intpos_to_pos(foreignpoint->IntPos, pdat.searchcenter, posdiff);

      double rad2 = posdiff[0] * posdiff[0] + posdiff[1] * posdiff[1] + posdiff[2] * posdiff[2];
      if(rad2 > distsq || rad2 == 0)
        return;

      if(pdat.numngb >= MAX_NGBS)
        Terminate("pdat.numngb >= MAX_NGBS");

      int n = pdat.numngb++;

      Ngbhydrodat[n].SphCore      = &foreignpoint->SphCore;
      Ngbhydrodat[n].IntPos       = foreignpoint->IntPos;
      Ngbhydrodat[n].Mass         = foreignpoint->Mass;
      Ngbhydrodat[n].TimeBinHydro = foreignpoint->TimeBinHydro;
      Ngbhydrodat[n].ID           = foreignpoint->ID.get();
      numberofparticles++;
      numberofforeignparticles++;
    }
  else
    Terminate("unexpected");
}

inline void sph::sph_hydro_open_node(pinfo &pdat, ngbnode *nop, int mintopleafnode, int committed)
{
  /* open node */
  int p                 = nop->nextnode;
  unsigned char shmrank = nop->nextnode_shmrank;

  while(p != nop->sibling || (shmrank != nop->sibling_shmrank && nop->sibling >= MaxPart + D->NTopnodes))
    {
      if(p < 0)
        Terminate(
            "p=%d < 0  nop->sibling=%d nop->nextnode=%d shmrank=%d nop->sibling_shmrank=%d nop->foreigntask=%d  "
            "first_nontoplevelnode=%d",
            p, nop->sibling, nop->nextnode, shmrank, nop->sibling_shmrank, nop->OriginTask, MaxPart + D->NTopnodes);

      int next;
      unsigned char next_shmrank;
      char type;

      if(p < MaxPart) /* a local particle */
        {
          /* note: here shmrank cannot change */
          next         = get_nextnodep(shmrank)[p];
          next_shmrank = shmrank;
          type         = NODE_TYPE_LOCAL_PARTICLE;
        }
      else if(p < MaxPart + MaxNodes) /* an internal node  */
        {
          ngbnode *nop = get_nodep(p, shmrank);
          next         = nop->sibling;
          next_shmrank = nop->sibling_shmrank;
          type         = NODE_TYPE_LOCAL_NODE;
        }
      else if(p >= ImportedNodeOffset && p < EndOfTreePoints) /* an imported Treepoint particle  */
        {
          Terminate("not expected for SPH");
        }
      else if(p >= EndOfTreePoints && p < EndOfForeignNodes) /* an imported tree node */
        {
          ngbnode *nop = get_nodep(p, shmrank);
          next         = nop->sibling;
          next_shmrank = nop->sibling_shmrank;
          type         = NODE_TYPE_FETCHED_NODE;
        }
      else if(p >= EndOfForeignNodes) /* an imported particle below an imported tree node */
        {
          foreign_sphpoint_data *foreignpoint = get_foreignpointsp(p - EndOfForeignNodes, shmrank);

          next         = foreignpoint->Nextnode;
          next_shmrank = foreignpoint->Nextnode_shmrank;
          type         = NODE_TYPE_FETCHED_PARTICLE;
        }
      else
        {
          /* a pseudo point */
          Terminate(
              "should not happen: p=%d MaxPart=%d MaxNodes=%d  ImportedNodeOffset=%d  EndOfTreePoints=%d  EndOfForeignNodes=%d "
              "shmrank=%d",
              p, MaxPart, MaxNodes, ImportedNodeOffset, EndOfTreePoints, EndOfForeignNodes, shmrank);
        }

      sph_hydro_interact(pdat, p, type, shmrank, mintopleafnode, committed);

      p       = next;
      shmrank = next_shmrank;
    }
}

inline void sph::sph_hydro_interact(pinfo &pdat, int no, char no_type, unsigned char shmrank, int mintopleafnode, int committed)
{
  if(no_type <= NODE_TYPE_FETCHED_PARTICLE)  // we are interacting with a particle
    {
      sph_hydro_check_particle_particle_interaction(pdat, no, no_type, shmrank);
    }
  else  // we are interacting with a node
    {
      ngbnode *nop = get_nodep(no, shmrank);

      if(nop->not_empty == 0)
        return;

      if(no < MaxPart + MaxNodes)                // we have a top-levelnode
        if(nop->nextnode >= MaxPart + MaxNodes)  // if the next node is not a top-level, we have a leaf node
          mintopleafnode = no;

      int openflag = sph_hydro_evaluate_particle_node_opening_criterion(pdat, nop);

      if(openflag == NODE_OPEN) /* we need to open it */
        {
          if(nop->cannot_be_opened_locally.load(std::memory_order_acquire))
            {
              // are we in the same shared memory node?
              if(Shmem.GetNodeIDForSimulCommRank[nop->OriginTask] == Shmem.GetNodeIDForSimulCommRank[D->ThisTask])
                {
                  Terminate("this should not happen any more");
                }
              else
                {
                  tree_add_to_fetch_stack(nop, no, shmrank);  // will only add unique copies

                  tree_add_to_work_stack(pdat.target, no, shmrank, mintopleafnode);
                }
            }
          else
            {
              int min_buffer_space =
                  std::min<int>(MaxOnWorkStack - (NumOnWorkStack + NewOnWorkStack), MaxOnFetchStack - NumOnFetchStack);

              if(min_buffer_space >= committed + 8 * TREE_NUM_BEFORE_NODESPLIT)
                sph_hydro_open_node(pdat, nop, mintopleafnode, committed + 8 * TREE_NUM_BEFORE_NODESPLIT);
              else
                tree_add_to_work_stack(pdat.target, no, shmrank, mintopleafnode);
            }
        }
    }
}

void sph::hydro_forces_determine(int ntarget, int *targetlist)
{
  TIMER_STORE;
  TIMER_START(CPU_HYDRO);

  D->mpi_printf("SPH-HYDRO: Begin hydro-force calculation.  (presently allocated=%g MB)\n", Mem.getAllocatedBytesInMB());
  D->mpi_printf("SPH-HYDRO: global Nhydro=%llu (task zero: NumGas=%d, Nhydro=%d)\n", Tp->TimeBinsHydro.GlobalNActiveParticles,
                Tp->NumGas, ntarget);
  D->mpi_printf("MaxPartSph = %d\n", Tp->MaxPartSph);
  D->mpi_printf("NumPart = %d  TotNumPart = %d  TotNumGas = %d  MaxPart = %d  NActP = %d GNActP = %d\n", Tp->NumPart, Tp->TotNumPart,
                Tp->TotNumGas, Tp->MaxPart, Tp->TimeBinsHydro.NActiveParticles, Tp->TimeBinsHydro.GlobalNActiveParticles);

  double ta = Logs.second();

  // let's grab at most half the still available memory for imported points and nodes
  int nspace = (0.33 * Mem.FreeBytes) / (sizeof(ngbnode) + 8 * sizeof(foreign_sphpoint_data));

  MaxForeignNodes  = nspace;
  MaxForeignPoints = 8 * nspace;
  NumForeignNodes  = 0;
  NumForeignPoints = 0;

  sum_NumForeignNodes  = 0;
  sum_NumForeignPoints = 0;

  /* the following two arrays hold imported tree nodes and imported points to augment the local tree */
  Foreign_Nodes  = (ngbnode *)Mem.mymalloc_movable(&Foreign_Nodes, "Foreign_Nodes", MaxForeignNodes * sizeof(ngbnode));
  Foreign_Points = (foreign_sphpoint_data *)Mem.mymalloc_movable(&Foreign_Points, "Foreign_Points",
                                                                 MaxForeignPoints * sizeof(foreign_sphpoint_data));

  tree_initialize_leaf_node_access_info();

  max_ncycles = 0;

  prepare_shared_memory_access();

  if(All.ComovingIntegrationOn)
    {
      fac_mu       = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
      fac_vsic_fix = All.cf_hubble_a * pow(All.Time, 3 * GAMMA_MINUS1);
    }
  else
    {
      fac_mu       = 1.0;
      fac_vsic_fix = 1.0;
    }

  Ngbhydrodat = (ngbdata_hydro *)Mem.mymalloc("Ngbhydrodat", MAX_NGBS * sizeof(ngbdata_hydro));

#ifdef PBH_EFD /* Create list of scattering events. */
  scatter_list = (scatter_event *)Mem.mymalloc("scatter_list", 10 * Tp->TimeBinsHydro.GlobalNActiveParticles * sizeof(scatter_event));
  nscatterevents           = 0;
  numberofparticles        = 0;
  numberoflocalparticles   = 0;
  numberofforeignparticles = 0;
  pairsconsidered          = 0;
  n0vrelbefore             = 0;
  n0vrelafter              = 0;
  ti_step_to_phys          = 1 / (All.HubbleParam * Driftfac.hubble_function(All.Time));
  scatter_prob_to_phys     = All.HubbleParam * All.HubbleParam / pow(All.Time, 0);
#endif

  NumOnWorkStack         = 0;
  AllocWorkStackBaseLow  = std::max<int>(1.5 * (Tp->NumPart + NumPartImported), TREE_MIN_WORKSTACK_SIZE);
  AllocWorkStackBaseHigh = AllocWorkStackBaseLow + TREE_EXPECTED_CYCLES * TREE_MIN_WORKSTACK_SIZE;
  MaxOnWorkStack         = AllocWorkStackBaseLow;

  WorkStack = (workstack_data *)Mem.mymalloc("WorkStack", AllocWorkStackBaseHigh * sizeof(workstack_data));

  for(int i = 0; i < ntarget; i++)
    {
      int target = targetlist[i];

      clear_hydro_result(&Tp->SphP[target]);

      WorkStack[NumOnWorkStack].Target         = target;
      WorkStack[NumOnWorkStack].Node           = MaxPart;
      WorkStack[NumOnWorkStack].ShmRank        = Shmem.Island_ThisTask;
      WorkStack[NumOnWorkStack].MinTopLeafNode = MaxPart + D->NTopnodes;
      NumOnWorkStack++;
    }

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  workstack_data *WorkStackBak = (workstack_data *)Mem.mymalloc("WorkStackBak", NumOnWorkStack * sizeof(workstack_data));
  int NumOnWorkStackBak        = NumOnWorkStack;
  memcpy(WorkStackBak, WorkStack, NumOnWorkStack * sizeof(workstack_data));
#endif

  // set a default size of the fetch stack equal to half the work stack (this may still be somewhat too large)
  MaxOnFetchStack = std::max<int>(0.1 * (Tp->NumPart + NumPartImported), TREE_MIN_WORKSTACK_SIZE);
  StackToFetch    = (fetch_data *)Mem.mymalloc_movable(&StackToFetch, "StackToFetch", MaxOnFetchStack * sizeof(fetch_data));

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  for(int rep = 0; rep < 2; rep++)
    {
      if(rep == 0)
        {
          skip_actual_force_computation = true;
        }
      else
        {
          skip_actual_force_computation = false;
          NumOnWorkStack                = NumOnWorkStackBak;
          memcpy(WorkStack, WorkStackBak, NumOnWorkStack * sizeof(workstack_data));
        }
#endif

      while(NumOnWorkStack > 0)  // repeat until we are out of work
        {
          NewOnWorkStack  = 0;  // gives the new entries
          NumOnFetchStack = 0;
          MaxOnWorkStack  = std::min<int>(AllocWorkStackBaseLow + max_ncycles * TREE_MIN_WORKSTACK_SIZE, AllocWorkStackBaseHigh);

          TIMER_START(CPU_HYDROWALK);

          int item = 0;

          while(item < NumOnWorkStack)
            {
              int committed = 8 * TREE_NUM_BEFORE_NODESPLIT;
              int min_buffer_space =
                  std::min<int>(MaxOnWorkStack - (NumOnWorkStack + NewOnWorkStack), MaxOnFetchStack - NumOnFetchStack);
              if(min_buffer_space >= committed)
                {
                  int target     = WorkStack[item].Target;
                  int no         = WorkStack[item].Node;
                  int shmrank    = WorkStack[item].ShmRank;
                  int mintopleaf = WorkStack[item].MinTopLeafNode;
                  item++;

                  pinfo pdat;
                  get_pinfo(target, pdat);

                  if(no == MaxPart)
                    {
                      // we have a pristine particle that's processed for the first time
                      sph_hydro_interact(pdat, no, NODE_TYPE_LOCAL_NODE, shmrank, mintopleaf, committed);
                    }
                  else
                    {
                      // we have a node that we previously could not open
                      ngbnode *nop = get_nodep(no, shmrank);

                      if(nop->cannot_be_opened_locally)
                        {
                          Terminate("item=%d:  no=%d  now we should be able to open it!", item, no);
                        }
                      else
                        sph_hydro_open_node(pdat, nop, mintopleaf, committed);
                    }
#ifdef PBH_EFD
                  scatter_evaluate_kernel(pdat);
#else
              hydro_evaluate_kernel(pdat);
#endif
                }
              else
                break;
            }

          if(item == 0 && NumOnWorkStack > 0)
            Terminate("Can't even process a single particle");

          TIMER_STOP(CPU_HYDROWALK);

          TIMER_START(CPU_HYDROFETCH);

          tree_fetch_foreign_nodes(FETCH_SPH_HYDRO);

          TIMER_STOP(CPU_HYDROFETCH);

          printf("NewOnWorkStack = %d  sum_NumForeignPoints = %lld\n", NewOnWorkStack, sum_NumForeignPoints);

          /* now reorder the workstack such that we are first going to do residual pristine particles, and then
           * imported nodes that hang below the first leaf nodes */
          NumOnWorkStack = NumOnWorkStack - item + NewOnWorkStack;
          memmove(WorkStack, WorkStack + item, NumOnWorkStack * sizeof(workstack_data));

          /* now let's sort such that we can go deep on top-level node branches, allowing us to clear them out eventually */
          mycxxsort(WorkStack, WorkStack + NumOnWorkStack, compare_workstack);

          max_ncycles++;
        }

#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
    }
#endif
#ifdef PBH_EFD
  scatter_list_evaluate(scatter_list, nscatterevents);

  D->mpi_printf("Number of particles = %d  Number of local particles = %d  Number of foreign particles = %d\n", numberofparticles,
                numberoflocalparticles, numberofforeignparticles);
  D->mpi_printf("Number of scattering events = %d  Pairs considered = %d\n", nscatterevents, pairsconsidered);
  D->mpi_printf("Number of 0 vrel pairs = %d  Remaining after check = %d\n", n0vrelbefore, n0vrelafter);
#endif
  Mem.myfree(StackToFetch);
#ifdef PRESERVE_SHMEM_BINARY_INVARIANCE
  Mem.myfree(WorkStackBak);
#endif
  Mem.myfree(WorkStack);
#ifdef PBH_EFD
  Mem.myfree(scatter_list);
#endif
  Mem.myfree(Ngbhydrodat);

  /* now factor in a prefactor for the computed rates */
  for(int i = 0; i < ntarget; i++)
    {
      int target = targetlist[i];

      double fac = GAMMA_MINUS1 / (All.cf_atime2_hubble_a * pow(Tp->SphP[target].Density, GAMMA_MINUS1));

      Tp->SphP[target].DtEntropy *= fac;
    }

  /* Now the tree-based hydrodynamical force computation is finished,
   * output some performance metrics
   */

  TIMER_START(CPU_HYDROIMBALANCE);

  MPI_Allreduce(MPI_IN_PLACE, &max_ncycles, 1, MPI_INT, MPI_MAX, D->Communicator);

  TIMER_STOP(CPU_HYDROIMBALANCE);

  cleanup_shared_memory_access();

  /* free temporary buffers */
  Mem.myfree(Foreign_Points);
  Mem.myfree(Foreign_Nodes);

  double tb = Logs.second();

  TIMER_STOPSTART(CPU_HYDRO, CPU_LOGS);

  D->mpi_printf("SPH-HYDRO: hydro-force computation done. took %8.3f\n", Logs.timediff(ta, tb));

  struct detailed_timings
  {
    double tree, wait, fetch, all;
    double numnodes;
    double NumForeignNodes, NumForeignPoints;
    double fillfacFgnNodes, fillfacFgnPoints;
  };
  detailed_timings timer, tisum, timax;

  timer.tree             = TIMER_DIFF(CPU_HYDROWALK);
  timer.wait             = TIMER_DIFF(CPU_HYDROIMBALANCE);
  timer.fetch            = TIMER_DIFF(CPU_HYDROFETCH);
  timer.all              = timer.tree + timer.wait + timer.fetch + TIMER_DIFF(CPU_HYDRO);
  timer.numnodes         = NumNodes;
  timer.NumForeignNodes  = NumForeignNodes;
  timer.NumForeignPoints = NumForeignPoints;
  timer.fillfacFgnNodes  = NumForeignNodes / ((double)MaxForeignNodes);
  timer.fillfacFgnPoints = NumForeignPoints / ((double)MaxForeignPoints);

  MPI_Reduce((double *)&timer, (double *)&tisum, (int)(sizeof(detailed_timings) / sizeof(double)), MPI_DOUBLE, MPI_SUM, 0,
             D->Communicator);
  MPI_Reduce((double *)&timer, (double *)&timax, (int)(sizeof(detailed_timings) / sizeof(double)), MPI_DOUBLE, MPI_MAX, 0,
             D->Communicator);

  All.TotNumHydro += Tp->TimeBinsHydro.GlobalNActiveParticles;

  printf("ThisTask = %d\n", D->ThisTask);

  if(D->ThisTask == 0)
    {
      fprintf(Logs.FdHydro, "Nf=%9lld  highest active timebin=%d  total-Nf=%lld\n", Tp->TimeBinsHydro.GlobalNActiveParticles,
              All.HighestActiveTimeBin, All.TotNumHydro);
      fprintf(Logs.FdHydro, "   work-load balance: %g   part/sec: raw=%g, effective=%g\n",
              timax.tree / ((tisum.tree + 1e-20) / D->NTask), Tp->TimeBinsGravity.GlobalNActiveParticles / (tisum.tree + 1.0e-20),
              Tp->TimeBinsGravity.GlobalNActiveParticles / ((timax.tree + 1.0e-20) * D->NTask));
      fprintf(Logs.FdHydro,
              "   maximum number of nodes: %g, filled: %g  NumForeignNodes: max=%g avg=%g fill=%g NumForeignPoints: max=%g avg=%g "
              "fill=%g  cycles=%d\n",
              timax.numnodes, timax.numnodes / MaxNodes, timax.NumForeignNodes, tisum.NumForeignNodes / D->NTask,
              timax.fillfacFgnNodes, timax.NumForeignPoints, tisum.NumForeignPoints / D->NTask, timax.fillfacFgnPoints, max_ncycles);
      fprintf(Logs.FdHydro, "   avg times: <all>=%g  <tree>=%g  <wait>=%g  <fetch>=%g  sec\n", tisum.all / D->NTask,
              tisum.tree / D->NTask, tisum.wait / D->NTask, tisum.fetch / D->NTask);
      myflush(Logs.FdHydro);
    }

  TIMER_STOP(CPU_LOGS);
}

#ifdef EXPLICIT_VECTORIZATION
void sph::hydro_evaluate_kernel(pinfo &pdat)
{
#ifndef LEAN
  particle_data *P_i        = &Tp->P[pdat.target];
  sph_particle_data *SphP_i = &Tp->SphP[pdat.target];

  /* the particles needs to be active */
  if(P_i->getTimeBinHydro() > All.HighestSynchronizedTimeBin)
    Terminate("bummer");

  double shinv, shinv3, shinv4;
  kernel_hinv(SphP_i->Hsml, &shinv, &shinv3, &shinv4);

  Vec4d hinv(shinv);
  Vec4d hinv3(shinv3);
  Vec4d hinv4(shinv4);

  Vec4d dwnorm(NORM * shinv3);
  Vec4d dwknorm(NORM * shinv4);

  Vec4d rho_i(SphP_i->Density);

#ifdef PRESSURE_ENTROPY_SPH
  Vec4d p_over_rho2_i((double)SphP_i->Pressure / ((double)SphP_i->PressureSphDensity * (double)SphP_i->PressureSphDensity));
#else
  Vec4d p_over_rho2_i((double)SphP_i->Pressure / ((double)SphP_i->Density * (double)SphP_i->Density));
#endif

  Vec4d sound_i(SphP_i->Csnd);
  Vec4d h_i(SphP_i->Hsml);

  Vec4d v_i[3];
  for(int i = 0; i < NUMDIMS; i++)
    {
      v_i[i] = SphP_i->VelPred[i];
    }
  Vec4d DhsmlDensityFactor_i(SphP_i->DhsmlDensityFactor);
#ifdef PRESSURE_ENTROPY_SPH
  Vec4d DhsmlDerivedDensityFactor_i(SphP_i->DhsmlDerivedDensityFactor);
  Vec4d EntropyToInvGammaPred_i(SphP_i->EntropyToInvGammaPred);
#endif

#if !defined(NO_SHEAR_VISCOSITY_LIMITER) && !defined(TIMEDEP_ART_VISC)
  Vec4d f_i(fabs(SphP_i->DivVel) / (fabs(SphP_i->DivVel) + SphP_i->CurlVel + 0.0001 * SphP_i->Csnd / SphP_i->Hsml / fac_mu));
#endif

#ifdef TIMEDEP_ART_VISC
  Vec4d alpha_i(SphP_i->Alpha);
#endif
  /* Now start the actual SPH computation for this particle */

  double dacc[3]     = {0};
  double dentr       = 0;
  Vec4d MaxSignalVel = sound_i;

  const int vector_length = 4;
  const int array_length  = (pdat.numngb + vector_length - 1) & (-vector_length);

  for(int n = pdat.numngb; n < array_length; n++) /* fill up neighbour array so that sensible data is accessed */
    Ngbhydrodat[n] = Ngbhydrodat[0];

  for(int n = 0; n < array_length; n += vector_length)
    {
      sph_particle_data_hydrocore *ngb0 = Ngbhydrodat[n + 0].SphCore;
      sph_particle_data_hydrocore *ngb1 = Ngbhydrodat[n + 1].SphCore;
      sph_particle_data_hydrocore *ngb2 = Ngbhydrodat[n + 2].SphCore;
      sph_particle_data_hydrocore *ngb3 = Ngbhydrodat[n + 3].SphCore;

      ngbdata_hydro *P0_j = &Ngbhydrodat[n + 0];
      ngbdata_hydro *P1_j = &Ngbhydrodat[n + 1];
      ngbdata_hydro *P2_j = &Ngbhydrodat[n + 2];
      ngbdata_hydro *P3_j = &Ngbhydrodat[n + 3];

      /* converts the integer distance to floating point */
      Vec4d dpos[NUMDIMS];
      double posdiff[array_length][3];
      for(int i = 0; i < 4; i++)
        {
          Tp->nearest_image_intpos_to_pos(P_i->IntPos, Ngbhydrodat[n + i].IntPos, &(posdiff[i][0]));
        }

      for(int i = 0; i < NUMDIMS; i++)
        {
          dpos[i] = Vec4d(posdiff[0][i], posdiff[1][i], posdiff[2][i], posdiff[3][i]);
        }

      Vec4d r2(0);

      for(int i = 0; i < NUMDIMS; i++)
        {
          r2 += dpos[i] * dpos[i];
        }

      Vec4d r = sqrt(r2);

      Vec4d v_j[NUMDIMS];
      for(int i = 0; i < NUMDIMS; i++)
        {
          v_j[i] = Vec4d(ngb0->VelPred[i], ngb1->VelPred[i], ngb2->VelPred[i], ngb3->VelPred[i]);
        }

      Vec4d pressure(ngb0->Pressure, ngb1->Pressure, ngb2->Pressure, ngb3->Pressure);
      Vec4d rho_j(ngb0->Density, ngb1->Density, ngb2->Density, ngb3->Density);
#ifdef PRESSURE_ENTROPY_SPH
      Vec4d rho_press_j(ngb0->PressureSphDensity, ngb1->PressureSphDensity, ngb2->PressureSphDensity, ngb3->PressureSphDensity);
      Vec4d p_over_rho2_j = pressure / (rho_press_j * rho_press_j);
#else
      Vec4d p_over_rho2_j = pressure / (rho_j * rho_j);
#endif

      Vec4d wk_i, dwk_i;
      Vec4d u = r * hinv;
      kernel_main_vector(u, dwnorm, dwknorm, &wk_i, &dwk_i);
      Vec4db decision = (r < h_i);
      Vec4d fac       = select(decision, 1., 0.);
      wk_i *= fac;
      dwk_i *= fac;

      Vec4d h_j(ngb0->Hsml, ngb1->Hsml, ngb2->Hsml, ngb3->Hsml);
      Vec4d hinv_j = 1 / h_j;
#ifdef THREEDIMS
      Vec4d hinv3_j = hinv_j * hinv_j * hinv_j;
#endif

#ifdef TWODIMS
      Vec4d hinv3_j = hinv_j * hinv_j;
#endif

#ifdef ONEDIMS
      Vec4d hinv3_j = hinv_j;
#endif
      Vec4d hinv4_j = hinv3_j * hinv_j;

      Vec4d wk_j, dwk_j;
      u = r * hinv_j;
      kernel_main_vector(u, NORM * hinv3_j, NORM * hinv4_j, &wk_j, &dwk_j);
      decision = (r < h_j);
      fac      = select(decision, 1., 0.);
      wk_j *= fac;
      dwk_j *= fac;

      Vec4d sound_j(ngb0->Csnd, ngb1->Csnd, ngb2->Csnd, ngb3->Csnd);
      Vec4d vsig = sound_i + sound_j;
      if(n + vector_length > pdat.numngb)
        {
          wk_i.cutoff(vector_length - (array_length - pdat.numngb));
          dwk_i.cutoff(vector_length - (array_length - pdat.numngb));
          wk_j.cutoff(vector_length - (array_length - pdat.numngb));
          dwk_j.cutoff(vector_length - (array_length - pdat.numngb));
          vsig.cutoff(vector_length - (array_length - pdat.numngb));
        }

      Vec4d dwk_ij = 0.5 * (dwk_i + dwk_j);

      MaxSignalVel = max(MaxSignalVel, vsig);

      Vec4d visc(0);

      Vec4d dv[NUMDIMS];
      for(int i = 0; i < NUMDIMS; i++)
        {
          dv[i] = v_i[i] - v_j[i];
        }

      Vec4d vdotr2(0);
      for(int i = 0; i < NUMDIMS; i++)
        {
          vdotr2 += dv[i] * dpos[i];
        }

      if(All.ComovingIntegrationOn)
        vdotr2 += All.cf_atime2_hubble_a * r2;

      decision            = (vdotr2 < 0);
      Vec4d viscosity_fac = select(decision, 1, 0);

      /* ... artificial viscosity */

      Vec4d mu_ij = fac_mu * vdotr2 / r;

      vsig -= 3 * mu_ij;

#if defined(NO_SHEAR_VISCOSITY_LIMITER) || defined(TIMEDEP_ART_VISC)
      Vec4d f_i(1);
      Vec4d f_j(1);
#else
      Vec4d DivVel_j(ngb0->DivVel, ngb1->DivVel, ngb2->DivVel, ngb3->DivVel);
      Vec4d CurlVel_j(ngb0->CurlVel, ngb1->CurlVel, ngb2->CurlVel, ngb3->CurlVel);
      Vec4d f_j = abs(DivVel_j) / (abs(DivVel_j) + CurlVel_j + 0.0001 * sound_j / fac_mu * hinv_j);
#endif

#ifdef TIMEDEP_ART_VISC
      Vec4d alpha_j(ngb0->Alpha, ngb1->Alpha, ngb2->Alpha, ngb3->Alpha);
      Vec4d BulkVisc_ij = 0.5 * (alpha_i + alpha_j);

#else
      Vec4d BulkVisc_ij(All.ArtBulkViscConst);
#endif
      Vec4d rho_ij_inv = 2.0 / (rho_i + rho_j);
      visc             = 0.25 * BulkVisc_ij * vsig * (-mu_ij) * rho_ij_inv * (f_i + f_j);
      Vec4d mass_j(P0_j->Mass, P1_j->Mass, P2_j->Mass, P3_j->Mass);
#ifdef VISCOSITY_LIMITER_FOR_LARGE_TIMESTEPS
      Vec4i timeBin_i(P_i->TimeBinHydro);
      Vec4i timeBin_j(P0_j->TimeBinHydro, P1_j->TimeBinHydro, P2_j->TimeBinHydro, P3_j->TimeBinHydro);

      Vec4i timebin = max(timeBin_i, timeBin_j);
      Vec4i integer_time(((integertime)1) << timebin[0], ((integertime)1) << timebin[1], ((integertime)1) << timebin[2],
                         ((integertime)1) << timebin[3]);

      Vec4ib decision_i    = (timebin != 0);
      Vec4i factor_timebin = select(decision_i, Vec4i(1), Vec4i(0));
      Vec4d dt             = to_double(2 * integer_time * factor_timebin) * All.Timebase_interval;

      decision = (dt > 0 && dwk_ij < 0);

      Vec4d visc_alternavtive = 0.5 * fac_vsic_fix * vdotr2 / ((P_i->getMass() + mass_j) * dwk_ij * r * dt);

      Vec4d visc2 = select(decision, visc_alternavtive, visc);
      visc        = min(visc, visc2);
#endif

      Vec4d hfc_visc = mass_j * visc * dwk_ij / r * viscosity_fac;

#ifndef PRESSURE_ENTROPY_SPH
      /* Formulation derived from the Lagrangian */
      dwk_i *= DhsmlDensityFactor_i;
      Vec4d DhsmlDensityFactor_j(ngb0->DhsmlDensityFactor, ngb1->DhsmlDensityFactor, ngb2->DhsmlDensityFactor,
                                 ngb3->DhsmlDensityFactor);
      dwk_j *= DhsmlDensityFactor_j;

      Vec4d hfc = mass_j * (p_over_rho2_i * dwk_i + p_over_rho2_j * dwk_j) / r + hfc_visc;
#else
      Vec4d EntropyToInvGammaPred_j(ngb0->EntropyToInvGammaPred, ngb1->EntropyToInvGammaPred, ngb2->EntropyToInvGammaPred,
                                    ngb3->EntropyToInvGammaPred);
      Vec4d DhsmlDerivedDensityFactor_j(ngb0->DhsmlDerivedDensityFactor, ngb1->DhsmlDerivedDensityFactor,
                                        ngb2->DhsmlDerivedDensityFactor, ngb3->DhsmlDerivedDensityFactor);
      /* leading order term */
      Vec4d hfc = mass_j *
                  (p_over_rho2_i * dwk_i * EntropyToInvGammaPred_j / EntropyToInvGammaPred_i +
                   p_over_rho2_j * dwk_j * EntropyToInvGammaPred_i / EntropyToInvGammaPred_j) /
                  r;

      /* grad-h term */
      hfc += mass_j *
             (p_over_rho2_i * dwk_i * SphP_i->DhsmlDerivedDensityFactor + p_over_rho2_j * dwk_j * DhsmlDerivedDensityFactor_j) / r;

      /* add viscous term */
      hfc += hfc_visc;
#endif

      for(int i = 0; i < NUMDIMS; i++)
        {
          dacc[i] += horizontal_add(-hfc * dpos[i]);
        }
      dentr += horizontal_add(0.5 * (hfc_visc)*vdotr2);
    }

  SphP_i->HydroAccel[0] += dacc[0];
  SphP_i->HydroAccel[1] += dacc[1];
  SphP_i->HydroAccel[2] += dacc[2];
  SphP_i->DtEntropy += dentr;

  for(int i = 0; i < 4; i++)
    {
      if(SphP_i->MaxSignalVel < MaxSignalVel[i])
        SphP_i->MaxSignalVel = MaxSignalVel[i];
    }
#endif
}

#else

/*! This function is the 'core' of the SPH force computation. A target
 *  particle is specified which may either be local, or reside in the
 *  communication buffer.
 */
void sph::hydro_evaluate_kernel(pinfo &pdat)
{
#ifndef LEAN
  particle_data *P_i = &Tp->P[pdat.target];
  sph_particle_data *SphP_i = &Tp->SphP[pdat.target];

  /* the particles needs to be active */
  if(P_i->getTimeBinHydro() > All.HighestSynchronizedTimeBin)
    Terminate("bummer");

#ifdef PRESSURE_ENTROPY_SPH
  double p_over_rho2_i = (double)SphP_i->Pressure / ((double)SphP_i->PressureSphDensity * (double)SphP_i->PressureSphDensity);
#else
  double p_over_rho2_i = (double)SphP_i->Pressure / ((double)SphP_i->Density * (double)SphP_i->Density);
#endif

  kernel_hydra kernel;

  kernel.sound_i = SphP_i->Csnd;
  kernel.h_i = SphP_i->Hsml;

  /* Now start the actual SPH computation for this particle */

  double daccx = 0;
  double daccy = 0;
  double daccz = 0;
  double dentr = 0;
  double MaxSignalVel = kernel.sound_i;

  for(int n = 0; n < pdat.numngb; n++)
    {
      sph_particle_data_hydrocore *SphP_j = Ngbhydrodat[n].SphCore;
      ngbdata_hydro *P_j = &Ngbhydrodat[n];

      /* converts the integer distance to floating point */
      double posdiff[3];
      Tp->nearest_image_intpos_to_pos(P_i->IntPos, P_j->IntPos, posdiff);

      kernel.dx = posdiff[0];
      kernel.dy = posdiff[1];
      kernel.dz = posdiff[2];

      double r2 = kernel.dx * kernel.dx + kernel.dy * kernel.dy + kernel.dz * kernel.dz;
      kernel.h_j = SphP_j->Hsml;

      if(r2 < kernel.h_i * kernel.h_i || r2 < kernel.h_j * kernel.h_j)
        {
          kernel.r = sqrt(r2);
          if(kernel.r > 0)
            {
#ifdef PRESSURE_ENTROPY_SPH
              double p_over_rho2_j =
                  (double)SphP_j->Pressure / ((double)SphP_j->PressureSphDensity * (double)SphP_j->PressureSphDensity);
#else
              double p_over_rho2_j = (double)SphP_j->Pressure / ((double)SphP_j->Density * (double)SphP_j->Density);
#endif

              kernel.sound_j = SphP_j->Csnd;

              kernel.dvx = SphP_i->VelPred[0] - SphP_j->VelPred[0];
              kernel.dvy = SphP_i->VelPred[1] - SphP_j->VelPred[1];
              kernel.dvz = SphP_i->VelPred[2] - SphP_j->VelPred[2];
              kernel.vdotr2 = kernel.dx * kernel.dvx + kernel.dy * kernel.dvy + kernel.dz * kernel.dvz;
              kernel.rho_ij_inv = 2.0 / (SphP_i->Density + SphP_j->Density);

              if(All.ComovingIntegrationOn)
                kernel.vdotr2 += All.cf_atime2_hubble_a * r2;

              double hinv, hinv3, hinv4;
              if(kernel.r < kernel.h_i)
                {
                  kernel_hinv(kernel.h_i, &hinv, &hinv3, &hinv4);
                  double u = kernel.r * hinv;
                  kernel_main(u, hinv3, hinv4, &kernel.wk_i, &kernel.dwk_i, COMPUTE_DWK);
                }
              else
                {
                  kernel.dwk_i = 0;
                  kernel.wk_i = 0;
                }

              if(kernel.r < kernel.h_j)
                {
                  kernel_hinv(kernel.h_j, &hinv, &hinv3, &hinv4);
                  double u = kernel.r * hinv;
                  kernel_main(u, hinv3, hinv4, &kernel.wk_j, &kernel.dwk_j, COMPUTE_DWK);
                }
              else
                {
                  kernel.dwk_j = 0;
                  kernel.wk_j = 0;
                }

              kernel.dwk_ij = 0.5 * (kernel.dwk_i + kernel.dwk_j);

              kernel.vsig = kernel.sound_i + kernel.sound_j;

              if(kernel.vsig > MaxSignalVel)
                MaxSignalVel = kernel.vsig;

              double visc = 0;

              if(kernel.vdotr2 < 0) /* ... artificial viscosity */
                {
                  double mu_ij = fac_mu * kernel.vdotr2 / kernel.r;

                  kernel.vsig -= 3 * mu_ij;

#if defined(NO_SHEAR_VISCOSITY_LIMITER) || defined(TIMEDEP_ART_VISC)
                  double f_i = 1.;
                  double f_j = 1.;
#else
                  double f_i =
                      fabs(SphP_i->DivVel) / (fabs(SphP_i->DivVel) + SphP_i->CurlVel + 0.0001 * SphP_i->Csnd / SphP_i->Hsml / fac_mu);

                  double f_j =
                      fabs(SphP_j->DivVel) / (fabs(SphP_j->DivVel) + SphP_j->CurlVel + 0.0001 * kernel.sound_j / fac_mu / kernel.h_j);
#endif

#ifdef TIMEDEP_ART_VISC
                  double BulkVisc_ij = 0.5 * (SphP_i->Alpha + SphP_j->Alpha);

#else
                  double BulkVisc_ij = All.ArtBulkViscConst;
#endif

                  visc = 0.25 * BulkVisc_ij * kernel.vsig * (-mu_ij) * kernel.rho_ij_inv * (f_i + f_j);
#ifdef VISCOSITY_LIMITER_FOR_LARGE_TIMESTEPS
                  int timebin = std::max<int>(P_i->TimeBinHydro, P_j->TimeBinHydro);

                  double dt = 2 * (timebin ? (((integertime)1) << timebin) : 0) * All.Timebase_interval;

                  if(dt > 0 && kernel.dwk_ij < 0)
                    {
                      visc = std::min<double>(
                          visc, 0.5 * fac_vsic_fix * kernel.vdotr2 / ((P_i->getMass() + P_j->Mass) * kernel.dwk_ij * kernel.r * dt));
                    }
#endif
                }

              double hfc_visc = P_j->Mass * visc * kernel.dwk_ij / kernel.r;

#ifndef PRESSURE_ENTROPY_SPH
              /* Formulation derived from the Lagrangian */
              kernel.dwk_i *= SphP_i->DhsmlDensityFactor;
              kernel.dwk_j *= SphP_j->DhsmlDensityFactor;

              double hfc = P_j->Mass * (p_over_rho2_i * kernel.dwk_i + p_over_rho2_j * kernel.dwk_j) / kernel.r + hfc_visc;
#else
              /* leading order term */
              double hfc = P_j->Mass *
                           (p_over_rho2_i * kernel.dwk_i * SphP_j->EntropyToInvGammaPred / SphP_i->EntropyToInvGammaPred +
                            p_over_rho2_j * kernel.dwk_j * SphP_i->EntropyToInvGammaPred / SphP_j->EntropyToInvGammaPred) /
                           kernel.r;

              /* grad-h term */
              hfc += P_j->Mass *
                     (p_over_rho2_i * kernel.dwk_i * SphP_i->DhsmlDerivedDensityFactor +
                      p_over_rho2_j * kernel.dwk_j * SphP_j->DhsmlDerivedDensityFactor) /
                     kernel.r;

              /* add viscous term */
              hfc += hfc_visc;
#endif

              daccx += (-hfc * kernel.dx);
              daccy += (-hfc * kernel.dy);
              daccz += (-hfc * kernel.dz);
              dentr += (0.5 * (hfc_visc)*kernel.vdotr2);
            }
        }
    }

  SphP_i->HydroAccel[0] += daccx;
  SphP_i->HydroAccel[1] += daccy;
  SphP_i->HydroAccel[2] += daccz;
  SphP_i->DtEntropy += dentr;

  if(SphP_i->MaxSignalVel < MaxSignalVel)
    SphP_i->MaxSignalVel = MaxSignalVel;

#endif /* LEAN */
}

void sph::scatter_evaluate_kernel(pinfo &pdat)
{
#ifndef LEAN
  particle_data *P_i = &Tp->P[pdat.target];
  sph_particle_data *SphP_i = &Tp->SphP[pdat.target];

  // if((int )P_i->ID.get() != pdat.target)
  //    Terminate("pdat.target is not the same as the ID"); //They are not the same. But what does that mean for accessing particles
  //    later on?

  /* the particles needs to be active */
  if(P_i->getTimeBinHydro() > All.HighestSynchronizedTimeBin)
    Terminate("bummer");

  kernel_hydra kernel;

  kernel.sound_i = SphP_i->Csnd;
  kernel.h_i = SphP_i->Hsml;

  /* Now start the actual scatter computation for this particle */
  /*  double Delta_velx_i = 0;
    double Delta_vely_i = 0;
    double Delta_velz_i = 0;  */

  int timebin_i = P_i->TimeBinHydro;
  double dt_i = (timebin_i ? (((integertime)1) << timebin_i) : 0) * All.Timebase_interval;
  double dt_i_phys = dt_i / ti_step_to_phys;

  SphP_i->scatter_occurrence = 0;
  for(int g = 0; g < 3; g++)
    SphP_i->scatter_delta_vel[g] = 0;

  for(int n = 0; n < pdat.numngb; n++)
    {
      sph_particle_data_hydrocore *SphP_j = Ngbhydrodat[n].SphCore;
      ngbdata_hydro *P_j = &Ngbhydrodat[n];
      //      D->mpi_printf("ID i = %d  ID j = %d\n", P_i->ID.get(), P_j->ID); //just to test that passing the IDs works; it does.

      if(P_j->ID >= P_i->ID.get())
        continue;
      else
        {
          pairsconsidered++;
          /* converts the integer distance to floating point */
          double posdiff[3];
          Tp->nearest_image_intpos_to_pos(P_i->IntPos, P_j->IntPos, posdiff);

          kernel.dx = posdiff[0];
          kernel.dy = posdiff[1];
          kernel.dz = posdiff[2];

          double r2 = kernel.dx * kernel.dx + kernel.dy * kernel.dy + kernel.dz * kernel.dz;
          kernel.h_j = SphP_j->Hsml;

          /*      Delta_velx_i = SphP_i->VelPred[0];
                Delta_vely_i = SphP_i->VelPred[1];
                Delta_velz_i = SphP_i->VelPred[2];  */

          if(r2 < kernel.h_i * kernel.h_i || r2 < kernel.h_j * kernel.h_j)
            {
              kernel.r = sqrt(r2);
              if(kernel.r > 0)
                {
                  kernel.sound_j = SphP_j->Csnd;

                  kernel.dvx = SphP_i->VelPred[0] - SphP_j->VelPred[0];
                  kernel.dvy = SphP_i->VelPred[1] - SphP_j->VelPred[1];
                  kernel.dvz = SphP_i->VelPred[2] - SphP_j->VelPred[2];
                  kernel.dv2 = kernel.dvx * kernel.dvx + kernel.dvy * kernel.dvy + kernel.dvz * kernel.dvz;
                  kernel.dv = sqrt(kernel.dv2);
                  kernel.dvinv3 = 1.0 / (kernel.dv * kernel.dv2);

                  if(kernel.dv == 0)
                    n0vrelbefore++;

                  double dist_vrel_check = std::min(SphP_i->dist_over_time, SphP_j->dist_over_time) * kernel.r / kernel.dv;

                  //              D->mpi_printf("vrel = %f  dist_vrel_check = %f  d_o_t_i = %f  d_o_t_j = %f\n", kernel.dv,
                  //              dist_vrel_check, SphP_i->dist_over_time, SphP_j->dist_over_time);

                  if(3000.0 * All.Time < dist_vrel_check) /*Not very efficient, only need to compute 1/a once. */
                    continue;
                  else
                    {
                      if(kernel.dv == 0)
                        n0vrelafter++;
                      double hinv, hinv3, hinv4;
                      if(kernel.r < kernel.h_i)
                        {
                          kernel_hinv(kernel.h_i, &hinv, &hinv3, &hinv4);
                          double u = kernel.r * hinv;
                          kernel_main(u, hinv3, hinv4, &kernel.wk_i, &kernel.dwk_i, COMPUTE_WK);
                        }
                      else
                        {
                          kernel.dwk_i = 0;
                          kernel.wk_i = 0;
                        }

                      if(kernel.r < kernel.h_j)
                        {
                          kernel_hinv(kernel.h_j, &hinv, &hinv3, &hinv4);
                          double u = kernel.r * hinv;
                          kernel_main(u, hinv3, hinv4, &kernel.wk_j, &kernel.dwk_j, COMPUTE_WK);
                        }
                      else
                        {
                          kernel.dwk_j = 0;
                          kernel.wk_j = 0;
                        }

                      int timebin_j = P_j->TimeBinHydro;
                      double dt_j = (timebin_j ? (((integertime)1) << timebin_j) : 0) * All.Timebase_interval;
                      double dt_j_phys = dt_j / ti_step_to_phys;

                      double scatter_prob;

                      if(P_i->getMass() == P_j->Mass && dt_i_phys == dt_j_phys && kernel.h_i == kernel.h_j)
                        {
                          scatter_prob = kernel.dvinv3 * All.SigmaOverM * dt_i_phys * P_j->Mass * kernel.wk_i * scatter_prob_to_phys;
                          //                  scatter_prob =  kernel.dvinv3 * All.SigmaOverM * dt_i_phys * P_j->Mass * kernel.wk_i;
                        }
                      else
                        {
                          double scatter_prob_i_on_j = P_j->Mass * kernel.wk_j * dt_j_phys;
                          double scatter_prob_j_on_i = P_i->getMass() * kernel.wk_i * dt_i_phys;
                          scatter_prob =
                              (scatter_prob_i_on_j + scatter_prob_j_on_i) * kernel.dvinv3 * All.SigmaOverM * scatter_prob_to_phys / 2;
                          //                    scatter_prob = (scatter_prob_i_on_j + scatter_prob_j_on_i) * kernel.dvinv3 *
                          //                    All.SigmaOverM / 2;
                        }
                      //                D->mpi_printf("Scatter prob = %f  vrel = %f  r = %f  hi = %f  hj = %f  wki = %f  wkj = %f\n",
                      //                scatter_prob, kernel.dv, kernel.r, kernel.h_i, kernel.h_j, kernel.wk_i, kernel.wk_j);

                      double rand_u = get_random_number();
                      if(rand_u <= scatter_prob)
                        {
                          int particle_i_index = get_index_from_ID(P_i->ID.get(), 0);
                          int particle_j_index = get_index_from_ID(P_j->ID, 1);

                          if(particle_i_index < 0 || particle_j_index < 0)
                            {
                              //                        D->mpi_printf("pdat.target = %d  index of P_i = %d  ID of P_i = %d  vrel =
                              //                        %f\n", pdat.target, particle_i_index, P_i->ID.get(), kernel.dv);
                              continue;
                            }

                          scatter_list[nscatterevents].scatter_partner_one = particle_i_index;
                          scatter_list[nscatterevents].scatter_partner_two = particle_j_index;
                          //                    scatter_list[nscatterevents].scatter_partner_one = P_i->ID.get();
                          //                    scatter_list[nscatterevents].scatter_partner_two = P_j->ID;
                          scatter_list[nscatterevents].scattering_probability = scatter_prob;
                          nscatterevents++;
                          //                    D->mpi_printf("pdat.target = %d  unsigned int of P_i->ID.get = %d  unsigned int of
                          //                    P_j->ID = %d\n", pdat.target, P_i->ID.get(), P_j->ID); D->mpi_printf("pdat.target = %d
                          //                    index of P_i = %d  index of P_j = %d\n", pdat.target, particle_i_index,
                          //                    particle_j_index);

                          /* Calculate new velocities after scattering. */
                          /*  double mass_sum = P_i->getMass() + P_j->Mass;
                            double center_of_mass_velx = ( P_i->getMass() * SphP_i->VelPred[0] + P_j->Mass * SphP_j->VelPred[0] ) /
                            mass_sum; double center_of_mass_vely = ( P_i->getMass() * SphP_i->VelPred[1] + P_j->Mass *
                            SphP_j->VelPred[1] ) / mass_sum; double center_of_mass_velz = ( P_i->getMass() * SphP_i->VelPred[2] +
                            P_j->Mass * SphP_j->VelPred[2] ) / mass_sum;

                            double after_scatter_velx_i = center_of_mass_velx + P_j->Mass / mass_sum * kernel.dv * rand_x;
                            double after_scatter_vely_i = center_of_mass_vely + P_j->Mass / mass_sum * kernel.dv * rand_y;
                            double after_scatter_velz_i = center_of_mass_velz + P_j->Mass / mass_sum * kernel.dv * rand_z;

                            Delta_velx_i = after_scatter_velx_i - Delta_velx_i;
                            Delta_vely_i = after_scatter_vely_i - Delta_vely_i;
                            Delta_velz_i = after_scatter_velz_i - Delta_velz_i;

                            ScatterOccurence = 1;
                            numberofscatterings++; */

                          //                  SphP_i->HydroAccel[0] += Delta_velx_i / dt_i;
                          //                  SphP_i->HydroAccel[1] += Delta_vely_i / dt_i;
                          //                  SphP_i->HydroAccel[2] += Delta_velz_i / dt_i;

                          /* Calculate them also for j even though we can't currently save them.
                          double after_scatter_velx_j = center_of_mass_velx - P_i->getMass() / mass_sum * kernel.dv * rand_x
                          double after_scatter_vely_j = center_of_mass_vely - P_i->getMass() / mass_sum * kernel.dv * rand_y
                          double after_scatter_velz_j = center_of_mass_velz - P_i->getMass() / mass_sum * kernel.dv * rand_z

                          double Delta_velx_j = after_scatter_velx_j - SphP_j->VelPred[0]
                          double Delta_vely_j = after_scatter_vely_j - SphP_j->VelPred[1]
                          double Delta_velz_j = after_scatter_velz_j - SphP_j->VelPred[2]
                          */
                        }
                    }
                }
            }
        }
    }
/*  if(ScatterOccurence == 1)
  {
  SphP_i->HydroAccel[0] += Delta_velx_i / dt_i;
  SphP_i->HydroAccel[1] += Delta_vely_i / dt_i;
  SphP_i->HydroAccel[2] += Delta_velz_i / dt_i;
  } */
#endif /* LEAN */
}
#endif

inline int sph::get_index_from_ID(MyIDType ID, int h)
{
  int particle_index = 0; /*This might return zero if a particle index cannot be found, e.g. if Tp->NumPart is not the goal we want to
                             reach, so be careful. */
  for(int z = 0; z < Tp->TimeBinsHydro.NActiveParticles; z++)
    //  for(int z = 0; z < Tp->TotNumPart; z++)
    {
      int test_index = Tp->TimeBinsHydro.ActiveParticleList[z];
      if(Tp->P[test_index].ID.get() == ID)
        //      return z;
        particle_index = test_index;
      /*    else
            {
            D->mpi_printf("NumPart = %d  TotNumPart = %d  TotNumGas = %d  MaxPart = %d  NActP = %d GNActP = %d\n", Tp->NumPart,
         Tp->TotNumPart, Tp->TotNumGas, Tp->MaxPart, Tp->TimeBinsHydro.NActiveParticles, Tp->TimeBinsHydro.GlobalNActiveParticles);
            Tp->print_particle_info_from_ID(ID);
            Terminate("Couldn't find index of particle %d\n", ID);
            } */
    }
  if(particle_index == 0 && Tp->P[0].ID.get() != ID)
    {
      /*    if(h == 0)
            D->mpi_printf("Index of particle %d couldn't be found, returning zero (particle i)\n", ID);
          else
            D->mpi_printf("Index of particle %d couldn't be found, returning zero (particle j)\n", ID); */
      particle_index = -1;
    }
  return particle_index;
}

/* this routine clears the fields in the SphP particle structure that are additively computed by the SPH density loop
 * by summing over neighbours
 */
inline void sph::clear_hydro_result(sph_particle_data *SphP)
{
  for(int k = 0; k < 3; k++)
    SphP->HydroAccel[k] = 0;

  SphP->DtEntropy    = 0;
  SphP->MaxSignalVel = 0;
}

void sph::scatter_list_evaluate(scatter_event *scatter_list, int nscatterevents)
{
  mycxxsort(scatter_list, scatter_list + nscatterevents, by_scatter_prob);
  D->mpi_printf("max_scatter prob = %f\n", scatter_list[0].scattering_probability);
  int nvelpredinits = 0;
  for(int l = 0; l < nscatterevents; l++)
    {
      //      if(scatter_list[l].scattering_probability < INFINITY)
      //        Terminate("particle i = %d  particle j = %d  scatter prob = %f\n", scatter_list[l].scatter_partner_one,
      //        scatter_list[l].scatter_partner_two, scatter_list[l].scattering_probability);
      //      D->mpi_printf("particle i = %d  particle j = %d  scatter prob = %f\n", scatter_list[l].scatter_partner_one,
      //      scatter_list[l].scatter_partner_two, scatter_list[l].scattering_probability);
      particle_data *P_i        = &Tp->P[scatter_list[l].scatter_partner_one];
      sph_particle_data *SphP_i = &Tp->SphP[scatter_list[l].scatter_partner_one];

      particle_data *P_j        = &Tp->P[scatter_list[l].scatter_partner_two];
      sph_particle_data *SphP_j = &Tp->SphP[scatter_list[l].scatter_partner_two];

      /*      if(P_i->ID.get() != scatter_list[l].scatter_partner_one || P_j->ID.get() != scatter_list[l].scatter_partner_two)
              D->mpi_printf("P_i ID = %d  Scatter partner one ID = %d  P_j ID = %d  Scatter_partner_two = %d\n", P_i->ID.get(),
         scatter_list[l].scatter_partner_one, P_j->ID.get(), scatter_list[l].scatter_partner_two); */

      double vrel_x = SphP_i->VelPred[0] - SphP_j->VelPred[0];
      double vrel_y = SphP_i->VelPred[1] - SphP_j->VelPred[1];
      double vrel_z = SphP_i->VelPred[2] - SphP_j->VelPred[2];
      double vrel2  = vrel_x * vrel_x + vrel_y * vrel_y + vrel_z * vrel_z;
      double vrel   = sqrt(vrel2);

      double mass_sum            = P_i->getMass() + P_j->getMass();
      double center_of_mass_velx = (P_i->getMass() * SphP_i->VelPred[0] + P_j->getMass() * SphP_j->VelPred[0]) / mass_sum;
      double center_of_mass_vely = (P_i->getMass() * SphP_i->VelPred[1] + P_j->getMass() * SphP_j->VelPred[1]) / mass_sum;
      double center_of_mass_velz = (P_i->getMass() * SphP_i->VelPred[2] + P_j->getMass() * SphP_j->VelPred[2]) / mass_sum;

      /* Pick a random point on a sphere. */
      double rand_v     = get_random_number();
      double rand_w     = get_random_number();
      double rand_theta = 2 * M_PI * rand_v;
      double rand_phi   = acos(2 * rand_w - 1);
      double rand_x     = cos(rand_phi) * sin(rand_theta);
      double rand_y     = sin(rand_phi) * sin(rand_theta);
      double rand_z     = cos(rand_theta);

      /* Calculate new velocities after scattering. */
      double after_scatter_velx_i = center_of_mass_velx + P_j->getMass() / mass_sum * vrel * rand_x;
      double after_scatter_vely_i = center_of_mass_vely + P_j->getMass() / mass_sum * vrel * rand_y;
      double after_scatter_velz_i = center_of_mass_velz + P_j->getMass() / mass_sum * vrel * rand_z;

      double after_scatter_velx_j = center_of_mass_velx - P_i->getMass() / mass_sum * vrel * rand_x;
      double after_scatter_vely_j = center_of_mass_vely - P_i->getMass() / mass_sum * vrel * rand_y;
      double after_scatter_velz_j = center_of_mass_velz - P_i->getMass() / mass_sum * vrel * rand_z;

      /* In preparation of the velocity update and lacking a better method. */
      if(SphP_i->scatter_delta_vel[0] == 0 && SphP_i->scatter_delta_vel[1] == 0 && SphP_i->scatter_delta_vel[2] == 0)
        {
          for(int b = 0; b < 3; b++)
            SphP_i->scatter_delta_vel[b] = SphP_i->VelPred[b];
          nvelpredinits++;
        }
      if(SphP_j->scatter_delta_vel[0] == 0 && SphP_j->scatter_delta_vel[1] == 0 && SphP_j->scatter_delta_vel[2] == 0)
        {
          for(int q = 0; q < 3; q++)
            SphP_j->scatter_delta_vel[q] = SphP_j->VelPred[q];
          nvelpredinits++;
        }

      SphP_i->scatter_delta_vel[0] = after_scatter_velx_i - SphP_i->scatter_delta_vel[0];
      SphP_i->scatter_delta_vel[1] = after_scatter_vely_i - SphP_i->scatter_delta_vel[1];
      SphP_i->scatter_delta_vel[2] = after_scatter_velz_i - SphP_i->scatter_delta_vel[2];

      SphP_j->scatter_delta_vel[0] = after_scatter_velx_j - SphP_j->scatter_delta_vel[0];
      SphP_j->scatter_delta_vel[1] = after_scatter_vely_j - SphP_j->scatter_delta_vel[1];
      SphP_j->scatter_delta_vel[2] = after_scatter_velz_j - SphP_j->scatter_delta_vel[2];

      SphP_i->scatter_occurrence = 1;
      SphP_j->scatter_occurrence = 1;
    }

  D->mpi_printf("nvelpredinits = %d\n", nvelpredinits);

  for(int c = 0; c < Tp->TimeBinsHydro.NActiveParticles; c++)
    {
      //      particle_data *P_i = &Tp->P[c];
      sph_particle_data *SphP_i = &Tp->SphP[c];

      /* Apply changes to HydroAccel as accelerations. */
      if(SphP_i->scatter_occurrence == 1)
        {
          //          int timebin_i = P_i->TimeBinHydro;
          //          double dt_i = 2 * (timebin_i ? (((integertime)1) << timebin_i) : 0) * All.Timebase_interval;
          for(int o = 0; o < 3; o++)
            //          SphP_i->HydroAccel[o] += SphP_i->scatter_delta_vel[o]/ dt_i;
            SphP_i->HydroAccel[o] += SphP_i->scatter_delta_vel[o];
        }
    }
}

/*
      particle_data *P_j = &Tp->P[c];
      sph_particle_data *SphP_j = &Tp->SphP[c];

      int timebin_j = P_j->TimeBinHydro;
      double dt_j = 2 * (timebin_j ? (((integertime)1) << timebin_j) : 0) * All.Timebase_interval;

      SphP_j->HydroAccel[o] += SphP_j->scatter_delta_vel[o]/ dt_j;
*/
