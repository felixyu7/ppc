#include <map>
#include <deque>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sys/time.h>

#ifndef __CUDACC__
#define XCPU
#endif

#ifdef XCPU
#include <cmath>
#include <cstring>
#endif

#ifdef USE_I3_LOGGING
#include "icetray/I3Logging.h"
#else
#define log_info_stream(msg) \
  do { std::cerr << msg << std::endl; } while (0)
#endif

using namespace std;

namespace xppc{
#if defined(__APPLE_CC__)
  void sincosf(float x, float * s, float * c){ __sincosf(x, s, c); }
#endif

#include "ini.cxx"
#include "pro.cu"

  void initialize(float enh = 1.f){ m.set(); q.eff*=enh; }

#ifdef TABULATE
  void tab_init(){
    char *ev;
    ev=getenv("TAB_SRC_X"); d.tab_src[0]=ev?atof(ev):0;
    ev=getenv("TAB_SRC_Y"); d.tab_src[1]=ev?atof(ev):0;
    ev=getenv("TAB_SRC_Z"); d.tab_src[2]=ev?atof(ev):0;
    ev=getenv("TAB_DIR_X"); d.tab_dir[0]=ev?atof(ev):0;
    ev=getenv("TAB_DIR_Y"); d.tab_dir[1]=ev?atof(ev):0;
    ev=getenv("TAB_DIR_Z"); d.tab_dir[2]=ev?atof(ev):-1;
    float dn=sqrtf(d.tab_dir[0]*d.tab_dir[0]+d.tab_dir[1]*d.tab_dir[1]+d.tab_dir[2]*d.tab_dir[2]);
    if(dn>0){ d.tab_dir[0]/=dn; d.tab_dir[1]/=dn; d.tab_dir[2]/=dn; }
    // perpDir (CLSim convention)
    { float dx=d.tab_dir[0], dy=d.tab_dir[1], dz=d.tab_dir[2];
      float perpz=sqrtf(dx*dx+dy*dy);
      if(perpz>1e-8f){ d.tab_perp[0]=-dx*dz/perpz; d.tab_perp[1]=-dy*dz/perpz; d.tab_perp[2]=perpz; }
      else { d.tab_perp[0]=1; d.tab_perp[1]=0; d.tab_perp[2]=0; }
      // perpDir2 = tab_dir × tab_perp (for signed phi via atan2)
      d.tab_perp2[0]=d.tab_dir[1]*d.tab_perp[2]-d.tab_dir[2]*d.tab_perp[1];
      d.tab_perp2[1]=d.tab_dir[2]*d.tab_perp[0]-d.tab_dir[0]*d.tab_perp[2];
      d.tab_perp2[2]=d.tab_dir[0]*d.tab_perp[1]-d.tab_dir[1]*d.tab_perp[0];
    }
    // Spatial-bin counts and ranges
    ev=getenv("TAB_N_RHO");   d.tab_n_rho=ev?atoi(ev):64;
    ev=getenv("TAB_N_PHI");   d.tab_n_phi=ev?atoi(ev):16;
    ev=getenv("TAB_N_L");     d.tab_n_l=ev?atoi(ev):64;
    ev=getenv("TAB_RHO_MAX"); d.tab_rho_max=ev?atof(ev):500;
    ev=getenv("TAB_RHO0");    d.tab_rho0=ev?atof(ev):5;
    ev=getenv("TAB_PHI_MAX"); d.tab_phi_max=ev?atof(ev):2*FPI;
    ev=getenv("TAB_L_MIN");   d.tab_l_min=ev?atof(ev):-500;
    ev=getenv("TAB_L_MAX");   d.tab_l_max=ev?atof(ev):500;
    ev=getenv("TAB_L0");      d.tab_l0=ev?atof(ev):5;
    ev=getenv("TAB_T_MAX");   d.tab_t_max=ev?atof(ev):7000;
    // Feature-uniform spatial warps: rho uniform in log1p(rho/rho0), l uniform
    // in asinh(l/l0) — the same compressions the kernel's feature map uses.
    d.tab_inv_log1p_rho=1.0f/log1pf(d.tab_rho_max/d.tab_rho0);
    {
      float ylo=asinhf(d.tab_l_min/d.tab_l0);
      d.tab_asinh_l_lo=ylo;
      d.tab_inv_asinh_l=1.0f/(asinhf(d.tab_l_max/d.tab_l0)-ylo);
    }
    // Strict-physics geometric-time coefficients from the ice model
    // (mid-wavelength). Decomposes t_geo = l*ocv + rho*((ocm-coschr*ocv)/sinchr).
    {
      float coschr=z.w[WNUM/2].coschr, sinchr=z.w[WNUM/2].sinchr;
      float ocm=z.w[WNUM/2].ocm;
      d.tab_t_geo_l=d.ocv;
      d.tab_t_geo_r=(ocm-coschr*d.ocv)/sinchr;
    }
    ev=getenv("TAB_K"); d.tab_n_t_bins=ev?atoi(ev):512;
    ev=getenv("TAB_T_SCALE"); d.tab_t_scale=ev?(float)atof(ev):0.5f;
    if(d.tab_t_scale<=0.0f){
      cerr<<"Error: TAB_T_SCALE must be > 0, got "<<d.tab_t_scale<<endl; exit(2);
    }
    ev=getenv("TAB_T_PAD"); d.tab_t_pad=ev?(float)atof(ev):0.0f;
    if(d.tab_t_pad<0.0f){
      cerr<<"Error: TAB_T_PAD must be >= 0, got "<<d.tab_t_pad<<endl; exit(2);
    }
    ev=getenv("TAB_STEP"); d.tab_step=ev?(float)atof(ev):1.0f;
    if(d.tab_step<=0.0f){
      cerr<<"Error: TAB_STEP must be > 0, got "<<d.tab_step<<endl; exit(2);
    }
    ev=getenv("TAB_RAND_PHASE"); d.tab_rand_phase=ev?atoi(ev):0;
    ev=getenv("TAB_N_ETA"); d.tab_n_eta=ev?atoi(ev):1;
    if(d.tab_n_t_bins<=0 || d.tab_n_eta<1 || (long long)d.tab_n_t_bins*d.tab_n_eta>65536){
      cerr<<"Error: need TAB_K>0, TAB_N_ETA>=1, TAB_K*TAB_N_ETA<=65536, got K="
          <<d.tab_n_t_bins<<" n_eta="<<d.tab_n_eta<<endl; exit(2);
    }
    d.tab_n_row=d.tab_n_t_bins*d.tab_n_eta;
    { // standard-DOM wavelength response (see pro.cu): standard OMs are type (-1,0)
      irde & mt=irdes[make_pair(-1,0)];
      if(mt.rat.size()!=WNUM || mt.rmax<=0){
        cerr<<"Error: unsupported standard-DOM RDE ("<<mt.rat.size()<<" slices)"<<endl; exit(2);
      }
      float avg=0;
      for(int i=0; i<WNUM; i++) avg+=d.tab_rde[i]=mt.rat[i]/mt.rmax;
      cerr<<"Tabulating standard-DOM hits: mean RDE/envelope="<<avg/WNUM;
      for(map<pair<int,int>, irde>::iterator i=irdes.begin(); i!=irdes.end(); ++i)
        if(i->first!=make_pair(-1,0) && i->second.rat.size()==WNUM){
          float r=0; for(int k=0; k<WNUM; k++) r+=i->second.rat[k];
          cerr<<"; OM class ("<<i->first.first<<","<<i->first.second<<") / standard = "<<r/avg;
        }
      cerr<<endl;
    }
    d.tab_inv_t_scale=1.0f/d.tab_t_scale;
    d.tab_y0=asinhf(-d.tab_t_pad*d.tab_inv_t_scale);
    d.tab_inv_yspan=1.0f/(asinhf((d.tab_t_max-d.tab_t_pad)*d.tab_inv_t_scale)
                          -d.tab_y0);
    {
      unsigned long long n_spatial=(unsigned long long)d.tab_n_rho*d.tab_n_phi*d.tab_n_l;
      unsigned long long mem=n_spatial*d.tab_n_row*sizeof(unsigned int);
      cerr<<"Tabulation: "<<d.tab_n_rho<<"x"<<d.tab_n_phi<<"x"<<d.tab_n_l
          <<" = "<<n_spatial<<" spatial bins, K="<<d.tab_n_t_bins
          <<" asinh time bins, dense buffer "<<mem<<" bytes"<<endl
          <<"  t_scale="<<d.tab_t_scale<<" t_max="<<d.tab_t_max
          <<" inv_yspan="<<d.tab_inv_yspan
          <<" rho_max="<<d.tab_rho_max
          <<" rho0="<<d.tab_rho0
          <<" l=["<<d.tab_l_min<<","<<d.tab_l_max<<"] l0="<<d.tab_l0
          <<" t_pad="<<d.tab_t_pad
          <<" step="<<d.tab_step<<" rand_phase="<<d.tab_rand_phase
          <<" n_eta="<<d.tab_n_eta<<" t_geo_l="<<d.tab_t_geo_l<<" t_geo_r="<<d.tab_t_geo_r<<endl;
    }
  }
#endif

  unsigned int pmax, pmxo, pn, pk, hquo;

  void setq(){
    char * HQUO=getenv("HQUO");
    hquo=HQUO==NULL?1:atoi(HQUO);
    cerr<<"HQUO(photons/max number of hits)="<<hquo<<endl;
  }

#ifdef XCPU
  dats *e;  // pointer to a copy of "d" on device
  int nblk, nthr, ntot;

  void ini(){
    setq();
    rs_ini();
    pn=0; pk=0;

    ntot=nblk*nthr;
    pmax=ntot*NPHO;
    pmxo=pmax/OVER;
    pmax=pmxo*OVER;
    d.hnum=pmax/hquo;

    d.gdev=0; d.gnum=1;
    d.gini=0; d.gspc=pmax; d.gtot=pmax; d.gdiv=1;

    {
      d.hits = q.hits = new hit[d.hnum];
      d.pz = q.pz = new photon[pmxo];
      d.bf = new pbuf[pmax];
    }

    {
      d.z=&z; d.oms=q.oms; e=&d;
    }


    {
      unsigned int size=d.rsize, need=seed+1;
      if(size<need) cerr<<"Error: not enough multipliers: asked for "<<seed<<"-th out of "<<size<<"!"<<endl;
    }
  }

  void fin(){
    delete d.pz;
    delete d.hits;
    delete d.bf;
  }

#ifdef XLIB
  size_t getMaxBunchSize(){ return pmxo; }
  size_t getWorkgroupSize(){ return nthr; }
  float getTotalDeviceTime(){ return 0.f; }
#endif

#else
  bool xgpu=false;

  void checkError(cudaError result){
    if(result!=cudaSuccess){
      cerr<<"CUDA Error: "<<cudaGetErrorString(result)<<endl;
      exit(2);
    }
  }

  struct gpu{
    dats d;
    dats *e;  // pointer to a copy of "d" on device

    int device;
    int nblk, nthr, ntot;
    unsigned int npho, pmax, pmxo;

    float dt, deviceTime, threadMin, threadMax;
    int clockRate;
    cudaDeviceProp prop;
    cudaStream_t stream;
    cudaEvent_t evt1, evt2;

    unsigned int old, num;

    gpu(int device) : deviceTime(0), threadMin(0), threadMax(0), clockRate(0), old(0), npho(NPHO){
      this->device=device;

      {
	ostringstream o; o<<"NPHO_"<<device;
	char * nph=getenv(o.str().c_str());
	if(nph==NULL) nph=getenv("NPHO");
	if(nph!=NULL) if(*nph!=0){
	    npho=atoi(nph);
	    cerr<<"Setting NPHO="<<npho<<endl;
	    if(npho<=0){
	      cerr<<"Not using device # "<<device<<"!"<<endl;
	      return;
	    }
	  }
      }

      checkError(cudaSetDevice(device));
      checkError(cudaGetDeviceProperties(&prop, device));
      checkError(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, device));

#if CUDART_VERSION >= 3000
      checkError(cudaFuncSetCacheConfig(propagate, cudaFuncCachePreferL1));
#endif

      cudaFuncAttributes attr;
      checkError(cudaFuncGetAttributes (&attr, propagate));

      nblk=prop.multiProcessorCount, nthr=attr.maxThreadsPerBlock;
      cerr<<"Running on "<<nblk<<" MPs x "<<nthr<<" threads"<<endl;

      fprintf(stderr, "Kernel uses: l=%lu r=%d s=%lu c=%lu\n", (unsigned long)attr.localSizeBytes,
	      attr.numRegs, (unsigned long)attr.sharedSizeBytes, (unsigned long)attr.constSizeBytes);
    }

    void ini(){
      rs_ini();
      d=xppc::d;

      d.gdev=device;

      {
	d.blockIdx=-1, d.gridDim=nblk;
	ostringstream o; o<<"BADMP_"<<device;
	char * bmp=getenv(o.str().c_str());
	if(bmp==NULL) bmp=getenv("BADMP");
	if(bmp!=NULL) if(*bmp!=0){
	    d.blockIdx=atoi(bmp), d.gridDim--;
	    cerr<<"Not using MP #"<<d.blockIdx<<endl;
	  }
      }

      ntot=nblk*nthr;

      {
	unsigned long xmem=prop.totalGlobalMem;

	while(npho>0){
	  pmax=ntot*npho;
	  pmxo=pmax/OVER;
	  pmax=pmxo*OVER;
	  d.hnum=pmax/hquo;

	  unsigned long mtot=sizeof(datz)+sizeof(dats)+d.gsize*sizeof(DOM);
	  mtot+=d.hnum*sizeof(hit);
	  mtot+=pmxo*sizeof(photon);
	  mtot+=pmax*sizeof(pbuf);

	  if(mtot>xmem) npho/=2; else break;
	}
      }

      {
	checkError(cudaStreamCreate(&stream));
	checkError(cudaEventCreateWithFlags(&evt1, cudaEventBlockingSync));
	checkError(cudaEventCreateWithFlags(&evt2, cudaEventBlockingSync));
      }

      {
	unsigned int size=d.rsize;
	if(size<ntot) cerr<<"Error: not enough multipliers: only have "<<size<<" (need "<<ntot<<")!"<<endl;
	else d.rsize=ntot;
      }

      unsigned long tot=0, cnt=0;

      {
	unsigned long size=sizeof(datz); tot+=size;
	checkError(cudaMalloc((void**) &d.z, size));
	checkError(cudaMemcpy(d.z, &z, size, cudaMemcpyHostToDevice));
      }

      {
	unsigned long size=d.hnum*sizeof(hit); tot+=size;
	checkError(cudaMalloc((void**) &d.hits, size));
      }

      {
	unsigned long size=pmxo*sizeof(photon); tot+=size;
	checkError(cudaMalloc((void**) &d.pz, size));
      }

      {
	unsigned long size=pmax*sizeof(pbuf); tot+=size;
	checkError(cudaMalloc((void**) &d.bf, size));
      }

      {
	unsigned long size=d.gsize*sizeof(DOM); cnt+=size;
	checkError(cudaMalloc((void**) &d.oms, size));
	checkError(cudaMemcpy(d.oms, &q.oms, size, cudaMemcpyHostToDevice));
	// checkError(cudaMemcpyToSymbol(oms, q.oms, size));
      }

#ifdef TABULATE
      {
	unsigned long long n_cells=(unsigned long long)d.tab_n_rho*d.tab_n_phi
	                          *d.tab_n_l*d.tab_n_row;
	unsigned long size=(unsigned long)(n_cells*sizeof(unsigned int)); tot+=size;
	checkError(cudaMalloc((void**) &d.tab_hist_st, size));
	checkError(cudaMemset(d.tab_hist_st, 0, size));
	cerr<<"Tabulation buffer: "<<n_cells<<" uint32 cells ("<<size<<" bytes)"<<endl;
      }
#endif

      {
	unsigned long size=sizeof(dats); tot+=size;
	checkError(cudaMalloc((void**) &e, size));
	checkError(cudaMemcpy(e, &d, size, cudaMemcpyHostToDevice));
      }

      cerr<<"Total GPU memory usage: "<<tot<<"  const: "<<cnt<<"  (npho="<<npho<<")"<<endl;
    }

    void fin(){
      checkError(cudaFree(d.z));
      checkError(cudaFree(d.hits));
      checkError(cudaFree(d.pz));
      checkError(cudaFree(d.bf));
      checkError(cudaFree(d.oms));
#ifdef TABULATE
      checkError(cudaFree(d.tab_hist_st));
#endif
      checkError(cudaFree(e));
      checkError(cudaEventDestroy(evt1));
      checkError(cudaEventDestroy(evt2));
      checkError(cudaStreamDestroy(stream));
    }

    void set(){
      if(xgpu) checkError(cudaSetDevice(device));
    }

    void kernel_i(){
      {
	checkError(cudaStreamSynchronize(stream));
	checkError(cudaMemcpy(&d, e, 7*sizeof(int), cudaMemcpyDeviceToHost));

	checkError(cudaEventElapsedTime(&dt, evt1, evt2)); deviceTime+=dt;

	if(d.ab>0){
	  cerr<<"Error: TOT was a nan or an inf "<<d.ab<<" times! Bad GPU "<<device<<" MP";
	  for(int i=0; i<min(d.ab, 4); i++) cerr<<" #"<<d.bmp[i]; cerr<<endl;
	}
	if(d.mp!=d.gridDim){ cerr<<"Error: did not encounter MP #"<<d.blockIdx<<endl; exit(4); }
	if(threadMax!=-1){
	  if((unsigned long long)(dt*clockRate)<0x100000000ULL){
	    threadMin+=d.tn/(float)clockRate;
	    threadMax+=d.tx/(float)clockRate;
	  }
	  else threadMin=-1, threadMax=-1;
	}

	if(d.hidx>d.hnum){ cerr<<"Error: data buffer overflow occurred: "<<d.hidx<<">"<<d.hnum<<"!"<<endl; d.hidx=d.hnum; }
      }

      {
	unsigned int size=d.hidx*sizeof(hit);
	checkError(cudaMemcpyAsync(&q.hits[xppc::d.hidx], d.hits, size, cudaMemcpyDeviceToHost, stream));
	xppc::d.hidx+=d.hidx;
      }
    }

    void kernel_c(){
      if(old>0) checkError(cudaStreamSynchronize(stream));
      if(num>0){
	unsigned int size=pk*sizeof(photon);
	checkError(cudaMemcpyAsync(d.pz, q.pz, size, cudaMemcpyHostToDevice, stream));
      }
    }

    void kernel_f(){
      checkError(cudaStreamSynchronize(stream));
      if(num>0){
	checkError(cudaEventRecord(evt1, stream));

	propagate<<< 1, 1, 0, stream >>>(e, 0);
	checkError(cudaGetLastError());

	propagate<<< nblk, nthr, 0, stream >>>(e, num);
	checkError(cudaGetLastError());

	checkError(cudaEventRecord(evt2, stream));
      }
    }

    void stop(){
      fprintf(stderr, "Device time: %2.1f (in-kernel: %2.1f...%2.1f) [ms]\n", deviceTime, threadMin, threadMax);
      checkError(cudaDeviceReset());
    }

    void ini_f(unsigned int & c, unsigned int t, unsigned int div = 1){
      unsigned int aux=pmax/div;
      d.gini=c, d.gspc=aux, d.gtot=t/div; d.gdiv=div; c+=aux;

      checkError(cudaMemcpy(e, &d, 14*sizeof(int), cudaMemcpyHostToDevice));

      cerr<<" "<<d.gspc;
    }
  };

  int gcd(int a, int b){
    if(a==0) return b;
    return gcd(b%a, a);
  }

  vector<gpu> gpus;

  void ini(){
    setq();

    d.hnum=0; d.gnum=gpus.size();
    pmax=0, pmxo=0, pn=0; pk=0;

    unsigned int div=1;
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++){
      i->set();
      i->ini(); if(xgpu) sv++;
      d.hnum+=i->d.hnum;
      pmax+=i->pmax;
      if(pmxo==0 || pmxo>i->pmxo) pmxo=i->pmxo;

      if(i->device==0) div=i->pmax;
      else div=gcd(i->pmax, div);
    }

    unsigned int gsum=0; cerr<<"Relative GPU loadings:";
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->ini_f(gsum, pmax, div);
    d.gini=0; d.gspc=gsum; d.gtot=gsum; d.gdiv=div; cerr<<endl;

    {
      unsigned long size=d.hnum*sizeof(hit);
      checkError(cudaMallocHost((void**) &q.hits, size));
    }

    {
      unsigned long size=pmxo*sizeof(photon);
      checkError(cudaMallocHost((void**) &q.pz, size));
    }
  }

#ifdef XLIB
  int lcm(int a, int b){
    return a*b/gcd(a, b);
  }

  size_t getMaxBunchSize() { return pmxo; }

  size_t getWorkgroupSize()
  {
    size_t workgroupSize = 0;
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) {
      if (workgroupSize == 0) {
	workgroupSize = i->nthr;
      } else {
	workgroupSize = lcm(workgroupSize, i->nthr);
      }
    }

    if(getMaxBunchSize()%workgroupSize != 0){
      cerr<<"MaxBunchSize is no good!"<<endl;
      exit(2);
    }
    return workgroupSize;
  }

  float getTotalDeviceTime()
  {
    float total = 0;
    for(vector<gpu>::const_iterator i=gpus.begin(); i!=gpus.end(); i++) {
      total += i->deviceTime;
    }
    return total;
  }
#endif

  void fin(){
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->fin();
    checkError(cudaFreeHost(q.hits));
    checkError(cudaFreeHost(q.pz));
  }

  void listDevices(){
    int deviceCount, driver, runtime;
    cudaGetDeviceCount(&deviceCount);
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    fprintf(stderr, "Found %d devices, driver %d, runtime %d\n", deviceCount, driver, runtime);
    for(int device=0; device<deviceCount; ++device){
      cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
      int clockRate=0, timeout=0, overlap=0, computeMode=0;
      cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, device);
      cudaDeviceGetAttribute(&timeout, cudaDevAttrKernelExecTimeout, device);
      cudaDeviceGetAttribute(&overlap, cudaDevAttrGpuOverlap, device);
      cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, device);
      fprintf(stderr, "%d(%d.%d): %s %g GHz G(%lu) S(%lu) C(%lu) R(%d) W(%d)\n"
	      "\tl%d o%d c%d h%d i%d m%d a%lu M(%lu) T(%d: %d,%d,%d) G(%d,%d,%d)\n",
	      device, prop.major, prop.minor, prop.name, clockRate/1.e6,
	      (unsigned long)prop.totalGlobalMem, (unsigned long)prop.sharedMemPerBlock,
	      (unsigned long)prop.totalConstMem, prop.regsPerBlock, prop.warpSize,
	      timeout, overlap, computeMode,
	      prop.canMapHostMemory, prop.integrated, prop.multiProcessorCount,
	      (unsigned long)prop.textureAlignment,
	      (unsigned long)prop.memPitch, prop.maxThreadsPerBlock,
	      prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
	      prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    fprintf(stderr, "\n");
  }

  static unsigned int old=0;
#endif

#ifdef TABULATE
  // Sparse CSR wire format (matches candela's encode_sparse_hist pre-zstd
  // layout, so the Python writer can pass it through directly):
  //   [uint32 magic 0xC0DEC0FE][uint32 n_occ]
  //   [uint32 × n_occ] spatial_idx (sorted ascending)
  //   [uint32 × n_occ] nnz_per_cell
  //   [uint16 × nnz_total] time_ids (ascending within a cell)
  //   [uint32 × nnz_total] time_counts
  //
  // Table mode is single-device. The dense [N_SPATIAL × B*K] histogram is
  // sparsified on-device (`tab_sparsify_count` + `tab_sparsify_write` in
  // pro.cu); the write kernel zeroes every cell it consumes, which resets the
  // histogram for the next source. Buffers persist across dumps.
  static unsigned int *tab_d_nnz, *tab_d_off, *tab_d_cnts, *tab_h_nnz, *tab_h_off, *tab_h_occ, *tab_h_cnts;
  static unsigned short *tab_d_tids, *tab_h_tids;
  static size_t tab_cap=0;

  void tab_print_histogram(){
    unsigned int N=d.tab_n_rho*d.tab_n_phi*d.tab_n_l, K=d.tab_n_row;
    unsigned int grid=(N*32+255)/256;
    unsigned int * hist=gpus[0].d.tab_hist_st;
    if(!tab_d_nnz){
      checkError(cudaMalloc((void**)&tab_d_nnz, N*sizeof(unsigned int)));
      checkError(cudaMalloc((void**)&tab_d_off, N*sizeof(unsigned int)));
      checkError(cudaHostAlloc((void**)&tab_h_nnz, N*sizeof(unsigned int), cudaHostAllocDefault));
      checkError(cudaHostAlloc((void**)&tab_h_off, N*sizeof(unsigned int), cudaHostAllocDefault));
      checkError(cudaHostAlloc((void**)&tab_h_occ, N*sizeof(unsigned int), cudaHostAllocDefault));
    }

    tab_sparsify_count<<<grid, 256>>>(hist, N, K, tab_d_nnz);
    checkError(cudaMemcpy(tab_h_nnz, tab_d_nnz, N*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int n_occ=0; size_t tot=0;
    for(unsigned int s=0; s<N; s++){
      unsigned int n=tab_h_nnz[s];
      tab_h_off[s]=tot; tot+=n;
      if(n) tab_h_occ[n_occ]=s, tab_h_nnz[n_occ++]=n; // compact in place (n_occ <= s)
    }
    checkError(cudaMemcpy(tab_d_off, tab_h_off, N*sizeof(unsigned int), cudaMemcpyHostToDevice));

    if(tot>tab_cap){
      if(tab_cap){
        checkError(cudaFree(tab_d_tids)); checkError(cudaFree(tab_d_cnts));
        checkError(cudaFreeHost(tab_h_tids)); checkError(cudaFreeHost(tab_h_cnts));
      }
      tab_cap=max(tot, tab_cap*3/2);
      checkError(cudaMalloc((void**)&tab_d_tids, tab_cap*sizeof(unsigned short)));
      checkError(cudaMalloc((void**)&tab_d_cnts, tab_cap*sizeof(unsigned int)));
      checkError(cudaHostAlloc((void**)&tab_h_tids, tab_cap*sizeof(unsigned short), cudaHostAllocDefault));
      checkError(cudaHostAlloc((void**)&tab_h_cnts, tab_cap*sizeof(unsigned int), cudaHostAllocDefault));
    }
    if(tot>0){
      tab_sparsify_write<<<grid, 256>>>(hist, N, K, tab_d_nnz, tab_d_off, tab_d_tids, tab_d_cnts);
      checkError(cudaMemcpy(tab_h_tids, tab_d_tids, tot*sizeof(unsigned short), cudaMemcpyDeviceToHost));
      checkError(cudaMemcpy(tab_h_cnts, tab_d_cnts, tot*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    unsigned int head[2]={0xC0DEC0FE, n_occ};
    fwrite(head, sizeof(unsigned int), 2, stdout);
    fwrite(tab_h_occ, sizeof(unsigned int), n_occ, stdout);
    fwrite(tab_h_nnz, sizeof(unsigned int), n_occ, stdout);
    fwrite(tab_h_tids, sizeof(unsigned short), tot, stdout);
    fwrite(tab_h_cnts, sizeof(unsigned int), tot, stdout);
    fflush(stdout);
  }
  void tab_set_source(float sx, float sy, float sz, float dx, float dy, float dz){
    d.tab_src[0]=sx; d.tab_src[1]=sy; d.tab_src[2]=sz;
    d.tab_dir[0]=dx; d.tab_dir[1]=dy; d.tab_dir[2]=dz;
    float dn=sqrtf(dx*dx+dy*dy+dz*dz);
    if(dn>0){ d.tab_dir[0]/=dn; d.tab_dir[1]/=dn; d.tab_dir[2]/=dn; }
    // Compute perpDir (CLSim convention)
    { float perpz=sqrtf(d.tab_dir[0]*d.tab_dir[0]+d.tab_dir[1]*d.tab_dir[1]);
      if(perpz>1e-8f){
        d.tab_perp[0]=-d.tab_dir[0]*d.tab_dir[2]/perpz;
        d.tab_perp[1]=-d.tab_dir[1]*d.tab_dir[2]/perpz;
        d.tab_perp[2]=perpz;
      } else { d.tab_perp[0]=1; d.tab_perp[1]=0; d.tab_perp[2]=0; }
      // perpDir2 = tab_dir × tab_perp (for signed phi via atan2)
      d.tab_perp2[0]=d.tab_dir[1]*d.tab_perp[2]-d.tab_dir[2]*d.tab_perp[1];
      d.tab_perp2[1]=d.tab_dir[2]*d.tab_perp[0]-d.tab_dir[0]*d.tab_perp[2];
      d.tab_perp2[2]=d.tab_dir[0]*d.tab_perp[1]-d.tab_dir[1]*d.tab_perp[0];
    }
    gpu & g=gpus[0];
    for(int k=0;k<3;k++){
      g.d.tab_src[k]=d.tab_src[k];
      g.d.tab_dir[k]=d.tab_dir[k];
      g.d.tab_perp[k]=d.tab_perp[k];
      g.d.tab_perp2[k]=d.tab_perp2[k];
    }
    checkError(cudaMemcpy(g.e, &g.d, sizeof(dats), cudaMemcpyHostToDevice));
  }
#endif

  void print();

  void kernel(unsigned int num){
#ifdef XCPU
    unsigned int & old = num;
#endif
    if(old>0){
      d.hidx=0;
#ifdef XCPU
      for(d.blockIdx=0, d.gridDim=nblk, blockDim.x=nthr; d.blockIdx<d.gridDim; d.blockIdx++)
	for(threadIdx.x=0; threadIdx.x<blockDim.x; threadIdx.x++) propagate(e, num);

      if(d.hidx>d.hnum){ cerr<<"Error: data buffer overflow occurred: "<<d.hidx<<">"<<d.hnum<<"!"<<endl; d.hidx=d.hnum; }
#else
      for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->kernel_i();
#endif
      log_info_stream("photons: "<<old<<"  hits: "<<d.hidx);
    }

#ifndef XCPU
    {
      unsigned int res=num/d.gspc;
      for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->num=res*i->d.gspc; res=num-res*d.gspc;
      for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++){
	unsigned int del=min(res, i->d.gspc);
	i->num+=del; res-=del;
      }
    }

    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->kernel_c();

    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->kernel_f();
#endif

    if(old>0) print();
#ifndef XCPU
    old=num;
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->old=i->num;
#endif
  }

#ifdef XCPU
  void start(){}
  void stop(){}
  void choose(int device){
    char * seed_offset=getenv("PPC_SEED_OFFSET");
    sv+=device+(seed_offset?(unsigned int)strtoul(seed_offset, NULL, 10):0U);
    seed=device;
    nblk=NBLK, nthr=NTHR;
  }
  void listDevices(){}
#else
  void start(){
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
  }

  void stop(){
    fprintf(stderr, "\n");
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->stop();
  }

  void choose(int device){
    char * seed_offset=getenv("PPC_SEED_OFFSET");
    if(seed_offset) sv+=(unsigned int)strtoul(seed_offset, NULL, 10);
    if(device<0){
      int deviceCount; cudaGetDeviceCount(&deviceCount);
      for(int device=0; device<deviceCount; ++device){
	gpus.push_back(gpu(device));
	if(gpus.back().npho<=0) gpus.pop_back();
      }
    }
    else{
      sv+=device;
      gpus.push_back(gpu(device));
      if(gpus.back().npho<=0) gpus.pop_back();
    }
    if(gpus.size()<=0){
      cerr<<"No active GPU(s) selected!"<<endl;
      exit(5);
    }
    xgpu=gpus.size()>1;
#ifdef TABULATE
    if(xgpu){ cerr<<"Error: table mode runs on a single device"<<endl; exit(5); }
#endif
  }
#endif

#include "f2k.cxx"
}

#ifndef XLIB
using namespace xppc;

float zdh;

float zshift(float4 r){
  zdh=d.dh;
  return zshift(d, r, zdh);
}

int main(int arg_c, char *arg_a[]){
  start();
  if(arg_c<=1){
    listDevices();
    fprintf(stderr, "Use: %s [device] (f2k muons)\n"
	    "     %s [str] [om] [num] [device] (flasher)\n", arg_a[0], arg_a[0]);
  }
  else if(0==strcmp(arg_a[1], "-")){
    initialize();
    ices & w = z.w[WNUM/2];
    cerr<<"For wavelength="<<q.wvs[w.wvl].w<<" [nm]  np="<<(1/w.coschr)<<"  cm="<<1/w.ocm<<" [m/ns]"<<endl;
    float4 r;
    r.w=0;
    if(arg_c==4){
      r.x=atof(arg_a[2]);
      r.y=atof(arg_a[3]);
    }
    else r.x=0, r.y=0;
    for(int i=0; i<d.size; i++){
      float z=d.hmin+d.dh*i;
      r.z=z; for(int j=0; j<10; j++) r.z=z+zshift(r); z=r.z;
      cout<<z<<" "<<w.z[i].abs<<" "<<w.z[i].sca*(1-d.g)<<" "<<d.az[i].ra*d.sum<<endl;
    }
  }
  else if(0==strcmp(arg_a[1], "=")){
    initialize();
    ices & w = z.w[WNUM/2];
    cerr<<"For wavelength="<<q.wvs[w.wvl].w<<" [nm]  np="<<(1/w.coschr)<<"  cm="<<1/w.ocm<<" [m/ns]"<<endl;
    float4 r;
    r.w=0;
    string in;
    while(getline(cin, in)){
      if(3==sscanf(in.c_str(), "%f %f %f", &r.x, &r.y, &r.z)){
	float dz=zshift(r);
	cout<<in<<" "<<dz<<" "<<zdh<<endl;
      }
    }
  }
  else if(0==strcmp(arg_a[1], "_")){
    initialize();
    float4 r;
    r.w=0;
    for(r.x=-750.f; r.x<751.f; r.x+=3.f) for(r.y=-750.f; r.y<751.f; r.y+=3.f) for(float z=-750.f; z<751.f; z+=6.f){
	  r.z=z; for(int j=0; j<10; j++) r.z=z+zshift(r);
	  cout<<z<<" "<<r.x<<" "<<r.y<<" "<<(r.z-z)<<endl;
	}
  }
  else if(arg_c<=2){
    int device=0;
    if(arg_c>1) device=atoi(arg_a[1]);
    initialize();
#ifdef TABULATE
    tab_init();
#endif
    choose(device);
    fprintf(stderr, "Processing f2k muons from stdin on device %d\n", device);
    f2k();
  }
  else{
    int str=0, dom=0, device=0, itr=0;
    unsigned long long num=1000000ULL;

    if(arg_c>1) str=atoi(arg_a[1]);
    if(arg_c>2) dom=atoi(arg_a[2]);
    if(arg_c>3){
      num=(unsigned long long) atof(arg_a[3]);
      char * sub = strchr(arg_a[3], '*');
      if(sub!=NULL) itr=(int) atof(++sub);
    }
    if(arg_c>4) device=atoi(arg_a[4]);
    initialize();
    choose(device);
    fprintf(stderr, "Running flasher simulation on device %d\n", device);
    flasher(str, dom, num, itr);
  }

  stop();
}
#endif
