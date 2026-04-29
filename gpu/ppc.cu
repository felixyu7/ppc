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
    ev=getenv("TAB_RHO_MAX");   d.tab_rho_max=ev?atof(ev):500;
    ev=getenv("TAB_RHO_POWER"); d.tab_rho_power=ev?atof(ev):3;
    ev=getenv("TAB_PHI_MAX"); d.tab_phi_max=ev?atof(ev):2*M_PI;
    ev=getenv("TAB_L_MIN");   d.tab_l_min=ev?atof(ev):-500;
    ev=getenv("TAB_L_MAX");   d.tab_l_max=ev?atof(ev):500;
    ev=getenv("TAB_T_MAX");   d.tab_t_max=ev?atof(ev):7000;
    // Strict-physics geometric-time coefficients from the ice model
    // (mid-wavelength). Decomposes t_geo = l*ocv + rho*((ocm-coschr*ocv)/sinchr).
    {
      float coschr=z.w[WNUM/2].coschr, sinchr=z.w[WNUM/2].sinchr;
      float ocm=z.w[WNUM/2].ocm;
      d.tab_t_geo_l=d.ocv;
      d.tab_t_geo_r=(ocm-coschr*d.ocv)/sinchr;
    }
    ev=getenv("TAB_K"); d.tab_n_t_bins=ev?atoi(ev):512;
    ev=getenv("TAB_TAU"); d.tab_tau=ev?(float)atof(ev):1.0f;
    if(d.tab_tau<=0.0f){
      cerr<<"Error: TAB_TAU must be > 0, got "<<d.tab_tau<<endl; exit(2);
    }
    if(d.tab_n_t_bins<=0 || d.tab_n_t_bins>65536){
      cerr<<"Error: TAB_K must be in (0, 65536], got "<<d.tab_n_t_bins<<endl; exit(2);
    }
    d.tab_inv_log1p_T=1.0f/log1pf(d.tab_t_max/d.tab_tau);
    {
      unsigned long long n_spatial=(unsigned long long)d.tab_n_rho*d.tab_n_phi*d.tab_n_l;
      unsigned long long mem=n_spatial*d.tab_n_t_bins*sizeof(unsigned int);
      cerr<<"Tabulation: "<<d.tab_n_rho<<"x"<<d.tab_n_phi<<"x"<<d.tab_n_l
          <<" = "<<n_spatial<<" spatial bins, K="<<d.tab_n_t_bins
          <<" log1p time bins, dense buffer "<<mem<<" bytes"<<endl
          <<"  τ="<<d.tab_tau<<" t_max="<<d.tab_t_max
          <<" inv_log1p_T="<<d.tab_inv_log1p_T
          <<" rho_max="<<d.tab_rho_max
          <<" l=["<<d.tab_l_min<<","<<d.tab_l_max<<"]"
          <<" t_geo_l="<<d.tab_t_geo_l<<" t_geo_r="<<d.tab_t_geo_r<<endl;
    }
  }
  void tab_reset_histogram();
  void tab_print_histogram();
  void tab_set_source(float sx, float sy, float sz, float dx, float dy, float dz);
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

#ifdef TABULATE
    {
      unsigned long long n_cells=(unsigned long long)d.tab_n_rho*d.tab_n_phi
                                 *d.tab_n_l*d.tab_n_t_bins;
      d.tab_hist_st=new unsigned int[n_cells];
      memset(d.tab_hist_st, 0, n_cells*sizeof(unsigned int));
    }
#endif

    {
      unsigned int size=d.rsize, need=seed+1;
      if(size<need) cerr<<"Error: not enough multipliers: asked for "<<seed<<"-th out of "<<size<<"!"<<endl;
    }
  }

  void fin(){
    delete d.pz;
    delete d.hits;
    delete d.bf;
#ifdef TABULATE
    delete[] d.tab_hist_st;
#endif
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
    cudaDeviceProp prop;
    cudaStream_t stream;
    cudaEvent_t evt1, evt2;

    unsigned int old, num;

    gpu(int device) : deviceTime(0), threadMin(0), threadMax(0), old(0), npho(NPHO){
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
      }

#ifdef TABULATE
      {
	unsigned long long n_cells=(unsigned long long)d.tab_n_rho*d.tab_n_phi
	                          *d.tab_n_l*d.tab_n_t_bins;
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
	  if((unsigned long long)(dt*prop.clockRate)<0x100000000ULL){
	    threadMin+=d.tn/(float)prop.clockRate;
	    threadMax+=d.tx/(float)prop.clockRate;
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
      fprintf(stderr, "%d(%d.%d): %s %g GHz G(%lu) S(%lu) C(%lu) R(%d) W(%d)\n"
	      "\tl%d o%d c%d h%d i%d m%d a%lu M(%lu) T(%d: %d,%d,%d) G(%d,%d,%d)\n",
	      device, prop.major, prop.minor, prop.name, prop.clockRate/1.e6,
	      (unsigned long)prop.totalGlobalMem, (unsigned long)prop.sharedMemPerBlock,
	      (unsigned long)prop.totalConstMem, prop.regsPerBlock, prop.warpSize,
	      prop.kernelExecTimeoutEnabled, prop.deviceOverlap, prop.computeMode,
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
  // Sparse CSR wire format (matches neural_ppc.data.encode_sparse_hist
  // pre-zstd layout, so the Python writer can pass it through directly):
  //   [uint32 magic 0xC0DEC0FE][uint32 n_occ]
  //   [uint32 × n_occ] spatial_idx (sorted ascending)
  //   [uint32 × n_occ] nnz_per_cell
  //   [uint16 × nnz_total] time_ids
  //   [uint32 × nnz_total] time_counts
  //
  // For CUDA, the dense [N_SPATIAL × K] device histogram is sparsified
  // on-device (`tab_sparsify_count` + `tab_sparsify_write` in pro.cu), so
  // only the small CSR arrays are copied D2H — no dense GPU→host transfer
  // and no dense host scan. Multi-GPU outputs are merged per row on host.
  void tab_print_histogram(){
    unsigned int magic=0xC0DEC0FE;
    unsigned int K=(unsigned int)d.tab_n_t_bins;
    unsigned int N_SPATIAL=(unsigned int)(d.tab_n_rho*d.tab_n_phi*d.tab_n_l);

    vector<unsigned int> occ;
    vector<unsigned int> nnz_per_cell;
    vector<unsigned short> time_ids;
    vector<unsigned int> time_counts;

#ifdef XCPU
    {
      const unsigned int * hist = d.tab_hist_st;
      for(unsigned int s=0; s<N_SPATIAL; s++){
        const unsigned int * row = hist + (size_t)s * K;
        unsigned int row_nnz=0;
        for(unsigned int k=0; k<K; k++) if(row[k]) row_nnz++;
        if(row_nnz==0) continue;
        occ.push_back(s);
        nnz_per_cell.push_back(row_nnz);
        for(unsigned int k=0; k<K; k++) if(row[k]){
          time_ids.push_back((unsigned short)k);
          time_counts.push_back(row[k]);
        }
      }
    }
#else
    struct PerGpu {
      vector<unsigned int> nnz_per_row;     // [N_SPATIAL]
      vector<unsigned int> row_offsets;     // [N_SPATIAL+1]
      vector<unsigned short> tids;          // [total_nnz]
      vector<unsigned int> cnts;            // [total_nnz]
    };
    vector<PerGpu> per_gpu(gpus.size());
    const unsigned int block=128;
    const unsigned int grid=(N_SPATIAL+block-1)/block;

    for(size_t gi=0; gi<gpus.size(); gi++){
      gpus[gi].set();
      PerGpu & pg = per_gpu[gi];
      pg.nnz_per_row.resize(N_SPATIAL);
      pg.row_offsets.resize(N_SPATIAL+1);

      unsigned int * d_nnz_per_row=NULL;
      checkError(cudaMalloc((void**)&d_nnz_per_row,
                            N_SPATIAL*sizeof(unsigned int)));
      tab_sparsify_count<<<grid, block>>>(
        gpus[gi].d.tab_hist_st, N_SPATIAL, K, d_nnz_per_row);
      checkError(cudaMemcpy(pg.nnz_per_row.data(), d_nnz_per_row,
                            N_SPATIAL*sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));

      unsigned int total_nnz=0;
      for(unsigned int s=0; s<N_SPATIAL; s++){
        pg.row_offsets[s] = total_nnz;
        total_nnz += pg.nnz_per_row[s];
      }
      pg.row_offsets[N_SPATIAL] = total_nnz;

      if(total_nnz>0){
        unsigned int * d_row_offsets=NULL;
        unsigned short * d_tids=NULL;
        unsigned int * d_cnts=NULL;
        checkError(cudaMalloc((void**)&d_row_offsets,
                              (N_SPATIAL+1)*sizeof(unsigned int)));
        checkError(cudaMemcpy(d_row_offsets, pg.row_offsets.data(),
                              (N_SPATIAL+1)*sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        checkError(cudaMalloc((void**)&d_tids, total_nnz*sizeof(unsigned short)));
        checkError(cudaMalloc((void**)&d_cnts, total_nnz*sizeof(unsigned int)));
        tab_sparsify_write<<<grid, block>>>(
          gpus[gi].d.tab_hist_st, N_SPATIAL, K, d_row_offsets, d_tids, d_cnts);
        pg.tids.resize(total_nnz);
        pg.cnts.resize(total_nnz);
        checkError(cudaMemcpy(pg.tids.data(), d_tids,
                              total_nnz*sizeof(unsigned short),
                              cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(pg.cnts.data(), d_cnts,
                              total_nnz*sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        checkError(cudaFree(d_tids));
        checkError(cudaFree(d_cnts));
        checkError(cudaFree(d_row_offsets));
      }
      checkError(cudaFree(d_nnz_per_row));
    }

    if(per_gpu.size()==1){
      // Single-GPU fast path: no merge needed.
      const PerGpu & pg = per_gpu[0];
      for(unsigned int s=0; s<N_SPATIAL; s++){
        unsigned int n = pg.nnz_per_row[s];
        if(n==0) continue;
        occ.push_back(s);
        nnz_per_cell.push_back(n);
      }
      time_ids = pg.tids;
      time_counts = pg.cnts;
    } else {
      // Per-row k-way merge with sums on collisions. Per-GPU rows are
      // already sorted by time_id ascending.
      vector<unsigned int> heads(per_gpu.size());
      vector<unsigned int> ends(per_gpu.size());
      for(unsigned int s=0; s<N_SPATIAL; s++){
        unsigned int row_start = (unsigned int)time_ids.size();
        for(size_t g=0; g<per_gpu.size(); g++){
          heads[g] = per_gpu[g].row_offsets[s];
          ends[g]  = per_gpu[g].row_offsets[s+1];
        }
        while(true){
          unsigned int min_tid = 0xFFFFFFFFu;
          for(size_t g=0; g<per_gpu.size(); g++){
            if(heads[g] < ends[g]){
              unsigned int t = per_gpu[g].tids[heads[g]];
              if(t < min_tid) min_tid = t;
            }
          }
          if(min_tid > 65535u) break;
          unsigned int sum = 0;
          for(size_t g=0; g<per_gpu.size(); g++){
            if(heads[g] < ends[g] &&
               per_gpu[g].tids[heads[g]] == (unsigned short)min_tid){
              sum += per_gpu[g].cnts[heads[g]];
              heads[g]++;
            }
          }
          time_ids.push_back((unsigned short)min_tid);
          time_counts.push_back(sum);
        }
        unsigned int row_nnz = (unsigned int)time_ids.size() - row_start;
        if(row_nnz>0){
          occ.push_back(s);
          nnz_per_cell.push_back(row_nnz);
        }
      }
    }
#endif

    unsigned int n_occ=(unsigned int)occ.size();
    fwrite(&magic, sizeof(unsigned int), 1, stdout);
    fwrite(&n_occ, sizeof(unsigned int), 1, stdout);
    if(n_occ>0){
      fwrite(occ.data(), sizeof(unsigned int), n_occ, stdout);
      fwrite(nnz_per_cell.data(), sizeof(unsigned int), n_occ, stdout);
      fwrite(time_ids.data(), sizeof(unsigned short), time_ids.size(), stdout);
      fwrite(time_counts.data(), sizeof(unsigned int), time_counts.size(), stdout);
    }
    fflush(stdout);
  }
  void tab_reset_histogram(){
    unsigned long long n_cells=(unsigned long long)d.tab_n_rho*d.tab_n_phi
                              *d.tab_n_l*d.tab_n_t_bins;
    unsigned long bytes=(unsigned long)(n_cells*sizeof(unsigned int));
#ifndef XCPU
    for(size_t gi=0; gi<gpus.size(); gi++){
      gpus[gi].set();
      checkError(cudaMemset(gpus[gi].d.tab_hist_st, 0, bytes));
    }
#else
    memset(d.tab_hist_st, 0, bytes);
#endif
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
#ifndef XCPU
    for(size_t gi=0; gi<gpus.size(); gi++){
      gpus[gi].set();
      for(int k=0;k<3;k++){
        gpus[gi].d.tab_src[k]=d.tab_src[k];
        gpus[gi].d.tab_dir[k]=d.tab_dir[k];
        gpus[gi].d.tab_perp[k]=d.tab_perp[k];
        gpus[gi].d.tab_perp2[k]=d.tab_perp2[k];
      }
      checkError(cudaMemcpy(gpus[gi].e, &gpus[gi].d, sizeof(dats), cudaMemcpyHostToDevice));
    }
#endif
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
    sv+=device;
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
