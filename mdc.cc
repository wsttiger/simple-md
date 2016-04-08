#include <cstdio>
#include <tuple>
#include <random>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <chrono>
#include <algorithm>

using namespace std;

const double PI = 3.1415926535897932384626433832795028841971693993751;

double L = 320;
double dt = 0.1;
int natoms = 10000;
int nsteps = 10;
double C = 0.1;
double alpha = 0.01;

const int nprint = 10;

void forces(int ncoords, double* x, double* y, double* z, double* fx, double* fy, double* fz, double* pe) {
  double pe_t1 = 0.0;
  #pragma omp parallel for reduction(+:pe_t1)
  for (int i = 0; i < ncoords; i++) {
    double fxi = 0.0; double fyi = 0.0; double fzi = 0.0;
    for (int j = 0; j < ncoords; j++) {
      auto dx = x[i]-x[j];
      auto dy = y[i]-y[j];
      auto dz = z[i]-z[j];
      if (dx > L/2) dx -= L;
      else if (dx < -L/2) dx += L;
      if (dy > L/2) dy -= L;
      else if (dy < -L/2) dy += L;
      if (dz > L/2) dz -= L;
      else if (dz < -L/2) dz += L;
      double fxij=0.0; double fyij=0.0; double fzij=0.0; double vij=0.0;
      double alpha_rsq = alpha*(dx*dx+dy*dy+dz*dz);
      if (alpha_rsq < 15.0) {
        vij = C*exp(-alpha_rsq);
        fxij = 2.0*dx*alpha*vij;
        fyij = 2.0*dy*alpha*vij;
        fzij = 2.0*dz*alpha*vij;
      }
      pe_t1 += vij;
      fxi += fxij; fyi += fyij; fzi += fzij;
    }
    fx[i] = fxi; fy[i] = fyi; fz[i] = fzi;
  } 
  *pe = pe_t1;
}

void read_positions_from_file(const string& fname, int ncoords, double* x, double* y, double* z, double* vx, double* vy, double* vz) {
  ifstream fil(fname.c_str());
  assert(fil.is_open());
  int natoms2 = -1;
  fil >> natoms2;
  assert(ncoords == natoms2);
  for (int i = 0; i < natoms2; i++) {
    fil >> x[i];
    fil >> y[i];
    fil >> z[i];
    fil >> vx[i];
    fil >> vy[i];
    fil >> vz[i];
  }
}

void write_positions_to_file(const string& fname, int ncoords, double* x, double* y, double* z, double* vx, double* vy, double* vz) {
  ofstream fil(fname.c_str());
  fil.precision(10);
  assert(fil.is_open());
  fil << ncoords << endl;
  for (int i = 0; i < ncoords; i++) {
    fil.width(20);
    fil << std::scientific;
    fil << x[i] << " ";
    fil.width(20);
    fil << y[i] << " ";
    fil.width(20);
    fil << z[i] << " ";
    fil.width(20);
    fil << vx[i] << " ";
    fil.width(20);
    fil << vy[i] << " ";
    fil.width(20);
    fil << vz[i];
    fil.width(20);
    fil << endl;
  }
}

void create_particles(int ncoords, double* x, double* y, double* z, double* vx, double* vy, double* vz) {
  // create particles and intialize their positions and velocities
  default_random_engine generator;
  uniform_real_distribution<double> udist(0.0,1.0);
  normal_distribution<double> vdist(5.0,1.5);
  for (int i = 0; i < ncoords; i++) {
    x[i]       = udist(generator)*L;    
    y[i]       = udist(generator)*L;    
    z[i]       = udist(generator)*L;    
    auto v     = vdist(generator);
    auto phi   = udist(generator)*2*PI;
    auto theta = udist(generator)*PI;
    vx[i]      = v*sin(theta)*cos(phi);
    vy[i]      = v*sin(theta)*sin(phi);
    vz[i]      = v*cos(theta);
  }
}

void iterate(int nsteps, int ncoords, double* x, double* y, double* z, double* vx, double* vy, double* vz) {
  double* fx = new double[ncoords];
  double* fy = new double[ncoords];
  double* fz = new double[ncoords];
  double pe = 0.0;
  forces(ncoords, x, y, z, fx, fy, fz, &pe);
  for (auto istep = 0; istep < nsteps; istep++) {
    for (auto i = 0; i < ncoords; i++) {
      // update velocities at t + dt/2
      vx[i] += fx[i]*dt*0.5;
      vy[i] += fy[i]*dt*0.5;
      vz[i] += fz[i]*dt*0.5;
      // update positions at t + dt/2
      x[i] += vx[i]*dt;
      y[i] += vy[i]*dt;
      z[i] += vz[i]*dt;
      // periodic boundary conditions
      if (x[i] >= L) x[i] -= L;
      else if (x[i] < 0.0) x[i] += L;
      if (y[i] >= L) y[i] -= L;
      else if (y[i] < 0.0) y[i] += L;
      if (z[i] >= L) z[i] -= L;
      else if (z[i] < 0.0) z[i] += L;
    }
    double ke = 0.0;
    forces(ncoords, x, y, z, fx, fy, fz, &pe);
    for (int i = 0; i < ncoords; i++) {
      vx[i] += fx[i]*dt*0.5;
      vy[i] += fy[i]*dt*0.5;
      vz[i] += fz[i]*dt*0.5;
      // energies
      ke += 0.5*(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
    } 
    // if (istep%nprint == 0) {
    //   printf("KE:     %15.8e   PE:     %15.8e   TE:     %15.8e\n", ke, pe, ke+pe);
    // } 
  }
  delete [] fx;
  delete [] fy;
  delete [] fz;
}

// test code by first integrating forward in time, and then reversing the direction
// of the particles.
void test() {
  L = 320;
  dt = 0.1;
  natoms = 10000;
  nsteps = 20;
  C = 0.1;
  alpha = 0.01;

  bool doprint = false;

  // create coordinates and velocities
  double* x = new double[natoms];
  double* y = new double[natoms];
  double* z = new double[natoms];
  double* xinit = new double[natoms];
  double* yinit = new double[natoms];
  double* zinit = new double[natoms];
  double* vx = new double[natoms];
  double* vy = new double[natoms];
  double* vz = new double[natoms];
  double* vxinit = new double[natoms];
  double* vyinit = new double[natoms];
  double* vzinit = new double[natoms];
  create_particles(natoms, x, y, z, vx, vy, vz);
  if (doprint) {
    printf("Initial coords. and vels.\n");
    for (int i = 0; i < natoms; i++) {
      printf("%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", 
        x[i], y[i], z[i], vx[i], vy[i], vz[i]);
    }
  }
  // copy coordinates and velocities
  for (int i = 0; i < natoms; i++) {
    xinit[i] = x[i];
    yinit[i] = y[i];
    zinit[i] = z[i];
    vxinit[i] = vx[i];
    vyinit[i] = vy[i];
    vzinit[i] = vz[i];
  }
  {
    const auto tstart = chrono::system_clock::now();
    iterate(nsteps, natoms, x, y, z, vx, vy, vz);
    const auto tstop = chrono::system_clock::now();
    const chrono::duration<double> time_elapsed = tstop - tstart;
    cout << "(iterating forward) Time elapsed: " << time_elapsed.count() << " s" << endl << endl;
  }
  if (doprint) {
    printf("\n");
    printf("Coords. and Vels. after nsteps\n");
    for (int i = 0; i < natoms; i++) {
      printf("%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", 
        x[i], y[i], z[i], vx[i], vy[i], vz[i]);
    }
  }
  // reverse velocities
  for (int i = 0; i < natoms; i++) {
    vx[i] = -vx[i];
    vy[i] = -vy[i];
    vz[i] = -vz[i];
  }
  if (doprint) {
    printf("\n");
    printf("Coords. and Vels. after reversing velocities\n");
    for (int i = 0; i < natoms; i++) {
      printf("%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", 
        x[i], y[i], z[i], vx[i], vy[i], vz[i]);
    }
  }
  {
    const auto tstart = chrono::system_clock::now();
    iterate(nsteps, natoms, x, y, z, vx, vy, vz);
    const auto tstop = chrono::system_clock::now();
    const chrono::duration<double> time_elapsed = tstop - tstart;
    cout << "(iterating backwards) Time elapsed: " << time_elapsed.count() << " s" << endl << endl;
  }
  if (doprint) {
    printf("\n");
    printf("Coords. and Vels. after reversing velocities (should be intial coords. again)\n");
    for (int i = 0; i < natoms; i++) {
      printf("%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", 
        x[i], y[i], z[i], vx[i], vy[i], vz[i]);
    }
    printf("\n");
    printf("x-coord (initial) x-coord x-error x-coord (initial) y-coord y-error z-coord (initial) z-coord z-error\n");
    for (int i = 0; i < natoms; i++) {
      printf("%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", 
        x[i], xinit[i], x[i]-xinit[i], y[i], yinit[i], y[i]-yinit[i], z[i], zinit[i], z[i]-zinit[i]);
    }
    printf("\n");
    printf("x-vel (initial) x-vel x-error x-vel (initial) y-vel y-error z-vel (initial) z-vel z-error\n");
    for (int i = 0; i < natoms; i++) {
      printf("%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", 
        vx[i], vxinit[i], vx[i]+vxinit[i], vy[i], vyinit[i], vy[i]+vyinit[i], vz[i], vzinit[i], vz[i]+vzinit[i]);
    }
  }
  bool iscorrect = true;
  double tol = 1e-10;
  for (int i = 0; i < natoms; i++) {
    iscorrect = iscorrect &&
      (fabs(x[i] - xinit[i]) < tol) &&  
      (fabs(y[i] - yinit[i]) < tol) && 
      (fabs(z[i] - zinit[i]) < tol) && 
      (fabs(vx[i] + vxinit[i]) < tol) &&  
      (fabs(vy[i] + vyinit[i]) < tol) && 
      (fabs(vz[i] + vzinit[i]) < tol);
      if (!iscorrect) printf("failed on i: %d\n", i);
  }
  if (iscorrect) printf("PASSED!\n"); else printf("FAILED!\n");

  delete [] x;
  delete [] y;
  delete [] z;
  delete [] xinit;
  delete [] yinit;
  delete [] zinit;
  delete [] vx;
  delete [] vy;
  delete [] vz;
  delete [] vxinit;
  delete [] vyinit;
  delete [] vzinit;
}

void doit() {
  double* x = new double[natoms];
  double* y = new double[natoms];
  double* z = new double[natoms];
  double* vx = new double[natoms];
  double* vy = new double[natoms];
  double* vz = new double[natoms];
  bool from_scratch = false;
  if (from_scratch) {
    create_particles(natoms, x, y, z, vx, vy, vz);
  } else {
    read_positions_from_file("start.data", natoms, x, y, z, vx, vy, vz);
  }
  const auto tstart = chrono::system_clock::now();
  iterate(nsteps, natoms, x, y, z, vx, vy, vz);
  const auto tstop = chrono::system_clock::now();
  const chrono::duration<double> time_elapsed = tstop - tstart;
  cout << "Time elapsed: " << time_elapsed.count() << " s" << endl;
  write_positions_to_file("end.data", natoms, x, y, z, vx, vy, vz);

  delete [] x;
  delete [] y;
  delete [] z;
  delete [] vx;
  delete [] vy;
  delete [] vz;
}

int main(int argc, char** argv) {
  test();
  return 0;
}


