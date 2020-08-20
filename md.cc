//////////////////////////
// NOTE : BUILD WITH c++14
//////////////////////////
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

using coord_3d = std::tuple<double,double,double>;
using vel_3d = std::tuple<double,double,double>;
using force_3d = std::tuple<double,double,double>;

const double PI = 3.1415926535897932384626433832795028841971693993751;

// default parameters
double L = 320;
double dt = 0.1;
double natoms = 10000;
int nsteps = 10;
double C = 0.1;
double alpha = 0.1;

const int nprint = 10;

auto force(double x, double y, double z) {
  double fx = 0.0; double fy = 0.0; double fz = 0.0;
  double vij = 0.0;
  double alpha_rsq = alpha*(x*x+y*y+ z*z);
  if (alpha_rsq < 15.0) {
    vij = C*exp(-alpha_rsq);
    fx = 2.0*x*alpha*vij;
    fy = 2.0*y*alpha*vij;
    fz = 2.0*z*alpha*vij;
  }
  return std::make_tuple(fx,fy,fz,vij);
}

auto forces(std::vector<coord_3d> coords) {
  int ncoords = coords.size();
  double pe = 0.0;
  std::vector<force_3d> vforces(ncoords);
  //#pragma omp parallel for reduction(+:pe)
  for (int i = 0; i < ncoords; i++) {
    auto [xi, yi, zi] = coords[i];
    double fxi = 0.0; double fyi = 0.0; double fzi = 0.0;
    //#pragma omp parallel for reduction(+:fxi,fyi,fzi,pe)
    for (int j = 0; j < ncoords; j++) {
      auto [xj, yj, zj] = coords[j];
      auto dx = xi-xj;
      auto dy = yi-yj;
      auto dz = zi-zj;
      if (dx > L/2) dx -= L;
      else if (dx < -L/2) dx += L;
      if (dy > L/2) dy -= L;
      else if (dy < -L/2) dy += L;
      if (dz > L/2) dz -= L;
      else if (dz < -L/2) dz += L;
      auto [fx, fy, fz, vij] = force(dx,dy,dz);
      pe += vij;
      fxi += fx; fyi += fy; fzi += fz;
    }
    vforces[i] = std::make_tuple(fxi,fyi,fzi);
  } 
  return std::make_tuple(vforces,pe);
}

auto forces_tiled(std::vector<coord_3d> coords, int tsize = 120) {
  int ncoords = coords.size();
  double pe = 0.0;
  std::vector<force_3d> vforces(ncoords);
  for (int ii = 0; ii < ncoords; ii+=tsize) {
    for (int jj = 0; jj < ncoords; jj+=tsize) {
    int imax = std::min(ii+tsize, ncoords);
    int jmax = std::min(jj+tsize, ncoords);
      #pragma omp for 
      for (int i = ii; i < imax; i++) {
        auto [xi, yi, zi] = coords[i];
        double fxi = 0.0; double fyi = 0.0; double fzi = 0.0;
        for (int j = jj; j < jmax; j++) {
          auto [xj, yj, zj] = coords[j];
          auto dx = xi-xj;
          auto dy = yi-yj;
          auto dz = zi-zj;
          if (dx > L/2) dx -= L;
          else if (dx < -L/2) dx += L;
          if (dy > L/2) dy -= L;
          else if (dy < -L/2) dy += L;
          if (dz > L/2) dz -= L;
          else if (dz < -L/2) dz += L;
          auto [fx, fy, fz, vij] = force(dx,dy,dz);
          pe += vij;
          fxi += fx; fyi += fy; fzi += fz;
        }
        vforces[i] = std::make_tuple(fxi,fyi,fzi);
      }
    }
  }
  return std::make_tuple(vforces,pe);
}

void read_positions_from_file(const std::string& fname, std::vector<coord_3d>& coords, std::vector<coord_3d>& vels) {
  std::ifstream fil(fname.c_str());
  assert(fil.is_open());
  int natoms2 = -1;
  double x, y, z, vx, vy, vz;
  fil >> natoms2;
  assert(natoms == natoms2);
  for (int i = 0; i < natoms2; i++) {
    fil >> x;
    fil >> y;
    fil >> z;
    fil >> vx;
    fil >> vy;
    fil >> vz;
    coords[i] = std::make_tuple(x,y,z);
    vels[i] = std::make_tuple(vx,vy,vz);
  }
}

void write_positions_to_file(const std::string& fname, const std::vector<coord_3d>& coords, const std::vector<vel_3d>& vels) {
  std::ofstream fil(fname.c_str());
  fil.precision(10);
  assert(fil.is_open());
  int natoms = coords.size();
  int natoms2 = vels.size();
  assert(natoms == natoms2);
  fil << natoms2 << std::endl;
  for (int i = 0; i < natoms; i++) {
    auto [x, y, z] = coords[i];
    auto [vx, vy, vz] = vels[i];
    fil.width(20);
    fil << std::scientific;
    fil << x << " ";
    fil.width(20);
    fil << y << " ";
    fil.width(20);
    fil << z << " ";
    fil.width(20);
    fil << vx << " ";
    fil.width(20);
    fil << vy << " ";
    fil.width(20);
    fil << vz;
    fil.width(20);
    fil << std::endl;
  }
}

void create_particles(std::vector<coord_3d>& coords, std::vector<vel_3d>& vels) {
  // create particles and intialize their positions and velocities
  std::default_random_engine generator;
  std::uniform_real_distribution<double> udist(0.0,1.0);
  std::normal_distribution<double> vdist(5.0,1.5);
  for (int i = 0; i < natoms; i++) {
    auto x =     udist(generator)*L;    
    auto y =     udist(generator)*L;    
    auto z =     udist(generator)*L;    
    auto v =     vdist(generator);
    auto phi =   udist(generator)*2*PI;
    auto theta = udist(generator)*PI;
    auto vx = v*sin(theta)*cos(phi);
    auto vy = v*sin(theta)*sin(phi);
    auto vz = v*cos(theta);
    coords[i] = std::make_tuple(x,y,z);
    vels[i] = std::make_tuple(vx,vy,vz);
  }
}

void iterate(const int& nsteps, std::vector<coord_3d>& coords, std::vector<vel_3d>& vels) {
  auto [f, pe] = forces(coords);
  for (auto istep = 0; istep < nsteps; istep++) {
    for (auto i = 0; i < natoms; i++) {
      // get position 
      auto [x, y, z] = coords[i];
      // get velocity
      auto [vx, vy, vz] = vels[i];
      // get forces
      auto [fx, fy, fz] = f[i];
      // update velocities at t + dt/2
      vx += fx*dt*0.5;
      vy += fy*dt*0.5;
      vz += fz*dt*0.5;
      // update positions at t + dt/2
      x += vx*dt;
      y += vy*dt;
      z += vz*dt;
      // periodic boundary conditions
      if (x >= L) x -= L;
      else if (x < 0.0) x += L;
      if (y >= L) y -= L;
      else if (y < 0.0) y += L;
      if (z >= L) z -= L;
      else if (z < 0.0) z += L;
      coords[i] = std::make_tuple(x,y,z);
      vels[i] = std::make_tuple(vx,vy,vz);
    }
    auto ke = 0.0;
    std::tie(f, pe) = forces(coords);
    for (int i = 0; i < natoms; i++) {
      // get force
      double fx, fy, fz;
      std::tie(fx, fy, fz) = f[i];
      // get velocity
      double vx, vy, vz;
      std::tie(vx, vy, vz) = vels[i];
      vx += fx*dt*0.5;
      vy += fy*dt*0.5;
      vz += fz*dt*0.5;
      // energies
      ke += 0.5*(vx*vx + vy*vy + vz*vz);
      // update velocities
      vels[i] = std::make_tuple(vx, vy, vz);
    } 
    if (istep%nprint == 0) {
      printf("KE:     %15.8e   PE:     %15.8e   TE:     %15.8e\n", ke, pe, ke+pe);
    } 
  }
}

void test_tiles() {
  auto niter = 1;
  std::vector<coord_3d> coords(natoms);
  std::vector<coord_3d> vels(natoms);
  create_particles(coords, vels);
  for (auto it = 1; it <= 20; it++) {
    auto tsize = 120*it;
    const auto tstart = std::chrono::system_clock::now();
    for (auto iter = 0; iter < niter; iter++) {
      auto result = forces_tiled(coords, tsize);
    }
    const auto tstop = std::chrono::system_clock::now();
    const std::chrono::duration<double> time_elapsed = tstop - tstart;
    std::cout << "tsize: " << tsize << "     Time elapsed: " << time_elapsed.count() << " s" << std::endl;
  }
}

void test() {
  L = 32;
  dt = 0.1;
  natoms = 20;
  nsteps = 10;
  C = 0.1;
  alpha = 0.01;

  // create coordinates and velocities
  std::vector<coord_3d> coords(natoms);
  std::vector<coord_3d> vels(natoms);
  create_particles(coords, vels);
  // copy coordinates and velocities
  std::vector<coord_3d> coords_init(coords);
  std::vector<coord_3d> vels_init(vels);
  iterate(nsteps, coords, vels);
  for (int i = 0; i < natoms; i++) {
    auto [vx, vy, vz] = vels[i];
    vx = -vx; vy = -vy; vz = -vz;
    vels[i] = std::make_tuple(vx,vy,vz);
  }
  iterate(nsteps, coords, vels);
  for (int i = 0; i < natoms; i++) {
    auto [x1,y1,z1] = coords[i]; auto [x2,y2,z2] = coords_init[i];
    double xerr = x1-x2; double yerr = y1-y2; double zerr = z1-z2;
    printf("%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", x1, x2, xerr, y1, y2, yerr, z1, z2, zerr);
  }
}

void doit() {
  std::vector<coord_3d> coords(natoms);
  std::vector<coord_3d> vels(natoms);
  bool from_scratch = false;
  if (from_scratch) {
    create_particles(coords, vels); 
  } else {
    read_positions_from_file("start.data", coords, vels);
  }
  const auto tstart = std::chrono::system_clock::now();
  iterate(nsteps, coords, vels);
  const auto tstop = std::chrono::system_clock::now();
  const std::chrono::duration<double> time_elapsed = tstop - tstart;
  std::cout << "Time elapsed: " << time_elapsed.count() << " s" << std::endl;
  write_positions_to_file("end.data", coords, vels);
}

int main(int argc, char** argv) {
  //doit();
  //test_tiles();
  test();
  return 0;
}


