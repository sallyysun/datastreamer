// KL divergence per tick between the Gaussian and t4 copulas at matched rho_S = 0.582 (d = 0.6), each
// direction against the best-fitting member of the other family (grid over its correlation, 0.40-0.80):
// the drift of the family bank's Hedge log-weight ratio after a Gaussian <-> t switch.
// Notes: notes/studies/thesis_bank_family.md section 2.3.
// Usage (from experiments/): g++ -std=c++20 -O2 -pthread probe_kl_shape.cpp -o probe_kl_shape && ./probe_kl_shape
#include <cstdio>
#include "tail_common.hpp"
int main(){ using namespace copula; const int n=200000;
 for (std::string f : {"t","gaussian"}) {
  auto fx = tail::fix(f,0.6); tail::Sampler sm(fx, 3);
  std::vector<std::pair<double,double>> d(n); for(auto& x:d) x=sm.next();
  double own=0; for(auto [u,v]:d) own += f=="t"? StudentT::log_density(u,v,fx.param,4): Gaussian::log_density(u,v,fx.param); own/=n;
  double best=-1e9, br=0;
  for(double r=0.40;r<=0.80;r+=0.01){ double a=0; for(auto [u,v]:d) a += f=="t"? Gaussian::log_density(u,v,r): StudentT::log_density(u,v,r,4); a/=n; if(a>best){best=a;br=r;} }
  std::printf("data %s: own %.4f, best other (r=%.2f) %.4f, KL per tick %.4f nats\n", f.c_str(), own, br, best, own-best);
 }}
