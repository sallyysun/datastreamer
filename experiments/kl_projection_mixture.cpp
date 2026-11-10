// KL projection of the Clayton-Gumbel mixture (both calibrated to rho_S = 0.5819) onto single families.
#include <cmath>
#include <cstdio>
#include <vector>
#include "estimators/dependence/family_bank.hpp"
using namespace copula;
int main(){
  const double rs = 6/M_PI*std::asin(0.3), q=0.05, nu=4;
  const double thC = param_from_spearman(Family::Clayton, rs), thG = param_from_spearman(Family::Gumbel, rs);
  const int G=300; const double L=7.0; std::vector<double> x(G), wx(G), U(G);
  for(int i=0;i<G;i++){ x[i]=-L+2*L*(i+0.5)/G; U[i]=special::normal_cdf(x[i]); wx[i]=2*L/G*std::exp(-0.5*x[i]*x[i])/std::sqrt(2*M_PI); }
  auto mixd=[&](double pi,double u,double v){ return pi*std::exp(log_density(Family::Clayton,u,v,thC))+(1-pi)*std::exp(log_density(Family::Gumbel,u,v,thG)); };
  printf("rho_S = %.4f; Clayton theta %.3f, Gumbel theta %.3f\n",rs,thC,thG);
  for(double pi : {0.75,0.5,0.25}){
    const auto tc=exact_tails(Shape::Clayton,thC,nu,q), tg=exact_tails(Shape::Gumbel,thG,nu,q);
    const double mL=pi*tc.chi_L+(1-pi)*tg.chi_L, mU=pi*tc.chi_U+(1-pi)*tg.chi_U;
    printf("\npi = %.2f (Clayton weight): mixture chi_L(0.05) = %.3f, chi_U(0.05) = %.3f\n",pi,mL,mU);
    std::vector<double> cm(G*G), w2(G*G), lcm(G*G); double ent=0;   // mixture density once per pi
    for(int i=0;i<G;i++) for(int j=0;j<G;j++){ int k=i*G+j; cm[k]=mixd(pi,U[i],U[j]); w2[k]=wx[i]*wx[j]*cm[k]; lcm[k]=cm[k]>0?std::log(cm[k]):0; ent+=w2[k]*lcm[k]; }
    struct F { const char* name; Family f; Shape s; double lo, hi; };
    for (F f : {F{"Gaussian",Family::Gaussian,Shape::Gaussian,0.05,0.95}, F{"t4",Family::StudentT,Shape::StudentT,0.05,0.95},
                F{"Clayton",Family::Clayton,Shape::Clayton,0.05,6}, F{"Gumbel",Family::Gumbel,Shape::Gumbel,1.01,4}}) {
      auto KL=[&](double p){ double cr=0; for(int i=0;i<G;i++) for(int j=0;j<G;j++){ int k=i*G+j; if(w2[k]>0) cr+=w2[k]*log_density(f.f,U[i],U[j],p,nu);} return ent-cr; };
      double a=f.lo, b=f.hi, g=(std::sqrt(5.0)-1)/2, c=b-g*(b-a), d=a+g*(b-a), fc=KL(c), fd=KL(d);   // golden section
      for(int it=0;it<40;it++){ if(fc<fd){ b=d; d=c; fd=fc; c=b-g*(b-a); fc=KL(c);} else { a=c; c=d; fc=fd; d=a+g*(b-a); fd=KL(d);} }
      const double bp=0.5*(a+b), best=KL(bp); const auto tp=exact_tails(f.s,bp,nu,q);
      printf("  %-8s KL* = %.4f nats  theta* = %.3f (rho_S %.3f)  chi_L %.3f  chi_U %.3f\n",f.name,best,bp,spearman_from_param(f.f,bp,nu),tp.chi_L,tp.chi_U);
      fflush(stdout);
    }
  }
}
