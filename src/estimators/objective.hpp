#pragma once

// Placeholder for the parametric branch (roadmap: online MLE / Fisher scoring).
//
// Intended home for the bivariate copula log-likelihood and its derivatives,
// shared by the parametric online MLE, the exponential-forgetting (RLS) variant,
// and later the Bayesian update:
//
//   double log_likelihood(double u, double v, double theta);   // copula density
//   double score(double u, double v, double theta);            // s_t(theta) = d/dtheta log c
//   double fisher_information(double theta);                   // I(theta)
//
// Empty until the parametric branch is implemented; kept as a stub so the
// intended interface is discoverable. Not yet referenced anywhere.
