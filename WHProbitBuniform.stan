data {                      // Data block
  int<lower=1> G;           // Number of groups 
  int<lower=1> A;           // Number of animals
  int<lower=1> T;           // Number of tests
  int<lower=1> E;           // Number of terms in likelihood/response (2^T)
  int<lower=1> gind[A];     // Index to group that animal belongs to
  int          y[A];        // Response - defined test response for each animal
  matrix[E,T]  ind;         // Indicator variable used to define response and likelihood
 }

parameters {                // Parameters block
  vector<lower=0,upper=1>[G]  prevalence;     // Prevalance in each group
  vector<lower=-8,upper=0>[T] a0;    // Independence parameters (uninfected)
  vector<lower=-1,upper=8>[T] a1;    // Independence parameters (infected)
}

// Use transformed parameter block to calculate sensitivity, specificity
transformed parameters {
vector<lower=0,upper=1>[T] sensitivity = Phi(a1);
vector<lower=0,upper=1>[T] specificity = 1-Phi(a0);
}

model {// Model block
  int c=1;
 
  // Uniform priors
 
  // likelihood
  
  for(a in 1:A){
    vector[E] p;

    for(e in 1:E)
    {
      real s_prod = 1;
      real p_prod = 1;
      
      for(t in 1:T)
      {
      s_prod = s_prod*(ind[e,t] * Phi(a1[t]) + (1-ind[e,t]) * (1-Phi(a1[t])));
      p_prod = p_prod*(ind[e,t] * Phi(a0[t]) + (1-ind[e,t]) * (1-Phi(a0[t])));
      }
      
      p[e] = (prevalence[gind[a]] * (s_prod) + (1-prevalence[gind[a]]) * (p_prod));

    }
   
    1 ~ bernoulli(p[y[a]]);
    
  }
}

generated quantities {
  vector[A] log_lik;
  vector[A] post_I;
  
  for(a in 1:A){
    vector[E] p;
    vector[E] p_i;

     for(e in 1:E)
    {
      real s_prod = 1;
      real p_prod = 1;
      
      for(t in 1:T)
      {
      s_prod = s_prod*(ind[e,t] * Phi(a1[t]) + (1-ind[e,t]) * (1-Phi(a1[t])));
      p_prod = p_prod*(ind[e,t] * Phi(a0[t]) + (1-ind[e,t]) * (1-Phi(a0[t])));
      }
      
      p[e] = (prevalence[gind[a]] * (s_prod) + (1-prevalence[gind[a]]) * (p_prod));
      p_i[e] = prevalence[gind[a]] * (s_prod);
    }
    log_lik[a] = bernoulli_lpmf(1 | p[y[a]]);
    post_I[a] = p_i[y[a]]/p[y[a]]; 
    
}
}
