data {                      // Data block
  int<lower=1> G;           // Number of groups 
  int<lower=1> A;           // Number of animals
  int<lower=1> T;           // Number of tests
  int<lower=1> E;           // Number of terms in likelihood/response (2^T)
  int<lower=1> D;           // Number of unique diagnostic test groups
  
  int<lower=1> tind[T];     // Index to group that diagnostic test belongs to
  int<lower=1> gind[A];     // Index to group that animal belongs to
  int          y[A];        // Response - defined test response for each animal
  matrix[E,T]  ind;         // Indicator variable used to define response and likelihood
  int          direction[D]; // Indicator variable to flip prior assumption about ordering of sensitivity/specificity
 }

parameters {                // Parameters block
  vector<lower=0,upper=1>[G]  prevalence;     // Prevalance in each group
  vector<lower=-8,upper=8>[T] a0;    // Independence parameters (uninfected)
  vector<lower=0,upper=8>[D]  sigma;
}

// Use transformed parameter block to calculate sensitivity, specificity
transformed parameters {
vector<lower=0,upper=1>[T] sensitivity;
vector<lower=0,upper=1>[T] specificity = 1 - Phi(a0);

for(t in 1:T)
{
sensitivity[t] = Phi(inv_Phi(Phi(a0[t]))+direction[tind[t]]*sigma[tind[t]]);
}
}

model {// Model block
  int c=1;
 
  prevalence  ~ beta(1,1);
 
  a0 ~ normal(0,1);
  sigma ~ normal(0,1);
  
// likelihood
  
  for(a in 1:A){
    vector[E] p;

    for(e in 1:E)
    {
      real s_prod = 1;
      real p_prod = 1;
      
      for(t in 1:T)
      {
      s_prod = s_prod*(ind[e,t] * sensitivity[t] + (1-ind[e,t]) * (1-sensitivity[t]));
      p_prod = p_prod*(ind[e,t] * (1-specificity[t]) + (1-ind[e,t]) * specificity[t]);
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
      s_prod = s_prod*(ind[e,t] * sensitivity[t] + (1-ind[e,t]) * (1-sensitivity[t]));
      p_prod = p_prod*(ind[e,t] * (1-specificity[t]) + (1-ind[e,t]) * specificity[t]);
      }
      
      p[e] = (prevalence[gind[a]] * (s_prod) + (1-prevalence[gind[a]]) * (p_prod));
      p_i[e] = prevalence[gind[a]] * (s_prod);
    }
    log_lik[a] = bernoulli_lpmf(1 | p[y[a]]);
    post_I[a] = p_i[y[a]]/p[y[a]]; 
    
}
}
