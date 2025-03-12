functions {
  vector gp_pred_rng(array[] real x_pred,
                     array[] real x,
                     vector y, 
                     real lengthscale,
                     real alpha) {
    int N = rows(y);
    int N_pred = size(x_pred);
    vector[N_pred] f_pred;
    {
      matrix[N, N] L;
      vector[N] K_div_y;
      matrix[N, N_pred] k_x_x_pred;
      matrix[N, N_pred] v_pred;
      vector[N_pred] f_pred_mu;
      matrix[N_pred, N_pred] f_pred_cov;
      matrix[N, N] K;
      K = gp_matern52_cov(x, alpha, lengthscale);
      
      L = cholesky_decompose(K);
      K_div_y = mdivide_left_tri_low(L, y);
      K_div_y = mdivide_right_tri_low(K_div_y', L)';
      k_x_x_pred = gp_matern52_cov(x, x_pred, alpha, lengthscale);
      f_pred_mu = (k_x_x_pred' * K_div_y);
      v_pred = mdivide_left_tri_low(L, k_x_x_pred);
      f_pred_cov = gp_matern52_cov(x_pred, alpha, lengthscale) - v_pred' * v_pred;

      f_pred = multi_normal_rng(f_pred_mu, add_diag(f_pred_cov, rep_vector(1e-9, N_pred)));
    }
    return f_pred;
  }
  matrix gp_pred_multi_rng(array[] real x_pred,
                           array[] real x,
                           matrix y, 
                           real lengthscale,
                           vector alpha, 
                           matrix L_Omega){
    int N = rows(y);
    int D = cols(y);
    int N_pred = size(x_pred);
    matrix[D, D] corr = diag_pre_multiply(alpha, L_Omega)';
    matrix[N, D] y_U = y / corr;

    matrix[N, N] K = gp_matern52_cov(x, 1.0, lengthscale);
    matrix[N, N] L = cholesky_decompose(add_diag(K, rep_vector(1e-9, N)));

    matrix[N, N_pred] k_x_x_pred = gp_matern52_cov(x, x_pred, 1.0, lengthscale);
    matrix[N, N_pred] v_pred = mdivide_left_tri_low(L, k_x_x_pred);
    matrix[N_pred, N_pred] f_pred_cov = gp_matern52_cov(x_pred, 1.0, lengthscale) - v_pred' * v_pred;

    matrix[N_pred, D] f_pred;
    matrix[N_pred, D] f_pred_U;
    vector[N] K_div_y;
    vector[N_pred] f_pred_mu;
    for (d in 1:D) {
      K_div_y = mdivide_left_tri_low(L, y_U[, d]);
      K_div_y = mdivide_right_tri_low(K_div_y', L)';
      
      f_pred_mu = (k_x_x_pred' * K_div_y);

      f_pred_U[, d] = multi_normal_rng(f_pred_mu, add_diag(f_pred_cov, rep_vector(1e-9, N_pred)));
    }
    f_pred = f_pred_U * corr;
    return f_pred;
  }
}
data {
  int<lower=1> N_times;
  int<lower=1> N_times_pred;
  int<lower=1> N_patients;
  int<lower=1> N_dimensions;
  int<lower=1> N_visits;
  int<lower=1> N_groups;
  array[N_times] real times;
  array[N_times_pred] real times_pred;
  array[N_visits, N_patients, N_times, N_dimensions] int mask;
  array[N_visits, N_patients] matrix[N_times, N_dimensions] y;  
  array[N_patients] int groups;
  real<lower=0> lengthscale_common;
  real<lower=0> lengthscale_patient;
  real<lower=0> sigma_beta;
}
parameters {
  array[N_groups, N_visits] vector[N_dimensions] mu_common;
  array[N_groups, N_visits] vector<lower=0>[N_dimensions] sigma_patient;
  array[N_visits] matrix[N_patients, N_dimensions] mu_patient;
  array[N_groups, N_visits] matrix[N_times, N_dimensions] eta_common;
  array[N_visits, N_patients] matrix[N_times, N_dimensions] eta_patient;

  array[N_groups, N_visits] vector<lower=0>[N_dimensions] alpha_common; // common/shared temporal pattern: kernel scale (sqrt of kernel variance)
  array[N_visits, N_patients] vector<lower=0>[N_dimensions] alpha_patient; // patient-specific temporal pattern: kernel scale (sqrt of kernel variance)

  array[N_groups, N_visits] cholesky_factor_corr[N_dimensions] L_Omega_common;
  array[N_visits, N_patients] cholesky_factor_corr[N_dimensions] L_Omega_patient;

  vector<lower=0>[N_dimensions] sigma; // observation noise scale
}
transformed parameters {
  array[N_groups, N_visits] matrix[N_times, N_dimensions] f_common; // convert it to vector with N_times
  {
    matrix[N_times, N_times] K_common = gp_matern52_cov(times, 1.0, lengthscale_common);
    matrix[N_times, N_times] L_common = cholesky_decompose(add_diag(K_common, rep_vector(1e-9, N_times)));
    for (g in 1:N_groups) {
      for (v in 1:N_visits) {
        f_common[g, v] = L_common * eta_common[g, v] * diag_pre_multiply(alpha_common[g, v], L_Omega_common[g, v])';
      }
    }
  }

  array[N_visits, N_patients] matrix[N_times, N_dimensions] f_patient;
  {
    matrix[N_times, N_times] K_patient = gp_matern52_cov(times, 1.0, lengthscale_patient);
    matrix[N_times, N_times] L_patient = cholesky_decompose(add_diag(K_patient, rep_vector(1e-9, N_times)));
    for (p in 1:N_patients) {
      for (v in 1:N_visits) {
        f_patient[v, p] = L_patient * eta_patient[v, p] * diag_pre_multiply(alpha_patient[v, p], L_Omega_patient[v, p])';
      }
    }
  }

  array[N_visits, N_patients] matrix[N_times, N_dimensions] f;
  for (v in 1:N_visits) {
    for (p in 1:N_patients) {
      for (t in 1:N_times) {
        for (d in 1:N_dimensions) {
          f[v, p, t, d] = mu_patient[v, p, d] // patient's deviation in baseline
                          + f_common[groups[p], v, t, d] //f_common_response
                          + f_patient[v, p, t, d]; //f_patient_deviation
        }
      }
    }
  }
}
model {
  for (v in 1:N_visits) {
    for (g in 1:N_groups) {
      mu_common[g, v] ~ normal(0, 1);
      sigma_patient[g, v] ~ inv_gamma(1, 0.5);
      to_vector(eta_common[g, v]) ~ normal(0, 1);
      L_Omega_common[g, v] ~ lkj_corr_cholesky(1);
      alpha_common[g, v] ~ inv_gamma(1, 1);
    }
    for (p in 1:N_patients) {
      for (d in 1:N_dimensions) {
        mu_patient[v, p, d] ~ normal(mu_common[groups[p], v, d], sigma_patient[groups[p], v, d]);
      }
    }
  }

  for (p in 1:N_patients) {
    for (v in 1:N_visits) {
      to_vector(eta_patient[v, p]) ~ normal(0, 1);
      L_Omega_patient[v, p] ~ lkj_corr_cholesky(1);
      alpha_patient[v, p] ~ inv_gamma(1, 0.1);
    }
  }

  sigma ~ inv_gamma(1, sigma_beta);

  for (v in 1:N_visits) {
    for (p in 1:N_patients) {
      for (t in 1:N_times) {
        for (d in 1:N_dimensions) {
          if (mask[v, p, t, d] == 1) {
            if (t == 1) {
              y[v, p, t, d] ~ normal(mu_patient[v, p, d], sigma[d] / 10);
              0 ~ normal(f_common[groups[p], v, t, d], sqrt(pow(sigma[d], 2) * 0.495));
              0 ~ normal(f_patient[v, p, t, d], sqrt(pow(sigma[d], 2) * 0.495));
            }
            else {
              y[v, p, t, d] ~ normal(f[v, p, t, d], sigma[d]);
            }
          }
        }
      }
    }
  }
}
generated quantities {
  array[N_groups, N_visits] matrix[N_dimensions, N_dimensions] Omega_common;
  array[N_visits, N_patients] matrix[N_dimensions, N_dimensions] Omega_patient;
  array[N_groups, N_visits] matrix[N_times_pred, N_dimensions] f_common_pred;
  array[N_visits, N_patients] matrix[N_times_pred, N_dimensions] f_patient_pred;
  for (v in 1:N_visits) {
    for (g in 1:N_groups) {
      Omega_common[g, v] = L_Omega_common[g, v] * L_Omega_common[g, v]';
      f_common_pred[g, v] = gp_pred_multi_rng(times_pred, times, f_common[g, v], lengthscale_common, alpha_common[g, v], L_Omega_common[g, v]);
    }
    for (p in 1:N_patients) {
      Omega_patient[v, p] = L_Omega_patient[v, p] * L_Omega_patient[v, p]';
      f_patient_pred[v, p] = gp_pred_multi_rng(times_pred, times, f_patient[v, p], lengthscale_patient, alpha_patient[v, p], L_Omega_patient[v, p]);
    }
  }
  array[N_visits, N_patients] matrix[N_times_pred, N_dimensions] f_pred;
  array[N_visits, N_patients] matrix[N_times_pred, N_dimensions] y_pred;
  for (v in 1:N_visits) {
    for (p in 1:N_patients) {
      for (t in 1:N_times_pred) {
        for (d in 1:N_dimensions) {
          f_pred[v, p, t, d] = mu_patient[v, p, d] + f_common_pred[groups[p], v, t, d] + f_patient_pred[v, p, t, d];
          y_pred[v, p, t, d] = normal_rng(f_pred[v, p, t, d], sigma[d]);
        }
      }
    }
  }   
}
