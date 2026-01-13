# Main script for permutation max-t testing
#
# Neurobiology research unit, 2021-2022

list.of.packages <- c("nlme","parallel","tictoc")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
library("nlme")
library("parallel")
library("tictoc")

setwd("/indirect/staff/anderssolsen/p3_spectral")
source("spectral_analysis/scripts_spectral_analysis/Rfunction_permlme.R")
# dt <- read.csv("data/results/spectra_networks_rois_agg.csv")

# nperm = 10

# # initialize output dataframe and corresponding counter
# outdf <- data.frame(matrix(ncol = 9, nrow = sum(unique(dt$N_centroids))))
# colnames(outdf) <- c("strategy","network","frequency","coefintercept","coefcov","coefCIcovlow","coefCIcovhigh","pval","pval_perm")
# counter = 1
# k <- 25 #number of frequencies


# Function that can be parallelized in x = permutations.
# The function permutes observations and runs the permlme_function
# the latter being an implementation of Lee2012 (Biometrics).
# dtk is the subset of data for a given k

LRTapply <- function(x, df_reduced, controlled_variable, n_obs, e_lmeH0, e_lmeH1){
  index.perm <- sample(1:n_obs)
  # index.perm <- n_obs:1
  k <- length(unique(df_reduced[[controlled_variable]]))
  # LRT <- 1:k
  # initialize as nan
  LRT <- rep(NA, k)
  for (i in 1:k) {
    controlled <- unique(df_reduced[[controlled_variable]])[i]
    df_reduced_controlled <- df_reduced[df_reduced[[controlled_variable]] == controlled, ]
    
    LRT[i] <- permlme_function(
      e_lmeH0[[i]], e_lmeH1[[i]], data=df_reduced_controlled, index.perm=index.perm
    )
  }
  # print LRT values for debugging
  print(LRT)
  return(LRT)
}

main <- function(df, target_variable, covariates, nuisance_regressors,
                 uncontrolled_variable, controlled_variable,
                 nperm = NULL) {
  outdf <- data.frame(
    covariate = character(),
    uncontrolled = character(),
    controlled = character(),
    coefintercept = numeric(),
    coefcovariate = numeric(),
    coefCIcovariatelow = numeric(),
    coefCIcovariatehigh = numeric(),
    pval = numeric(),
    pval_perm = numeric(),
    stringsAsFactors = FALSE
  )
  # loop through nuisance regressors and rescale
  for (nuisance in nuisance_regressors){
    # if nuisance is not numeric, skip
    if (!is.numeric(df[[nuisance]])){
      next
    }
    df[[nuisance]] <- scale(df[[nuisance]])
  }

  formula1 <- paste0(target_variable, " ~ 1 + ", paste(nuisance_regressors, collapse = " + "))
  counter <- 1

  for (covariate in covariates){
    formula2 <- paste0(formula1, " + ", covariate)
    not_covariate <- setdiff(covariates, covariate)
    df_covariate <- df[, !colnames(df) %in% not_covariate, drop = FALSE]
    # df_covariate <- na.omit(df_covariate)
    # omit nans in target variable, nuisance regressors, covariate
    df_covariate <- df_covariate[!is.na(df_covariate[[target_variable]]), ]
    for (nuisance in nuisance_regressors){
      df_covariate <- df_covariate[!is.na(df_covariate[[nuisance]]), ]
    }
    df_covariate <- df_covariate[!is.na(df_covariate[[covariate]]), ]

    for (uncontrolled in unique(df_covariate[[uncontrolled_variable]])){
      message("Processing covariate: ", covariate, ", uncontrolled variable: ", uncontrolled)
      df_reduced <- try(df_covariate[df_covariate[[uncontrolled_variable]] == uncontrolled, ],silent=TRUE)
      if(inherits(df_reduced, "try-error")){
        browser()
      }
      
      # Initialize variables because R
      k <- length(unique(df_reduced[[controlled_variable]]))
      e_lmeH0 <- as.list(1:k)
      e_lmeH1 <- as.list(1:k)
      LRT_init <- as.list(1:k)
      pval_init = as.list(1:k)
      perm_p = as.list(1:k)
      
      for (i in 1:k) {
        message(" Controlled variable level: ", i, " of ", k)
        controlled <- unique(df_reduced[[controlled_variable]])[i]
        df_reduced_controlled <- df_reduced[df_reduced[[controlled_variable]] == controlled, ]
        
        n_obs = nrow(df_reduced_controlled)
        # message(n_obs)
        
        # Compute initial model
        e_lmeH0[[i]] <- try(nlme::lme(as.formula(formula1), random =~1|subject, data = df_reduced_controlled, method="ML"), silent = TRUE)
        e_lmeH1[[i]] <- try(nlme::lme(as.formula(formula2), random =~1|subject, data = df_reduced_controlled, method="ML"), silent = TRUE)
        
        if(inherits(e_lmeH0[[i]], "try-error") || inherits(e_lmeH1[[i]], "try-error")){
          # browser()
          LRT_init[[i]] <- NA
          pval_init[[i]] = NA
          next
        }
        
        LRT_init[[i]] <- as.double(2*(logLik(e_lmeH1[[i]])-logLik(e_lmeH0[[i]])))
        pval_init[[i]] = pchisq(LRT_init[[i]],df=1,lower.tail = FALSE)
      }
      
      ######################### Run permutation testing for all states ################################
      if (nperm > 1){
        message("Running permutation testing with nperm = ", nperm)
        # browser()
        LRT_list = mclapply(1:nperm,LRTapply,df_reduced=df_reduced, controlled_variable=controlled_variable, n_obs=n_obs,e_lmeH0=e_lmeH0,e_lmeH1=e_lmeH1,mc.cores=8)
        # LRT_list = LRTapply(x=1,df_reduced=df_reduced, controlled_variable=controlled_variable, n_obs=n_obs,e_lmeH0=e_lmeH0,e_lmeH1=e_lmeH1)
        message("Permutation testing done, processing results")
        LRT_matrix = matrix(unlist(LRT_list),ncol=k,byrow=TRUE)
        # Find max statistic per permutation
        LRTmax <- apply(LRT_matrix,1,max)
        LRTmax = LRTmax[!is.na(LRTmax)]
      }
      
      # find permutation p-value and paste information into output dataframe
      for (i in 1:k) {
        controlled <- unique(df_reduced[[controlled_variable]])[i]

        if (nperm > 1){
          perm_p = sum(LRTmax>=LRT_init[[i]])/nperm
        } else {
          perm_p = NA
        }
        
        # confidence intervals for lme fails when fit is singular
        if (inherits(e_lmeH1[[i]], "try-error")){
          ci <- c(NA, NA)
          coef_intercept = NA
          coef_covariate = NA
        } else {
        coefci <- try(intervals(e_lmeH1[[i]]), silent = TRUE)
        if(inherits(coefci,"try-error")){
          ci <- c(NA, NA)
        }else{
          ci <- coefci$fixed[covariate, c(1, 3)]
        }
        coef_intercept  = e_lmeH1[[i]]["coefficients"]$coefficients$fixed["(Intercept)"]
        coef_covariate  = e_lmeH1[[i]]["coefficients"]$coefficients$fixed[covariate]
        }
        outdf[counter, ] <- list(
          covariate, uncontrolled, controlled,
          coef_intercept, coef_covariate,
          ci[1], ci[2],
          pval_init[[i]], perm_p
        )
        
        counter = counter + 1
      }
      print(paste0("Done with model ",uncontrolled))
    }

  }
  return(outdf)
}
