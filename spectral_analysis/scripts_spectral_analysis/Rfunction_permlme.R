
# Implementation of max-T permutation testing
#
# Neurobiology research unit, 2021-2022
permlme_function <- function(lme0.ML, lme1.ML, data = NULL, index.perm = NULL, random_formula = NULL){ 
  
  
  ## ** 1- extract key quantities from input
  n.obs <- NROW(data) ## number of observations
  
  subject <- getGroups(lme0.ML) ## to which cluster (patient) each observation belongs
  unique_subject <- levels(subject)
  index_subject <- lapply(unique_subject, function(c){which(subject==c)}) ## used to restaure proper ordering after tapply
  n_subject <- length(unique_subject)
  
  Y <- getResponse(lme1.ML) ## response
  target_variable <- all.vars(formula(lme1.ML))[[1]] ## name of the response variable
  X0 <- model.matrix(formula(lme0.ML),data) ## design matrix
  
  Omega0 <- getVarCov(lme0.ML, type = "marginal", individuals = levels(subject)) ## residual variance-covariance matrix
  beta0 <- fixef(lme0.ML) ## regression coefficients
  
  ## ** 2- compute residuals
  Xbeta0 <- X0 %*% beta0
  residuals0 <- as.double(Y - X0 %*% beta0)
  
  ## ** 3- compute normalized residuals
  sqrtOmega0 <- lapply(Omega0,function(iOmega0){t(chol(iOmega0))})
  sqrtOmega0M1 <- lapply(sqrtOmega0,solve)
  
  residuals0N <- vector(length=n.obs, mode = "numeric")
  for(isubject in 1:n_subject){
    residuals0N[index_subject[[isubject]]] <- sqrtOmega0M1[[isubject]] %*% residuals0[index_subject[[isubject]]]
  }
  
  ## ** 4- estimate the distribution of the test statistics under the null
  # Anders edit: done elsewhere
  
  data.perm <- data
  
  ## permute residuals and fixed effects
  
  residuals0N.perm <- residuals0N[index.perm]
  
  ## rescale residuals
  for(isubject in 1:n_subject){ ## isubject <- 1
    data.perm[[target_variable]][index_subject[[isubject]]] <- sqrtOmega0[[isubject]] %*% residuals0N.perm[index_subject[[isubject]]]
  }
  ## add permuted fixed effects
  # browser()
  data.perm[[target_variable]] <- try(data.perm[[target_variable]] + Xbeta0[index.perm,,drop=FALSE], silent = TRUE)
  # if(inherits(data.perm[[target_variable]],"try-error")){
  #   browser()
  # }
  
  # compute new models using permuted observations
  # browser()
  formula1 <- formula(lme0.ML)
  formula2 <- formula(lme1.ML)
  lme0.permML <- try(update(lme0.ML, data = data.perm, method = "ML"), silent = TRUE)
  lme1.permML <- try(update(lme1.ML, data = data.perm, method = "ML"), silent = TRUE)
  if(inherits(lme0.permML,"try-error")||inherits(lme1.permML,"try-error")){
    LRT.stat <- NA
  }else{
    LRT.stat <- as.double(2*(logLik(lme1.permML)-logLik(lme0.permML)))
  }
  
  return(LRT.stat)
}
