# Load required packages
library(jsonlite)
library(readr)
library(dplyr)

# set working directory
setwd("/indirect/staff/anderssolsen/p3_spectral")

# Source the R function
source("spectral_analysis/scripts_spectral_analysis/Rfunction_permmaxT.R")

# Assume your R script defines: main <- function(df, target_variable, covariates, nuisance_regressors, uncontrolled_variable, controlled_variable, nperm) { ... }

# Load configuration
config <- fromJSON("config.json")

denoising_strategies <- config$strategies
covariates <- config$covariates
nuisance_regressors <- config$nuisance_regressors
nuisance_regressors_nomotion <- config$nuisance_regressors_nomotion
  
# Loop through permutation setting (only True in your Python code)
for (do_perm in c(TRUE)) {
  if (do_perm) {
    add_perm <- "_perm"
    nperm <- config$num_permutations
  } else {
    add_perm <- ""
    nperm <- 0
  }
  
  # # Loop through strategies
  # for (strategy in denoising_strategies) {
  #   if (!strategy %in% c('high-pass-only','high-pass-motion','acompcor','9p')) {
  #     next
  #   }
  #   cat("Running stats for strategy:", strategy, "\n")
  #   #################### log-power, frequency, network #######################
  #   # Read data
  #   df_frequencies_networks <- read_csv(
  #     paste0("data/results/spectra_by_frequency_network_agg_", strategy, ".csv")
  #   )
  # 
  #   # Rename column if needed
  #   if ("PPL_mcg/L" %in% names(df_frequencies_networks)) {
  #     df_frequencies_networks <- df_frequencies_networks %>%
  #       rename(PPL_mcg_L = `PPL_mcg/L`)
  #   }
  #   # rename "ratio_outliers_fd0.5_std_dvars1.5" to "ratio_outliers_fd0_5_std_dvars1_5" if needed
  #   if ("ratio_outliers_fd0.5_std_dvars1000" %in% names(df_frequencies_networks)) {
  #     df_frequencies_networks <- df_frequencies_networks %>%
  #       rename(ratio_outliers_fd0_5_std_dvars1000 = `ratio_outliers_fd0.5_std_dvars1000`)
  #   }
  # 
  #   target_variable <- config$target_variable
  #   uncontrolled_variable <- "network"
  #   controlled_variable <- "frequency"
  # 
  #   # Call the R function 'main'
  #   result_r <- main(
  #     df_frequencies_networks,
  #     target_variable,
  #     covariates,
  #     nuisance_regressors,
  #     uncontrolled_variable,
  #     controlled_variable,
  #     nperm = nperm
  #   )
  # 
  #   # Save output
  #   out_path <- paste0("data/results/logpower_stats_by_frequency_network_", strategy, add_perm, ".csv")
  #   write_csv(result_r, out_path, na = "NaN")
  # 
  #   # Call the R function 'main'
  #   result_r <- main(
  #     df_frequencies_networks,
  #     target_variable,
  #     covariates,
  #     nuisance_regressors_nomotion,
  #     uncontrolled_variable,
  #     controlled_variable,
  #     nperm = nperm
  #   )
  # 
  #   # Save output
  #   out_path <- paste0("data/results/logpower_stats_by_frequency_network_nomotion_", strategy, add_perm, ".csv")
  #   write_csv(result_r, out_path, na = "NaN")
  # }

  # Loop through strategies
  for (strategy in denoising_strategies) {
    if (!strategy %in% c('high-pass-only','high-pass-motion','acompcor','9p')) {
      next
    }
    cat("Running stats for strategy:", strategy, "\n")
    #################### log-power, band, parcel #######################
    # Read data
    df_bands_parcels <- read_csv(
      paste0("data/results/spectra_by_band_parcel_", strategy, ".csv")
    )

    # Rename column if needed
    if ("PPL_mcg/L" %in% names(df_bands_parcels)) {
      df_bands_parcels <- df_bands_parcels %>%
        rename(PPL_mcg_L = `PPL_mcg/L`)
    }
    # rename "ratio_outliers_fd0.5_std_dvars1000" to "ratio_outliers_fd0_5_std_dvars1000" if needed
    if ("ratio_outliers_fd0.5_std_dvars1000" %in% names(df_bands_parcels)) {
      df_bands_parcels <- df_bands_parcels %>%
        rename(ratio_outliers_fd0_5_std_dvars1000 = `ratio_outliers_fd0.5_std_dvars1000`)
    }

    target_variable <- config$target_variable
    uncontrolled_variable <- "band"
    controlled_variable <- "roi"

    # Call the R function 'main'
    result_r <- main(
      df_bands_parcels,
      target_variable,
      covariates,
      nuisance_regressors,
      uncontrolled_variable,
      controlled_variable,
      nperm = nperm
    )

    # Save output
    out_path <- paste0("data/results/logpower_stats_by_band_parcel_", strategy, add_perm, ".csv")
    write_csv(result_r, out_path, na = "NaN")
  }

  # Loop through strategies
  for (strategy in denoising_strategies) {
    if (!strategy %in% c('high-pass-only','high-pass-motion','acompcor','9p')) {
      next
    }
    cat("Running stats for strategy:", strategy, "\n")
    #################### entropy, parcel #######################
    # Read data
    df_entropy <- read_csv(
      paste0("data/results/entropy_by_parcel_", strategy, ".csv")
    )
    df_entropy$ones <- 1  # add a column of ones for uncontrolled variable

    # Rename column if needed
    if ("PPL_mcg/L" %in% names(df_entropy)) {
      df_entropy <- df_entropy %>%
        rename(PPL_mcg_L = `PPL_mcg/L`)
    }
    # rename "ratio_outliers_fd0.5_std_dvars1000" to "ratio_outliers_fd0_5_std_dvars1000" if needed
    if ("ratio_outliers_fd0.5_std_dvars1000" %in% names(df_entropy)) {
      df_entropy <- df_entropy %>%
        rename(ratio_outliers_fd0_5_std_dvars1000 = `ratio_outliers_fd0.5_std_dvars1000`)
    }

    target_variable <- "entropy"
    uncontrolled_variable <- "ones"
    controlled_variable <- "roi"

    # Call the R function 'main'
    result_r <- main(
      df_entropy,
      target_variable,
      covariates,
      nuisance_regressors,
      uncontrolled_variable,
      controlled_variable,
      nperm = nperm
    )

    # Save output
    out_path <- paste0("data/results/entropy_stats_by_parcel_", strategy, add_perm, ".csv")
    write_csv(result_r, out_path, na = "NaN")
  }
}