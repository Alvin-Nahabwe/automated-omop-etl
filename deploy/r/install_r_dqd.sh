#!/bin/sh
# Installs R, Java and the OHDSI packages (DatabaseConnector, Achilles, DataQualityDashboard) for
# omop-agent's quality.use_r_dqd. Used by Dockerfile.r; also works as root on an Ubuntu 24.04
# server. Tested 2026-10-02: R 4.3.3, OpenJDK 21, DatabaseConnector 8.0.0, SqlRender 1.19.7,
# ParallelLogger 3.5.1, Achilles 1.7.2, DataQualityDashboard 2.8.9. Takes about 3 minutes.
set -eu
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  r-base-core r-base-dev default-jdk-headless libpostgresql-jdbc-java \
  r-cran-rjava r-cran-dplyr r-cran-jsonlite r-cran-readr r-cran-tidyr r-cran-stringr \
  r-cran-lubridate r-cran-checkmate r-cran-rlang r-cran-digest r-cran-r6 r-cran-dbi \
  r-cran-bit64 r-cran-dbplyr r-cran-urltools r-cran-snow r-cran-xml2 r-cran-rjson \
  r-cran-data.table r-cran-plyr r-cran-magrittr r-cran-tidyselect r-cran-purrr r-cran-vroom \
  r-cran-tibble r-cran-glue r-cran-cli r-cran-vctrs r-cran-pillar r-cran-httr r-cran-curl \
  r-cran-openssl r-cran-triebeard r-cran-stringi r-cran-backports r-cran-lifecycle \
  r-cran-generics r-cran-withr r-cran-cpp11 r-cran-crayon r-cran-hms r-cran-tzdb r-cran-clipr \
  r-cran-bit r-cran-blob r-cran-progress r-cran-timechange r-cran-zip \
  r-cran-lattice r-cran-quadprog r-cran-zoo r-cran-xts r-cran-ttr r-cran-quantmod \
  r-cran-tseries r-cran-rstudioapi

R CMD javareconf

# DatabaseConnector 8.0.0 needs dbplyr >= 2.6.0; dbplyr 2.6.0 calls dplyr::filter_out (dplyr >= 1.2,
# not declared in its DESCRIPTION); dplyr 1.2.1 needs rlang >= 1.1.7, vctrs >= 0.7.1,
# lifecycle >= 1.0.5. install.packages() does not upgrade outdated dependencies, so list them.
Rscript -e '
options(repos = c(CRAN = "https://cloud.r-project.org"))
pkgs <- c("rlang", "lifecycle", "vctrs", "tidyselect", "dplyr", "dbplyr", "memuse",
          "SqlRender", "DatabaseConnector", "ParallelLogger", "Achilles", "DataQualityDashboard")
install.packages(pkgs, lib = "/usr/local/lib/R/site-library", Ncpus = 3,
                 dependencies = c("Depends", "Imports", "LinkingTo"))
stopifnot(all(vapply(c("DatabaseConnector", "Achilles", "DataQualityDashboard"),
                     requireNamespace, logical(1), quietly = TRUE)))'

# DatabaseConnector wants exactly ONE file matching ^postgresql-.*\.jar$ in the folder:
# /usr/share/java itself fails (postgresql-jdbc3.jar and postgresql-jdbc4.jar also match ->
# "Error in if (key %in% ls(jdbcDrivers)): the condition has length > 1").
mkdir -p /opt/jdbc
ln -sf "$(readlink -f /usr/share/java/postgresql.jar)" /opt/jdbc/
echo 'R ready: set DATABASECONNECTOR_JAR_FOLDER=/opt/jdbc'
