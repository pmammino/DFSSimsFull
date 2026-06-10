#!/usr/bin/env Rscript
# convert_rds_to_csv.R
# ====================
# Convert a Statcast RDS file (e.g., all_2024.rds, all_pitches.rds) into a
# slim BIP-only CSV ready for the Python pipeline.
#
# Usage:
#   Rscript convert_rds_to_csv.R input.rds output.csv [SEASON]
#
# If SEASON is supplied, all rows are tagged with that season (used for
# single-season files like all_2024.rds). If omitted, the script assumes
# the RDS already contains a Season column (e.g., all_pitches.rds).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript convert_rds_to_csv.R input.rds output.csv [SEASON]")
}
input_rds  <- args[1]
output_csv <- args[2]
season_arg <- if (length(args) >= 3) as.integer(args[3]) else NA

cat("Loading", input_rds, "...\n")
df <- readRDS(input_rds)
cat("Loaded", nrow(df), "rows\n")

# Filter to BIP only
df <- df[df$description %in% c("hit_into_play",
                                "hit_into_play_no_out",
                                "hit_into_play_score"), ]
cat("After BIP filter:", nrow(df), "rows\n")

keep <- c("batter", "pitcher", "events", "stand",
          "launch_speed", "launch_angle", "hc_x", "hc_y", "home_team")
df <- df[, keep]

if (!is.na(season_arg)) {
  df$Season <- season_arg
} else if (!"Season" %in% names(df)) {
  stop("Input has no 'Season' column and no SEASON argument provided.")
}

# Spray angle: home plate at origin, then angle from y-axis
df$x <- df$hc_x - 125.42
df$y <- 198.27 - df$hc_y
df$spray_angle <- atan(df$x / df$y) * 180 / pi
df$adjusted_angle <- ifelse(df$stand == "L", -df$spray_angle, df$spray_angle)

df <- df[, c("Season", "batter", "pitcher", "events", "stand",
             "launch_speed", "launch_angle", "adjusted_angle", "home_team")]
df <- unique(df)
cat("After dedup:", nrow(df), "rows\n")

write.csv(df, output_csv, row.names = FALSE)
cat("Saved", output_csv, "\n")
