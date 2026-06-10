#!/usr/bin/env Rscript
# Memory-efficient conversion of large all_pitches.rds
# Subsets columns before filtering to keep memory low.

args <- commandArgs(trailingOnly = TRUE)
input_rds  <- args[1]
output_csv <- args[2]

cat("Loading", input_rds, "...\n")
df <- readRDS(input_rds)
cat("Loaded", nrow(df), "rows, dropping unused columns first...\n")

# Keep only the columns we need from the very start to free memory
keep_cols <- c("Season","batter","pitcher","events","stand","description",
               "launch_speed","launch_angle","hc_x","hc_y","home_team")
df <- df[, intersect(keep_cols, names(df))]
gc()
cat("Mem trimmed. Filtering to BIP only...\n")

df <- df[df$description %in% c("hit_into_play","hit_into_play_no_out","hit_into_play_score"), ]
cat("After BIP filter:", nrow(df), "rows\n")
gc()

df$x <- df$hc_x - 125.42
df$y <- 198.27 - df$hc_y
df$spray_angle <- atan(df$x / df$y) * 180 / pi
df$adjusted_angle <- ifelse(df$stand == "L", -df$spray_angle, df$spray_angle)
df <- df[, c("Season","batter","pitcher","events","stand",
             "launch_speed","launch_angle","adjusted_angle","home_team")]
df <- unique(df)
cat("After dedup:", nrow(df), "rows\n")

write.csv(df, output_csv, row.names = FALSE)
cat("Saved", output_csv, "\n")
