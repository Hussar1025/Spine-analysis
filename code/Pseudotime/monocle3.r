library(monocle3)
library(Matrix)
library(readr)
library(dplyr)
library(monocle3)
library(Matrix)

out_dir <- "yourpath"

counts <- read.csv(
  file.path(out_dir, "yourdata"),
  row.names = 1,
  check.names = FALSE
)

cell_metadata <- read.csv(
  file.path(out_dir, "yourdata"),
  row.names = 1,
  check.names = FALSE
)

gene_metadata <- read.csv(
  file.path(out_dir, "yourdata"),
  row.names = 1,
  check.names = FALSE
)

counts_mat <- as(as.matrix(counts), "sparseMatrix")

cds <- new_cell_data_set(
  expression_data = counts_mat,
  cell_metadata = cell_metadata,
  gene_metadata = gene_metadata
)

cds <- preprocess_cds(cds, num_dim = 50)
cds <- reduce_dimension(cds, reduction_method = "UMAP")
cds <- cluster_cells(cds)
cds <- learn_graph(cds)

cds <- order_cells(cds)

pseudotime_df <- data.frame(
  cell_id = colnames(cds),
  pseudotime = pseudotime(cds),
  monocle_partition = partitions(cds),
  monocle_cluster = clusters(cds)
)

write.csv(
  pseudotime_df,
  file.path(out_dir, "yourdata"),
  row.names = FALSE
)

dynamic_genes <- graph_test(
  cds,
  neighbor_graph = "principal_graph",
  cores = 4
)

dynamic_genes <- dynamic_genes[order(dynamic_genes$q_value), ]

write.csv(
  dynamic_genes,
  file.path(out_dir, "yourdata")
)

safe_save_monocle <- function(cds, save_name = "yourdata") {
  
  safe_out_dir <- "yourpath"
  
  if (!dir.exists(safe_out_dir)) {
    dir.create(safe_out_dir, recursive = TRUE)
  }
  
  time_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")
  target_dir <- file.path(safe_out_dir, paste0(save_name, "_", time_tag))

  if (dir.exists(target_dir)) {
    unlink(target_dir, recursive = TRUE, force = TRUE)
  }
  
  message("Saving Monocle3 object to: ", target_dir)
  
  tryCatch({
    save_monocle_objects(cds, target_dir)
    message("save_monocle_objects saved successfully.")
  }, error = function(e) {
    message("save_monocle_objects failed.")
    message("Error message: ", e$message)

    fallback_rds <- file.path(safe_out_dir, paste0(save_name, "_fallback_", time_tag, ".rds"))
    saveRDS(cds, fallback_rds)
    message("Fallback saveRDS saved successfully: ", fallback_rds)
  })
}

safe_save_monocle(cds)