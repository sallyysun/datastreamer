# Circular buffer backed by a pre-allocated matrix.
# O(1) push, O(1) random access, fixed memory.

new_ring_buffer <- function(capacity, ncol = 2L) {
  env <- new.env(parent = emptyenv())
  env$capacity <- as.integer(capacity)
  env$ncol     <- as.integer(ncol)
  env$data     <- matrix(NA_real_, nrow = capacity, ncol = ncol)
  env$head     <- 0L
  env
}

ring_push <- function(buf, row) {
  idx <- (buf$head %% buf$capacity) + 1L
  buf$data[idx, ] <- row
  buf$head <- buf$head + 1L
  invisible(buf)
}

ring_size <- function(buf) {
  min(buf$head, buf$capacity)
}

ring_get <- function(buf, i) {
  buf$data[((i - 1L) %% buf$capacity) + 1L, ]
}

ring_tail <- function(buf, n = NULL) {
  sz <- ring_size(buf)
  if (is.null(n)) n <- sz
  n <- min(n, sz)
  if (n == 0L) return(matrix(NA_real_, nrow = 0, ncol = buf$ncol))

  start <- buf$head - n + 1L
  indices <- seq.int(start, buf$head)
  idx <- ((indices - 1L) %% buf$capacity) + 1L
  buf$data[idx, , drop = FALSE]
}
