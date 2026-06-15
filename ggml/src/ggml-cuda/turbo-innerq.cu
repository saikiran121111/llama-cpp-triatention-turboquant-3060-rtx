#include "turbo-innerq.cuh"

// Host-side shared state for InnerQ cross-TU communication
// Definitions now live in ggml-turbo-quant.c (ggml-base) so both
// llama.dll and ggml-cuda.dll can link to them.
// Declarations are provided by turbo-innerq.cuh.
