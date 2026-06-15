#pragma once

// TurboQuant InnerQ per-channel equalization — cross-TU shared state
// All state now lives in ggml-turbo-quant.c (ggml-base) so both
// llama.dll and ggml-cuda.dll can link to them.

#define INNERQ_MAX_CHANNELS 128

#ifdef __cplusplus
extern "C" {
#endif

// Host-side shared state (defined in ggml-turbo-quant.c, exported via GGML_API)
extern int   g_innerq_finalized;
extern float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];

// Called from set-rows.cu after InnerQ finalization to publish scale_inv
void turbo_innerq_publish(const float * scale_inv, int group_size);

// Called from llama-kv-cache.cpp (or equivalent) to check if tensor needs update
// Returns nonzero if there are new scale_inv values to upload
int turbo_innerq_needs_tensor_update(void);

// Called after tensor update to clear the flag
void turbo_innerq_mark_tensor_updated(void);

#ifdef __cplusplus
}
#endif
