# Appendix

## Glossary

| Term | Description |
|---|---|
| FSDP2 | PyTorch's Fully Sharded Data Parallel implementation based on DTensor; parameters, gradients, and optimizer states are all sharded according to the DeviceMesh |
| TP | Tensor parallelism, which shards the weight matrices within a single layer across multiple devices by column/row |
| EP | Expert parallelism for MoE models, which shards experts across multiple devices along the expert dimension |
| CP | Context parallelism, which shards the sequence dimension across multiple devices to support extremely long sequence training |
| Ulysses | A context parallelism approach based on sharding along the attention head dimension |
| EFSDP | Expert FSDP2, which further fully shards the expert parameters after EP sharding |
| DeviceMesh | PyTorch device mesh that describes the multi-dimensional parallelism topology (such as `dp × fsdp × tp × ep`) |
| Dispatcher | The expert computation scheduler in expert parallelism, which determines how token dispatch and expert computation are organized |
| Recompute | Gradient checkpointing technique that does not save activations during the forward pass and recomputes them during the backward pass to reduce memory usage |
| Chunk Loss | Processes loss computation in a chunked pipeline to avoid fully materializing logits for large-vocabulary scenarios |
| MXFP8 | Microscaling FP8 quantization format supporting variants such as E4M3/E5M2/HIF8 (hifloat8) |
| MC2 | Communication-computation fusion acceleration operator library provided by Ascend, used to accelerate expert computation on the EP path |
