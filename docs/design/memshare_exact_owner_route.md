# MemShare Exact Owner Route

The common runtime spine stays unchanged through `Batch -> dependency access`.
The dependency provider is the specialization boundary.

MemShare/master computes exact per-owner counts from sorted UID/EID partitions,
exchanges those counts synchronously, allocates exact receive buffers, exchanges
IDs, gathers owner-local values, and launches the response payload
asynchronously. There is no fixed-capacity padding and count completion is not
hidden behind another collective.

StarryGL already follows that exact dynamic request route. Node feature,
memory, and mailbox reads may reuse one UID request; edge features use their
own EID request. For historical memory/mailbox responses, the four homogeneous
floating tensors are flattened and concatenated once, sent through one existing
asynchronous response route, and viewed back into their original shapes when
the batch is consumed. Ownership, cache freshness, state commit, negative
sampling, model math, and collective ordering do not change.

Torch `cat` and views match MemShare's layout. DGL has no applicable route
packing operator, and no custom C++/CUDA kernel is needed. Mixed-dtype combined
state is rejected at the provider boundary instead of using raw-byte packing;
the current TGN memory, mailbox, and timestamps are homogeneous floating data.
