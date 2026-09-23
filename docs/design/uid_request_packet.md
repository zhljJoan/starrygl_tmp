# UID Request Packet

The canonical runtime remains:

`window row -> prepared slice -> negatives -> graph access -> Batch -> dependencies -> model -> task -> state update`

The specialization boundary is dependency access. Node feature, memory,
mailbox and recurrent-state providers share one UID owner-request operation;
their response tensors remain provider-specific. Edge features keep the normal
EID route because padding by the global edge count is not economical.

For a UID request, each destination receives one fixed-capacity row:

`[count, uid_0, ..., uid_(count-1), padding]`

Capacity is the global node count, so no overflow fallback or rank-local
decision is required. One equal-split all-to-all replaces the count collective
followed by the UID collective. The received count header supplies response
split sizes; tensor masking compacts received IDs. Every rank enters exactly
one request collective, including empty requests.

This uses existing Torch tensor operators and `CommScheduler`. DGL has no
corresponding owner-route primitive. A custom C++/CUDA kernel cannot eliminate
the network phase, and is outside the current kernel constraint. The cost is
padding proportional to `world_size * num_nodes`; acceptance therefore depends
on the four-GPU trace and it must not be generalized to EID without evidence.
