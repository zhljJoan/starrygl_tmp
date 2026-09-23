# Dependency Route Pipeline

The shared Event/Snapshot path remains:

`window -> sample/materialize -> Batch -> dependency access -> model -> task -> state update`

Feature, state and mailbox payloads stay provider-specific. Their distributed
request keys use only two route kinds: node IDs (UID) and edge IDs (EID). A
dynamic route has three ordered phases under the existing `CommScheduler`:

1. compute owner order and per-peer counts with tensor operators;
2. launch the fixed-size count collective and retain its handle;
3. at the dependency boundary, finish counts, launch UID/EID payload exchange,
   then launch provider responses.

Every rank enters the same phases, including empty payloads. No global barrier,
new process group, P2P protocol, DGL reconstruction, or custom kernel is added.
The first migration unit only exposes the count handle while preserving the
old blocking wrapper. Moving that handle one loader slot ahead is a separate
performance change and must be accepted by a four-GPU trace before replacing
the retained path.

`OwnerRequest` records its exact requested IDs. Reuse between node features and
state is valid only when the requested UID sequence is identical; multi-layer
state must otherwise issue its own scheduled UID route. Cache policy changes
where a dependency can be read, not route ordering or ownership.
