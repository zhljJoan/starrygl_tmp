# Execution Spine

Module-local contracts are authoritative. Start with:

- [Package contract index](../../../src/starrygl/CONTRACT.md)
- [Runtime Stage A/B/C](../../../src/starrygl/runtime/CONTRACT.md)
- [DataLoader queues](../../../src/starrygl/runtime/dataloader/CONTRACT.md)
- [T-CSR sampling](../../../src/starrygl/runtime/sample/CONTRACT.md)
- [Snapshot-CSC](../../../src/starrygl/runtime/snapshot/CONTRACT.md)
- [State access and commit](../../../src/starrygl/runtime/state/CONTRACT.md)

The canonical flow is:

```text
window row -> prepared task slice -> negatives -> bound graph accessor
  -> request_queue -> materialize and await D_remote -> ready_queue
  -> model -> task -> backward -> state commit
```

This file is an index only. Do not duplicate module contracts here.

