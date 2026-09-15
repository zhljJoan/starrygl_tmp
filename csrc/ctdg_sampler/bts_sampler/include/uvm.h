#pragma once

#include "extension.h"

at::Storage uvm_storage_new(const std::size_t size_bytes, const at::DeviceIndex cuda_device);
at::Storage uvm_storage_to_cuda(const at::Storage& storage, const at::DeviceIndex cuda_device);
at::Storage uvm_storage_to_cpu(const at::Storage& storage);

void uvm_storage_advise(const at::Storage& storage, const cudaMemoryAdvise advise);
void uvm_storage_prefetch(const at::Storage& storage);