#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#include <cuda.h>

#define STRINGIFY(x) #x
#define SYMBOL_STRING(x) STRINGIFY(x)

enum HookSymbols {
    CU_GET_PROC_ADDRESS,
    CU_MEM_ALLOC,
    CU_MEM_ALLOC_HOST,
    CU_MEM_ALLOC_MANAGED,
    CU_MEM_ALLOC_PITCH,
    CU_MEM_FREE,
    CU_MEM_FREE_HOST,
    CU_MEM_HOST_ALLOC,
    CU_MEMCPY,
    CU_MEMCPY_ASYNC,
    CU_MEMCPY_DTOD,
    CU_MEMCPY_DTOD_ASYNC,
    CU_MEMCPY_DTOH,
    CU_MEMCPY_DTOH_ASYNC,
    CU_MEMCPY_HTOD,
    CU_MEMCPY_HTOD_ASYNC,
    CU_MEMCPY_PEER,
    CU_MEMCPY_PEER_ASYNC,
    CU_LAUNCH_COOPERATIVE_KERNEL,
    CU_LAUNCH_KERNEL,
    NUM_HOOK_SYMBOLS
};

struct cudaHookInfo {
    int debug_mode;
    void *func_prehook[NUM_HOOK_SYMBOLS];
    // void *func_proxy[NUM_HOOK_SYMBOLS];
    void *func_actual[NUM_HOOK_SYMBOLS];
    void *func_posthook[NUM_HOOK_SYMBOLS];

    cudaHookInfo() {
        debug_mode = 0;
#ifdef _DEBUG
        debug_mode = 1;
#endif
    }
};
CUresult CUDAAPI cuMemAlloc_hook( CUdeviceptr* dptr, size_t bytesize) {
  cuMemAlloc(dptr,bytesize); 
  printf("allocate %zu bytes.\n", bytesize);

  return CUDA_SUCCESS;

}
static struct cudaHookInfo cuda_hook_info;

extern "C" {
void *__libc_dlsym(void *map, const char *name);
void *__libc_dlopen_mode(const char *name, int mode);
}

void *libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);
void *libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
void *libcudnnHandle = __libc_dlopen_mode("libcudnn.so", RTLD_LAZY);

void *actualDlsym(void *handle, const char *symbol) {
    typedef void *(*fnDlsym)(void *, const char *);
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(libdlHandle, "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

void *dlsym(void *handle, const char *symbol) {
#ifdef _DEBUG
    //printf("dlsym() hook %s\n", symbol);
#endif
    if (strncmp(symbol, "cu", 2) != 0) {
        return actualDlsym(handle, symbol);
    }

    if (strcmp(symbol, SYMBOL_STRING(cuGetProcAddress)) == 0) {
        return (void *)(&cuGetProcAddress);
    }

    return actualDlsym(handle, symbol);
}


CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
#ifdef _DEBUG
    //printf("Enter %s\n", SYMBOL_STRING(cuGetProcAddress));
    //printf("symbol %s, cudaVersion %d, flags %lu\n", symbol, cudaVersion, flags);
#endif
    printf("Enter %s\n", SYMBOL_STRING(cuGetProcAddress));
    printf("symbol %s, cudaVersion %d, flags %lu\n", symbol, cudaVersion, flags);
    typedef decltype(&cuGetProcAddress) funcType;
    funcType actualFunc;
    if(!cuda_hook_info.func_actual[CU_GET_PROC_ADDRESS])
        actualFunc = (funcType)actualDlsym(libcudaHandle, SYMBOL_STRING(cuGetProcAddress));
    else
        actualFunc = (funcType)cuda_hook_info.func_actual[CU_GET_PROC_ADDRESS];
    CUresult result = actualFunc(symbol, pfn, cudaVersion, flags);

    if(strcmp(symbol, SYMBOL_STRING(cuGetProcAddress)) == 0) {
        cuda_hook_info.func_actual[CU_GET_PROC_ADDRESS] = *pfn;
        *pfn = (void*)(&cuGetProcAddress);

#pragma push_macro("cuMemAlloc")
#undef cuMemAlloc
    } else if (strcmp(symbol, SYMBOL_STRING(cuMemAlloc)) == 0) {
#pragma pop_macro("cuMemAlloc")
        cuda_hook_info.func_actual[CU_MEM_ALLOC] = *pfn;
        *pfn = (void *)(&cuMemAlloc_hook);

#pragma push_macro("cuMemAllocManaged")
#undef cuMemAllocManaged
    } else if (strcmp(symbol, SYMBOL_STRING(cuMemAllocManaged)) == 0) {
#pragma pop_macro("cuMemAllocManaged")
        cuda_hook_info.func_actual[CU_MEM_ALLOC_MANAGED] = *pfn;
        *pfn = (void *)(&cuMemAllocManaged);

#pragma push_macro("cuMemFree")
#undef cuMemFree
    } else if (strcmp(symbol, SYMBOL_STRING(cuMemFree)) == 0) {
#pragma pop_macro("cuMemFree")
        cuda_hook_info.func_actual[CU_MEM_FREE] = *pfn;
        *pfn = (void *)(&cuMemFree);
    }

#ifdef _DEBUG
    //printf("Leave %s\n", SYMBOL_STRING(cuGetProcAddress));
#endif
    return (result);
}

