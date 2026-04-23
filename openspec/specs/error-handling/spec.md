# Error Handling & Resource Management

> **Version**: v1.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2025-04-16

## Requirement: Robust Error Handling
**Name**: error-handling
**Text**: Provide robust error handling and resource management for safe usage in production code.

### Scenario: CUDA Allocation Failure
**WHEN** CUDA memory allocation fails
**THEN** should return descriptive error code and release any allocated resources

### Scenario: Kernel Launch Failure
**WHEN** kernel launch fails
**THEN** should capture CUDA error and propagate to caller

### Scenario: Async Error Handling
**WHEN** SpMV operation completes
**THEN** should synchronize properly and check for asynchronous errors

### Scenario: RAII Resource Management
**WHEN** using GPU memory allocation
**THEN** should provide RAII-style resource management for automatic cleanup

### Scenario: Input Validation
**WHEN** given invalid matrix dimensions or mismatched vector sizes
**THEN** should validate inputs before GPU operations and return appropriate error codes

---

## Error Code Enum

```cpp
enum class SpMVError {
    SUCCESS = 0,              // Operation successful
    INVALID_DIMENSION = -1,   // Matrix or vector dimension mismatch
    CUDA_MALLOC = -2,         // GPU memory allocation failed
    CUDA_MEMCPY = -3,         // GPU memory copy failed
    KERNEL_LAUNCH = -4,       // CUDA kernel launch/execution failed
    INVALID_FORMAT = -5,      // Invalid sparse matrix format
    FILE_IO = -6,             // File read/write error
    OUT_OF_MEMORY = -7,       // Host/device out of memory
    INVALID_ARGUMENT = -8     // Invalid argument provided
};

const char* spmv_error_string(SpMVError err);
```

## CUDA Check Macros

```cpp
#define CUDA_CHECK_MALLOC(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return static_cast<int>(SpMVError::CUDA_MALLOC); \
    } \
} while(0)

#define CUDA_CHECK_MEMCPY(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return static_cast<int>(SpMVError::CUDA_MEMCPY); \
    } \
} while(0)

// Backward compatible alias
#define CUDA_CHECK(call) CUDA_CHECK_MALLOC(call)
```

## RAII Template

```cpp
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();  // Automatically frees GPU memory

    // Non-copyable
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Movable
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    // Accessors
    T* get();
    const T* get() const;
    size_t size() const;

    // Memory operations
    void copyFromHost(const T* host_ptr, size_t count);
    void copyToHost(T* host_ptr, size_t count);
    void memset(int value);
    void fill(const T& value);
};
```

## Memory Ownership

All matrix structures track memory ownership:

```cpp
struct CSRMatrix {
    // ... data pointers ...
    bool owns_host_memory;     // Free host memory on destroy?
    bool owns_device_memory;   // Free device memory on destroy?
};
```

**Guidelines:**
- Use `*_create()` and `*_destroy()` for lifecycle management
- Use `CudaBuffer<T>` for automatic GPU memory management
- Never use raw `cudaMalloc`/`cudaFree` in new code

## Test Coverage

All property tests validate error handling as part of their execution.

## See Also

- [Public API](../public-api/spec.md) - API error conventions
