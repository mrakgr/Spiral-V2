// Basic reverse mode AD on the GPU. This v2 of Spiral is focused on convolutional operations.

module Spiral

#if INTERACTIVE
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/ManagedCuda.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/NVRTC.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaBlas.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaRand.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/NPP.dll"
#r "../packages/ManagedCuda-CudaDNN.3.0/lib/net45/CudaDNN.dll"
#endif

// Open up the namespaces.
open ManagedCuda
open ManagedCuda.BasicTypes
open ManagedCuda.VectorTypes
open ManagedCuda.CudaBlas
open ManagedCuda.CudaRand
open ManagedCuda.NVRTC
open ManagedCuda.CudaDNN

open System
open System.Collections.Generic

// Initialize the context. Analogous to a CPU process. Cuda tries to offload as much as possible during context creation so there aren't
// any unexpected delays later.
let ctx = new CudaContext()
let numSm = ctx.GetDeviceInfo().MultiProcessorCount // The number of streaming multiprocessors on the device.

// Make a stream class.
let str = new CudaStream()
// Set the Cuda libraries handles to the above stream.
let cublas = CudaBlas(str.Stream,PointerMode.Host,AtomicsMode.Allowed) // Better performance for some solver functions with atomics allowed. The Spiral library does not use them though.
let cudnn = new CudaDNN.CudaDNNContext()
cudnn.SetStream(str)
let cudaRandom = new CudaRand.CudaRandDevice(GeneratorType.PseudoDefault)
cudaRandom.SetStream(str.Stream)

// I'll skip aliasing float32 to floatType for this iteration of the library. There is not point to it as Cuda native functions cannot be overloaded this way.

// Helper functions
/// Copies a host array to device.
let inline to_dev (host_ar: 't []) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

/// Copies a device array to host.
let inline to_host (dev_ar: CudaDeviceVariable<'t>) =
    let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
    dev_ar.CopyToHost(h_a)
    h_a

/// Copies the device array to host. Extends the CudaDeviceVariable class.
type CudaDeviceVariable<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> with
    member inline this.Gather() =
        to_host this

/// Allocates a new device array without initializing it.
let inline new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
    new CudaDeviceVariable<'t>(SizeT n)

/// The float scalar type
type Df = 
    {
    P : float32 ref // primal
    A : float32 ref // adjoint
    }

    static member inline create P =
        {P=P;A=ref 0.0f}

/// The main matrix type.
type d4M =
    {
    mutable num_images:int
    mutable num_channels:int
    mutable num_rows:int
    mutable num_cols:int
    mutable P: CudaDeviceVariable<float32> // primal
    mutable A: CudaDeviceVariable<float32> option // adjoint
    }  

    /// Throws an exception if it tries to allocate an array of size 0.
    static member create(num_images: int, num_channels:int, num_rows: int, num_cols: int) =
        let size = num_images*num_channels*num_rows*num_cols
        let p = new_dev size
        let a = new_dev size |> Some
        {num_images=num_images; num_channels=num_channels; num_rows=num_rows; num_cols=num_cols; P=p; A=a}

    /// Copies a host to a device array.
    /// Throws an exception if it tries to allocate an array of size 0.
    static member create(num_images: int, num_channels:int, num_rows: int, num_cols: int, dArray: float32[]) =
        let size = num_images*num_channels*num_rows*num_cols
        if dArray.Length <> size then failwith "Invalid size in dMatrix construction."
        let p = to_dev dArray
        let a = new_dev size |> Some
        {num_images=num_images; num_channels=num_channels; num_rows=num_rows; num_cols=num_cols; P=p; A=a}

    /// Throws an exception if it tries to allocate an array of size 0.
    /// Does not allocate the adjoint array.
    static member inline createConstant(num_images: int, num_channels:int, num_rows: int, num_cols: int) =
        let size = num_images*num_channels*num_rows*num_cols
        let p = new_dev size
        let a = None
        {num_images=num_images; num_channels=num_channels; num_rows=num_rows; num_cols=num_cols; P=p; A=a}

    /// Copies a host to a device array.
    /// Throws an exception if it tries to allocate an array of size 0.
    /// Does not allocate the adjoint array.
    static member createConstant(num_images: int, num_channels:int, num_rows: int, num_cols: int, dArray: float32[]) =
        let size = num_images*num_channels*num_rows*num_cols
        if dArray.Length <> size then failwith "Invalid size in dMatrix construction."
        let p = to_dev dArray
        let a = None
        {num_images=num_images; num_channels=num_channels; num_rows=num_rows; num_cols=num_cols; P=p; A=a}

    /// Returns a new instance of an empty dMatrix.
    /// Unlike the let statements, the member statements are always reevaluated.
    static member createEmpty = {num_images=0; num_channels=0; num_rows=0; num_cols=0;P=CudaDeviceVariable.Null; A=Some CudaDeviceVariable.Null}

    /// Returns a new instance of an empty dMatrix without the adjoint allocated.
    /// Unlike the let statements, the member statements are always reevaluated.
    static member createEmptyConstant = {num_images=0; num_channels=0; num_rows=0; num_cols=0;P=CudaDeviceVariable.Null; A=None}

    /// Returns nhwc as a tuple
    member inline t.nchw = t.num_images, t.num_channels, t.num_rows, t.num_cols
    /// Returns the n*h*w*c
    member inline t.size = t.num_images * t.num_channels * t.num_rows * t.num_cols

    /// Returns the nchw, primal tuple.
    member inline t.P' = t.nchw, t.P
    /// Returns the nchw, adjoint tuple. Throws an exception if the adjoint is None.
    member inline t.A' = t.nchw, t.A.Value

    /// Sets the primal to zero.
    member inline t.setZeroPrimal() = t.P.MemsetAsync(0u,str.Stream)
    /// Sets the adjoint to zero.
    member inline t.setZeroAdjoint() = 
        match t.A with
        | Some A -> A.MemsetAsync(0u,str.Stream)
        | None -> ()

    /// Set the matrix to a value.
    member inline t.setPrimal (x: float32) = 
        let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
        t.P.MemsetAsync(v,str.Stream)
    /// Creates a copy of this matrix with all the values set to zero.
    member inline t.zeroLike() =
        match t.A with
        | Some A -> 
            d4M.create(t.num_images,t.num_rows,t.num_cols, t.num_channels)
            |> fun x ->
                x.setZeroPrimal()
                x.setZeroAdjoint()
                x
        | None -> 
            d4M.createConstant(t.num_images,t.num_rows,t.num_cols, t.num_channels)
            |> fun x ->
                x.setZeroPrimal()
                x
    /// Copies a matrix.
    member inline t.copy() =
        match t.A with
        | Some A -> 
            t.nchw 
            |> d4M.create 
            |> fun c ->
                c.P.AsyncCopyToDevice(t.P,str)
                A.AsyncCopyToDevice(A,str)
                c
        | None -> 
            t.nchw 
            |> d4M.createConstant
            |> fun c ->
                c.P.AsyncCopyToDevice(t.P,str)
                c

    /// Resized the dArray if the current one is less than nn*nh*nw*nc. Otherwise it only adjusts num_images, num_rows, num_cols, num_channels.
    member inline t.ReplaceIf (nn, nc, nh, nw) is_constant =
        let new_size = nn*nh*nw*nc
        if int t.P.Size < new_size
        then
            (t :> IDisposable).Dispose()
            t.num_images <- nn
            t.num_channels <- nc
            t.num_rows <- nh
            t.num_cols <- nw
            t.P <- new_dev new_size
            t.A <- 
                match t.A with
                | Some A -> new_dev new_size |> Some
                | None -> 
                    match is_constant with
                    | true -> t.A
                    | false -> new_dev new_size |> Some
        else
            t.num_images <- nn
            t.num_channels <- nc
            t.num_rows <- nh
            t.num_cols <- nw
            t.A <-
                match t.A with
                | Some A -> t.A
                | None -> 
                    match is_constant with
                    | true -> t.A
                    | false -> new_dev new_size |> Some

    /// Copies the primal matrix to a host array.
    member inline t.GatherPrimal() =
        let h_a = Array.zeroCreate<float32> t.size
        t.P.CopyToHost(h_a,SizeT 0, SizeT 0, SizeT t.size * t.P.TypeSize)
        h_a

    /// Copies the adjoint matrix to a host array.
    member inline t.GatherAdjoint() =
        let h_a = Array.zeroCreate<float32> t.size
        t.A.Value.CopyToHost(h_a,SizeT 0, SizeT 0, SizeT t.size * t.P.TypeSize)
        h_a

    /// The unmanaged Cuda memory has to be freed explicitly or by letting go of the context by resetting  the F# Interactive.
    /// Finalizers work really poorly and can lead to unpredictable bugs when used to manage Cuda memory.
    /// Also do not bother to check whether an array is Null using Equals or =. Just hit Dispose().
    interface IDisposable with
        member t.Dispose() = 
                t.P.Dispose()
                match t.A with
                | Some A -> A.Dispose()
                | None -> ()

type ObjectPool() =
    let d4MPool = ResizeArray()
    let d4Mp = ref 0
    let workspacePool = ResizeArray()
    let wp = ref 0
    let tensorDescriptorPool = ResizeArray()
    let tp = ref 0
    let filterDescriptorPool = ResizeArray()
    let fp = ref 0
    let convolutionDescriptorPool = ResizeArray()
    let cp = ref 0
    let poolingDescriptorPool = ResizeArray()
    let pp = ref 0

    static member inline private getFromPool (pool : ResizeArray<_>) (pointer_to_pool : int ref) (creation_function) =
        if pool.Count > !pointer_to_pool then
            let t = pool.[!pointer_to_pool]
            pointer_to_pool := !pointer_to_pool+1
            t
        else
            let t = creation_function()
            pool.Add(t)
            pointer_to_pool := !pointer_to_pool+1
            t

    member t.getWorkspace n = 
        let t' = ObjectPool.getFromPool workspacePool wp (fun _ -> new_dev<byte> n)
        if int t'.Size < n then // Resize the object if less than n
            t'.Dispose()
            let t'' = new_dev<byte> n
            workspacePool.[!wp-1] <- t''
            t''
        else t'
    member t.getd4M is_constant (n:int,c:int,h:int,w:int as p) =
        let t' = 
            match is_constant with
            | false -> ObjectPool.getFromPool d4MPool d4Mp (fun _ -> d4M.createEmpty)
            | true -> ObjectPool.getFromPool d4MPool d4Mp (fun _ -> d4M.createEmptyConstant)

        t'.ReplaceIf p is_constant
        t'

    member t.getTensorDescriptor = ObjectPool.getFromPool tensorDescriptorPool tp (fun _ -> new TensorDescriptor())
    member t.getFilterDescriptor = ObjectPool.getFromPool filterDescriptorPool fp (fun _ -> new FilterDescriptor())
    member t.getConvolutionDescriptor = ObjectPool.getFromPool convolutionDescriptorPool cp (fun _ -> new ConvolutionDescriptor())
    member t.getPoolingDescriptor = ObjectPool.getFromPool poolingDescriptorPool pp (fun _ -> new PoolingDescriptor())

    /// Sets only the object pool pointers to zero.
    member t.ResetPointers() =
        d4Mp := 0
        wp := 0
        tp := 0
        fp := 0
        cp := 0
        pp := 0

    /// Zeroes out the adjoints in preparation for the backprop step and sets all the object pool pointers to zero.
    member t.Reset () =
        for i= !d4Mp-1 downto 0 do
            let x : d4M = d4MPool.[i]
            x.setZeroAdjoint()
        t.ResetPointers()


let ObjectPool = new ObjectPool() // In the past iteration of the library, the object pool's role was taken by the tape. Not anymore.

type d4M with
    /// Copies a matrix.
    /// Uses the object pool.
    member inline t.copy'() =
        match t.A with
        | Some A -> 
            t.nchw 
            |> ObjectPool.getd4M false
            |> fun c ->
                c.P.AsyncCopyToDevice(t.P,str)
                A.AsyncCopyToDevice(A,str)
                c
        | None -> 
            t.nchw 
            |> ObjectPool.getd4M true
            |> fun c ->
                c.P.AsyncCopyToDevice(t.P,str)
                c



let tape = new Stack<_>(1000) // Nice and simple way of passing in the closures for the backprop step.

let T = Operation.Transpose
let nT = Operation.NonTranspose

let defaultLayout = cudnnTensorFormat.NCHW
let defaultType = cudnnDataType.Float

type TensorDescriptor with
    /// Extended method that works according to the bound defaultLayout and defaultType variables.
    member inline t.SetTensor4dDescriptor(n,c,h,w) = t.SetTensor4dDescriptor(defaultLayout,defaultType,n,c,h,w)

type FilterDescriptor with
    /// Extended method that works according to the bound defaultType variable.
    member inline t.SetFilter4dDescriptor(n,c,h,w) = t.SetFilter4dDescriptor(defaultType,n,c,h,w)

type ConvolutionParameters = {
    pad_h : int
    pad_w : int
    stride_h : int
    stride_w : int
    upscale_h : int
    upscale_w : int
    mode : cudnnConvolutionMode
    }

type PoolingParameters =
    {
    mode : cudnnPoolingMode
    windowHeight : int
    windowWidth : int
    verticalPadding : int
    horizontalPadding : int
    verticalStride : int
    horizontalStride : int
    }

type PoolingDescriptor with
    member inline t.SetPooling2dDescriptor (p : PoolingParameters) =
        t.SetPooling2dDescriptor(p.mode,p.windowHeight,p.windowWidth,p.verticalPadding,p.horizontalPadding,p.verticalStride,p.horizontalStride)

    /// WARNING: The GetPooling2dForwardOutputDim method returns incorrect results! There is a bug in the cuDNN v3 library!
    member inline t.GetPooling2dForwardOutputDim s =
        let mutable n,c,h,w = 0,0,0,0
        t.GetPooling2dForwardOutputDim(s,&n,&c,&h,&w)
        n,c,h,w

let defaultConvPar = 
    {
    pad_h = 0
    pad_w = 0
    stride_h = 1
    stride_w = 1
    upscale_h = 1
    upscale_w = 1
    mode = cudnnConvolutionMode.Convolution
    }

type ConvolutionDescriptor with
    member inline t.SetConvolution2dDescriptor (p : ConvolutionParameters) =
        t.SetConvolution2dDescriptor(p.pad_h,p.pad_w,p.stride_h,p.stride_w,p.upscale_h,p.upscale_w,p.mode)
    member inline t.GetConvolution2dForwardOutputDim (s,f) =
        let mutable n,c,h,w = 0,0,0,0
        t.GetConvolution2dForwardOutputDim(s,f,&n,&c,&h,&w)
        n,c,h,w

let inline divup a b = (a-1)/b+1 // Integer division with rounding up. (a+b-1)/b is another variant on this.

let kernels_dir = IO.Path.Combine(__SOURCE_DIRECTORY__,"Cuda Kernels")
IO.Directory.CreateDirectory(kernels_dir) // Creates the Cuda Kernels directory if it does not exist. WriteAllBytes would otherwise throw an exception.

let inline size_nchw (n:int,c,h,w) = n*c*h*w

let load_kernel kernel_code kernel_name = 
    let kernel_path = IO.Path.Combine(kernels_dir,kernel_name)
        
    if IO.File.Exists(kernel_path) 
    then
        ctx.LoadKernelPTX(kernel_path,kernel_name) // For all the modules, it takes roughly 0.35s to compile them. Loading them from drive takes less than a millisecond.
    else
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,kernel_name)
        try k.Compile([|"-arch=compute_30"|])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()
        let ptx = k.GetPTX()
        IO.File.WriteAllBytes(kernel_path,ptx)
        ctx.LoadKernelPTX(ptx,kernel_name)

/// o <- f(x)
type DeviceUnaryTransformModule(op: string, unique_name : string) = 
    let block_size = 256

    let kernel_name = "Map1Kernel"+unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member inline t.A((x_nchw, x: CudaDeviceVariable<float32>), (o_nchw, o: CudaDeviceVariable<float32>)) =
        if x_nchw <> o_nchw then failwith "x_nchw <> o_nchw in DeviceUnaryTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore

/// o <- f(x,y)
type DeviceBinaryTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map2Kernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x, floatType y)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, const floatType* B, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i],B[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""
    
    let kernel = load_kernel kernel_code kernel_name

    member inline t.A((x_nchw, x: CudaDeviceVariable<float32>),(y_nchw, y: CudaDeviceVariable<float32>), (o_nchw, o: CudaDeviceVariable<float32>)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceBinaryTransformModule"
        if y_nchw <> o_nchw then failwith "y_nchw <> o_nchw in DeviceBinaryTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore

/// o <- f(x,y,z)
type DeviceTrinaryTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map3Kernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x, floatType y, floatType z)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, const floatType* B, const floatType* C, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i],B[i],C[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A((x_nchw, x: CudaDeviceVariable<float32>), (y_nchw, y: CudaDeviceVariable<float32>), (z_nchw, z: CudaDeviceVariable<float32>), (o_nchw, o: CudaDeviceVariable<float32>)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceTrinaryTransformModule"
        if y_nchw <> z_nchw then failwith "y_nchw <> z_nchw in DeviceTrinaryTransformModule"
        if z_nchw <> o_nchw then failwith "z_nchw <> o_nchw in DeviceTrinaryTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,z.DevicePointer,o.DevicePointer,n) |> ignore


/// o <- sum(f(x))
type DeviceUnaryMapSumModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map1SumKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x)
            {
                return ";op;"
            }
        
            __device__ inline floatType warpDownReduce(floatType value){
                #pragma unroll
	            for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	            return value;
            }

            // Device code
            __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int N)
            {
	            int i = blockDim.x * blockIdx.x + threadIdx.x;
	            const int stride = blockDim.x * gridDim.x;
	            __shared__ floatType temp[32];
                if (threadIdx.x < 32) {
                    temp[threadIdx.x] = 0.0f; 
                    if (blockIdx.x == 0) O[0] = 0.0f;
                    }
                
                floatType acc = 0.0f;
	            while (i < N)
	            {
		            acc += op(A[i]);
		            i += stride;
	            }
	            __syncthreads(); 
                floatType out_partial = warpDownReduce(acc);
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	            __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	            if (threadIdx.x == 0) atomicAdd(O, out_partial);
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    let o = new_dev<float32> 1

    member inline t.A((x_nchw, x: CudaDeviceVariable<float32>)) =
        let n = size_nchw x_nchw
        
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)

        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore
        o.[SizeT 0]

/// o <- sum(f(x,y))
type DeviceBinaryMapSumModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map2SumKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x, floatType y)
            {
                return ";op;"
            }
        
            __device__ inline floatType warpDownReduce(floatType value){
                #pragma unroll
	            for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	            return value;
            }

            // Device code
            __global__ void ";kernel_name;"(const floatType* A, const floatType* B, floatType* O, const int N)
            {
	            int i = blockDim.x * blockIdx.x + threadIdx.x;
	            const int stride = blockDim.x * gridDim.x;
	            __shared__ floatType temp[32]; 
                if (threadIdx.x < 32) {
                    temp[threadIdx.x] = 0.0f; 
                    if (blockIdx.x == 0) O[0] = 0.0f;
                    }    
                floatType acc = 0.0f;
	            while (i < N)
	            {
		            acc += op(A[i],B[i]);
		            i += stride;
	            }
	            __syncthreads(); 
                floatType out_partial = warpDownReduce(acc);
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	            __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	            if (threadIdx.x == 0) atomicAdd(O, out_partial);
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    let o = new_dev<float32> 1

    member inline t.A((x_nchw, x: CudaDeviceVariable<float32>),(y_nchw, y: CudaDeviceVariable<float32>)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceBinaryMapSumModule"
        let n = size_nchw x_nchw
        
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore
        o.[SizeT 0]

/// o <- f(coef_x,x)
type DeviceUnaryCoefTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map1CoefKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType coef_x, floatType x)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(coef_A,A[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = 
        let kernel_path = IO.Path.Combine(kernels_dir,kernel_name)
        
        if IO.File.Exists(kernel_path) 
        then
            ctx.LoadKernelPTX(kernel_path,kernel_name) // For all the modules, it takes roughly 0.35s to compile them. Loading them from drive takes less than a millisecond.
        else
            let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,kernel_name)
            try k.Compile([||])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()
            let ptx = k.GetPTX()
            IO.File.WriteAllBytes(kernel_path,ptx)
            ctx.LoadKernelPTX(ptx,kernel_name)

    member inline t.A(coef_x: float32, (x_nchw, x: CudaDeviceVariable<float32>), (o_nchw, o: CudaDeviceVariable<float32>)) =
        if x_nchw <> o_nchw then failwith "x.nchw <> o.nchw in DeviceUnaryCoefTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,o.DevicePointer,n) |> ignore

/// o <- f(coef_x,x,coef_y,y)
type DeviceBinaryCoefTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map2CoefKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;

            __device__ inline floatType op(floatType coef_x, floatType x, floatType coef_y, floatType y)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, const floatType coef_B, const floatType* B, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(coef_A,A[i],coef_B,B[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = 
        let kernel_path = IO.Path.Combine(kernels_dir,kernel_name)
        
        if IO.File.Exists(kernel_path) 
        then
            ctx.LoadKernelPTX(kernel_path,kernel_name) // For all the modules, it takes roughly 0.35s to compile them. Loading them from drive takes less than a millisecond.
        else
            let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,kernel_name)
            try k.Compile([||])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()
            let ptx = k.GetPTX()
            IO.File.WriteAllBytes(kernel_path,ptx)
            ctx.LoadKernelPTX(ptx,kernel_name)

    member inline t.A(coef_x: float32, (x_nchw, x: CudaDeviceVariable<float32>), coef_y: float32, (y_nchw, y: CudaDeviceVariable<float32>), (o_nchw, o: CudaDeviceVariable<float32>)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceBinaryCoefTransformModule"
        if y_nchw <> o_nchw then failwith "y_nchw <> o_nchw in DeviceBinaryCoefTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,coef_y,y.DevicePointer,o.DevicePointer,n) |> ignore

/// o <- f(coef_x,x,coef_y,y,coef_z,z)
type DeviceTrinaryCoefTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map3CoefKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType coef_x, floatType x, floatType coef_y, floatType y, floatType coef_z, floatType z)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, const floatType coef_B, const floatType* B, const floatType coef_C, const floatType* C, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(coef_A,A[i],coef_B,B[i],coef_C,C[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = 
        let kernel_path = IO.Path.Combine(kernels_dir,kernel_name)
        
        if IO.File.Exists(kernel_path) 
        then
            ctx.LoadKernelPTX(kernel_path,kernel_name) // For all the modules, it takes roughly 0.35s to compile them. Loading them from drive takes less than a millisecond.
        else
            let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,kernel_name)
            try k.Compile([||])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()
            let ptx = k.GetPTX()
            IO.File.WriteAllBytes(kernel_path,ptx)
            ctx.LoadKernelPTX(ptx,kernel_name)

    member inline t.A(coef_x: float32, (x_nchw, x: CudaDeviceVariable<float32>), coef_y: float32, (y_nchw, y: CudaDeviceVariable<float32>), coef_z: float32, (z_nchw, z: CudaDeviceVariable<float32>), (o_nchw, o: CudaDeviceVariable<float32>,n)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceTrinaryCoefTransformModule"
        if y_nchw <> z_nchw then failwith "y_nchw <> z_nchw in DeviceTrinaryCoefTransformModule"
        if z_nchw <> o_nchw then failwith "z_nchw <> o_nchw in DeviceTrinaryCoefTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,coef_y,y.DevicePointer,coef_z,z.DevicePointer,o.DevicePointer,n) |> ignore

/// o <- max_col(x)
/// Sets all except one of the max of a column to zero.
type DeviceMaxColumnActivationModule() = 
    let block_size = 128

    let kernel_name = "MaxColumnKernel"
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            #define INIT __int_as_float(0xff800000) // The constant init for the reduce operations. This is float negative infinity.
            // The max reduce version.
            __device__ inline floatType warpReduce(floatType value){
                #pragma unroll
	            for (int i=1; i<32; i*=2) {
                    floatType tmp = __shfl_xor(value, i);
                    value = (tmp > value) ? tmp : value;
                    }
	            return value;
            }
              
            __device__ inline floatType blockReduce(floatType value){
	            __shared__ floatType temp[32];
                if (threadIdx.x < 32) temp[threadIdx.x] = INIT; 
                floatType out_partial = warpReduce(value);
                __syncthreads();
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
                __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpReduce(temp[threadIdx.x]);
                return out_partial;
            }

            // Device code
            __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int num_rows, const int num_cols)
            {
                int row = threadIdx.x;
                //const int col = blockIdx.x;
                int col_idx = blockIdx.x*num_rows; 
                floatType max = INIT; // This is the negative infinity for floats.
                int index = -1;
                while (row < num_rows)
                {
                   if (A[row+col_idx] > max) {
                        max = A[row+col_idx];
                        index = row;
                        }
                    row += blockDim.x;
                }
                
                __shared__ floatType max_index;
                if (max == blockReduce(max)) max_index = index;
                __syncthreads();
                index = max_index; // These last four lines are to make absolutely sure that only one max is selected in case there is more than one.
                row = threadIdx.x;
                while (row < num_rows)
                {
                    O[row+col_idx] = (row == index) ? max : 0.0f;
                    row += blockDim.x;
                }
            }
        }

        "|] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(((n : int,c : int,h,w as x_nchw), x: CudaDeviceVariable<float32>), (o_nchw, o: CudaDeviceVariable<float32>)) =
        if x_nchw <> o_nchw then failwith "x_nchw <> o_nchw"
        let m = c*h*w
        kernel.GridDimensions <- dim3(n)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream,x.DevicePointer,o.DevicePointer,m,n) |> ignore


// The gradient clipping module.
let gradclipModule = lazy DeviceUnaryCoefTransformModule("(x < -coef_x) ? -coef_x : (x > coef_x ? coef_x : x);", "GradClip") // Unique names like GradClip are necessary for load and saving to drive. Be very careful of collisions.

// coef_x = scale
// coef_y = location
// y does not get used.
let randMapModule = lazy DeviceBinaryCoefTransformModule("coef_x*(x-0.5f)+coef_y;","RandMapper")

/// Fills matrix by sampling from a random uniform distribution in <-1.0f,1.0f]
let fillRandomUniformMatrix (x_nchw, x : CudaDeviceVariable<float32> as x') (scaling_factor : float32) location =
        let weights_total_size = size_nchw x_nchw

        cudaRandom.GenerateUniform(x)
        // 2.0f*scaling_factor ensures that it is rescaled around zero if the scaling_factor is 1.0f.
        randMapModule.Value.A(2.0f*scaling_factor,x',location,x',x')

// y <- alpha * x + y
let saxpy (alpha:float32) (x_nchw, x:CudaDeviceVariable<float32>) (y_nchw, y:CudaDeviceVariable<float32>) =
    if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in saxpy"
    cublas.Axpy(alpha,x,1,y,1)

/// General matrix-matrix addition. Inplace version.
/// N and C dimensions are restricted to 1 in this as the function is intended to be used for transposes.
#nowarn "49"
let geam transa transb (alpha: float32) ((A_num_images, A_num_channels, A_num_rows, A_num_cols), A:CudaDeviceVariable<float32>) beta ((B_num_images, B_num_channels, B_num_rows, B_num_cols), B:CudaDeviceVariable<float32>) ((C_num_images, C_num_channels, C_num_rows, C_num_cols), C:CudaDeviceVariable<float32>) =
    if A_num_images <> 1 || A_num_channels <> 1 then failwith "A_num_images <> 1 || A_num_channels <> 1"
    if B_num_images <> 1 || B_num_channels <> 1 then failwith "B_num_images <> 1 || B_num_channels <> 1"
    if C_num_images <> 1 || C_num_channels <> 1 then failwith "C_num_images <> 1 || C_num_channels <> 1"

    let a_row = if transa = nT then A_num_rows else A_num_cols
    let a_col = if transa = nT then A_num_cols else A_num_rows
    let b_row = if transb = nT then B_num_rows else B_num_cols
    let b_col = if transb = nT then B_num_cols else B_num_rows
        
    if a_row <> b_row then failwith (sprintf "a_row <> b_row in geam2! %i <> %i" a_row b_row)
    if a_col <> b_col then failwith (sprintf "a_col <> b_col in geam2! %i <> %i" a_col b_col)

    if a_row <> C_num_rows then failwith (sprintf "a_row <> C_num_rows in geam2! %i <> %i" a_col b_col)
    if a_col <> C_num_cols then failwith (sprintf "a_col <> C_num_cols in geam2! %i <> %i" a_col b_col)

    let lda = if transa = nT then a_row else a_col
    let ldb = if transa = nT then b_row else b_col
    let ldc = a_row

    cublas.Geam(transa, transb, a_row, a_col, alpha, A, lda, B, ldb, beta, C, ldc)

/// General matrix-matrix multiply from cuBLAS. Inplace version
let gemm transa transb (alpha: float32) ((A_num_images, A_num_channels, A_num_rows, A_num_cols), A:CudaDeviceVariable<float32>) ((B_num_images, B_num_channels, B_num_rows, B_num_cols), B:CudaDeviceVariable<float32>) beta ((C_num_images, C_num_channels, C_num_rows, C_num_cols), C:CudaDeviceVariable<float32>) =
    let gemm (A_num_rows, A_num_cols) (B_num_rows, B_num_cols) (C_num_rows, C_num_cols) =
        let a_col = if transa = nT then A_num_cols else A_num_rows
        let b_row = if transb = nT then B_num_rows else B_num_cols
        if a_col <> b_row then failwithf "a_col <> b_row in gemm! %i <> %i" a_col b_row
        let m = if transa = nT then A_num_rows else A_num_cols
        let n = if transb = nT then B_num_cols else B_num_rows
        let k = a_col
        let lda = if transa = nT then m else k
        let ldb = if transb = nT then k else n
        let ldc = m

        if m <> C_num_rows || n <> C_num_cols then failwith "m <> C_num_rows || n <> C_num_cols"

        cublas.Gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

    gemm (A_num_channels*A_num_cols*A_num_rows,A_num_images) (B_num_channels*B_num_cols*B_num_rows,B_num_images) (C_num_channels*C_num_cols*C_num_rows,C_num_images)

/// Matrix-matrix multiply.
let matmult (a: d4M) (b:d4M) =
    let c = 
        let num_rows = a.num_channels*a.num_cols*a.num_rows
        let num_cols = b.num_images
        ObjectPool.getd4M false (num_cols,num_rows,1,1)
        
    gemm nT nT 1.0f a.P' b.P' 0.0f c.P'

    let matmult_backward_left () = gemm nT T 1.0f c.A' b.P' 1.0f a.A'
    let matmult_backward_right () = gemm T nT 1.0f a.A' c.A' 1.0f b.A'
    if a.A.IsSome then tape.Push(matmult_backward_left)
    if b.A.IsSome then tape.Push(matmult_backward_right)
    c


/// Can be used to add matrices or for (4D)matrix-vector broadcast addition.
/// The output dimensions are based on the left argument.
/// Those dimenions the size of 1 of the right argument are broadcasted.
let inline private tensor_add' add_to_left alpha (left : d4M) beta (right : d4M) =
    let leftDesc = ObjectPool.getTensorDescriptor
    left.nchw |> leftDesc.SetTensor4dDescriptor
    let rightDesc = ObjectPool.getTensorDescriptor
    right.nchw |> rightDesc.SetTensor4dDescriptor

    let output = if add_to_left = false then left.nchw |> ObjectPool.getd4M false else left

    cudnn.AddTensor(alpha,leftDesc,left.P,0.0f,leftDesc,output.P)
    cudnn.AddTensor(beta,rightDesc,right.P,1.0f,leftDesc,output.P)

    let tensor_add_right_backwards () = 
        if left.nchw = right.nchw then
            saxpy beta output.A' right.A'
        else
            cudnn.ConvolutionBackwardBias(beta,leftDesc,output.A.Value,1.0f,rightDesc,right.A.Value)
    let tensor_add_left_backwards () = 
        saxpy alpha output.A' left.A'

    if right.A.IsSome then tape.Push(tensor_add_right_backwards)
    if add_to_left = false && left.A.IsSome then tape.Push(tensor_add_left_backwards)
    output

let tensor_add = tensor_add' false

/// The activation function. Zeroes out the target primal during the call.
let activation_forward mode (input : d4M)  =
    let input_sizes = input.nchw
    let srcTensorDesc = ObjectPool.getTensorDescriptor
    input_sizes |> srcTensorDesc.SetTensor4dDescriptor

    let alpha = 1.0f
    let beta = 1.0f

    let output = ObjectPool.getd4M false input_sizes

    cudnn.ActivationForward(mode,alpha,srcTensorDesc,input.P,0.0f,srcTensorDesc,output.P)

    let activation_backward () =
        cudnn.ActivationBackward(mode,alpha,srcTensorDesc,output.P,srcTensorDesc,output.A.Value,srcTensorDesc,input.P,beta,srcTensorDesc,input.A.Value)

    if input.A.IsSome then tape.Push(activation_backward)
    output

/// The pooling function. Zeroes out the target primal during the call.
let pooling_forward p (input : d4M) =
    let poolingDescriptor = ObjectPool.getPoolingDescriptor
    poolingDescriptor.SetPooling2dDescriptor p

    let srcTensorDesc = ObjectPool.getTensorDescriptor
    let input_sizes = input.nchw
    input_sizes |> srcTensorDesc.SetTensor4dDescriptor
    //let dest_sizes = poolingDescriptor.GetPooling2dForwardOutputDim srcTensorDesc // Buggy cuDNN function.
    let dest_sizes =
        input_sizes
        |> fun (n,c,h,w) ->
            n,c,
            1 + divup (h + 2*p.verticalPadding - p.windowHeight) p.verticalStride,
            1 + divup (h + 2*p.horizontalPadding - p.windowWidth) p.horizontalStride

    let output = ObjectPool.getd4M false dest_sizes

    let dstTensorDesc = ObjectPool.getTensorDescriptor
    dest_sizes |> dstTensorDesc.SetTensor4dDescriptor

    let alpha, beta = 1.0f, 1.0f

    cudnn.PoolingForward(poolingDescriptor,alpha,srcTensorDesc,input.P,0.0f,dstTensorDesc,output.P)

    let pooling_backward () =
        cudnn.PoolingBackward(poolingDescriptor,alpha,srcTensorDesc,output.P,srcTensorDesc,output.A.Value,dstTensorDesc,input.P,beta,dstTensorDesc,input.A.Value)

    if input.A.IsSome then tape.Push(pooling_backward)
    output

let squareModule = lazy new DeviceUnaryTransformModule("x*x;","Square")
//y = error
//z = previous adjoint value
let squareErrorModule = lazy new DeviceTrinaryTransformModule("2.0f*x*y + z;","SquareError")
let square (a:d4M) =
    let c = a.nchw |> ObjectPool.getd4M false
    squareModule.Value.A(a.P',c.P')

    let square_backward () = squareErrorModule.Value.A(a.P',c.A',a.A',a.A')
    if a.A.IsSome then tape.Push square_backward
    c

let sumModule = lazy new DeviceUnaryMapSumModule("x;", "Sum")
let sumErrorModule = lazy new DeviceUnaryCoefTransformModule("coef_x + x;", "SumError")
let sum (a:d4M) =
    let c = Df.create (ref 0.0f)
    c.P := sumModule.Value.A(a.P')

    let sum_backward () = sumErrorModule.Value.A(!c.A,a.A',a.A')
    if a.A.IsSome then tape.Push sum_backward
    c

let scale (alpha: float32) (a:Df) =
    let c = Df.create (ref 0.0f)
    c.P := alpha * !a.P

    let scale_backward () = a.A := alpha * !c.A + !a.A
    tape.Push scale_backward
    c

let inline private convolutional_forward' (prev_output: ((int*int*int*int)*d4M) option) (convPar, data : d4M, filter : d4M) =
    let data_sizes = data.nchw
    let filter_sizes = filter.nchw

    let srcTensorDesc = ObjectPool.getTensorDescriptor
    let dstTensorDesc = ObjectPool.getTensorDescriptor
    let filterDesc = ObjectPool.getFilterDescriptor
    let convDesc = ObjectPool.getConvolutionDescriptor

    data_sizes |> srcTensorDesc.SetTensor4dDescriptor
    filter_sizes |> filterDesc.SetFilter4dDescriptor
    convPar |> convDesc.SetConvolution2dDescriptor 

    let dims, output = 
        let dims = convDesc.GetConvolution2dForwardOutputDim(srcTensorDesc,filterDesc)
        match prev_output with
        | Some (prev_dims, prev_output) ->
            if dims <> prev_dims then failwith "dims <> prev_dims in linear_layer_conv"
            prev_dims, prev_output
        | None ->
            dims |> dstTensorDesc.SetTensor4dDescriptor
            dims, dims |> ObjectPool.getd4M false

    let algo = cudnn.GetConvolutionForwardAlgorithm(srcTensorDesc,filterDesc,convDesc,dstTensorDesc,cudnnConvolutionFwdPreference.PreferFastest,SizeT 0)
    let workspace = 
        cudnn.GetConvolutionForwardWorkspaceSize(srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo) |> int
        |> ObjectPool.getWorkspace

    let alpha = 1.0f
    let beta = 1.0f

    cudnn.ConvolutionForward(alpha,srcTensorDesc,data.P,filterDesc,filter.P,convDesc,algo,workspace,0.0f,dstTensorDesc,output.P)

    let convolution_backwards_filter () =
        let algo = cudnn.GetConvolutionBackwardFilterAlgorithm(srcTensorDesc,dstTensorDesc,convDesc,filterDesc,cudnnConvolutionBwdFilterPreference.PreferFastest,SizeT 0)
        let workspace =
            cudnn.GetConvolutionBackwardFilterWorkspaceSize(srcTensorDesc,dstTensorDesc,convDesc,filterDesc,algo) |> int
            |> ObjectPool.getWorkspace
        cudnn.ConvolutionBackwardFilter(alpha,srcTensorDesc,data.P,dstTensorDesc,output.A.Value,convDesc,algo,workspace,beta,filterDesc,filter.A.Value)

    let convolution_backwards_data () =
        let algo = cudnn.GetConvolutionBackwardDataAlgorithm(filterDesc,dstTensorDesc,convDesc,srcTensorDesc,cudnnConvolutionBwdDataPreference.PreferFastest,SizeT 0)
        let workspace =
            cudnn.GetConvolutionBackwardDataWorkspaceSize(filterDesc,dstTensorDesc,convDesc,srcTensorDesc,algo) |> int
            |> ObjectPool.getWorkspace
        cudnn.ConvolutionBackwardData(alpha,filterDesc,filter.P,dstTensorDesc,output.A.Value,convDesc,beta,algo,workspace,srcTensorDesc,data.A.Value)

    if filter.A.IsSome then tape.Push(convolution_backwards_filter)
    if data.A.IsSome then tape.Push(convolution_backwards_data)

    (dims,output) |> Some

/// The convolutional function. Zeroes out the target primal during the call.
let convolution_forward convPar (data : d4M) (filter : d4M) = 
    convolutional_forward' None (convPar,data,filter)
    |> fun x -> x.Value |> snd

let linear_layer_conv (convs: (ConvolutionParameters*d4M*d4M) []) (bias: d4M option) =
    convs
    |> Array.fold convolutional_forward' None
    |> fun (output) ->
        let _, left = output.Value
        match bias with
        | None -> left
        | Some right -> tensor_add' true 1.0f left 1.0f right

let hadamaradMultiplicationModule = lazy new DeviceBinaryTransformModule("x*y;", "HadMult")
let hadamaradMultiplicationErrorModule = lazy new DeviceTrinaryTransformModule("x*y+z;", "HadMultError")
/// Hadamarad (elementwise) multiplication function.
let inline private hadmult' (prev_output : d4M option) ((a,b): d4M*d4M) =
    let c = 
        match prev_output with
        | Some v -> v
        | None -> ObjectPool.getd4M false a.nchw

    hadamaradMultiplicationModule.Value.A(a.P', b.P', c.P')

    let hadmult_backward_left () = hadamaradMultiplicationErrorModule.Value.A(b.P',c.A',a.A',a.A')
    let hadmult_backward_right () = hadamaradMultiplicationErrorModule.Value.A(a.P',c.A',b.A',b.A')
    if a.A.IsSome then tape.Push hadmult_backward_left
    if b.A.IsSome then tape.Push hadmult_backward_right
    Some c

let hadmult (a: d4M) (b: d4M) = hadmult' None (a, b) |> fun x -> x.Value
let linear_layer_hadmult (hads: (d4M*d4M)[]) = hads |> Array.fold hadmult' None

let squared_error_cost target activations =
    tensor_add 1.0f target -1.0f activations
    |> square
    |> sum
    |> scale (0.5f/ float32 target.num_images)


let maxColumnModule = lazy new DeviceMaxColumnActivationModule()
let accuracyModule = lazy new DeviceBinaryMapSumModule("(x*y == 0.0f) ? 0.0f : 1.0f;","Accuracy")
let get_accuracy (targets : d4M) (activations : d4M) =
    let o = ObjectPool.getd4M true targets.nchw
    maxColumnModule.Value.A(activations.P',o.P')
    accuracyModule.Value.A(targets.P',o.P')

type d4M with
    static member makeUniformRandomNode (n,c,h,w as nchw) =
        let scale = (1.0f / sqrt(size_nchw nchw |> float32))
        let p = d4M.create(n,c,h,w)
        fillRandomUniformMatrix p.P' scale 0.0f
        p

// A convolutional feedforward layer of neurons
type ConvolutionalFeedforwardLayer =
    {
    W:d4M  // Input weight matrix
    b:d4M  // Bias vector
    a:cudnnActivationMode
    } with     // Activation function
     
    static member fromArray (a : d4M[]) act =
        {
         W = a.[0]
         b = a.[1]
         a = act
        }

    static member createRandomLayer (n,c,h,w as nchw) act =
        {
         W = d4M.makeUniformRandomNode nchw
         b = d4M.makeUniformRandomNode (1,n,1,1)
         a = act
        } 

    member l.runLayer (convPar,x:d4M) =
        //linear_layer_matmult [|l.W,x|] (Some l.b) |> l.a // TODO: Make optimize linear layer functions.
        //matmult l.W x
//        convolution_forward convPar x l.W
//        |> fun x -> tensor_add 1.0f x 1.0f l.b
        linear_layer_conv [|convPar,x,l.W|] (Some l.b)
        |> activation_forward l.a

    member l.ToArray = [|l.W;l.b|]
    member t.ResetAdjoints () = t.W.setZeroAdjoint(); t.b.setZeroAdjoint()
    member t.SGD learning_rate = saxpy -learning_rate t.W.A' t.W.P'; saxpy -learning_rate t.b.A' t.b.P'

// A fully connected feedforward layer of neurons
type FeedforwardLayer =
    {
    W:d4M  // Input weight matrix
    b:d4M  // Bias vector
    a:cudnnActivationMode
    } with     // Activation function
     
    static member fromArray (a : d4M[]) act =
        {
         W = a.[0]
         b = a.[1]
         a = act
        }

    static member createRandomLayer (n,c,h,w as nchw) act =
        {
         W = d4M.makeUniformRandomNode nchw
         b = d4M.makeUniformRandomNode (1,c,1,1)
         a = act
        } 

    member l.runLayer (convPar,x:d4M) =
        matmult l.W x
        |> fun x -> tensor_add 1.0f x 1.0f l.b
        |> activation_forward l.a

    member l.ToArray = [|l.W;l.b|]
    member t.ResetAdjoints () = t.W.setZeroAdjoint(); t.b.setZeroAdjoint()
    member t.SGD learning_rate = saxpy -learning_rate t.W.A' t.W.P'; saxpy -learning_rate t.b.A' t.b.P'


let load_data file_name is_constant =
    use stream_data = IO.File.OpenRead(file_name)
    use reader_data = new IO.BinaryReader(stream_data)

    let m = reader_data.ReadInt32()
    if m <> 929856 then failwith "Wrong file type in load_weights"

    let l = reader_data.ReadInt32()
    let weights = [|
        for i=1 to l do
            let num_rows = reader_data.ReadInt32()
            let num_cols = reader_data.ReadInt32()
            let ar = [|for x=1 to num_rows*num_cols do yield reader_data.ReadSingle()|]
            match is_constant with
            | true -> yield d4M.createConstant(num_cols,num_rows,1,1,ar)
            | false -> yield d4M.create(num_cols,num_rows,1,1,ar)
        |]

    weights
