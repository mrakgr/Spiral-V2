// Basic reverse mode AD on the GPU. This v2 of Spiral is focused on convolutional operations.

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

/// The main matrix type.
type d4Matrix =
    {
    mutable num_images:int
    mutable num_channels:int
    mutable num_rows:int
    mutable num_cols:int
    mutable dArray: CudaDeviceVariable<float32>
    }  

    /// The main create function. A substitute for the constructor.
    static member create(num_images: int, num_channels:int, num_rows: int, num_cols: int, dArray: CudaDeviceVariable<float32>) =
        {num_images=num_images; num_channels=num_channels; num_rows=num_rows; num_cols=num_cols;dArray=dArray}

    /// Throws an exception if it tries to allocate an array of size 0.
    static member create(num_images: int, num_channels:int, num_rows: int, num_cols: int) =
        let q = (num_images*num_channels*num_rows*num_cols) |> SizeT
        let t = new CudaDeviceVariable<float32>(q)
        {num_images=num_images; num_channels=num_channels; num_rows=num_rows; num_cols=num_cols; dArray=t}

    /// Copies a host to a device array.
    /// Throws an exception if it tries to allocate an array of size 0.
    static member create(num_images: int, num_channels:int, num_rows: int, num_cols: int, dArray: float32[]) =
        let q = num_images*num_channels*num_rows*num_cols
        if dArray.Length <> q then failwith "Invalid size in dMatrix construction."
        let t = to_dev dArray
        {num_images=num_images; num_channels=num_channels; num_rows=num_rows; num_cols=num_cols;dArray=t}

    /// Returns a new instance of an (dMatrix.createEmpty) dMatrix.
    /// Unlike the let statements, the member statements are always reevaluated.
    static member createEmpty = d4Matrix.create(0,0,0,0,CudaDeviceVariable.Null)

    /// Returns nhwc as a tuple
    member inline t.nchw = t.num_images, t.num_channels, t.num_rows, t.num_cols
    /// Returns the n*h*w*c
    member inline t.size = t.num_images * t.num_channels * t.num_rows * t.num_cols
    /// Sets the matrix to zero.
    member inline t.setZero() = t.dArray.MemsetAsync(0u,str.Stream)
    /// Set the matrix to a value.
    member inline t.set (x: float32) = 
        let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
        t.dArray.MemsetAsync(v,str.Stream)
    /// Creates a copy of this matrix with all the values set to zero.
    member inline t.zeroLike() =
        let c = d4Matrix.create(t.num_images,t.num_rows,t.num_cols, t.num_channels)
        c.setZero()
        c
    /// Copies a matrix.
    member inline t.copy() =
        let c = d4Matrix.create(t.num_images,t.num_rows,t.num_cols, t.num_channels)
        c.dArray.AsyncCopyToDevice(t.dArray,str)
        c

    /// Resized the dArray if the current one is less than nn*nh*nw*nc. Otherwise it only adjusts num_images, num_rows, num_cols, num_channels.
    member inline t.ReplaceIf (nn, nc, nh, nw) =
        let new_size = nn*nh*nw*nc
        if int t.dArray.Size < new_size
        then
            (t :> IDisposable).Dispose()
            t.num_images <- nn
            t.num_channels <- nc
            t.num_rows <- nh
            t.num_cols <- nw
            t.dArray <- new_dev new_size
        else
            t.num_images <- nn
            t.num_channels <- nc
            t.num_rows <- nh
            t.num_cols <- nw

    /// Copies a matrix to a host array.
    member inline t.Gather() =
        let h_a = Array.zeroCreate<float32> (int t.dArray.Size)
        t.dArray.CopyToHost(h_a)
        h_a

    member inline t.isEmpty = t.dArray.Equals(CudaDeviceVariable.Null)

    /// The unmanaged Cuda memory has to be freed explicitly or by letting go of the context by resetting  the F# Interactive.
    /// Finalizers work really poorly and can lead to unpredictable bugs when used to manage Cuda memory.
    interface IDisposable with
        member t.Dispose() = 
            if t.isEmpty = false then
                t.dArray.Dispose()

type d4M = {
    P : d4Matrix // primal
    A : d4Matrix // adjoint
    is_constant : bool
    }

type ObjectPool() =
    let d4MatrixPool = ResizeArray()
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
    member t.getd4Matrix (n:int,c:int,h:int,w:int as p) =
        let t' = ObjectPool.getFromPool d4MatrixPool d4Mp (fun _ -> d4Matrix.createEmpty)
        t'.ReplaceIf p
        t'
    member t.getd4M is_constant (n:int,c:int,h:int,w:int as p) =
        let t'1 = ObjectPool.getFromPool d4MatrixPool d4Mp (fun _ -> d4Matrix.createEmpty)
        let t'2 = ObjectPool.getFromPool d4MatrixPool d4Mp (fun _ -> d4Matrix.createEmpty)
        t'1.ReplaceIf p
        if is_constant = false then t'2.ReplaceIf p
        {P=t'1;A=t'2;is_constant=is_constant}
    member t.getTensorDescriptor = ObjectPool.getFromPool tensorDescriptorPool tp (fun _ -> new TensorDescriptor())
    member t.getFilterDescriptor = ObjectPool.getFromPool filterDescriptorPool fp (fun _ -> new FilterDescriptor())
    member t.getConvolutionDescriptor = ObjectPool.getFromPool convolutionDescriptorPool cp (fun _ -> new ConvolutionDescriptor())
    member t.getPoolingDescriptor = ObjectPool.getFromPool poolingDescriptorPool pp (fun _ -> new PoolingDescriptor())
    member t.reset () =
        d4Mp := 0
        wp := 0
        tp := 0
        fp := 0
        cp := 0
        pp := 0

let ObjectPool = new ObjectPool() // In the past iteration of the library, the object pool's role was taken by the tape. Not anymore.
let tape = new Stack<_>(1000) // Nice and simple way of passing in the closures for the backprop step.

let T = Operation.Transpose
let nT = Operation.NonTranspose

let defaultLayout = cudnnTensorFormat.NHWC
let defaultType = cudnnDataType.Float

type TensorDescriptor with
    /// Extended method that works according to the bound defaultLayout and defaultType variables.
    member t.SetTensor4dDescriptor(n,c,h,w) = t.SetTensor4dDescriptor(defaultLayout,defaultType,n,c,h,w)

type FilterDescriptor with
    /// Extended method that works according to the bound defaultType variable.
    member t.SetFilter4dDescriptor(n,c,h,w) = t.SetFilter4dDescriptor(defaultType,n,c,h,w)

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
    mode = cudnnConvolutionMode.CrossCorrelation
    }

type ConvolutionDescriptor with
    member inline t.SetConvolution2dDescriptor (p : ConvolutionParameters) =
        t.SetConvolution2dDescriptor(p.pad_h,p.pad_w,p.stride_h,p.stride_w,p.upscale_h,p.upscale_w,p.mode)
    member inline t.GetConvolution2dForwardOutputDim (s,f) =
        let mutable n,c,h,w = 0,0,0,0
        t.GetConvolution2dForwardOutputDim(s,f,&n,&c,&h,&w)
        n,c,h,w

let inline divup a b = (a-1)/b+1 // Integer division with rounding up. (a+b-1)/b is another variant on this.

let convolution_forward convPar (data : d4M) (filter : d4M) =
    let data_sizes = data.P.nchw
    let filter_sizes = filter.P.nchw

    let srcTensorDesc = ObjectPool.getTensorDescriptor
    let dstTensorDesc = ObjectPool.getTensorDescriptor
    let filterDesc = ObjectPool.getFilterDescriptor
    let convDesc = ObjectPool.getConvolutionDescriptor

    data_sizes |> srcTensorDesc.SetTensor4dDescriptor
    filter_sizes |> filterDesc.SetFilter4dDescriptor
    convPar |> convDesc.SetConvolution2dDescriptor 

    let output =
        let t = convDesc.GetConvolution2dForwardOutputDim(srcTensorDesc,filterDesc)
        t |> dstTensorDesc.SetTensor4dDescriptor
        t |> ObjectPool.getd4M false

    let algo = cudnn.GetConvolutionForwardAlgorithm(srcTensorDesc,filterDesc,convDesc,dstTensorDesc,cudnnConvolutionFwdPreference.PreferFastest,SizeT 0)
    let workspace = 
        cudnn.GetConvolutionForwardWorkspaceSize(srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo) |> int
        |> ObjectPool.getWorkspace

    let alpha = 1.0f
    let beta = 1.0f

    cudnn.ConvolutionForward(alpha,srcTensorDesc,data.P.dArray,filterDesc,filter.P.dArray,convDesc,algo,workspace,beta,dstTensorDesc,output.P.dArray)

    let convolution_backwards_filter () =
        let algo = cudnn.GetConvolutionBackwardFilterAlgorithm(srcTensorDesc,dstTensorDesc,convDesc,filterDesc,cudnnConvolutionBwdFilterPreference.PreferFastest,SizeT 0)
        let workspace =
            cudnn.GetConvolutionBackwardFilterWorkspaceSize(srcTensorDesc,dstTensorDesc,convDesc,filterDesc,algo) |> int
            |> ObjectPool.getWorkspace
        cudnn.ConvolutionBackwardFilter(alpha,srcTensorDesc,data.P.dArray,dstTensorDesc,output.A.dArray,convDesc,algo,workspace,beta,filterDesc,filter.A.dArray)

    let convolution_backwards_data () =
        let algo = cudnn.GetConvolutionBackwardDataAlgorithm(filterDesc,dstTensorDesc,convDesc,srcTensorDesc,cudnnConvolutionBwdDataPreference.PreferFastest,SizeT 0)
        let workspace =
            cudnn.GetConvolutionBackwardDataWorkspaceSize(filterDesc,dstTensorDesc,convDesc,srcTensorDesc,algo) |> int
            |> ObjectPool.getWorkspace
        cudnn.ConvolutionBackwardData(alpha,filterDesc,filter.P.dArray,dstTensorDesc,output.A.dArray,convDesc,beta,algo,workspace,srcTensorDesc,data.A.dArray)

    tape.Push(convolution_backwards_filter)
    if data.is_constant = false then tape.Push(convolution_backwards_data)

    output

let activation_forward mode (input : d4M)  =
    let input_sizes = input.P.nchw
    let srcTensorDesc = ObjectPool.getTensorDescriptor
    input_sizes |> srcTensorDesc.SetTensor4dDescriptor

    let alpha = 1.0f
    let beta = 1.0f

    let output = ObjectPool.getd4M false input_sizes

    cudnn.ActivationForward(mode,alpha,srcTensorDesc,input.P.dArray,beta,srcTensorDesc,output.P.dArray)

    let activation_backward () =
        cudnn.ActivationBackward(mode,alpha,srcTensorDesc,input.P.dArray,srcTensorDesc,input.A.dArray,srcTensorDesc,output.P.dArray,beta,srcTensorDesc,output.A.dArray)

    tape.Push(activation_backward)
    output

let pooling_forward p (input : d4M) =
    let poolingDescriptor = ObjectPool.getPoolingDescriptor
    poolingDescriptor.SetPooling2dDescriptor p

    let srcTensorDesc = ObjectPool.getTensorDescriptor
    let input_sizes = input.P.nchw
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

    cudnn.PoolingForward(poolingDescriptor,alpha,srcTensorDesc,input.P.dArray,beta,dstTensorDesc,output.P.dArray)

    let pooling_backward () =
        cudnn.PoolingBackward(poolingDescriptor,alpha,srcTensorDesc,input.P.dArray,srcTensorDesc,input.A.dArray,dstTensorDesc,output.P.dArray,beta,dstTensorDesc,output.A.dArray)

    tape.Push(pooling_backward)
    output

let kernels_dir = IO.Path.Combine(__SOURCE_DIRECTORY__,"Cuda Kernels")
IO.Directory.CreateDirectory(kernels_dir) // Creates the Cuda Kernels directory if it does not exist. WriteAllBytes would otherwise throw an exception.

/// o <- f(x)
type DeviceUnaryTransformModule(op: string) = 
    let block_size = 256

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
            __global__ void Map1Kernel(const floatType* A, floatType* O, const int N)
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

    let kernel = 
        let kernel_name = "Map1Kernel"
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

    member t.A(x: CudaDeviceVariable<float32>, o: CudaDeviceVariable<float32>, n) =
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore

    /// Uses the object pool.
    member t.A(x: d4Matrix) =
        let o = x.nchw |> ObjectPool.getd4Matrix
        t.A(x,o)
        o

    member t.A(x: d4Matrix, o: d4Matrix) =
        if x.nchw <> o.nchw then failwith "x.nchw <> o.nchw in DeviceUnaryTransformModule"
        t.A(x.dArray,o.dArray,x.size)

/// o <- f(x,y)
type DeviceBinaryTransformModule(op: string) = 
    let block_size = 256

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
            __global__ void Map2Kernel(const floatType* A, const floatType* B, floatType* O, const int N)
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
    
    let kernel = 
        let kernel_name = "Map2Kernel"
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

    member t.A(x: CudaDeviceVariable<float32>, y: CudaDeviceVariable<float32>, o: CudaDeviceVariable<float32>, n) =
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore

    /// Uses the object pool.
    member t.A(x: d4Matrix, y: d4Matrix) =
        let o = x.nchw |> ObjectPool.getd4Matrix
        t.A(x,y,o)
        o

    member t.A(x: d4Matrix, y: d4Matrix, o: d4Matrix) =
        if x.nchw <> y.nchw then failwith "x.nchw <> y.nchw in DeviceBinaryTransformModule"
        if y.nchw <> o.nchw then failwith "y.nchw <> o.nchw in DeviceBinaryTransformModule"
        t.A(x.dArray,y.dArray,o.dArray,x.size)

/// o <- f(x,y,z)
type DeviceTrinaryTransformModule(op: string) = 
    let block_size = 256

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
            __global__ void Map3Kernel(const floatType* A, const floatType* B, const floatType* C, floatType* O, const int N)
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

    let kernel = 
        let kernel_name = "Map3Kernel"
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

    member t.A(x: CudaDeviceVariable<float32>, y: CudaDeviceVariable<float32>, z: CudaDeviceVariable<float32>, o: CudaDeviceVariable<float32>,n) =
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,z.DevicePointer,o.DevicePointer,n) |> ignore

    /// Uses the object pool.
    member t.A(x: d4Matrix, y: d4Matrix, z: d4Matrix) =
        let o = x.nchw |> ObjectPool.getd4Matrix
        t.A(x,y,z,o)
        o

    member t.A(x: d4Matrix, y: d4Matrix, z: d4Matrix, o: d4Matrix) =
        if x.nchw <> y.nchw then failwith "x.nchw <> y.nchw in DeviceTrinaryTransformModule"
        if y.nchw <> z.nchw then failwith "y.nchw <> z.nchw in DeviceTrinaryTransformModule"
        if z.nchw <> o.nchw then failwith "z.nchw <> o.nchw in DeviceTrinaryTransformModule"
        t.A(x.dArray,y.dArray,z.dArray,o.dArray,x.size)

/// o <- sum(f(x))
type DeviceUnaryMapSumModule(op: string) = 
    let block_size = 256

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
            __global__ void Map1SumKernel(const floatType* A, floatType* O, const int N)
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

    let kernel = 
        let kernel_name = "Map1SumKernel"
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

    let o = new_dev<float32> 1

    member t.A(x: CudaDeviceVariable<float32>, n) =
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore
        o.[SizeT 0]

    member t.A(x: d4Matrix) =
        t.A(x.dArray, x.size)

/// o <- sum(f(x,y))
type DeviceBinaryMapSumModule(op: string) = 
    let block_size = 256

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
            __global__ void Map2SumKernel(const floatType* A, const floatType* B, floatType* O, const int N)
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

    let kernel = 
        let kernel_name = "Map2SumKernel"
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

    let o = new_dev<float32> 1

    member t.A(x: CudaDeviceVariable<float32>,y: CudaDeviceVariable<float32>,n) =
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore
        o.[SizeT 0]

    member t.A(x: d4Matrix,y: d4Matrix) =
        if x.nchw <> y.nchw then failwith "x.nchw <> y.nchw in DeviceBinaryMapSumModule"
        t.A(x.dArray,y.dArray,x.size)

/// o <- f(coef_x,x)
type DeviceUnaryCoefTransformModule(op: string) = 
    let block_size = 256

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
            __global__ void Map1CoefKernel(const floatType coef_A, const floatType* A, floatType* O, const int N)
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
        let kernel_name = "Map1CoefKernel"
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

    member t.A(coef_x: float32, x: CudaDeviceVariable<float32>, o: CudaDeviceVariable<float32>,n) =
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,o.DevicePointer,n) |> ignore

    member t.A(coef_x, x: d4Matrix) =
        let o = x.nchw |> ObjectPool.getd4Matrix
        t.A(coef_x,x,o)
        o

    member t.A(coef_x, x: d4Matrix, o: d4Matrix) =
        if x.nchw <> o.nchw then failwith "x.nchw <> o.nchw in DeviceUnaryCoefTransformModule"
        t.A(coef_x,x.dArray,o.dArray,x.size)

/// o <- f(coef_x,x,coef_y,y)
type DeviceBinaryCoefTransformModule(op: string) = 
    let block_size = 256

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
            __global__ void Map2CoefKernel(const floatType coef_A, const floatType* A, const floatType coef_B, const floatType* B, floatType* O, const int N)
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
        let kernel_name = "Map2CoefKernel"
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

    member t.A(coef_x: float32, x: CudaDeviceVariable<float32>, coef_y: float32, y: CudaDeviceVariable<float32>, o: CudaDeviceVariable<float32>,n) =
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,coef_y,y.DevicePointer,o.DevicePointer,n) |> ignore

    member t.A(coef_x, x: d4Matrix, coef_y, y: d4Matrix) =
        let o = x.nchw |> ObjectPool.getd4Matrix
        t.A(coef_x,x,coef_y,y,o)
        o

    member t.A(coef_x, x: d4Matrix, coef_y, y: d4Matrix, o: d4Matrix) =
        if x.nchw <> y.nchw then failwith "x.nchw <> y.nchw in DeviceBinaryCoefTransformModule"
        if y.nchw <> o.nchw then failwith "y.nchw <> o.nchw in DeviceBinaryCoefTransformModule"
        t.A(coef_x,x.dArray,coef_y,y.dArray,o.dArray,x.size)

/// o <- f(coef_x,x,coef_y,y,coef_z,z)
type DeviceTrinaryCoefTransformModule(op: string) = 
    let block_size = 256

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
            __global__ void Map3CoefKernel(const floatType coef_A, const floatType* A, const floatType coef_B, const floatType* B, const floatType coef_C, const floatType* C, floatType* O, const int N)
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
        let kernel_name = "Map3CoefKernel"
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

    member t.A(coef_x: float32, x: CudaDeviceVariable<float32>, coef_y: float32, y: CudaDeviceVariable<float32>, coef_z: float32, z: CudaDeviceVariable<float32>, o: CudaDeviceVariable<float32>,n) =
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,coef_y,y.DevicePointer,coef_z,z.DevicePointer,o.DevicePointer,n) |> ignore

    /// Uses the object pool.
    member t.A(coef_x, x: d4Matrix, coef_y, y: d4Matrix, coef_z, z: d4Matrix) =
        let o = x.nchw |> ObjectPool.getd4Matrix
        t.A(coef_x,x,coef_y,y,coef_z,z,o)
        o

    member t.A(coef_x, x: d4Matrix, coef_y, y: d4Matrix, coef_z, z: d4Matrix, o: d4Matrix) =
        if x.nchw <> y.nchw then failwith "x.nchw <> y.nchw in DeviceTrinaryCoefTransformModule"
        if y.nchw <> z.nchw then failwith "y.nchw <> z.nchw in DeviceTrinaryCoefTransformModule"
        if z.nchw <> o.nchw then failwith "z.nchw <> o.nchw in DeviceTrinaryCoefTransformModule"
        t.A(coef_x,x.dArray,coef_y,y.dArray,coef_z,z.dArray,o.dArray,x.num_rows*x.num_cols)

// The gradient clipping module.
let gradclipModule = lazy DeviceUnaryCoefTransformModule "(x < -coef_x) ? -coef_x : (x > coef_x ? coef_x : x);"

// coef_x = scale
// coef_y = location
// y does not get used.
let randMapModule = lazy DeviceBinaryCoefTransformModule "coef_x*(x-0.5f)+coef_y;"

type d4Matrix with
    /// Generates a matrix sampled from a random uniform distribution in <-1.0f,1.0f]
    /// Uses the object pool.
    static member createRandomUniformMatrix' (nn,nc,nh,nw as p) (scaling_factor : float32) location =
        let weights_total_size = nn*nc*nh*nw
        
        let cudaBuffer = ObjectPool.getd4Matrix p
        cudaRandom.GenerateUniform(cudaBuffer.dArray)

        // 2.0f*scaling_factor ensures that it is rescaled around zero if the scaling_factor is 1.0f.
        randMapModule.Value.A(2.0f*scaling_factor,cudaBuffer.dArray,location,cudaBuffer.dArray,cudaBuffer.dArray,weights_total_size)

        cudaBuffer

    /// Generates a matrix sampled from a random uniform distribution in <-1.0f,1.0f]
    static member createRandomUniformMatrix (nn,nc,nh,nw as p) (scaling_factor : float32) location =
        let weights_total_size = nn*nc*nh*nw
        
        let cudaBuffer = p |> d4Matrix.create
        cudaRandom.GenerateUniform(cudaBuffer.dArray)

        // 2.0f*scaling_factor ensures that it is rescaled around zero if the scaling_factor is 1.0f.
        randMapModule.Value.A(2.0f*scaling_factor,cudaBuffer.dArray,location,cudaBuffer.dArray,cudaBuffer.dArray,weights_total_size)

        cudaBuffer

    /// Fills matrix by sampling from a random uniform distribution in <-1.0f,1.0f]
    member t.fillRandomUniformMatrix (scaling_factor : float32) location =
        let weights_total_size = t.num_images*t.num_channels*t.num_rows*t.num_cols

        cudaRandom.GenerateUniform(t.dArray)
        // 2.0f*scaling_factor ensures that it is rescaled around zero if the scaling_factor is 1.0f.
        randMapModule.Value.A(2.0f*scaling_factor,t,location,t,t)

type d4M with
    /// Generates a matrix sampled from a random uniform distribution in <-1.0f,1.0f]
    /// Uses the object pool.
    static member createRandomUniformDM' is_constant (nn,nc,nh,nw as p) (scaling_factor : float32) location =
        let pr = d4Matrix.createRandomUniformMatrix' p scaling_factor location // primal
        let ad = if is_constant then ObjectPool.getd4Matrix p else d4Matrix.createEmpty // adjoint
        ad.setZero()
        {P = pr; A = ad; is_constant=is_constant}

    /// Generates a matrix sampled from a random uniform distribution in <-1.0f,1.0f]
    static member createRandomUniformDM is_constant (nn,nc,nh,nw as p) (scaling_factor : float32) location =
        let pr = d4Matrix.createRandomUniformMatrix p scaling_factor location // primal
        let ad = if is_constant then ObjectPool.getd4Matrix p else d4Matrix.createEmpty // adjoint
        ad.setZero()
        {P = pr; A = ad; is_constant=is_constant}
        
type Df = {
    P : float32 ref
    A : float32 ref
    is_constant : bool
    } with

    static member create P =
        {P=P;A=ref 0.0f;is_constant=false}
    static member createConstant P =
        {P=P;A=ref 0.0f;is_constant=true}