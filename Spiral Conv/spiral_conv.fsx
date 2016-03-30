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
    mutable num_rows:int
    mutable num_cols:int
    mutable num_channels:int
    mutable dArray: CudaDeviceVariable<float32>
    }  

    /// The main create function. A substitute for the constructor.
    static member create(num_images: int,num_rows: int,num_cols: int, num_channels:int, dArray: CudaDeviceVariable<float32>) =
        {num_images=num_images; num_rows=num_rows; num_cols=num_cols;num_channels=num_channels;dArray=dArray}

    /// Throws an exception if it tries to allocate an array of size 0.
    static member create(num_images: int,num_rows: int,num_cols: int, num_channels:int) =
        let q = (num_images*num_rows*num_cols*num_channels) |> SizeT
        let t = new CudaDeviceVariable<float32>(q)
        {num_images=num_images; num_rows=num_rows; num_cols=num_cols;num_channels=num_channels;dArray=t}

    /// Copies a host to a device array.
    /// Throws an exception if it tries to allocate an array of size 0.
    static member create(num_images: int,num_rows: int,num_cols: int, num_channels:int,dArray: float32[]) =
        let q = num_images*num_rows*num_cols*num_channels
        if dArray.Length <> q then failwith "Invalid size in dMatrix construction."
        let t = to_dev dArray
        {num_images=num_images; num_rows=num_rows; num_cols=num_cols;num_channels=num_channels;dArray=t}

    /// Returns a new instance of an (dMatrix.createEmpty) dMatrix.
    /// Unlike the let statements, the member statements are always reevaluated.
    static member createEmpty = d4Matrix.create(0,0,0,0,CudaDeviceVariable.Null)

    /// Returns nhwc as a tuple
    member inline t.nhwc = t.num_images, t.num_rows, t.num_cols, t.num_channels
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

    /// Resized the dArray if the current one is less than nr*nc. Otherwise it only adjusts num_rows and num_cols.
    member inline t.ReplaceIf nn nh nw nc =
        let new_size = nn*nh*nw*nc
        if int t.dArray.Size < new_size
        then
            (t :> IDisposable).Dispose()
            t.num_images <- nn
            t.num_rows <- nh
            t.num_cols <- nw
            t.num_channels <- nc
            t.dArray <- new_dev new_size
        else
            t.num_images <- nn
            t.num_rows <- nh
            t.num_cols <- nw
            t.num_channels <- nc

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

let T = Operation.Transpose
let nT = Operation.NonTranspose

