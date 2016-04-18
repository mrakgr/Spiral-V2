// I made this example to check the correctness of BN in the feedforward case, but it seems it is entirely correct.
// This makes me wonder why it works so poorly on N puzzle.

open System
open System.IO

#if INTERACTIVE
#load "spiral_conv_v3.fsx"
#endif
open SpiralV2

open ManagedCuda
open ManagedCuda.BasicTypes
open ManagedCuda.CudaDNNv5

let wait_on_event (s : CudaStream) (occupied_ar : ResizeArray<CudaEvent>) =
        occupied_ar
        |> Seq.iter (fun x -> s.WaitEvent x.Event)

type d4MUnion =
    {
    mutable elements : ((int * int * int * int) * CUdeviceptr * CUdeviceptr option)[]
    mutable P: CudaDeviceVariable<float32> // primal
    mutable A: CudaDeviceVariable<float32> option // adjoint
    mutable is_dead : bool // flag to skip backprop
    mutable primal_occupied : ResizeArray<CudaEvent>
    mutable adjoint_occupied : ResizeArray<CudaEvent>
    }  

    static member private ar_scan (ar : (int * int * int * int)[]) = ar |> Array.scan (fun state e -> state + sizeof<float32> * size_nchw e) 0
    static member private ar_fold (ar : (int * int * int * int)[]) = ar |> Array.fold (fun state e -> state + sizeof<float32> * size_nchw e) 0
    static member private make_elements (p : CudaDeviceVariable<float32>) (a : CudaDeviceVariable<float32> option) (ar : (int * int * int * int)[]) (l : int[]) =
        [|
        for i=0 to l.Length-2 do
            let x = ar.[i]
            let size = l.[i+1] - l.[i]
            yield 
                x, 
                p.DevicePointer + SizeT l.[i], 
                match a with
                | Some a -> a.DevicePointer + SizeT l.[i] |> Some
                | None -> None
        |]

    static member private create' (ar : (int * int * int * int)[], is_constant) =
        let l = d4MUnion.ar_scan ar
        let p,a = l |> Array.last |> fun x -> x / sizeof<float32> |> SizeT |> fun x -> new CudaDeviceVariable<float32>(x), if is_constant = false then new CudaDeviceVariable<float32>(x) |> Some else None
        let elements = d4MUnion.make_elements p a ar l
        
        {elements = elements; P=p; A=a; is_dead=false; primal_occupied = ResizeArray(); adjoint_occupied = ResizeArray()}

    static member private create' (ar_data : ((int * int * int * int) * float32[]) [], is_constant) =
        let ar, data = Array.unzip ar_data
        let t = d4MUnion.create' (ar, is_constant)
        for i=0 to data.Length-1 do
            let x = data.[i]
            t.elements.[i]
            |> fun (nchw,p,a) ->
                let size = size_nchw nchw
                if size <> x.Length then failwithf "size(%i) <> data.[%i].Length(%i)" size i x.Length
                ctx.CopyToDevice(p,x)
        t

    static member inline create (ar : (int * int * int * int)[]) = d4MUnion.create'(ar, false)
    static member inline create (ar_data : ((int * int * int * int) * float32[]) []) = d4MUnion.create'(ar_data, false)
    static member inline createConstant (ar : (int * int * int * int)[]) = d4MUnion.create'(ar, true)
    static member inline createConstant (ar_data : ((int * int * int * int) * float32[]) []) = d4MUnion.create'(ar_data, true)

    // Constructors for the singular d4MUnion records.
    static member inline create (ar : int * int * int * int) = d4MUnion.create'([|ar|], false)
    static member inline create (ar_data : (int * int * int * int) * float32[]) = d4MUnion.create'([|ar_data|], false)
    static member inline createConstant (ar : int * int * int * int) = d4MUnion.create'([|ar|], true)
    static member inline createConstant (ar_data : (int * int * int * int) * float32[]) = d4MUnion.create'([|ar_data|], true)

    /// Checks if the type is singular and then returns the primal along with its dimensions.
    member t.P' = 
        if t.elements.Length <> 1 then failwithf "t.elements.Length(%i) <> 1" t.elements.Length
        else
            t.elements.[0]
            |> fun (nchw,p,_) -> nchw,p,t.primal_occupied

    /// Checks if the type is singular and then returns the adjoint along with its dimensions.
    member t.A' = 
        if t.elements.Length <> 1 then failwithf "t.elements.Length(%i) <> 1" t.elements.Length
        else
            t.elements.[0]
            |> fun (nchw,_,a) -> nchw,a.Value,t.primal_occupied

    /// Gets the slice by iteratively incrementing n if the other dimensions are equal.
    /// Does not do any layout transformation. The same as GetSliceAlongChannel.
    member t.GetSliceAlongImage (l,r) =
        let mutable (n,c,h,w),p,a = t.elements.[l]
        if l > r then failwith "l > r"
        for i=l+1 to r do
            t.elements.[i] 
            |> fun ((n',c',h',w'),_,_) -> 
                if c' <> c || h' <> h || w' <> w then failwithf "c'(%i) <> c(%i) || h'(%i) <> h(%i) || w'(%i) <> w(%i)" c' c h' h w' w
                n <- n + n'
        (n,c,h,w), p, a

    /// Gets the slice by iteratively incrementing c if the other dimensions are equal.
    /// Does not do any layout transformation. The same as GetSliceAlongImage.
    member t.GetSliceAlongChannel (l,r) =
        let mutable (n,c,h,w),p,a = t.elements.[l]
        if l > r then failwith "l > r"
        for i=l+1 to r do
            t.elements.[i] 
            |> fun ((n',c',h',w'),_,_) -> 
                if n' <> n || h' <> h || w' <> w then failwithf "n'(%i) <> n(%i) || h'(%i) <> h(%i) || w'(%i) <> w(%i)" n' n h' h w' w
                c <- c + c'
        (n,c,h,w), p, a

    member t.ReplaceIf (ar : (int * int * int * int)[]) =
        let l = d4MUnion.ar_scan ar
        let new_size = l |> Array.last
        if int t.P.SizeInBytes < new_size
        then
            (t :> IDisposable).Dispose()
            let t' = d4MUnion.create'(ar,t.A.IsNone)
            t.elements <- t'.elements
            t.P <- t'.P
            t.A <- t'.A
        else
            let p,a = t.P, t.A
            let elements = d4MUnion.make_elements p a ar l
            t.elements <- elements

    interface IDisposable with
        member t.Dispose() = 
            t.P.Dispose()
            match t.A with
            | Some A -> A.Dispose()
            | None -> ()

let gather_pointer (nchw, p) =
    let size = size_nchw nchw
    let t = Array.zeroCreate<float32> size
    ctx.CopyToHost(t,p)
    t

type StreamPool(num) =
    let ar = Array.init num (fun _ -> new CudaStream(), new CudaEvent(CUEventFlags.DisableTiming))
    let mutable p = -1

    // Pops a Stream, Event tuple from the ring buffer.
    member t.P = 
        p <- p + 1
        let (s,e as t) = ar.[p % num]
        //if e.Query() = false then e.Synchronize() // If the stream is in use, then block on the associated event.
        str.WaitEvent e.Event // Why not just use this instead? Edit: It speeds up the example 100% compared to the above.
        t

let StreamPool = new StreamPool(128) // 1024 of them take roughly 50Mb, so I've settled on the 128 number.

/// General matrix-matrix multiply from cuBLAS. Inplace version
/// c,h,w get multiplied together to form the first dimension. n is the second dimension.
let gemm transa transb (alpha: float32) ((A_num_images, A_num_channels, A_num_rows, A_num_cols), A:CUdeviceptr, occ_A : ResizeArray<CudaEvent>) ((B_num_images, B_num_channels, B_num_rows, B_num_cols), B:CUdeviceptr, occ_B : ResizeArray<CudaEvent>) beta ((C_num_images, C_num_channels, C_num_rows, C_num_cols), C:CUdeviceptr, occ_C : ResizeArray<CudaEvent>) =
    let inline gemm (A_num_rows, A_num_cols) (B_num_rows, B_num_cols) (C_num_rows, C_num_cols) =
        let a_col = if transa = nT then A_num_cols else A_num_rows
        let b_row = if transb = nT then B_num_rows else B_num_cols
        if a_col <> b_row then failwithf "a_col(%i) <> b_row(%i) in gemm!" a_col b_row
        let m = if transa = nT then A_num_rows else A_num_cols
        let n = if transb = nT then B_num_cols else B_num_rows
        let k = a_col
        let lda = if transa = nT then m else k
        let ldb = if transb = nT then k else n
        let ldc = m

        if m <> C_num_rows || n <> C_num_cols then failwithf "m(%i) <> C_num_rows(%i) || n(%i) <> C_num_cols(%i)" m C_num_rows n C_num_cols

        let str,event = StreamPool.P
        cublas.Stream <- str.Stream

        wait_on_event str occ_A
        wait_on_event str occ_B
        wait_on_event str occ_C

        use A = new CudaDeviceVariable<float32>(A,false,sizeof<float32> * A_num_rows * A_num_cols |> SizeT)
        use B = new CudaDeviceVariable<float32>(B,false,sizeof<float32> * B_num_rows * B_num_cols |> SizeT)
        use C = new CudaDeviceVariable<float32>(C,false,sizeof<float32> * C_num_rows * C_num_cols |> SizeT)
        cublas.Gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

        event.Record str.Stream
        occ_C.Add event
    gemm (A_num_channels*A_num_cols*A_num_rows,A_num_images) (B_num_channels*B_num_cols*B_num_rows,B_num_images) (C_num_channels*C_num_cols*C_num_rows,C_num_images)

/// Fills matrix by sampling from a random uniform distribution in <-1.0f,1.0f]
let fillRandomUniformMatrix ((A_num_images, A_num_channels, A_num_rows, A_num_cols as nchw), A:CUdeviceptr, occ_A : ResizeArray<CudaEvent>) (scaling_factor : float32) location =
        use x = new CudaDeviceVariable<float32>(A,false,sizeof<float32> * size_nchw nchw |> SizeT)
        cudaRandom.GenerateUniform(x)

let t =
    Array.init 
    <| 3000
    <| fun _ ->
        (128,256,1,1) |> d4MUnion.createConstant |> (fun x -> fillRandomUniformMatrix x.P' 1.0f 0.0f; x),
        (256,128,1,1) |> d4MUnion.createConstant |> (fun x -> fillRandomUniformMatrix x.P' 1.0f 0.0f; x),
        (256,256,1,1) |> d4MUnion.createConstant

for (a1,a2,a3) in t do
    a1.primal_occupied.Clear()
    a2.primal_occupied.Clear()
    a3.primal_occupied.Clear() // Cleaning up the occupied array is necessary. It seems that waiting on events is quite an expensive operation.
    a3.P.Memset(0uy)
#time
for i=1 to 1 do
    t
    |> Array.iter (fun (a,b,c) -> 
        gemm nT nT 1.0f a.P' b.P' 1.0f c.P'
        //ctx.Synchronize()
        ) // Having the ctx.Synchronize inside is 7x slower than having it outside.
    ctx.Synchronize()
#time

let s = t.[5] |> fun (_,_,x) -> x.P.Gather()
let s' = t.[5] |> fun (_,_,x) -> x.P.Gather() // This one is triggered after switching the ctx in the loop above and then running it again.
s = s' // This is enough to satisfy me that there aren't any data races. This thing really works. Amazing.
