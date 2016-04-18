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

type d4MUnion =
    {
    mutable elements : ((int * int * int * int) * CUdeviceptr * CUdeviceptr option)[]
    mutable P: CudaDeviceVariable<float32> // primal
    mutable A: CudaDeviceVariable<float32> option // adjoint
    mutable is_dead : bool // flag to skip backprop
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
        
        {elements = elements; P=p; A=a; is_dead=false}

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

    // Gets the slice by iteratively incrementing n if the other dimensions are equal.
    // Does not do any layout transformation. The same as GetSliceAlongChannel.
    member t.GetSliceAlongImage (l,r) =
        let mutable (n,c,h,w),p,a = t.elements.[l]
        if l > r then failwith "l > r"
        for i=l+1 to r do
            t.elements.[i] 
            |> fun ((n',c',h',w'),_,_) -> 
                if c' <> c || h' <> h || w' <> w then failwithf "c'(%i) <> c(%i) || h'(%i) <> h(%i) || w'(%i) <> w(%i)" c' c h' h w' w
                n <- n + n'
        (n,c,h,w), p, a

    // Gets the slice by iteratively incrementing c if the other dimensions are equal.
    // Does not do any layout transformation. The same as GetSliceAlongImage.
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

let c = d4MUnion.create [|(1,10,1,1),[|1.0f..10.0f|];(1,8,1,1),[|11.0f..18.0f|];(1,6,1,1),[|19.0f..24.0f|];(1,4,1,1),[|25.0f..28.0f|]|]

let gather_pointer (nchw, p) =
    let size = size_nchw nchw
    let t = Array.zeroCreate<float32> size
    ctx.CopyToHost(t,p)
    t

c.GetSliceAlongChannel (2,3) |> fun (nchw,x,_) -> gather_pointer (nchw,x)


