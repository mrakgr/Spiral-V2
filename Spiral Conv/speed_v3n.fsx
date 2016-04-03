// As setting the descriptors is a significant source of slowness, it would be great if I could place them in a dictionary and share them amongst
// the functions. Before I do that modification let me test here whether they can be shared.

open System
open System.IO

#if INTERACTIVE
#load "spiral_conv.fsx"
#endif
open SpiralV2

let minibatch_size = 128
let load_mnist filename =
    use f = File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read)
    use d = new BinaryReader(f)

    let magicnumber = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
    match magicnumber with
    | 2049 -> // Labels
        let n = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        d.ReadBytes n
        |> Array.collect (
            fun x -> 
                let t = Array.zeroCreate 10
                t.[int x] <- 1.0f
                t)
        |> Array.chunkBySize (minibatch_size*10)
        |> Array.map (fun x -> d4M.createConstant (x.Length/10,10,1,1,x))
    | 2051 -> // Images
        let n = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        let rows = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        let cols = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        d.ReadBytes(n * rows * cols)
        |> Array.map (fun x -> float32 x / 255.0f)
        |> Array.chunkBySize (minibatch_size*rows*cols)
        |> Array.map (fun x -> d4M.createConstant (x.Length/(rows*cols),rows*cols,1,1,x))
    | _ -> failwith "Given file is not in the MNIST format."

let [|test_images;test_labels;train_images;train_labels|] = 
    [|"t10k-images.idx3-ubyte";"t10k-labels.idx1-ubyte";"train-images.idx3-ubyte";"train-labels.idx1-ubyte"|]
    |> Array.map (fun x -> Path.Combine(__SOURCE_DIRECTORY__,x) |> load_mnist)



let l1 = FeedforwardLayer.createRandomLayer (784,2024,1,1) relu
let relu_clip x = clip 0.0f Single.PositiveInfinity x 0.0f
//let l1 = FeedforwardLayer.createRandomLayer (784,2024,1,1) relu_clip

let input = train_images.[0]

let input_sizes = input.nchw

let srcTensorDesc0 = 
    let t = ObjectPool.getTensorDescriptor
    input_sizes |> t.SetTensor4dDescriptor
    t

let srcTensorDesc1 = 
    let t = ObjectPool.getTensorDescriptor
    input_sizes |> t.SetTensor4dDescriptor
    t

let srcTensorDesc2 = 
    let t = ObjectPool.getTensorDescriptor
    input_sizes |> t.SetTensor4dDescriptor
    t

let srcTensorDesc3 = 
    let t = ObjectPool.getTensorDescriptor
    input_sizes |> t.SetTensor4dDescriptor
    t

let srcTensorDesc4 = 
    let t = ObjectPool.getTensorDescriptor
    input_sizes |> t.SetTensor4dDescriptor
    t

let output = ObjectPool.getd4M false input_sizes

let inline training_loop data = 
    let af (srcTensorDesc : ManagedCuda.CudaDNN.TensorDescriptor) =
        let alpha = 1.0f
        let beta = 1.0f

        cudnn.ActivationForward(ManagedCuda.CudaDNN.cudnnActivationMode.Relu,alpha,srcTensorDesc,input.P,0.0f,srcTensorDesc,output.P)

    for i=0 to 4 do // 20 of {af srcTensorDesc0} are faster than 4 of {af srcTensorDesc0,af srcTensorDesc1,af srcTensorDesc2,af srcTensorDesc3,af srcTensorDesc4}
        af srcTensorDesc0
        af srcTensorDesc1
        af srcTensorDesc2
        af srcTensorDesc3
        af srcTensorDesc4

let learning_rate = 0.03f

let test num_iterations =
    for i=1 to num_iterations do
        training_loop input
        ObjectPool.ResetPointers() // Resets all the adjoints from the top of the pointer in the object pool along with the pointers.
        tape.Clear()
        ctx.Synchronize()
#time
test 25000
#time
