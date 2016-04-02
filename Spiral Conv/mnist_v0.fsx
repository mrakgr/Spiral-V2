open System
open System.IO

#if INTERACTIVE
#load "spiral_conv.fsx"
#endif
open Spiral

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
        |> Array.map (fun x -> d4M.createConstant (x.Length/(rows*cols),1,rows,cols,x))
    | _ -> failwith "Given file is not in the MNIST format."

let [|test_images;test_labels;train_images;train_labels|] = 
    [|"t10k-images.idx3-ubyte";"t10k-labels.idx1-ubyte";"train-images.idx3-ubyte";"train-labels.idx1-ubyte"|]
    |> Array.map (fun x -> Path.Combine(__SOURCE_DIRECTORY__,x) |> load_mnist)

let relu = ManagedCuda.CudaDNN.cudnnActivationMode.Relu
let sigmoid = ManagedCuda.CudaDNN.cudnnActivationMode.Sigmoid

// This does not quite work for me yet.
// The library is too fresh. Let me try logistic regression.

let l1 = ConvolutionalFeedforwardLayer.createRandomLayer (32,1,5,5) relu
let l2 = ConvolutionalFeedforwardLayer.createRandomLayer (32,32,5,5) relu
let l3 = ConvolutionalFeedforwardLayer.createRandomLayer (32,32,5,5) relu
let l4 = ConvolutionalFeedforwardLayer.createRandomLayer (32,32,5,5) relu
let l5 = ConvolutionalFeedforwardLayer.createRandomLayer (10,32,4,4) sigmoid

let base_nodes = [|l1;l2;l3;l4;l5|]

let training_loop() = // For now, this is just checking if the new library can overfit on a single minibatch.
    [|
    defaultConvPar,l1
    defaultConvPar,l2
    {defaultConvPar with stride_h=2; stride_w=2},l3
    defaultConvPar,l4
    defaultConvPar,l5
    |] 
    |> Array.fold (fun x (convPars,layer) -> layer.runLayer (convPars,x)) train_images.[0]
    |> squared_error_cost train_labels.[0]

let learning_rate = 0.0001f

for i=1 to 100 do
    let r = training_loop() // Forward step
    ObjectPool.Reset() // Resets all the adjoints from the top of the pointer in the object pool
    base_nodes |> Array.iter (fun x -> x.ResetAdjoints())
    printfn "Squared error cost on the minibatch is %f at iteration %i" !r.P i

    r.A := 1.0f // Loads the 1.0f at the top
    while tape.Count > 0 do tape.Pop() |> fun x -> x() // The backpropagation step
    base_nodes |> Array.iter (fun x -> saxpy learning_rate x.W.A' x.W.P'; saxpy learning_rate x.b.A' x.b.P') // Stochastic gradient descent.
