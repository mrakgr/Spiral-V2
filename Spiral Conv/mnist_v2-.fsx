// Maybe the data needs a transpose first...
// No, it is not the tranpose.

// This narrows my error down to the convolutional functions.

open System
open System.IO

#if INTERACTIVE
#load "spiral_conv.fsx"
#endif
open Spiral

let minibatch_size = 128
let load_mnist filename =
    let inline transpose (t : d4M) = 
        let output = d4M.createConstant(1,1,t.num_cols,t.num_rows)
        geam T T 1.0f t.P' 0.0f t.P' output.P' // Transpose function
        output

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
        |> Array.map (fun x -> 
            let num_cols = x.Length/10
            let num_rows = 10
            d4M.createConstant (1,1,num_rows,num_cols,x)
            |> transpose
            |> fun x -> {num_images=num_cols; num_channels=num_rows;num_rows=1;num_cols=1;P=x.P;A=None})
    | 2051 -> // Images
        let n = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        let rows = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        let cols = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        d.ReadBytes(n * rows * cols)
        |> Array.map (fun x -> float32 x / 255.0f)
        |> Array.chunkBySize (minibatch_size*rows*cols)
        |> Array.map (fun x -> 
            let num_cols = x.Length/(rows*cols)
            let num_rows = rows*cols
            d4M.createConstant (1,1,num_rows,num_cols,x)
            |> transpose
            |> fun x -> {num_images=num_cols; num_channels=1;num_rows=rows;num_cols=cols;P=x.P;A=None})
    | _ -> failwith "Given file is not in the MNIST format."

let [|test_images;test_labels;train_images;train_labels|] = 
    [|"t10k-images.idx3-ubyte";"t10k-labels.idx1-ubyte";"train-images.idx3-ubyte";"train-labels.idx1-ubyte"|]
    |> Array.map (fun x -> Path.Combine(__SOURCE_DIRECTORY__,x) |> load_mnist)

let relu = ManagedCuda.CudaDNN.cudnnActivationMode.Relu
let sigmoid = ManagedCuda.CudaDNN.cudnnActivationMode.Sigmoid

let l1 = ConvolutionalFeedforwardLayer.createRandomLayer (10,1,28,28) sigmoid

let base_nodes = [|l1|]

let training_loop label data = // For now, this is just checking if the new library can overfit on a single minibatch.
    [|
    defaultConvPar,l1
    |] 
    |> Array.fold (fun x (convPars,layer) -> layer.runLayer (convPars,x)) data
    |> fun x -> lazy get_accuracy label x, squared_error_cost label x

let learning_rate = 0.03f

for i=1 to 100 do
    let mutable er = 0.0f
    for j=0 to train_images.Length-1 do
        let _,r = training_loop train_labels.[j] train_images.[j] // Forward step
        ObjectPool.Reset() // Resets all the adjoints from the top of the pointer in the object pool along with the pointers.
        base_nodes |> Array.iter (fun x -> x.ResetAdjoints())
        er <- er + !r.P
        //printfn "Squared error cost on the minibatch is %f at batch %i" !r.P j

        if !r.P |> Single.IsNaN then failwith "Nan!"

        r.A := 1.0f // Loads the 1.0f at the top
        while tape.Count > 0 do tape.Pop() |> fun x -> x() // The backpropagation step
        base_nodes |> Array.iter (fun x -> saxpy learning_rate x.W.A' x.W.P'; saxpy learning_rate x.b.A' x.b.P') // Stochastic gradient descent.
    printfn "-----"
    printfn "Squared error cost on the dataset is %f at iteration %i" (er / float32 train_images.Length) i

    let mutable acc = 0.0f
    for j=0 to test_images.Length-1 do
        let acc',r = training_loop test_labels.[j] test_images.[j] // Forward step
        ObjectPool.ResetPointers()
        tape.Clear()
        acc <- acc'.Value + acc

    printfn "Accuracy on the test set is %i/10000." (int acc)
    printfn "-----"
