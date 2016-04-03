﻿// The combination of there being a bugs in both tensor_add and the gradient descent part fucking demolished me.
// Not having negative beta combined with adding the opposite of the correct learning rate combined to make a slightly correct solution.
// This threw off my assumptions completely and even after I realized that there was an error in tensor_add I was completely bewildered to
// find that net was exploding after I'd fixed it. I could not go an extra step in my thinking and ended up wasting so much time.

// I think I now wasted an entire day hunting this thing down.

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

let l1 = ConvolutionalFeedforwardLayer.createRandomLayer(10,1,28,28) sigmoid

let base_nodes = [|l1|]

let training_loop label data = // For now, this is just checking if the new library can overfit on a single minibatch.
    [|
    defaultConvPar,l1
    |] 
    |> Array.fold (fun x (convPars,layer) -> layer.runLayer (convPars,x)) data
    |> fun x -> lazy get_accuracy label x, squared_error_cost label x

let learning_rate = 1.0f

let test() =
    for i=1 to 100 do
        let mutable er = 0.0f
        for j=0 to train_images.Length-1 do
            let _,r = training_loop train_labels.[j] train_images.[j] // Forward step
            er <- er + !r.P
            ObjectPool.Reset() // Resets all the adjoints from the top of the pointer in the object pool along with the pointers.
            base_nodes |> Array.iter (fun x -> x.ResetAdjoints())
            //printfn "Squared error cost on the minibatch is %f at batch %i" !r.P j

            if !r.P |> Single.IsNaN then failwith "Nan!"

            r.A := 1.0f // Loads the 1.0f at the top
            while tape.Count > 0 do tape.Pop() |> fun x -> x() // The backpropagation step
            base_nodes |> Array.iter (fun x -> x.SGD learning_rate) // Stochastic gradient descent.
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

test()
