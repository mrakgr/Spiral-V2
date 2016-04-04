// Spiral reverse AD example. Used for testing.
#load "spiral.fsx"
open Spiral

open System

open ManagedCuda
open ManagedCuda.BasicTypes
open ManagedCuda.VectorTypes
open ManagedCuda.CudaBlas
open ManagedCuda.CudaRand
open ManagedCuda.NVRTC
open ManagedCuda.CudaDNN

open System
open System.IO
open System.Collections

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
        |> Array.map (fun x -> 
            dMatrix.create (10,x.Length/10,x)
            |> DM.makeConstantNode)
    | 2051 -> // Images
        let n = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        let rows = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        let cols = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        d.ReadBytes(n * rows * cols)
        |> Array.map (fun x -> float32 x / 255.0f)
        |> Array.chunkBySize (minibatch_size*rows*cols)
        |> Array.map (fun x -> 
            dMatrix.create (rows*cols,x.Length/(rows*cols),x)
            |> DM.makeConstantNode)
    | _ -> failwith "Given file is not in the MNIST format."

let [|dtest_images;test_labels;train_images;train_labels|] = 
    [|"t10k-images.idx3-ubyte";"t10k-labels.idx1-ubyte";"train-images.idx3-ubyte";"train-labels.idx1-ubyte"|]
    |> Array.map (fun x -> Path.Combine(__SOURCE_DIRECTORY__,x) |> load_mnist)

let dtest = Array.zip dtest_images test_labels 
let dtrain = Array.zip train_images train_labels

let l1 = FeedforwardLayer.createRandomLayer 2048 784 relu
let l2 = FeedforwardLayer.createRandomLayer 2048 2048 relu
let l3 = FeedforwardLayer.createRandomLayer 2048 2048 relu
let l4 = FeedforwardLayer.createRandomLayer 10 2048 clipped_sigmoid

//let l1 = load_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl1")) false |> fun x -> FeedforwardLayer.fromArray x relu
//let l2 = load_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl2")) false |> fun x -> FeedforwardLayer.fromArray x relu
//let l3 = load_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl3")) false |> fun x -> FeedforwardLayer.fromArray x relu
//let l4 = load_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl4")) false |> fun x -> FeedforwardLayer.fromArray x clipped_sigmoid
//
//save_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl1")) l1.ToArray
//save_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl2")) l2.ToArray
//save_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl3")) l3.ToArray
//save_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl4")) l4.ToArray

//let l4 = FeedforwardLayer.createRandomLayer 10 1024 (clipped_steep_sigmoid 3.0f)
let layers = [|l1;l2;l3;l4|]

// This does not actually train it, it just initiates the tree for later training.
let training_loop (data: DM) (targets: DM) (layers: FeedforwardLayer[]) =
    let outputs = layers |> Array.fold (fun state layer -> layer.runLayer state) data
    // I make the accuracy calculation lazy. This is similar to returning a lambda function that calculates the accuracy
    // although in this case it will be calculated at most once.
    lazy get_accuracy targets.r.P outputs.r.P, cross_entropy_cost targets outputs 

let train_mnist_sgd num_iters learning_rate (layers: FeedforwardLayer[]) =
    [|
    let base_nodes = layers |> Array.map (fun x -> x.ToArray) |> Array.concat // Stores all the base nodes of the layer so they can later be reset.
    for i=1 to num_iters do
        let mutable r' = 0.0f
        for x in dtrain do
            let data, target = x
            let _,r = training_loop data target layers // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 dtrain.Length) // Adds the cost to the accumulator.
            if System.Single.IsNaN r' then failwith "Nan error"

            for x in base_nodes do x.r.A.setZero() // Resets the base adjoints
            tape.resetTapeAdjoint 0 // Resets the adjoints for the training select
            r.r.A := 1.0f // Pushes 1.0f from the top node
            tape.reversepropTape 0 // Resets the adjoints for the test select
            add_gradients_to_weights' base_nodes learning_rate // The optimization step
            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed for the simple recurrent and feedforward case.

        printfn "The training cost at iteration %i is %f" i r'
        let r1 = r'
        r' <- 0.0f
        let mutable acc = 0.0f

        for x in dtest do
            let data, target = x
            let lazy_acc,r = training_loop data target layers // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 dtest.Length) // Adds the cost to the accumulator.
            acc <- acc+lazy_acc.Value // Here the accuracy calculation is triggered by accessing it through the Lazy property.

            if System.Single.IsNaN r' then failwith "Nan error"

            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed.

        printfn "The validation cost at iteration %i is %f" i r'
        printfn "The accuracy is %i/10000" (int acc)
        let r2 = r'
        yield r1,r2
    |]
let num_iters = 25
let learning_rate = 0.03f
#time
let s = train_mnist_sgd num_iters learning_rate layers
#time

