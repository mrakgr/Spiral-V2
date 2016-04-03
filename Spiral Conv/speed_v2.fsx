// I think I finally got the last of them with that random initialization bug.
// Now I need to figure out why is the new library 2x slower than the old one.
// My best guess: cuDNN and all those damn descriptors it forces me to drag around.

// To get to the bottom of this, I'll compare each and every feature side by side starting from the forward step.

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

let dtrain = Array.zip train_images train_labels

let l1 = FeedforwardLayer.createRandomLayer 2048 784 relu

let data, target = dtrain.[0]

let a = 784*2048 |> new_dev<float32> |> fun x -> x.Memset(0uy) ; x
let b = data
let c = new_dev <| 2048*2048

// This does not actually train it, it just initiates the tree for later training.
let inline training_loop (data: DM) (targets: DM) =
    cublas.Gemm(nT, nT, 2048, 128, 784, 1.0f, a, 2048, b.r.P.dArray, 784, 0.0f, c, 2048)
    //linear_layer_matmult [|l1.W,data|] None |> ignore
    
    
let test num_iters =
    for i=1 to num_iters do
        training_loop data target // Builds the tape.
        //tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
        //tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed for the simple recurrent and feedforward case.

let num_iters = 10000
#time
let s = test num_iters
#time

