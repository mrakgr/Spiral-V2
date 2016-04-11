// I made this example to check the correctness of BN in the feedforward case, but it seems it is entirely correct.
// This makes me wonder why it works so poorly on N puzzle.

open System
open System.IO

#if INTERACTIVE
#load "spiral_conv_v3.fsx"
#endif
open SpiralV2

open ManagedCuda
open ManagedCuda.CudaDNNv5

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

let l1 = BNFullyConnectedLayer.create (784,2048,1,1) relu :> INNet
//let l2 = BNFullyConnectedLayer.create (2048,2048,1,1) relu :> INNet
//let l3 = BNFullyConnectedLayer.create (2048,2048,1,1) relu :> INNet
let l2 = BNResidualFullyConnectedLayer.create (2048,2048,1,1) relu relu :> INNet
let l4 = BNFullyConnectedLayer.create (2048,10,1,1) clipped_softmax :> INNet

let base_nodes = [|l1;l2;l4|]

let training_loop label data i =
    let increase_i_and_get_factor () = 
        let i' = !i
        i := i'+1
        1.0/(1.0 + float i')
    base_nodes
    |> Array.fold (fun x layer -> layer.train x increase_i_and_get_factor) data
    |> fun x -> lazy get_accuracy label x, cross_entropy_cost label x

let inference_loop label data = // For now, this is just checking if the new library can overfit on a single minibatch.
    base_nodes
    |> Array.fold (fun x layer -> layer.inference x ) data
    |> fun x -> lazy get_accuracy label x, cross_entropy_cost label x

let learning_rate = 0.03f

let test() =
    let c = ref 0
    for i=1 to 5 do
        let mutable er = 0.0f
        for j=0 to train_images.Length-1 do
            let _,r = training_loop train_labels.[j] train_images.[j] c // Forward step
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
            let acc',r = inference_loop test_labels.[j] test_images.[j] // Forward step
            ObjectPool.ResetPointers()
            tape.Clear()
            acc <- acc'.Value + acc
    
        printfn "Accuracy on the test set is %i/10000." (int acc)
        printfn "-----"

test()


