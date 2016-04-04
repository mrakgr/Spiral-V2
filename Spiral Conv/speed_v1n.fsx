// Tests for the new library.

// No, the convolutional functions are not right. The original feedforward example is so much better that it is not even funny.

// tensor_add is 30 times slower than it should be. This is ridiculous.

// Even when I pull the setTensor calls outside the function it is still too slow by about 13 times.

// Edit: Actually the reason it got 2x faster if because by accident I removed an add_tensor operation.
// So it really is 30x slower than it should be. Also sharing tensor decriptors helps performance a 
// tiny bit, but increases code clarity significantly.

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



//let l1 = FeedforwardLayer.createRandomLayer (784,2024,1,1) relu
let relu_clip x = clip 0.0f Single.PositiveInfinity x 0.0f
let l1 = FeedforwardLayer.createRandomLayer (784,2048,1,1) relu_clip

let data = train_images.[0]

let left = train_images.[0]
let right = train_images.[0]
let add_to_left = false

let alpha = 1.0f
let beta = -1.0f

let inline training_loop data = // For now, this is just checking if the new library can overfit on a single minibatch.
    let leftDesc = ObjectPool.getTensorDescriptor left.nchw
    let rightDesc = ObjectPool.getTensorDescriptor right.nchw

    let output = 
        if add_to_left = false 
        then 
            left.nchw |> ObjectPool.getd4M false 
            |> fun output -> cudnn.AddTensor(alpha,leftDesc,left.P,0.0f,leftDesc,output.P); output // Copy the left to output
        else 
            left

    cudnn.AddTensor(beta,rightDesc,right.P,1.0f,leftDesc,output.P) // Add right to output.
    
    //data |> squared_error_cost (train_images.[1]) |> ignore

let learning_rate = 0.03f

let test num_iterations =
    for i=1 to num_iterations do
        training_loop data
        ObjectPool.ResetPointers() // Resets all the adjoints from the top of the pointer in the object pool along with the pointers.
        tape.Clear()

#time
test 10000
#time

