// The way the first example is running is making me think that only the 
// uppermost layers are getting trained, but I cannot figure out what the
// problem with the batch normalization function is. I'll remove all but one layer 
// and try logistic regression.

// Edit: Logistic regression seems fine. Let me try an extra layer.

// Edit2: Fixed it after much hairpulling. See v3 for explanation.

open System
open System.IO

#if INTERACTIVE
#load "spiral_conv_v2.fsx"
#endif
open SpiralV2

open ManagedCuda
open ManagedCuda.CudaDNN

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

let batch_normalization_forward bnMode (bnScale : d4M) (bnBias : d4M) (bnRunningMean : d4M) (bnRunningVariance : d4M) exponentialAverageFactor do_inference (input : d4M) =
    let input_sizes = input.nchw
    let bias_sizes = bnBias.nchw
    let srcTensorDesc = ObjectPool.getTensorDescriptor input_sizes

    let bnDesc = 
        //bnBias.nchw |> ObjectPool.getTensorDescriptor
        ObjectPool.getBNDescriptor (input_sizes, bnMode, srcTensorDesc)

    let _ =
        let mutable d,n,c,h,w,sn,sc,sh,sw = cudnnDataType.Double,0,0,0,0,0,0,0,0
        bnDesc.GetTensor4dDescriptor(&d,&n,&c,&h,&w,&sn,&sc,&sh,&sw)
        let bn_nchw = n,c,h,w
        if bn_nchw <> bnScale.nchw then failwith "Tensor dimensions for bnScale are incorrect."
        if bn_nchw <> bnBias.nchw then failwith "Tensor dimensions for bnBias are incorrect."
        if bn_nchw <> bnRunningMean.nchw then failwith "Tensor dimensions for bnRunningMean are incorrect."
        if bn_nchw <> bnRunningVariance.nchw then failwith "Tensor dimensions for bnRunningVariance are incorrect."

    let alpha, beta = 1.0f, 0.0f
    let epsilon = 1e-5
//    let bnSavedMean = bias_sizes |> ObjectPool.getd4M true
//    let bnSavedVariance = bias_sizes |> ObjectPool.getd4M true
    let output = input_sizes |> ObjectPool.getd4M false

    if do_inference = false then
        cudnn.BatchNormalizationForwardTraining(bnMode,alpha,beta,srcTensorDesc,input.P,srcTensorDesc,output.P,bnDesc,bnScale.P,bnBias.P,exponentialAverageFactor,bnRunningMean.P,bnRunningVariance.P,epsilon,CudaDeviceVariable.Null,CudaDeviceVariable.Null)//,bnSavedMean.P,bnSavedVariance.P)
        if input.A.IsSome then 
            let batch_normalization_backward () =
                let dx_alpha, dx_beta = 1.0f, 1.0f
                let param_alpha, param_beta = 1.0f, 1.0f

                cudnn.BatchNormalizationBackward(bnMode,dx_alpha,dx_beta,param_alpha,param_beta,srcTensorDesc,input.P,srcTensorDesc,output.A.Value,srcTensorDesc,input.A.Value,bnDesc,bnScale.P,bnScale.A.Value,bnBias.A.Value,epsilon,CudaDeviceVariable.Null,CudaDeviceVariable.Null)//bnSavedMean.P,bnSavedVariance.P)
                
            tape.Push batch_normalization_backward
    else
        cudnn.BatchNormalizationForwardInference(bnMode,alpha,beta,srcTensorDesc,input.P,srcTensorDesc,output.P,bnDesc,bnScale.P,bnBias.P,bnRunningMean.P,bnRunningVariance.P, epsilon)
        
    output


/// The initialization parameter for this is based off the weights and not the constructor.
/// Can be used for feedforward nets assuming the last two dimensions are 1.
type BNConvolutionalLayer =
    {
    W : d4M  // Input weight tensor
    bnScale : d4M  // Scale tensor
    bnBias : d4M  // Bias tensor
    bnRunningMean : d4M  // Mean tensor
    bnRunningVariance : d4M  // Variance tensor
    a : d4M -> d4M // Activation function
    }      

    static member create weight_nchw a =
        let W = d4M.makeUniformRandomNode weight_nchw
        let bias_sizes = weight_nchw |> fun (n,c,h,w) -> (1,n,1,1)

        let bnScale = bias_sizes |> d4M.create 
        bnScale.setPrimal 0.1f // Initial scale value based on the Recurrent Batch Normalization paper by Cooijmans et al.
        bnScale.setZeroAdjoint()
        let bnBias = bias_sizes |> d4M.create
        bnBias.setZeroPrimal()
        bnBias.setZeroAdjoint()
        let bnRunningMean = bias_sizes |> d4M.createConstant
        let bnRunningVariance = bias_sizes |> d4M.createConstant

        { W = W; bnScale = bnScale; bnBias = bnBias; bnRunningMean = bnRunningMean; bnRunningVariance = bnRunningVariance; a=a  }

    member t.train (convPars,input:d4M) exponentialAverageFactor = 
        let bnMode = cudnnBatchNormMode.BatchNormSpatial
        convolution_forward convPars input t.W
        |> batch_normalization_forward bnMode t.bnScale t.bnBias t.bnRunningMean t.bnRunningVariance exponentialAverageFactor false
        |> t.a
    member t.inference (convPars,input:d4M) = 
        let bnMode = cudnnBatchNormMode.BatchNormSpatial
        convolution_forward convPars input t.W
        |> batch_normalization_forward bnMode t.bnScale t.bnBias t.bnRunningMean t.bnRunningVariance 1.0 true
        |> t.a

    member l.ToArray = [|l.W;l.bnScale;l.bnBias;l.bnRunningMean;l.bnRunningVariance|]
    member l.ResetAdjoints () = 
        l.W.setZeroAdjoint();l.bnScale.setZeroAdjoint();
        l.bnBias.setZeroAdjoint()
    member t.SGD learning_rate = 
        saxpy -learning_rate t.W.A' t.W.P'
        saxpy -learning_rate t.bnScale.A' t.bnScale.P'
        saxpy -learning_rate t.bnBias.A' t.bnBias.P'

let l1 = BNConvolutionalLayer.create (1024,1,28,28) relu
let l2 = BNConvolutionalLayer.create (10,1024,1,1) clipped_sigmoid

let base_nodes = [|l1;l2|]

let training_loop label data i = // For now, this is just checking if the new library can overfit on a single minibatch.
    let i' = !i
    i := i'+1
    let factor = 1.0/(1.0 + float i')
    [|
    defaultConvPar,l1
    defaultConvPar,l2
    |] 
    |> Array.fold (fun x (convPars,layer) -> layer.train (convPars,x) factor) data
    |> fun x -> lazy get_accuracy label x, cross_entropy_cost label x

let inference_loop label data = // For now, this is just checking if the new library can overfit on a single minibatch.
    [|
    defaultConvPar,l1
    defaultConvPar,l2
    |] 
    |> Array.fold (fun x (convPars,layer) -> layer.inference (convPars,x) ) data
    |> fun x -> lazy get_accuracy label x, cross_entropy_cost label x

let learning_rate = 0.5f

let test() =
    let c = ref 0
    for i=1 to 50 do
        let mutable er = 0.0f
        for j=0 to 0 do
            let _,r = training_loop train_labels.[j] train_images.[j] c // Forward step
            er <- er + !r.P
            ObjectPool.Reset() // Resets all the adjoints from the top of the pointer in the object pool along with the pointers.
            base_nodes |> Array.iter (fun x -> x.ResetAdjoints())
            printfn "Squared error cost on the minibatch is %f at batch %i" !r.P j

            if !r.P |> Single.IsNaN then failwith "Nan!"

            r.A := 1.0f // Loads the 1.0f at the top
            while tape.Count > 0 do tape.Pop() |> fun x -> x() // The backpropagation step
            base_nodes |> Array.iter (fun x -> x.SGD learning_rate) // Stochastic gradient descent.
//        printfn "-----"
//        printfn "Squared error cost on the dataset is %f at iteration %i" (er / float32 train_images.Length) i
//
//        let mutable acc = 0.0f
//        for j=0 to test_images.Length-1 do
//            let acc',r = inference_loop test_labels.[j] test_images.[j] // Forward step
//            ObjectPool.ResetPointers()
//            tape.Clear()
//            acc <- acc'.Value + acc
//    
//        printfn "Accuracy on the test set is %i/10000." (int acc)
//        printfn "-----"

test()


