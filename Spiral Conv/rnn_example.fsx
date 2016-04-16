// Does nothing for now except load Reber. The new cuDNN v5 RNN function are currently architecturally incompatible with the library.
// This will just be a placeholder until I decide to add the union type and streams.



// Spiral reverse AD example. Used for testing.
// Embedded Reber grammar LSTM example.

open System
open System.IO

#if INTERACTIVE
#load "spiral_conv_v3.fsx"
#endif
open SpiralV2

open ManagedCuda
open ManagedCuda.CudaDNNv5

#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"

open FSharp.Charting

#load "embedded_reber.fsx"
open Embedded_reber

let reber_set = make_reber_set 3000

let make_data_from_set target_length =
    let twenties = reber_set |> Seq.filter (fun (a,b,c) -> a.Length = target_length) |> Seq.toArray
    let batch_size = (twenties |> Seq.length)

    let d_training_data =
        [|
        for i=0 to target_length-1 do
            let input = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield input.[i] |] |> Array.concat
            yield d4M.createConstant(batch_size,7,1,1,input)|]

    let d_target_data =
        [|
        for i=1 to target_length-1 do // The targets are one less than the inputs. This has the effect of shifting them to the left.
            let output = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield output.[i] |] |> Array.concat
            yield d4M.createConstant(batch_size,7,1,1,output)|]

    d_training_data, d_target_data

let d_training_data_20, d_target_data_20 = make_data_from_set 20
let d_training_data_validation, d_target_data_validation = make_data_from_set 30

let rnn_forward =
    ()