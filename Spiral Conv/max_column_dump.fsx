// I've really just wasted time making these modules, but they might become useful at some point.

open System.Runtime.InteropServices

#nowarn "9"
// From what I can see, I pretty much created this for no reason. Using a 
// [-15.0f..16.0f] static layer after softmax was the right choice. Crap.
// ...I'll remove this in the next version of the library along with those maxColumn modules.
[<StructLayout(LayoutKind.Sequential)>]
type ValueIndex =
    struct
    val value : float32 // TODO : This might become misaligned when FP16 comes out. Make sure it stays right.
    val index : int
    end

    override t.ToString() = sprintf "(%f,%i)" t.value t.index

type dValueIndex =
    {
    mutable value_index_array_size:int
    mutable value_index_array: CudaDeviceVariable<ValueIndex> // primal
    }  

    /// Throws an exception if it tries to allocate an array of size 0.
    static member create n =
        let p = new_dev n
        { value_index_array_size=n; value_index_array=p }

    member inline t.ReplaceIf n =
        if int t.value_index_array.Size < n
        then
            (t :> IDisposable).Dispose()
            t.value_index_array_size <- n
            t.value_index_array <- new_dev n
        else
            t.value_index_array_size <- n

    interface IDisposable with
        member t.Dispose() = 
                t.value_index_array.Dispose()

type ObjectPool with
    let valueIndexPool = ResizeArray()
    let vip = ref 0

    member t.getdValueIndex n =
        let t' = ObjectPool.getFromPool valueIndexPool vip (fun _ -> dValueIndex.create n)
        t'.ReplaceIf n
        t'

/// o <- max_col_index(x)
/// Gets the maximum indices of each column.
type DeviceMaxColumnIndexModule() = 
    let block_size = 32

    let kernel_name = "MaxColumnIndexKernel"
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            #define INIT __int_as_float(0xff800000) // The constant init for the reduce operations. This is float negative infinity.
            // The max reduce version.
            __device__ inline floatType warpReduce(floatType value){
                #pragma unroll
	            for (int i=1; i<32; i*=2) {
                    floatType tmp = __shfl_xor(value, i);
                    value = (tmp > value) ? tmp : value;
                    }
	            return value;
            }
              
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int num_rows, const int num_cols)
            {
                int row = threadIdx.x;
                const int col = blockIdx.x;
                int col_idx = blockIdx.x*num_rows; 
                floatType max = INIT; // This is the negative infinity for floats.
                int index = -1;
                while (row < num_rows)
                {
                   if (A[row+col_idx] > max) {
                        max = A[row+col_idx];
                        index = row;
                        }
                    row += blockDim.x;
                }
                
                __shared__ floatType max_index;
                if (max == warpReduce(max)) max_index = index;
                __syncthreads();
                O[col] = max_index;
            }
        }

        "|] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(((n : int,c : int,h,w as x_nchw), x: CudaDeviceVariable<float32>), ((o_n,o_c,o_h,o_w), o: CudaDeviceVariable<float32>)) =
        if n <> o_n then failwith "n <> o_n"
        if o_c <> 1 || o_h <> 1 || o_w <> 1 then failwith "o_c <> 1 || o_h <> 1 || o_w <> 1"
        let m = c*h*w
        kernel.GridDimensions <- dim3(n)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream,x.DevicePointer,o.DevicePointer,m,n) |> ignore

/// o <- max_col(x)
/// Gets the maximum of each column.
type DeviceMaxColumnModule() = 
    let block_size = 32

    let kernel_name = "MaxColumnKernel"
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            #define INIT __int_as_float(0xff800000) // The constant init for the reduce operations. This is float negative infinity.
            // The max reduce version.
            __device__ inline floatType warpReduce(floatType value){
                #pragma unroll
	            for (int i=1; i<32; i*=2) {
                    floatType tmp = __shfl_xor(value, i);
                    value = (tmp > value) ? tmp : value;
                    }
	            return value;
            }
              
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int num_rows, const int num_cols)
            {
                int row = threadIdx.x;
                const int col = blockIdx.x;
                int col_idx = blockIdx.x*num_rows; 
                floatType max = INIT; // This is the negative infinity for floats.
                int index = -1;
                while (row < num_rows)
                {
                   if (A[row+col_idx] > max) {
                        max = A[row+col_idx];
                        index = row;
                        }
                    row += blockDim.x;
                }
                
                O[col] = warpReduce(max);
            }
        }

        "|] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(((n : int,c : int,h,w as x_nchw), x: CudaDeviceVariable<float32>), ((o_n,o_c,o_h,o_w), o: CudaDeviceVariable<float32>)) =
        if n <> o_n then failwith "n <> o_n"
        if o_c <> 1 || o_h <> 1 || o_w <> 1 then failwith "o_c <> 1 || o_h <> 1 || o_w <> 1"
        let m = c*h*w
        kernel.GridDimensions <- dim3(n)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream,x.DevicePointer,o.DevicePointer,m,n) |> ignore

/// o <- max_col_and_index(x)
/// Gets the maximum value and their respective indices of each column.
type DeviceMaxColumnAndIndexModule() = 
    let block_size = 32

    let kernel_name = "MaxColumnAndIndexKernel"
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            #define INIT __int_as_float(0xff800000) // The constant init for the reduce operations. This is float negative infinity.

            struct ValueIndex { // Hopefully the layout will be the same.
                floatType value; // TODO : This might become misaligned when FP16 comes out. Make sure it stays right.
                int index;
            };

            // The max reduce version.
            __device__ inline floatType warpReduce(floatType value){
                #pragma unroll
	            for (int i=1; i<32; i*=2) {
                    floatType tmp = __shfl_xor(value, i);
                    value = (tmp > value) ? tmp : value;
                    }
	            return value;
            }
              
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, ValueIndex* O, const int num_rows, const int num_cols)
            {
                int row = threadIdx.x;
                const int col = blockIdx.x;
                int col_idx = blockIdx.x*num_rows; 
                floatType max = INIT; // This is the negative infinity for floats.
                int index = -1;
                while (row < num_rows)
                {
                   if (A[row+col_idx] > max) {
                        max = A[row+col_idx];
                        index = row;
                        }
                    row += blockDim.x;
                }
                __shared__ int max_index;
                floatType max_2 = warpReduce(max);
                if (max == max_2) max_index = index;
                __syncthreads();
                ValueIndex value_index;
                value_index.value = max_2;
                value_index.index = max_index;
                O[col] = value_index;
            }
        }

        "|] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(((n : int,c : int,h,w as x_nchw), x: CudaDeviceVariable<float32>), o: dValueIndex) =
        if n <> o.value_index_array_size then failwithf "n(%i) <> o.value_index_array_size(%i)" n o.value_index_array_size
        let m = c*h*w
        kernel.GridDimensions <- dim3(n)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream,x.DevicePointer,o.value_index_array.DevicePointer,m,n) |> ignore

let maxColumnIndexModule = lazy new DeviceMaxColumnIndexModule()
let get_max_indices (x : d4M) =
    let o = ObjectPool.getd4M true (x.num_images,1,1,1)
    maxColumnIndexModule.Value.A(x.P',o.P')
    o

let maxColumnModule = lazy new DeviceMaxColumnModule()
let get_max (x : d4M) =
    let o = ObjectPool.getd4M true (x.num_images,1,1,1)
    maxColumnModule.Value.A(x.P',o.P')
    o

let maxColumnValueAndIndexModule = lazy new DeviceMaxColumnAndIndexModule()
let get_max_value_index (x : d4M) =
    let o = ObjectPool.getdValueIndex x.num_images
    maxColumnValueAndIndexModule.Value.A(x.P',o)
    o

