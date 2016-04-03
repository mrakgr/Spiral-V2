# Spiral V2

A redesign of my previous library with a focus on convolutional nets. Caches Cuda modules on disk and compiles lightning fast as a result. The design is much improved as well with several sections modularized. As in the previous iterations, it uses the object pool to avoid unnecessary allocations. Currently works with cuDNN v3, but I intend to write a wrapper for v4 and port it to that.

NOTE 4/2/2016: Done with a significant chunk of the new library. Mnist works well. Cross entropy will be done tomorrow. It is nothing less than astonishing how much can be done in just 5 days with concerted effort.

UPDATE 4/3/2016: All the features are complete, but work horribly at the present moment. Moving on to the intense debugging phase. I guess a single week is just too little time to make a ML library.

UPDATE OF UPDATE 4/3/2016: No, I am pretty close to the finish line in the terms of what I've done with this library. Now that I've fixed the random initialization bug, it clearly runs properly. The rest of the overhead seems to come from the buggy cuDNN functions and having used dictionaries for tensor descriptor objects. In regards to the later, I've tested that sharing them actually makes the activation function run faster.

Just as it is noted in the Julia documentation, TensorAdd is roughly 30 times slower than it ought to be, in fact it runs worse than the matrix multiply.

Tomorrow I'll implement the descriptor sharing via the object pool and then start on the wrapper.