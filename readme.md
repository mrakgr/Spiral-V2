# Spiral V2

A redesign of my previous library with a focus on convolutional nets. Caches Cuda modules on disk and compiles lightning fast as a result. The design is much improved as well with several sections modularized. As in the previous iterations, it uses the object pool to avoid unnecessary allocations. Currently works with cuDNN v3, but I intend to write a wrapper for v4 and port it to that.

NOTE 4/2/2016: Done with a significant chunk of the new library. Mnist works well. Cross entropy will be done tomorrow. It is nothing less than astonishing how much can be done in just 5 days with concerted effort.

UPDATE 4/3/2016: All the features are complete, but work horribly at the present moment. Moving on to the intense debugging phase. I guess a single week is just too little time to make a ML library.

UPDATE OF UPDATE 4/3/2016: No, I am pretty close to the finish line in the terms of what I've done with this library. Now that I've fixed the random initialization bug, it clearly runs properly. The rest of the overhead seems to come from the buggy cuDNN functions and having used dictionaries for tensor descriptor objects. In regards to the later, I've tested that sharing them actually makes the activation function run faster.

Just as it is noted in the Julia documentation, TensorAdd is roughly 30 times slower than it ought to be, in fact it runs worse than the matrix multiply.

Tomorrow I'll implement the descriptor sharing via the object pool and then start on the wrapper.

UPDATE 4/4/2016: Redesigned the object pool. It also seems that TensorAdd works properly in the previous library. When I remove the biases, the new library is actually a bit faster than the old one. At any rate, I now have everything I need to squeeze down the value function for the [N puzzle](https://github.com/mrakgr/N-Puzzle-Experiments), but let me deal with the v4 wrapper as per plan.

UPDATE 4/7/2016: Done with the v4 wrapper and the convolutional [batch normalization](http://arxiv.org/abs/1502.03167) example. It is quite something, it was worth doing the v4 wrapper just for it. I am definitely not getting the full use of it on Mnist. Also as luck would have it, the v5 RC of cuDNN came out so I might as well do that as well while I am at it. I was not planning on doing recurrent nets right now, but they will come in handy. I'd rather have functions for linear layers which would allow me to combine recurrent nets with batch normalization. At least, the new Winograd convolutional kernels are a reason to move to v5 already.

Also, a paper showing how to use [recurrent batch normalization](http://arxiv.org/abs/1603.09025) effectively came out very recently.