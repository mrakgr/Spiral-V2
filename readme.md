# Spiral V2

A redesign of my previous library with a focus on convolutional nets. Caches Cuda modules on disk and compiles lightning fast as a result. The design is much improved as well with several sections modularized. As in the previous iterations, it uses the object pool to avoid unnecessary allocations. Currently works with cuDNN v3, but I intend to write a wrapper for v4 and port it to that.

NOTE 4/2/2016: Done with a significant chunk of the new library. Mnist works well. Cross entropy will be done tomorrow. It is nothing less than astonishing how much can be done in just 5 days with concerted effort.

UPDATE 4/3/2016: All the features are complete, but work horribly at the present moment. Moving on to the intense debugging phase. I guess a single week is just too little time to make a ML library.

UPDATE OF UPDATE 4/3/2016: No, I am pretty close to the finish line in the terms of what I've done with this library. Now that I've fixed the random initialization bug, it clearly runs properly. The rest of the overhead seems to come from the buggy cuDNN functions and having used dictionaries for tensor descriptor objects. In regards to the later, I've tested that sharing them actually makes the activation function run faster.

Just as it is noted in the Julia documentation, TensorAdd is roughly 30 times slower than it ought to be, in fact it runs worse than the matrix multiply.

Tomorrow I'll implement the descriptor sharing via the object pool and then start on the wrapper.

UPDATE 4/4/2016: Redesigned the object pool. It also seems that TensorAdd works properly in the previous library. When I remove the biases, the new library is actually a bit faster than the old one. At any rate, I now have everything I need to squeeze down the value function for the [N puzzle](https://github.com/mrakgr/N-Puzzle-Experiments), but let me deal with the v4 wrapper as per plan.

UPDATE 4/7/2016: Done with the v4 wrapper and the convolutional [batch normalization](http://arxiv.org/abs/1502.03167) example. It is quite something, it was worth doing the v4 wrapper just for it. Well, I am exagerating. I am definitely not getting the full use of it on Mnist as it works only slightly better than a regular conv net on that, but I have great hope going forward.

Also as luck would have it, the v5 RC of cuDNN came out so I might as well do that as well while I am at it. I was not planning on doing recurrent nets right now, but they will come in handy. I'd rather have functions for linear layers which would allow me to combine recurrent nets with batch normalization. At least, the new Winograd convolutional kernels are a reason to move to v5 already.

Also, a paper showing how to use [recurrent batch normalization](http://arxiv.org/abs/1603.09025) effectively came out very recently.

UPDATE 4/8/2016: Done with the v5 wrapper. Ever since the first version of cuDNN was released last year, its feature set had grown by leaps and bounds. At this rate Spiral will become just a front end to it. I ran the v5 on the batch normalization example just to make sure it works properly, but I have not bothered to test then new RNN functions yet.

Maybe right now I finally have everything set to begin work on the N puzzle. To do all this just for that, I am a madman.

UPDATE 4/16/2016: Time to bring in the RNNs. The cuDNN ones do not have BN nor can they be extended in multiple dimensions, but they will do as a strong baseline. Given the success of residual nets, right now I am really suspecting that dimensionally folding standard feedforward into 2D nets could really ease the optimization burden. Hopefully some ML researcher will try it eventually and I will get to read about it.

Edit(3h later): "I pretty much spent the last two hours deep in thought. I am starting to lean towards skipping implementing these functions.

They are particularly troublesome. It is not that the instructions are unclear.

It is that I have to implement a union type to get them to work properly.

Before I can move forward with them, I basically have to ensure that the inputs are brought in the proper form.

Right now they simply aren't architecturally compatible with the library.

If I did a union type and the streams in addition to that, I might as well not even use those RNN functions as I would pretty much have everything to implement all the optimizations on my own. 

Basically, I know how to do the union type, but as I do not want to spend two weeks doing it and the streams, I'll defer those optimization actions until I actually need them."

The deal with the cuDNN v5 RNN functions is that both the inputs and the weights need to be stored into one array and clearly dealing with raw pointers is out of the question. I'll defer doing a union type for now.

UPDATE 4/18/2016: The fact that to do inference for Q Learning I need to have many different (perhaps hundreds) output layers has pushed the union type from something that would be good to have to make the LSTM faster to my number one priority. I'll do it and streams and in addition to that as most outputs will not be selected, I will have to make a mechanism to prune those dead paths. Seems simple enough. Let me get the basic blocks in place first...

Edit: Created both the union type and the stream pool. The stream pool example is quite interesting. I've managed to speed up a long matrix multiply sequence by 7x compared to the naive implementation and adding it to the library seems trivial. For the union type, I will have to rewrite the functions to take advantage of it. With it I might be able to use the new cuDNN RNN functions as well, but unfortunately, I can't use those for Deep Q Learning as the input in the following step depends on the previous.

At any rate, I will have to rewrite the library again, but I think that this time will really be it.

License: LGPL 3.