# Spiral V2

A redesign of my previous library with a focus on convolutional nets. Caches Cuda modules on disk and compiles lightning fast as a result. The design is much improved as well with several sections modularized. As in the previous iterations, it uses the object pool to avoid unnecessary allocations. Currently works with cuDNN v3, but I intend to write a wrapper for v4 and port it to that.

NOTE 4/2/2016: Done with a significant chunk of the new library. Mnist works well. Cross entropy will be done tomorrow. It is nothing less than astonishing how much can be done in just 5 days with concerted effort.

UPDATE 4/3/2016: All the features are complete, but work horribly at the present moment. Moving on to the intense debugging phase. I guess a single week is just too little time to make a ML library.