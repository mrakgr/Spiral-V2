Add concurrent streams. Some functions could benefit greatly from this. Mostly linear layers and especially the backwards pass. Residual nets would benefit. I can live without this for now though. The place where this would be important would be in linear layers for RNNs.

Also while at it, make is so the adjoints are reset on the first touch of the backwards pass instead having this be done manually. This has great synergy with the above mission.

Add a union type. Doing this would open the door to significant optimizations of recurrent networks.

