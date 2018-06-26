# Restricted-Boltzmann-Machine-Model-for-sequence-data-applications
RBM is a classical model for construct the Deep Architecture.

This package include Multi-type RBM for sequence data modeling, specificial include continuous RBM and flexible partially surpervised training method(Bengio et al.,2007). This model is adapt to the applications such as financial, atmosphere, geography sequence et al. In the near future, Gaussian Bernoulli RBM and Bernoulli  RBM will be added.

The great contribute of this package is the application of Continuous RBM combine with flexible partially surpervised method (Ref. Bengion et al., 2007). The partially surpervised for regression problem is using BP algorithm (Bengio 2007), and the partially surpervised for classification problem is using CD method (typically apply at the end layer) (Hinton 2006). 

The model is defined through a predefined architecture 'dbncreateopts', then create and initial the model using 'dbnsetup'.

opts = dbncreateopts('crbm',batchsize,epoch)
opts is very important for the RBM training, please setup the opts more carefully.

model = dbnsetup([layers_define], x_in, opts)

more details please see demo

If you use this package, please cite my papers ...... which are dragged by my supervisor more than one year (I have graudated from a PHD )......On the other way, please cite this web. 

reference: https://github.com/Piyush3dB/continuous-RBM/blob/master/RBM.py
           https://github.com/skaae/rbm_toolbox
           https://github.com/dustinstansbury/medal
           Chen. A continuous restricted Boltzmann machine with hardware-amenable learning algorithm 2002
           Bengio, Yoshua. Appendix of the paper “Greedy Layer-Wise Training of Deep Networks” 2007
           Hinton . A fast learning algorithm for deep belief nets 2006
