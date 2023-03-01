from typing import Any, Callable, Sequence
from tqdm.auto import tqdm, trange

import torch

from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
    compute_layer_gradients_and_eval,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage

class ConceptGradients(GradientAttribution):
    def __init__(self, forward_func: Callable, concept_forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
        """
        GradientAttribution.__init__(self, forward_func)
        self.concept_forward_func = concept_forward_func
        
    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        mode: str,
        target: TargetType = None,
        target_concept: TargetType = None,
        n_concepts: TargetType = 1,
        target_layer_name: str = None,
        concept_layer_name: str = None,
        abs: bool = False,
        additional_forward_args: Any = None,
        additional_concept_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            abs (bool, optional): Returns absolute value of gradients if set
                        to True, otherwise returns the (signed) gradients if
                        False.
                        Default: True
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        The gradients with respect to each input feature.
                        Attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # Generating random input with size 2x3x3x32
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Defining Saliency interpreter
            >>> saliency = Saliency(net)
            >>> # Computes saliency maps for class 3.
            >>> attribution = saliency.attribute(input, target=3)
        """
        
        assert mode in ['chain_rule_joint', 'chain_rule_independent',
                        'cav', 'inner_prod', 'cosine_similarity']
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # No need to format additional_forward_args here.
        # They are being formated in the `_run_forward` function in `common.py`
        if target_layer_name is None:
            dydxs = self.gradient_func(
                self.forward_func, inputs, target, additional_forward_args
            )
        else:
            target_layer = self.get_named_module(self.forward_func, target_layer_name)
            dydxs, acts = compute_layer_gradients_and_eval(
                self.forward_func, target_layer, inputs, target_ind=target,
                additional_forward_args=additional_forward_args, attribute_to_layer_input=True)
            del acts
        dydxs = tuple(dydxs_.detach().flatten(start_dim=1) for dydxs_ in dydxs)
        
        if target_concept is None:
            dcdxs = None
            for ci in range(n_concepts):
                if concept_layer_name is None:
                    dcdxs_ = self.gradient_func(
                        self.concept_forward_func, inputs, ci, additional_concept_forward_args
                    )
                else:
                    concept_layer = self.get_named_module(self.concept_forward_func, concept_layer_name)
                    dcdxs_, acts_ = compute_layer_gradients_and_eval(
                        self.concept_forward_func, concept_layer, inputs, target_ind=ci,
                        additional_forward_args=additional_concept_forward_args,
                        attribute_to_layer_input=True)
                    del acts_
                # dcdxs_ is a tuple
                # dcdxs_[0].shape = (bsize, *x.shape)
                
                dcdxs_ = tuple(dcdxs__.detach().flatten(start_dim=1) for dcdxs__ in dcdxs_)
                # crucial to detach here to avoid OOM
                # dcdxs_[0].shape = (bsize, -1)
                
                # crucial to preallocate memory to store the tensors to avoid OOM.
                if dcdxs is None:
                    dcdxs = tuple(torch.empty([dcdxs__.shape[0], n_concepts, dcdxs__.shape[1]]).to(dcdxs__.device) 
                                  for dcdxs__ in dcdxs_)
                
                for i in range(len(dcdxs_)):
                    dcdxs[i][:, ci, :] = dcdxs_[i]
            
            # convert list of tuples to tuple of (stacked) lists
            # dcdxs = tuple(torch.stack([dcdxs_[i] for dcdxs_ in dcdxs], dim=1) for i in range(len(dcdxs[0])))
        else:
            if concept_layer_name is None:
                dcdxs = self.gradient_func(
                    self.concept_forward_func, inputs, target_concept, additional_concept_forward_args
                )
            else:
                concept_layer = self.get_named_module(self.concept_forward_func, concept_layer_name)
                dcdxs, acts = compute_layer_gradients_and_eval(
                    self.concept_forward_func, concept_layer, inputs, target_ind=target_concept,
                    additional_forward_args=additional_concept_forward_args, attribute_to_layer_input=True)
                del acts
            dcdxs = tuple(dcdxs_.detach().flatten(start_dim=1).unsqueeze(1) for dcdxs_ in dcdxs)
        
        # dydxs.shape = [bsize, h_dim]
        # dcdxs.shape = [bsize, n_concept, h_dim]
        
        assert len(dydxs) == len(dcdxs)
        
        with torch.no_grad():
            if mode == 'chain_rule_joint':
                # torch.linalg.lstsq(A, B).solution == A.pinv() @ B
                gradients = tuple(torch.linalg.lstsq(torch.transpose(dcdx, 1, 2), dydx).solution
                                  for dydx, dcdx in zip(dydxs, dcdxs))
                # gradients = tuple(torch.bmm(dydx.unsqueeze(1), torch.linalg.pinv(dcdx)).squeeze(1)
                #                   for dydx, dcdx in zip(dydxs, dcdxs))
            elif mode == 'chain_rule_independent':
                gradients = tuple(torch.bmm(dcdx, dydx.unsqueeze(-1)).squeeze(-1) / (torch.norm(dcdx, dim=2)**2)
                                  for dydx, dcdx in zip(dydxs, dcdxs))
            elif mode == 'cav':
                gradients = tuple(torch.bmm(dcdx, dydx.unsqueeze(-1)).squeeze(-1) / torch.norm(dcdx, dim=2)
                                  for dydx, dcdx in zip(dydxs, dcdxs))
            elif mode == 'inner_prod':
                gradients = tuple(torch.bmm(dcdx, dydx.unsqueeze(-1)).squeeze(-1)
                                  for dydx, dcdx in zip(dydxs, dcdxs))
            elif mode == 'cosine_similarity':
                gradients = tuple(torch.bmm(dcdx, dydx.unsqueeze(-1)).squeeze(-1) / \
                                  (torch.norm(dcdx, dim=2) * torch.norm(dydx, dim=1, keepdim=True))
                                  for dydx, dcdx in zip(dydxs, dcdxs))
            else:
                raise NotImplementedError
            
        if abs:
            attributions = tuple(torch.abs(gradient) for gradient in gradients)
        else:
            attributions = gradients
        undo_gradient_requirements(inputs, gradient_mask)
        return _format_output(is_inputs_tuple, attributions)
    
    @staticmethod
    def get_named_module(model, name):
        for module_name, module in model.named_modules():
            if module_name == name:
                return module
        raise ValueError(f"{name} not found in model.")

class SmoothConceptGradients(ConceptGradients):
    
    @staticmethod
    def create_add_noise_hook(radius, seed=None):
        def _hook(self, input):
            if seed is not None:
                torch.manual_seed(seed)
            for input_ in input:
                noise = (torch.randn_like(input_, device=input_.device) - 0.5) * 2 * radius
                noise.requires_grad = True
                input_.add_(noise)
        return _hook
    
    @log_usage()
    def attribute(
        self,
        *args,
        nt_samples: int = 8,
        stdevs: float = 1e-2,
        share_seed: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> TensorOrTupleOfTensorsGeneric:
        
        if (kwargs['target_layer_name'] is None) or (kwargs['concept_layer_name'] is None):
            raise ValueError(f"Currently don't know how to deal with adding noise to the input in-place, since the input is a leaf variable and PyTorch complains about 'a leaf Variable that requires grad is being used in an in-place operation'.")
        
        target_module = self.forward_func if kwargs['target_layer_name'] is None else \
                        self.get_named_module(self.forward_func, kwargs['target_layer_name'])
        
        concept_module = self.concept_forward_func if kwargs['concept_layer_name'] is None else \
                         self.get_named_module(self.concept_forward_func, kwargs['concept_layer_name'])
        
        maxint = torch.iinfo(torch.int32).max
        attr = None
        for i in trange(nt_samples, leave=False, disable=(not verbose)):
            
            seed = torch.randint(low=0, high=maxint, size=(1,)).item() if share_seed else None
                
            all_handles = [
                target_module.register_forward_pre_hook(self.create_add_noise_hook(stdevs, seed=seed)),
                concept_module.register_forward_pre_hook(self.create_add_noise_hook(stdevs, seed=seed)),
            ]
            
            attr_ = super().attribute(*args, **kwargs)
            with torch.no_grad():
                if attr is None:
                    attr = torch.zeros_like(attr_)
                attr += attr_
                
            for handle in all_handles:
                handle.remove()
                
        smooth_attr = attr / nt_samples

        return smooth_attr
