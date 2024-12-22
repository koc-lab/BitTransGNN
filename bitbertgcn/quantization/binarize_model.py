from torch import nn
from copy import deepcopy
from .binary_layers import BitLinear, BitEmbedding

def replace_layer(module, num_states: int, linear_quantize: bool=True, quantize_embeddings: bool = False, num_bits_act: float = 8.0):
    """
    Replaces the Linear layers and the Embedding layers within the neural network or the transformer model by their quantized counterparts.
    If the module being passed is not one of Linear or Embedding modules, it is not modified.

    Args:
        module: the module being passed through the function
        num_states: number of states used to quantize the weights of the module
        linear_quantize: boolean variable to determine whether to quantize the weights of the linear layer
            Default: True
        quantize_embeddings: boolean variable to determine whether to quantize the weights of the embedding layer
            Default: False
        num_bits_act: number of bits used to quantize the activations
            Default: 8.0
    """
    if isinstance(module, nn.Linear):
        if linear_quantize:
            target_state_dict   = deepcopy(module.state_dict())
            bias                = True if module.bias is not None else False
            new_module          = BitLinear(
                                    in_features=module.in_features,
                                    out_features=module.out_features,
                                    bias=bias,
                                    num_states=num_states,
                                    num_bits_act=num_bits_act,
                                )
            new_module.load_state_dict(target_state_dict)
            return new_module
        else:
            return module
    elif isinstance(module, nn.Embedding):
        if quantize_embeddings:
            target_state_dict   = deepcopy(module.state_dict())
            new_module          = BitEmbedding(num_embeddings=module.num_embeddings,
                                            embedding_dim=module.embedding_dim,
                                            num_states=num_states)
            new_module.load_state_dict(target_state_dict)
            return new_module
        else:
            return module
    else:
        return module

def recursive_setattr(obj, attr, value):
    """
    Recursively sets the attribute of the object to the value being passed through the function.
    
    Args: 
        obj: the object being modified
        attr: the attribute being modified within the object
        value: the new value being assigned to the object's attribute
    """
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)

def quantize_tf_model(model, num_states: int = 2, linear_quantize: bool = True, quantize_embeddings: bool = False, num_bits_act: float = 8.0):
    """
    Iterates over the transformer model being passed through the function to quantize the weights of different modules within the model.

    Args: 
        model: the model being quantized
        num_states: number of states used to quantize the weights of the module
            Default: 2
        linear_quantize: boolean variable to determine whether to quantize the weights of the linear layer
            Default: True
        quantize_embeddings: boolean variable to determine whether to quantize the weights of the embedding layer
            Default: False
        num_bits_act: number of bits used to quantize the activations
            Default: 8.0
    """
    for name, module in tuple(model.named_modules()):
        if name:
            recursive_setattr(model, name, replace_layer(module, num_states, linear_quantize=linear_quantize, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act))
    print("Quantization completed.")
    print(f"Number of states for the quantized model: {num_states}")
    if quantize_embeddings and linear_quantize:
        print("Both linear layers and embedding weights are quantized.")
    elif not(quantize_embeddings) and linear_quantize:
        print("Linear layer weights are quantized but the embeddings are maintained full-precision.")
    elif quantize_embeddings and not(linear_quantize):
        print("Embeddings are quantized, however, linear layers are kept full-precision.")
    else:
        print("No quantization is implemented over the linear layers and the embedding layers.")
    return model

def quantize_bertgcn_architecture(model, ckpt, quantize_bert, bert_quant_type, quantize_gcn, num_states, joint_training, lmbd, quantize_embeddings, num_bits_act):
    """
    Quantizes the BERTGCN architecture being passed through the function. Uses the checkpoints from the fine-tuning stage of the BERT model to assign weights.

    Args: 
        model: the BERTGCN model being quantized
        ckpt: the BERT model checkpoint obtained from the initial fine-tuning stage of the transformer
        bert_quant_type: method used to fine-tune the quantized model
        num_states: number of states used to quantize the weights of the module
        quantize_embeddings: boolean variable to determine whether to quantize the weights of the embedding layer
        num_bits_act: number of bits used to quantize the activations

        #rest of the parameters are not used during quantization at this stage, and are only kept to give information regarding the model being used.
        quantize_gcn: whether the GCN is being quantized. Note that GCN quantization is done within the BERTGCNTorch model.
        joint_training: whether the transformer and the GCN are being jointly trained
        lmbd: the interpolation parameter
    """
    if quantize_bert:
        print("model before quantization: ", model)
        if bert_quant_type == "QAT":
            model.bert_model = quantize_tf_model(model=model.bert_model, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
            model.classifier = replace_layer(module=model.classifier, num_states=num_states, num_bits_act=num_bits_act)
            model.bert_model.load_state_dict(ckpt['bert_model'])
            model.classifier.load_state_dict(ckpt['classifier'])
        else:
            model.bert_model.load_state_dict(ckpt['bert_model'])
            model.classifier.load_state_dict(ckpt['classifier'])
            model.bert_model = quantize_tf_model(model=model.bert_model, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
            model.classifier = replace_layer(module=model.classifier, num_states=num_states, num_bits_act=num_bits_act)
    else:
        model.bert_model.load_state_dict(ckpt['bert_model'])
        model.classifier.load_state_dict(ckpt['classifier'])
    
    if quantize_gcn:
        print("A quantized GCN model is being used.")

    if quantize_bert or quantize_gcn:
        print("model after quantization: ", model)
    else:
        print("model: ", model)
    
    if joint_training:
        print(f'BERT model and the GCN model are jointly trained. The final outputs are interpolated by the parameter {lmbd}.')
        print(f'Cls logits from BERT are passed through a classifier and interpolated with the GCN output by the parameter {lmbd} to obtain the final output.')
    else:
        print('GCN model is trained and BERT model is only used to obtain the initial feature vectors for GCN.')
        print(f'Cls logits from BERT are passed through a classifier and interpolated with the GCN output by the parameter {lmbd} to obtain the final output.')
    return model

def quantize_bitbert_for_inference(model, ckpt, joint_training,
                                   quantize_bert, num_states, bitbert_quant_type, quantize_embeddings, 
                                   num_bits_act):
    if quantize_bert:
        if not(joint_training) and bitbert_quant_type=="PTQ": #if the quantized model has not been trained at all, then first load state dict then quantize
            model.bert_model.load_state_dict(ckpt['bert_model'])
            model.classifier.load_state_dict(ckpt['classifier'])
            model.bert_model = quantize_tf_model(model=model.bert_model, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
            model.classifier = replace_layer(module=model.classifier, num_states=num_states)
        else:
            model.bert_model = quantize_tf_model(model=model.bert_model, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
            model.classifier = replace_layer(module=model.classifier, num_states=num_states)
            model.bert_model.load_state_dict(ckpt['bert_model'])
            model.classifier.load_state_dict(ckpt['classifier'])
    else:
        model.bert_model.load_state_dict(ckpt['bert_model'])
        model.classifier.load_state_dict(ckpt['classifier'])
    return model


def quantize_bitbertgcn_for_inference(model, ckpt, joint_training,
                                      quantize_bert, num_states, bitbert_quant_type, quantize_embeddings, 
                                      num_bits_act):
    if quantize_bert:
        if not(joint_training) and bitbert_quant_type=="PTQ": #if the quantized model has not been trained at all, then first load state dict then quantize
            model.bert_model.load_state_dict(ckpt['bert_model'])
            model.classifier.load_state_dict(ckpt['classifier'])
            model.bert_model = quantize_tf_model(model=model.bert_model, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
            model.classifier = replace_layer(module=model.classifier, num_states=num_states)
        else:
            model.bert_model = quantize_tf_model(model=model.bert_model, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
            model.classifier = replace_layer(module=model.classifier, num_states=num_states)
            model.bert_model.load_state_dict(ckpt['bert_model'])
            model.classifier.load_state_dict(ckpt['classifier'])
    else:
        model.bert_model.load_state_dict(ckpt['bert_model'])
        model.classifier.load_state_dict(ckpt['classifier'])
    model.gcn.load_state_dict(ckpt["gcn"])
    return model


def quantize_teacher_architecture(teacher_model, teacher_ckpt, 
                                  joint_training, 
                                  quantize_teacher_bert, teacher_bert_quant_type, teacher_num_states, quantize_teacher_embeddings, 
                                  num_bits_act):
    if quantize_teacher_bert:
        if not(joint_training) and teacher_bert_quant_type == "PTQ": #if the quantized model has not been trained at all, then first load state dict then quantize
            teacher_model.bert_model.load_state_dict(teacher_ckpt['bert_model'])
            teacher_model.classifier.load_state_dict(teacher_ckpt['classifier'])
            teacher_model.gcn.load_state_dict(teacher_ckpt['gcn'])
            teacher_model.bert_model = quantize_tf_model(model=teacher_model.bert_model, num_states=teacher_num_states, quantize_embeddings=quantize_teacher_embeddings, num_bits_act=num_bits_act)
            teacher_model.classifier = replace_layer(module=teacher_model.classifier, num_states=teacher_num_states)
        else:
            teacher_model.bert_model = quantize_tf_model(model=teacher_model.bert_model, num_states=teacher_num_states, quantize_embeddings=quantize_teacher_embeddings, num_bits_act=num_bits_act)
            teacher_model.classifier = replace_layer(module=teacher_model.classifier, num_states=teacher_num_states)
            teacher_model.bert_model.load_state_dict(teacher_ckpt['bert_model'])
            teacher_model.classifier.load_state_dict(teacher_ckpt['classifier'])
            teacher_model.gcn.load_state_dict(teacher_ckpt['gcn'])
    else:
        teacher_model.bert_model.load_state_dict(teacher_ckpt['bert_model'])
        teacher_model.classifier.load_state_dict(teacher_ckpt['classifier'])
        teacher_model.gcn.load_state_dict(teacher_ckpt['gcn'])
    return teacher_model

def quantize_student_architecture(student_model, student_ckpt, quantize_student_bert, student_bert_quant_type, num_states, quantize_embeddings, num_bits_act):
    if quantize_student_bert:
        if student_bert_quant_type == "QAT":
            student_model.bert_model = quantize_tf_model(model=student_model.bert_model, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
            student_model.classifier = replace_layer(module=student_model.classifier, num_states=num_states)
            student_model.bert_model.load_state_dict(student_ckpt['bert_model'])
            student_model.classifier.load_state_dict(student_ckpt['classifier'])
        else:
            student_model.bert_model.load_state_dict(student_ckpt['bert_model'])
            student_model.classifier.load_state_dict(student_ckpt['classifier'])
            student_model.bert_model = quantize_tf_model(model=student_model.bert_model, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
            student_model.classifier = replace_layer(module=student_model.classifier, num_states=num_states)
    else:
        student_model.bert_model.load_state_dict(student_ckpt['bert_model'])
        student_model.classifier.load_state_dict(student_ckpt['classifier'])
    return student_model
