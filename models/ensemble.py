import torch.nn as nn

class EnsembleModel(nn.Module):
    """
    The ensemble of models.
    All the models should be put in the GPU, i.e., use .to(GPU) and then use this class to ensemble.
    """
    def __init__(self, basemodels=None):
        super(EnsembleModel, self).__init__()
        if basemodels == None:
            self.models =[]
        else:
            self.models = basemodels
    
    def get_output_scale(self, output):
        maxk = max((10,))
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        scale_list = (pred_val_out[:, 0] - pred_val_out[:, 1]).reshape([output.shape[0],-1])
        return scale_list

    def forward(self, x):
        assert(len(self.models)>0)
        prediction = self.models[0](x)
        for basemodel in self.models[1:]:
            tmp_prediction = basemodel(x)
            prediction += tmp_prediction
        
        prediction = prediction/len(self.models)
        return prediction
    
    def append_model(self, basemodel):
        self.models.append(basemodel)
    
    def eval(self):
        for model in self.models:
            model.eval()
    
    def train(self):
        for model in self.models:
            model.train()
    
    def parameters(self):
        param = list(self.models[0].parameters())
        for i in range(1, len(self.models)):
            param.extend(list(self.models[i].parameters()))
        return param