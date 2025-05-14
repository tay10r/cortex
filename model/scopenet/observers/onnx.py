import torch
from torch import nn

from scopenet.observer import Observer


class ONNXExporter(Observer):
    """
    This class will export the model, whenever it is updated, to an ONNX file.
    """

    def __init__(self, filename: str):
        self.__filename = filename

    def on_model_update(self, net: nn.Module, epoch: int):
        original_device = next(net.parameters()).device
        net.to('cpu')
        x = torch.randn((1, 3, 256, 256))
        torch.onnx.export(net,
                          x,
                          self.__filename,
                          input_names=['image'],
                          output_names=['reconstructed'],
                          dynamic_axes={
                              'image': {
                                  0: 'batch_size',
                                  2: 'width',
                                  3: 'height'
                              },
                              'reconstructed': {
                                  0: 'batch_size',
                                  2: 'width',
                                  3: 'height'
                              }
                          },
                          opset_version=11)
        net.to(original_device)

    def on_complete(self):
        pass
