import os
import torch
import numpy as np

from model import U2NET

def main():
    model_name = 'u2net'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '_human_seg', model_name + '_human_seg.pth')
    print(model_dir)

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    img = torch.randn(1, 3, 320, 320, requires_grad=False)
    img = img.to(torch.device('cpu'))

    output_dir = os.path.join('{}_human_seg.onnx'.format(model_name))
    torch.onnx.export(net, img, output_dir, opset_version=11)
    print('Finished!')

if __name__ == '__main__':
    main()
