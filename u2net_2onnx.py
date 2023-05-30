import os
import torch
import numpy as np
import argparse

from model import U2NET


def parsearg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default='u2net', help="Model name")
    parser.add_argument("-v", "--variant", type=str, default='human_seg', help="Model variant (human_seg or portrait)")
    parser.add_argument("-o", "--output", type=str, default='output.onnx', help="Output file", required=False)
    parser.add_argument("-s", "--size", type=str, default='512,512', help="Input size e.g, 512,512", required=False)
    return parser.parse_args()

def main():
    args = parsearg()
    model_name = args.name
    model_variant = args.variant
    input_sizes = args.size.split(',')
    model_dir = os.path.join(os.getcwd(), 'saved_models', f'{model_name}_{model_variant}', f'{model_name}_{model_variant}.pth')
    print(model_dir)

    input_size = [512, 512]
    if input_sizes == 2:
        input_size[0] = int(input_sizes[0])
        input_size[1] = int(input_sizes[1])

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    img = torch.randn(1, 3, input_size[0], input_size[1], requires_grad=False)
    img = img.to(torch.device('cpu'))

    output_dir = os.path.join(args.output)
    torch.onnx.export(net, img, output_dir, opset_version=11)
    print('Finished!')

if __name__ == '__main__':
    main()
