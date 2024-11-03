# load all parameters from checkpoint and extract the detection model

import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Extract detection model from checkpoint')
    parser.add_argument('checkpoint', help='path to the checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    except FileNotFoundError:
        print(f"Checkpoint file {args.checkpoint} not found.")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    if checkpoint.get('state_dict', None) is None:
        print('No state_dict found in the checkpoint')
        return
    
    state_dict = checkpoint['state_dict']
    det_state_dict = {}

    for k, v in state_dict.items():
        if 'detector' in k:
            new_key = k.replace('detector.', '')
            det_state_dict[new_key] = v

    # save the detection model
    new_path = os.path.join(os.path.dirname(args.checkpoint), 'det_model.pth')
    torch.save(det_state_dict, new_path)
    print(f'Save detection model to {new_path}')
    

if __name__ == '__main__':
    main()