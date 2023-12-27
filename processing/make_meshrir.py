import os
import gc
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from rich.progress import Progress

from utils.audio import from_audio
from utils.params import load_params
from utils.meshrir_utils import loadIR


def process_s1(path: str, n_fft: int, hop: int):
    # TODO
    return None

def process_s32(path: Path, n_fft: int, hop: int):
    
    pos_mic, pos_src, ir = loadIR(path)
    # print(ir.shape)

    spectrograms = np.memmap('s.temp', dtype=np.float32, mode="w+", shape=(32,441,257,257))
    phases = np.memmap('p.temp', dtype=np.float32, mode="w+", shape=(32,441,257,257))

    with Progress() as progress:
        task = progress.add_task("[cyan]Converting", total=ir.shape[0]*ir.shape[1])
        for m in range(pos_mic.shape[0]):
            for s in range(pos_src.shape[0]):
                sample = ir[s, m, :]
                spectrogram, phase = from_audio(sample, n_fft, hop)
                spectrograms[s, m] = spectrogram
                phases[s, m] = phase
                progress.update(task, advance=1)

    save_path = Path('dataset/mesh_rir/s32/')
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)


    print('Saving and cleaning up ...')
    gc.collect()
    np.save(save_path / 'spectrograms.npy', spectrograms)
    np.save(save_path / 'phases.npy', phases)
    del spectrograms
    del phases
    os.remove("s.temp")
    os.remove("p.temp")
    print('Done.')
    

def main():
    parser = argparse.ArgumentParser(description='Select which rooms to process.')
    parser.add_argument('-s1', help='Process room S1', action='store_true')
    parser.add_argument('-s32', help='Process room S2', action='store_true')

    args = parser.parse_args()

    params = load_params('params.yaml')

    data = Path(params['paths']['data'])
    s1_path = params['paths']['meshrir']['s1']
    s32_path = params['paths']['meshrir']['s32']

    s1_path = data / s1_path
    s32_path = data / s32_path

    # print(s1_path)
    # print(s32_path)

    n_fft = params['spectrograms']['n_fft']
    hop = n_fft // 4

    if args.s1:
        print('Converting room S1 to spectrograms')
        process_s1(s1_path, n_fft, hop)
    if args.s32:
        print('Converting room S32 to spectrograms')
        process_s32(s32_path, n_fft, hop)
    if not args.s1 and not args.s32:
        print('Converting room S1 and S32 to spectrograms')
        process_s1(s1_path, n_fft, hop)
        process_s32(s32_path, n_fft, hop)

if __name__ == "__main__":
    main()


