from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--input-dir", default="")
parser.add_argument("--output-dir", default="")


def main():
    flags = parser.parse_args()

    input_dir = flags.input_dir
    output_dir = flags.output_dir
    print(input_dir)
    print(output_dir)
    scene_dir = [x for x in sorted(os.listdir(input_dir))]
    print(scene_dir)
    for scene in scene_dir:
        print(scene)
        if scene[0] != '.':
            os.mkdir(output_dir + scene)
            files = [x for x in sorted(os.listdir(input_dir + scene))]
            for f in files:
                name = f.split('.')[0]
                img = Image.open(input_dir +scene + "/" + f)
                img.save(output_dir + scene + "/" + name + '.png')

if __name__ == '__main__':
    main()




