import os
import argparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/orig/PACS/kfold')
    parser.add_argument('--width', type=float, default=227)
    parser.add_argument('--height', type=int, default=227)
    parser.add_argument('--output_dir', type=str, default='data/pre/PACS/images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    domains = sorted([f.name for f in os.scandir(args.data_dir) if f.is_dir()])
    classes = sorted([f.name for f in os.scandir(os.path.join(args.data_dir,
                      domains[0])) if f.is_dir()])
    new_size = (args.width, args.height)

    for dom_name in domains:

        print(f'Processing {dom_name}', end=' ', flush=True)

        for i, class_name in enumerate(classes):
            # Get image names
            src_dir = os.path.join(args.data_dir, dom_name, class_name)
            files = sorted([f for f in os.listdir(src_dir)])

            out_dir = os.path.join(args.output_dir, dom_name, class_name)
            os.makedirs(out_dir, exist_ok=True)

            # Resize and save images
            for f in files:
                # Load input image
                path = os.path.join(src_dir, f)
                image = Image.open(path)
                input_size = (image.width, image.height)

                # Resize input image
                if input_size != new_size:
                    image = image.resize(new_size, Image.BILINEAR)
                image.save(os.path.join(out_dir, f))

        print('Ok')
