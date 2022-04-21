import os
import argparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--width', type=float, default=227)
    parser.add_argument('--height', type=int, default=227)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    domain_names = sorted([f.name for f in os.scandir(args.data_dir) if f.is_dir()])
    class_names = sorted([f.name for f in os.scandir(os.path.join(args.data_dir, domain_names[0])) if f.is_dir()])
    new_size = (args.width, args.height)

    print('\nResizing dataset images:\n')
    for domain_name in domain_names:
        print(f'\tDomain {domain_name}', end=' ', flush=True)

        for i, class_name in enumerate(class_names):
            # Get image names
            src_dir = os.path.join(args.data_dir, domain_name, class_name)
            files = sorted([f for f in os.listdir(src_dir)])

            out_dir = os.path.join(args.output_dir, domain_name, class_name)
            os.makedirs(out_dir, exist_ok=True)

            # Resize images and save
            for f in files:
                path = os.path.join(src_dir, f)
                image = Image.open(path)
                input_size = (image.width, image.height)

                if input_size != new_size:
                    image = image.resize(new_size, Image.BILINEAR)

                image.save(os.path.join(out_dir, f))

        print('Ok')

    print('\nDone.\n')
