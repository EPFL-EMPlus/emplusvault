from emv.pipelines.utils import create_optimized_media
import os

def main():
    base_path = "/mnt/rts/archives/"
    count = 0
    # find all underlying folders and execute create_optimized_media on them
    for root, dirs, files in os.walk(base_path):
        if not dirs:
            print(f"Processed {count} files. Current file {root}")
            count += 1
            try:
                r = create_optimized_media(root, root, False)
            except IndexError:
                print(f"Error processing {root}")
                continue
if __name__ == '__main__':
    main()