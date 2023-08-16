'''
    File name: view.py
    Author: Richard Dirauf
    Python Version: 3.8
    Description: Save the sequences as videos.
'''

import cv2
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        type=str,
                        required=True,
                        help="Path to directory.")
    parser.add_argument("--gray",
                        type=bool,
                        default=False,
                        help="Grayscale?.")
    args = parser.parse_args()

    # load seq
    if not os.path.isdir(args.dir):
        raise Exception()

    delay = int(1000 / 12.5)
    imgs = os.listdir(args.dir)
    imgs.sort()
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 12.5, (256, 128))

    for img in imgs:
        path = os.path.join(args.dir, img)
        frame = cv2.imread(path)
        if args.gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
        cv2.imshow("Polar Bears", frame)

        if cv2.waitKey(delay) == ord('q'):
            return

    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    main()
