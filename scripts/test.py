from pathlib import Path
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--directory', type=Path)
  parser.add_argument('--changed-files', type=Path, nargs="*")

  args = parser.parse_args()

  directory = args.directory
  changed_files = args.changed_files

  print(directory)
  print(list(directory.rglob("*")))
  print(changed_files)
