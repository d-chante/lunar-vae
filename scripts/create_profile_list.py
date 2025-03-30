import argparse
import os

from utils import CreateProfileListJson, CreateListJsonFromDir

parser = argparse.ArgumentParser(
    description="""
    Creates profile_list.json file with a list of all Profiles in a directory

    Example usage:
        ./create_profile_list -d /path/to/profiles -o /path/to/output/directory
    """,
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    '-d',
    '--directory',
    type=str,
    help="Path to Profiles directory",
    required=True
)
parser.add_argument(
    '-o',
    '--output',
    type=str,
    help="Path to output directory"
)
parser.add_argument(
    '-v',
    '--verify',
    action="store_true",
    default=False,
    help="Verify profile (for v2 data only)"
)
args = parser.parse_args()


def main():
    json_list = CreateListJsonFromDir(args.directory, args.verify)
    CreateProfileListJson(
        os.path.join(
            args.output,
            'profile_list.json'),
        json_list)


if __name__ == "__main__":
    main()
