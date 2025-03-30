import argparse

from utils import ConvertProfileListToCsv, LoadProfileList

parser = argparse.ArgumentParser(
    description="""
    Converts Profiles stored in multiple .json files into a single yyyymmdd-dataset.csv file
    Each line in the CSV will contain a JSON string representation of the profile data.
    """,
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    '-d',
    '--directory',
    type=str,
    help="Path to Profiles directory",
    required=True)
parser.add_argument(
    '-i',
    '--input',
    type=str,
    help="Path to input profile_list.json file",
    required=True)
parser.add_argument(
    '-o',
    '--output',
    type=str,
    help="Path to output directory",
    required=True)

args = parser.parse_args()


def main():
    profile_list = LoadProfileList(args.input)
    ConvertProfileListToCsv(args.directory, profile_list, args.output)


if __name__ == "__main__":
    main()
