import argparse
from archiving import promote_to_archive

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--notes", default="")
args = parser.parse_args()

archive_dir = promote_to_archive(args.run, args.name, args.notes)
print(f"Archived '{args.run}' as '{args.name}' -> {archive_dir}")