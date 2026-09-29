"""Skeleton entry point: evaluate_task4."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to task YAML configuration')
    parser.parse_args()
    parser.exit(2, 'Not implemented: complete the research decision and this pipeline first.\n')

if __name__ == '__main__':
    main()
