import argparse
import os

from alphai_delphi.performance.performance import OraclePerformance

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', help="The dir where the result are")
parser.add_argument('oracle_prefix', help="the prefixt of the oracle")

args = parser.parse_args()

oracle_performance = OraclePerformance(
    args.output_dir, args.oracle_prefix
)

oracle_performance.create_oracle_report()
