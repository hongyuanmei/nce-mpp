# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
organize logs
@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime
from itertools import chain

from nce_point_process.io.log import LogBatchReader

import argparse
__author__ = 'Hongyuan Mei'


def main():

    parser = argparse.ArgumentParser(description='organize logs')

    parser.add_argument(
        '-ds', '--Dataset', required=True, type=str, help='e.g. smallnhp, meme, etc'
    )
    parser.add_argument(
		'-cn', '--CSVName', required=True, type=str, 
		help='prefix name of meta and log csv'
	)
    parser.add_argument(
		'-rp', '--RootPath', type=str, default='../../',
		help='root path of project'
	)

    args = parser.parse_args()
    dict_args = vars(args)

    root_path = os.path.abspath(args.RootPath)
    args.PathLog = os.path.join(root_path, f"logs/{dict_args['Dataset']}")

    org = LogBatchReader(args.PathLog, args.CSVName)
    org.write_csv()


if __name__ == "__main__": main()
