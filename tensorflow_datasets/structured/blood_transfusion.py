# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Blood transfusion dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@ONLINE {blood_transfusion,
author = "I-Cheng Yeh, King-Jang Yang, Tao-Ming Ting",
title  = "blood transfusion service center",
month  = "may",
year   = "2015",
url    = "https://www.openml.org/d/1464"
}
"""

_DESCRIPTION = ("Dataset describing whether a person donated "
				"blood in March 2007 based on the features  "
				"months since last donation (V1), total number of "
				"donations (V2), total blood donated in ml (V3) and "
				"months since first donation (V4).")
				
_DONATEDMAR07_DICT = {"2": "yes", "1": "no"}

def convert_to_int(d):
  return -1 if d == "?" else np.int32(d)



FEATURE_DICT = collections.OrderedDict([
    ("V1", (tf.int32, convert_to_int)),
    ("V2", (tf.int32, convert_to_int)),
    ("V3", (tf.int32, convert_to_int)),
    ("V4", (tf.int32, convert_to_int)),
])

_URL = "https://www.openml.org/data/get_csv/1586225/php0iVrYT"

class Blood_transfusion(tfds.core.GeneratorBasedBuilder):
  """Blood transfusion dataset."""

  VERSION = tfds.core.Version("1.0.0")

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "donated_mar_07": tfds.features.ClassLabel(names=["yes", "no"]),
            "features": {name: dtype
                         for name, (dtype, func) in FEATURE_DICT.items()}
        }),
        supervised_keys=("features", "donated_mar_07"),
        urls=["https://www.openml.org/d/1464"],
        citation=_CITATION
		)
		
  def _split_generators(self, dl_manager):
    path = dl_manager.download(_URL)

    # There is no predefined train/val/test split for this dataset.
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=1,
            gen_kwargs={
                "file_path": path
            }),
    ]

  def _generate_examples(self, file_path):
    """Generate features and target given the directory path.
    Args:
      file_path: path where the csv file is stored
    Yields:
      The features and the target
    """

    with tf.io.gfile.GFile(file_path) as f:
      raw_data = csv.DictReader(f)
      for row in raw_data:
        donated_val = row.pop("donated_mar_07")
        yield {
            "donated_mar_07": convert_to_label(donated_val, _DONATEDMAR07_DICT),
            "features": {
                name: FEATURE_DICT[name][1](value)
                for name, value in row.items()
            }
}