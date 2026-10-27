# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the OneDrive / SharePoint ETA calculation.

Run with:
  python3 -m unittest tests.files.files_eta_test -v
"""

import unittest

import pandas as pd

from batching import files_eta
from util.constants import (
    EST_CORPUS_SIZE_COL,
    EST_FILE_COUNT_COL,
    FILES_GLOBAL_CORPUS_SIZE_LIMIT,
)


class FilesBatchRateTest(unittest.TestCase):
  """Verifies the per-batch rate follows the license tier with the buffer."""

  def test_rate_follows_license_tier_with_buffer(self):
    cases = [
        (0, 6.2), (500, 6.2), (1000, 6.2), (3000, 12.5), (10000, 18.7),
        (50000, 20.8), (50001, 22.3),
    ]
    for licenses, observed in cases:
      with self.subTest(licenses=licenses):
        rate = files_eta.get_files_batch_rate(
            {"totalAllotedUnits": {"User": licenses, "Company": 0}}
        )
        self.assertAlmostEqual(rate, observed * 0.8, places=9)

  def test_user_and_company_licenses_are_summed(self):
    rate = files_eta.get_files_batch_rate(
        {"totalAllotedUnits": {"User": 800, "Company": 300}}
    )
    self.assertAlmostEqual(rate, 12.5 * 0.8, places=9)

  def test_missing_license_metrics_use_lowest_tier(self):
    self.assertAlmostEqual(files_eta.get_files_batch_rate({}), 6.2 * 0.8, places=9)


class FilesEtaTest(unittest.TestCase):
  """Verifies batch time = max(files / rate, bytes / size limit), plus encrypted files."""

  def test_file_bound_batch(self):
    eta = files_eta.calculate_files_eta(total_size=1024, total_files=36_000, files_per_sec=10)
    self.assertAlmostEqual(eta, 1.0, places=6)

  def test_size_bound_batch(self):
    size = FILES_GLOBAL_CORPUS_SIZE_LIMIT * 3600 * 2  # Two hours of bytes.
    eta = files_eta.calculate_files_eta(total_size=size, total_files=100, files_per_sec=10)
    self.assertAlmostEqual(eta, 2.0, places=6)

  def test_encrypted_files_migrate_at_half_speed(self):
    plain = files_eta.calculate_files_eta(1024, 36_000, 10)
    encrypted = files_eta.calculate_files_eta(1024, 36_000, 10, encrypted_files=36_000, encrypted_size=1024)
    self.assertAlmostEqual(encrypted, plain * 2, places=6)

  def test_empty_batch_takes_no_time(self):
    self.assertEqual(files_eta.calculate_files_eta(0, 0, 10), 0)


class ResolveEtaColumnsTest(unittest.TestCase):
  """Verifies version-inclusive columns are used when present."""

  def test_resolve_eta_columns(self):
    plain = pd.DataFrame([{"Corpus Size": 1, "File Count": 1}])
    self.assertEqual(files_eta.resolve_eta_columns(plain), ("Corpus Size", "File Count"))
    with_est = plain.assign(**{EST_FILE_COUNT_COL: 10, EST_CORPUS_SIZE_COL: 10})
    self.assertEqual(
        files_eta.resolve_eta_columns(with_est), (EST_CORPUS_SIZE_COL, EST_FILE_COUNT_COL)
    )


if __name__ == "__main__":
  unittest.main()
