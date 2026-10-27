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

"""Tests for the OneDrive / SharePoint batching algorithm.

Run with:
  python3 -m unittest tests.files.files_batching_test -v
"""

import math
import random
import unittest
from unittest import mock

import pandas as pd

from batching import files_batching
from batching import files_eta
from util.constants import (
    FILES_GLOBAL_CORPUS_SIZE_LIMIT,
    FILES_MAX_SITES_PER_BATCH,
    SITE_SIMILARITY_MAX_RATIO,
    SITE_SIMILARITY_MIN_K,
)

GB = 1024**3
# 3,000 licenses -> 12.5 x 0.8 = 10 files/sec per batch (36,000 files/hour).
LICENSES = {"totalAllotedUnits": {"User": 3000, "Company": 0}}
COLUMNS = ["Site Id", "File Count", "Folder Count", "Resource Count", "Corpus Size"]


def _df(sites):
  """sites: list of (file_count, size_gb)."""
  return pd.DataFrame([
      {"Site Id": f"site{i}", "File Count": files, "Folder Count": 0,
       "Resource Count": files, "Corpus Size": size_gb * GB}
      for i, (files, size_gb) in enumerate(sites)
  ], columns=COLUMNS)


def _random_sites(n, seed):
  rnd = random.Random(seed)
  return [(int(rnd.lognormvariate(8, 1.6)), rnd.lognormvariate(21, 2) / GB) for _ in range(n)]


def _batch(df, parallel=1, min_sites=1, max_sites=1000, max_batches=100, **kwargs):
  return files_batching.calculate_files_migration_batches(
      df, LICENSES, min_sites=min_sites, max_sites=max_sites,
      num_parallel=parallel, max_batches=max_batches, **kwargs,
  )


def _fastest_possible(sites, lanes):
  """max(all sites as one batch / lanes, longest single site), in hours."""
  rate = files_eta.get_files_batch_rate(LICENSES)
  total = files_eta.calculate_files_eta(sum(s for _, s in sites) * GB, sum(f for f, _ in sites), rate)
  longest = max(files_eta.calculate_files_eta(s * GB, f, rate) for f, s in sites)
  return max(total / lanes, longest)


def _is_similar(batch):
  """At least min(K, sites) of the batch's largest sites within the ratio."""
  counts = sorted(batch["df_subset"]["File Count"], reverse=True)
  required = min(SITE_SIMILARITY_MIN_K, len(counts))
  return required <= 1 or counts[0] / max(counts[required - 1], 1) <= SITE_SIMILARITY_MAX_RATIO


class FilesBatchingTest(unittest.TestCase):
  """Verifies batching keeps the total ETA minimal and largest sites similar."""

  def test_every_site_is_batched_once(self):
    sites = [(100_000, 50)] * 12 + [(20_000, 300)] * 40 + [(500, 5)] * 300
    df_final, batches, _, _ = _batch(_df(sites), parallel=3, min_sites=10, max_sites=100)
    self.assertEqual(sum(b["sites"] for b in batches), len(sites))
    self.assertTrue((df_final["Suggested Batch"] != "").all())
    batched_ids = [sid for b in batches for sid in b["df_subset"]["Site Id"]]
    self.assertEqual(len(batched_ids), len(set(batched_ids)))

  def test_total_eta_matches_fastest_possible(self):
    """Similarity never makes the plan slower than balancing alone."""
    for seed, lanes in ((1, 7), (2, 5), (3, 3)):
      with self.subTest(seed=seed):
        sites = _random_sites(1500, seed)
        _, _, total_eta, _ = _batch(_df(sites), parallel=lanes, min_sites=25, max_sites=500,
                                    max_batches=50)
        self.assertLessEqual(total_eta, _fastest_possible(sites, lanes) * 1.001)

  def test_minimum_sites_does_not_slow_the_plan(self):
    """With 1,000-5,000 sites per batch and 2,000 sites, all 7 lanes are still used."""
    sites = _random_sites(2000, 4)
    _, batches, total_eta, buckets = _batch(_df(sites), parallel=7, min_sites=1000,
                                            max_sites=5000, max_batches=50)
    self.assertEqual(len(buckets), 7)
    self.assertLessEqual(total_eta, _fastest_possible(sites, 7) * 1.001)

  def test_largest_sites_in_each_batch_are_similar(self):
    """Clusters of similar sites: every batch's largest sites are within the ratio."""
    sites = ([(100_000, 50)] * 20 + [(30_000, 30)] * 20 + [(5_000, 10)] * 40
             + [(200, 1)] * 400)
    _, batches, _, _ = _batch(_df(sites), parallel=4, min_sites=10, max_sites=200)
    for batch in batches:
      self.assertTrue(_is_similar(batch), batch["name"])
      self.assertFalse(batch["largest_sites_dissimilar"], batch["name"])

  def test_seven_similar_sites_are_enough(self):
    """K ranges from 7 to 10: a batch led by 8 similar sites meets the rule."""
    sites = [(100_000, 10)] * 8 + [(1_500, 1)] * 50
    _, batches, _, _ = _batch(_df(sites), parallel=1, max_batches=1)
    self.assertEqual(len(batches), 1)
    self.assertFalse(batches[0]["largest_sites_dissimilar"])

  def test_fewer_than_seven_similar_sites_is_a_miss(self):
    sites = [(100_000, 10)] * 5 + [(1_500, 1)] * 50
    _, batches, _, _ = _batch(_df(sites), parallel=1, max_batches=1)
    self.assertTrue(batches[0]["largest_sites_dissimilar"])

  def test_one_large_site_is_not_mixed_into_a_batch_of_small_heads(self):
    """The customer case: 1 x 1M-file site and 49 x 10k-file sites."""
    sites = [(1_000_000, 10)] + [(10_000, 10)] * 49 + [(100, 1)] * 200
    _, batches, total_eta, _ = _batch(_df(sites), parallel=2, min_sites=10, max_sites=60)
    self.assertAlmostEqual(total_eta, _fastest_possible(sites, 2), places=3)
    for batch in batches:
      counts = sorted(batch["df_subset"]["File Count"], reverse=True)
      if counts[0] == 1_000_000:
        self.assertNotIn(10_000, counts)

  def test_giant_site_shares_its_batch_with_size_heavy_sites(self):
    """A file-bound giant takes size-bound fillers; its batch is logged."""
    log = mock.Mock()
    sites = [(1_000_000, 10)] + [(10_000, 10)] * 40 + [(100, 400)] * 60
    _, batches, total_eta, _ = _batch(_df(sites), parallel=2, log_fn=log)
    self.assertLessEqual(total_eta, _fastest_possible(sites, 2) * 1.01)
    giant_batch = next(b for b in batches if 1_000_000 in set(b["df_subset"]["File Count"]))
    self.assertGreater(giant_batch["sites"], 1)
    self.assertTrue(giant_batch["largest_sites_dissimilar"])
    messages = " ".join(call.args[0] for call in log.call_args_list)
    self.assertIn("no similar-size peers", messages)
    self.assertIn(giant_batch["name"], messages)

  def test_lanes_are_balanced_and_total_is_busiest_lane(self):
    sites = [(50_000, 100)] * 30 + [(5_000, 50)] * 100 + [(300, 2)] * 500
    _, _, total_eta, buckets = _batch(_df(sites), parallel=5, min_sites=10, max_sites=150)
    totals = [b["total"] for b in buckets]
    self.assertAlmostEqual(total_eta, max(totals), places=9)
    self.assertLessEqual(total_eta, _fastest_possible(sites, 5) * 1.001)
    for bucket in buckets:
      self.assertAlmostEqual(bucket["total"], sum(c["eta"] for c in bucket["batches"]), places=9)

  def test_never_more_than_5000_sites_per_batch(self):
    """The 5,000-site limit holds even if max batches or the setting would break it."""
    sites = [(100, 0.1)] * 12_000
    _, batches, _, _ = _batch(_df(sites), parallel=3, max_sites=10_000, max_batches=2)
    self.assertEqual(len(batches), math.ceil(12_000 / FILES_MAX_SITES_PER_BATCH))
    self.assertTrue(all(b["sites"] <= FILES_MAX_SITES_PER_BATCH for b in batches))

  def test_respects_max_sites_and_max_batches(self):
    sites = [(1_000, 1)] * 500
    _, batches, _, _ = _batch(_df(sites), parallel=3, min_sites=20, max_sites=50, max_batches=12)
    self.assertLessEqual(len(batches), 12)
    self.assertTrue(all(b["sites"] <= 50 for b in batches))

  def test_batches_are_named_by_start_time(self):
    sites = [(10_000, 10)] * 60 + [(100, 1)] * 200
    _, batches, _, _ = _batch(_df(sites), parallel=2, min_sites=10, max_sites=60)
    self.assertEqual([b["name"] for b in batches], [f"Batch {i + 1}" for i in range(len(batches))])
    starts = [(b["start_time"], b["bucket_idx"]) for b in batches]
    self.assertEqual(starts, sorted(starts))

  def test_version_estimate_columns_drive_eta(self):
    """Shallow Scan's version-inclusive columns are used when passed in."""
    df = _df([(10_000, 0)])
    df["Est Files"] = 30_000
    df["Est Size"] = 1024
    _, _, eta_active, _ = _batch(_df([(10_000, 0)]))
    _, _, eta_versions, _ = _batch(df, eta_size_col="Est Size", eta_file_col="Est Files")
    self.assertAlmostEqual(eta_versions, eta_active * 3, places=6)

  def test_folders_do_not_affect_eta_or_similarity(self):
    df = _df([(10_000, 10)] * 20)
    with_folders = df.assign(**{"Folder Count": 50_000, "Resource Count": 60_000})
    _, batches_a, eta_a, _ = _batch(df, parallel=2)
    _, batches_b, eta_b, _ = _batch(with_folders, parallel=2)
    self.assertAlmostEqual(eta_a, eta_b, places=9)
    self.assertEqual([b["sites"] for b in batches_a], [b["sites"] for b in batches_b])

  def test_worked_example(self):
    """The 10-site, 3-lane walkthrough (K = 3, 10 files/sec) reaches the true optimum.

    The best possible split of these sites over 3 lanes (found by trying every
    assignment) takes 12.0 h.
    """
    sites = [(360_000, 40), (108_000, 1600), (90_000, 800), (79_200, 1200), (36_000, 2000),
             (28_800, 400), (25_200, 2400), (7_200, 1600), (3_600, 800), (1_800, 3200)]
    df = _df(sites)
    df["Site Id"] = list("ABCDEFGHIJ")
    with mock.patch.object(files_batching, "SITE_SIMILARITY_MIN_K", 3), \
         mock.patch.object(files_batching, "SITE_SIMILARITY_MAX_K", 3):
      _, _, total_eta, _ = _batch(df, parallel=3, max_sites=10, max_batches=6)
    self.assertAlmostEqual(total_eta, 12.0, places=3)


class FilesBatchingEdgeCaseTest(unittest.TestCase):
  """Verifies unusual reports and settings are handled."""

  def test_empty_report(self):
    df_final, batches, total_eta, buckets = _batch(_df([]))
    self.assertEqual((len(df_final), batches, total_eta, buckets), (0, [], 0.0, []))
    self.assertIn("Suggested Batch", df_final.columns)

  def test_all_sites_empty(self):
    df_final, batches, total_eta, _ = _batch(_df([(0, 0)] * 50), parallel=3, max_sites=10)
    self.assertEqual(total_eta, 0)
    self.assertEqual(sum(b["sites"] for b in batches), 50)
    self.assertTrue((df_final["Suggested Batch"] != "").all())
    self.assertFalse(any(b["largest_sites_dissimilar"] for b in batches))

  def test_all_sites_very_small(self):
    """Sites under 1,000 files never head a batch and are exempt from the rule."""
    sites = [(random.Random(i).randint(0, 999), 0.5) for i in range(800)]
    _, batches, total_eta, _ = _batch(_df(sites), parallel=4, min_sites=25, max_sites=150,
                                      max_batches=50)
    self.assertLessEqual(total_eta, _fastest_possible(sites, 4) * 1.001)
    self.assertFalse(any(b["largest_sites_dissimilar"] for b in batches))
    self.assertTrue(all(b["sites"] <= 150 for b in batches))

  def test_single_site_and_fewer_sites_than_lanes(self):
    _, batches, total_eta, buckets = _batch(_df([(36_000, 1)]), parallel=3, min_sites=25)
    self.assertEqual((len(batches), len(buckets)), (1, 1))
    self.assertAlmostEqual(total_eta, 1.0, places=6)
    _, batches, _, buckets = _batch(_df([(5_000, 5), (3_000, 2)]), parallel=5)
    self.assertEqual(sum(b["sites"] for b in batches), 2)
    self.assertLessEqual(len(buckets), 2)

  def test_invalid_values_are_treated_as_zero(self):
    df = _df([(10_000, 10), (20_000, 10), (5_000, 1)])
    df["File Count"] = df["File Count"].astype(object)
    df.loc[0, "File Count"] = "N/A"
    df.loc[1, "Corpus Size"] = float("nan")
    df.loc[2, "File Count"] = -500
    df_final, batches, total_eta, _ = _batch(df, parallel=2)
    self.assertEqual(sum(b["sites"] for b in batches), 3)
    self.assertGreater(total_eta, 0)
    self.assertTrue((df_final["Suggested Batch"] != "").all())

  def test_duplicate_site_ids_each_get_a_batch(self):
    df = _df([(10_000, 10)] * 4)
    df["Site Id"] = "same-site"
    df_final, batches, _, _ = _batch(df, parallel=2, max_sites=2)
    self.assertEqual(sum(b["sites"] for b in batches), 4)
    self.assertTrue((df_final["Suggested Batch"] != "").all())

  def test_missing_site_id_column(self):
    df = _df([(10_000, 10)] * 3).drop(columns=["Site Id"])
    df_final, batches, _, _ = _batch(df)
    self.assertEqual(sum(b["sites"] for b in batches), 3)
    self.assertTrue((df_final["Suggested Batch"] != "").all())

  def test_invalid_settings_are_clamped(self):
    """Zero lanes or batches, and a minimum above the maximum, still plan safely."""
    sites = [(10_000, 10)] * 30
    _, batches, total_eta, _ = _batch(_df(sites), parallel=0, min_sites=100, max_sites=10,
                                      max_batches=0)
    self.assertEqual(sum(b["sites"] for b in batches), 30)
    self.assertTrue(all(b["sites"] <= 10 for b in batches))
    self.assertGreater(total_eta, 0)


class ShallowExclusionTest(unittest.TestCase):
  """Verifies sites with >500k-item entities are excluded from batching."""

  def test_large_entity_sites_are_marked_deep_scan_recommended(self):
    df = pd.DataFrame([
        {"Site Id": "large", "File Count": 600_000, "Entities with > 500k item count": 2,
         "Shortcut Count": "N/A"},
        {"Site Id": "small", "File Count": 100, "Entities with > 500k item count": 0,
         "Shortcut Count": "N/A"},
    ])
    batch_fn = mock.Mock(side_effect=lambda d, _lm: (
        d.assign(**{"Suggested Batch": "Batch 1"}), [{"shortcut_count": 0}], 1.0,
        [{"batches": [{"shortcut_count": 0}]}]))
    df_final, batches, total_eta, buckets = files_batching.batch_with_shallow_exclusions(
        df, {}, batch_fn
    )
    batch_of = dict(zip(df_final["Site Id"], df_final["Suggested Batch"]))
    self.assertEqual(batch_of, {"large": "Deep Scan Recommended", "small": "Batch 1"})
    self.assertEqual(list(batch_fn.call_args.args[0]["Site Id"]), ["small"])
    self.assertEqual(batches[0]["shortcut_count"], "N/A")
    self.assertEqual(buckets[0]["batches"][0]["shortcut_count"], "N/A")
    self.assertEqual(total_eta, 1.0)

  def test_all_sites_excluded(self):
    df = pd.DataFrame([{"Site Id": "large", "File Count": 600_000,
                        "Entities with > 500k item count": 1}])
    batch_fn = mock.Mock()
    df_final, batches, total_eta, buckets = files_batching.batch_with_shallow_exclusions(
        df, {}, batch_fn
    )
    batch_fn.assert_not_called()
    self.assertEqual((batches, total_eta, buckets), ([], 0.0, []))
    self.assertEqual(df_final["Suggested Batch"].iloc[0], "Deep Scan Recommended")


if __name__ == "__main__":
  unittest.main()
