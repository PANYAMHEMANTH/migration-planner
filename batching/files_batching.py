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

"""Batching algorithm for OneDrive and SharePoint sites (Deep and Shallow Scan).

Goals, in priority order:
  1. The total ETA (the busiest lane; lanes are batches running in parallel)
     is as low as possible. Similarity never makes a plan slower.
  2. In every batch, at least K (SITE_SIMILARITY_MIN_K to SITE_SIMILARITY_MAX_K)
     of the largest sites have similar file counts (largest <=
     SITE_SIMILARITY_MAX_RATIO x the K-th largest), so they finish around the
     same time and the batch doesn't wait on one site.
  3. Batches respect the configured minimum sites per batch where possible,
     and batches within a lane take a similar time.

Hard limits: at most min(max sites per batch, FILES_MAX_SITES_PER_BATCH)
sites per batch, and at most the configured number of batches (unless the
per-batch site limit needs more).

Steps:
  1. Sort sites by file count, largest first.
  2. For each candidate batch count N, set a target batch time (all sites
     together / N) and:
     a. Form heads: groups of up to SITE_SIMILARITY_MAX_K consecutive sites
        within the ratio, never longer than the target (+15%). Groups shorter
        than SITE_SIMILARITY_MIN_K are topped up if they stay within the
        near-miss ratio and the target; a site with no similar-size peers is
        a giant site. Sites with very few files never head a batch.
     b. The N largest heads start N batches. Every other site is a filler.
     c. Each filler goes to the batch whose time rises least. This mixes
        file-heavy and size-heavy sites so both throughput limits are used.
        Ties go to a batch whose largest sites stay similar, then to the
        batch that would finish soonest, which fills batches evenly.
     d. Batches go to lanes, longest first, each to the least busy lane.
     e. Improve: move or swap sites between lanes while that lowers the
        busiest lane, without breaking any batch's similarity. For the most
        promising plans, also merge batches within a lane when that is faster
        or fills a batch below the minimum sites.
     The same is done without heads (pure balancing) as a fallback, so the
     chosen plan is never slower than balancing alone.
  3. Keep the fastest plan; break ties by fewest similarity misses, then
     fewest batches below the minimum sites, then most even batches within
     lanes, then fewest batches.
"""

import heapq
import math
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from batching import files_eta
from util.constants import (
    BATCH_PLAN_ETA_TOLERANCE,
    BATCH_TARGET_TOLERANCE,
    FILES_MAX_SITES_PER_BATCH,
    SITE_SIMILARITY_MAX_K,
    SITE_SIMILARITY_MAX_RATIO,
    SITE_SIMILARITY_MIN_FILES,
    SITE_SIMILARITY_MIN_K,
    SITE_SIMILARITY_NEAR_MISS_RATIO,
)

# Limits on candidate batch counts, the coarse search and lane improvement, to
# keep planning fast for very large tenants.
_MAX_CANDIDATE_BATCH_COUNTS = 40
_COARSE_SEARCH_POINTS = 10
_SEARCH_IMPROVE_STEPS = 200  # Per candidate plan while searching.
_FINAL_IMPROVE_STEPS = 2000  # For the finalists.
_FINALISTS = 3
_IMPROVE_CANDIDATES = 40
_EPSILON = 1e-9
_DEEP_SCAN_RECOMMENDED = "Deep Scan Recommended"
_LARGE_ENTITY_COL = "Entities with > 500k item count"
_SHALLOW_NA_COLS = (
    "Shortcut Count",
    "Folder Count > Depth Limit 100",
    "File Count > Depth Limit 100",
)


class _Sites:
  """Per-site numbers needed for batching, indexed by position."""

  def __init__(self, files, sizes, enc_files, enc_sizes, files_per_sec, min_k, max_k):
    self.min_k = min_k
    self.max_k = max_k
    self.files = files
    self.sizes = sizes
    self.enc_files = enc_files
    self.enc_sizes = enc_sizes
    self.files_per_sec = files_per_sec
    self.times = [
        self.eta(files[i], sizes[i], enc_files[i], enc_sizes[i])
        for i in range(len(files))
    ]

  def eta(self, files, size, enc_files, enc_size):
    return files_eta.calculate_files_eta(
        size, files, self.files_per_sec, enc_files, enc_size
    )

  def top_ids(self, site_ids: Sequence[int], k: int) -> List[int]:
    """The k sites with the most files, largest first."""
    return heapq.nlargest(k, site_ids, key=self.files.__getitem__)

  def similarity_ratio(self, top_counts: Sequence[int], num_sites: int) -> float:
    """Largest file count / the K-th largest, with K = min(min_k, num_sites)."""
    required = min(self.min_k, num_sites, len(top_counts))
    if required <= 1:
      return 1.0
    return top_counts[0] / max(top_counts[required - 1], 1)

  def is_dissimilar(self, top_counts: Sequence[int], num_sites: int) -> bool:
    """True if fewer than K of a batch's largest sites are similar in size.

    Batches whose largest site is below SITE_SIMILARITY_MIN_FILES are exempt;
    such sites finish within minutes.
    """
    if not top_counts or top_counts[0] < SITE_SIMILARITY_MIN_FILES:
      return False
    return self.similarity_ratio(top_counts, num_sites) > SITE_SIMILARITY_MAX_RATIO


class _Batch:
  """A batch of sites with running totals, so its time updates cheaply."""

  def __init__(self, sites: _Sites, site_ids: Sequence[int]):
    self._s = sites
    self.site_ids: List[int] = []
    self.files = 0
    self.size = 0.0
    self.enc_files = 0
    self.enc_size = 0.0
    self.time = 0.0
    self.top_counts: List[int] = []  # Largest max_k file counts, descending.
    self.dissimilar = False
    for i in site_ids:
      self.add(i)

  def _eta_delta(self, add=None, remove=None):
    s = self._s
    files, size, enc_files, enc_size = self.files, self.size, self.enc_files, self.enc_size
    for i, sign in ((add, 1), (remove, -1)):
      if i is not None:
        files += sign * s.files[i]
        size += sign * s.sizes[i]
        enc_files += sign * s.enc_files[i]
        enc_size += sign * s.enc_sizes[i]
    return files, size, enc_files, enc_size

  def _refresh(self):
    self.time = self._s.eta(self.files, self.size, self.enc_files, self.enc_size)
    self.dissimilar = self._s.is_dissimilar(self.top_counts, len(self.site_ids))

  def add(self, i):
    self.site_ids.append(i)
    self.files, self.size, self.enc_files, self.enc_size = self._eta_delta(add=i)
    self.top_counts = sorted(self.top_counts + [self._s.files[i]], reverse=True)[:self._s.max_k]
    self._refresh()

  def remove(self, i):
    self.site_ids.remove(i)
    self.files, self.size, self.enc_files, self.enc_size = self._eta_delta(remove=i)
    self.top_counts = sorted(
        (self._s.files[j] for j in self._s.top_ids(self.site_ids, self._s.max_k)), reverse=True
    )
    self._refresh()

  def time_with(self, i):
    return self._s.eta(*self._eta_delta(add=i))

  def time_without(self, i):
    return self._s.eta(*self._eta_delta(remove=i))

  def time_swap(self, out_i, in_j):
    return self._s.eta(*self._eta_delta(add=in_j, remove=out_i))

  def dissimilar_with(self, i=None, without=None):
    """Whether the batch would be dissimilar after adding i and/or removing one."""
    s = self._s
    n = len(self.site_ids)
    if (without is None and i is not None and n >= s.min_k
        and s.files[i] <= self.top_counts[s.min_k - 1]):
      return self.dissimilar  # The site sits below the top K; nothing changes.
    if (i is None and without is not None and n - 1 >= s.min_k
        and s.files[without] < self.top_counts[s.min_k - 1]):
      return self.dissimilar  # The site sits below the top K; nothing changes.
    ids = [j for j in self.site_ids if j != without]
    if i is not None:
      ids.append(i)
    top = sorted((s.files[j] for j in s.top_ids(ids, s.max_k)), reverse=True)
    return s.is_dissimilar(top, len(ids))


def _form_heads(sites: _Sites, order: List[int], max_sites: int,
                cap_hours: float) -> List[List[int]]:
  """Splits sites (largest first) into heads of similar-size sites.

  A head grows while the next site's file count is within
  SITE_SIMILARITY_MAX_RATIO of its first site, up to SITE_SIMILARITY_MAX_K
  sites. A head's batch time never goes above cap_hours, so lanes can stay
  balanced (total time comes first).
  """
  files = sites.files
  ratio = SITE_SIMILARITY_MAX_RATIO
  min_k, max_k = sites.min_k, sites.max_k
  heads = []
  i, n = 0, len(order)
  while i < n:
    first = order[i]
    head = _Batch(sites, [first])
    j = i + 1
    while (j < n and len(head.site_ids) < max_k
           and files[order[j]] * ratio >= files[first]
           and head.time_with(order[j]) <= cap_hours):
      head.add(order[j])
      j += 1
    group = head.site_ids
    # Stopped by the time cap, not by running out of similar sites.
    reached_cap = (len(group) < max_k and j < n
                   and files[order[j]] * ratio >= files[first])

    # Short head: top it up to min_k with the next sites if that's a near
    # miss. Otherwise the first site has no similar-size peers (a giant site).
    is_giant = False
    if len(group) < min_k and j < n and not reached_cap:
      padded = group + order[j:j + min_k - len(group)]
      top = [files[x] for x in padded]
      near_miss = top[0] / max(top[-1], 1) <= SITE_SIMILARITY_NEAR_MISS_RATIO
      if near_miss and _Batch(sites, padded).time <= cap_hours:
        j += len(padded) - len(group)
        group = padded
      else:
        is_giant = not near_miss

    # Absorb a following run of similar sites that is too short to head its
    # own batch. They sit below this head's top K, so its ratio is unchanged.
    if not is_giant and not reached_cap and j < n:
      run = j
      while run < n and run - j < min_k and files[order[run]] * ratio >= files[order[j]]:
        run += 1
      following = [files[x] for x in order[j:run + min_k]]
      if (run - j < min_k and len(group) + (run - j) <= max_sites
          and following[0] / max(following[-1], 1) > SITE_SIMILARITY_NEAR_MISS_RATIO
          and _Batch(sites, group + order[j:run]).time <= cap_hours):
        group = group + order[j:run]
        j = run

    heads.append(group)
    i = j
  return heads


def _build_batches(sites: _Sites, order: List[int], n_batches: int, max_sites: int,
                   use_heads: bool) -> List[_Batch]:
  """Starts a batch per head (largest n), then packs the rest as fillers.

  Sites below SITE_SIMILARITY_MIN_FILES finish quickly, so they never head a
  batch. Without heads, every site is a filler (pure balancing).
  """
  heads = []
  if use_heads:
    target_hours = _Batch(sites, order).time / n_batches
    large = [i for i in order if sites.files[i] >= SITE_SIMILARITY_MIN_FILES]
    heads = _form_heads(sites, large, max_sites, target_hours * BATCH_TARGET_TOLERANCE)
  batches = [_Batch(sites, group) for group in heads[:n_batches]]
  batches += [_Batch(sites, []) for _ in range(n_batches - len(batches))]
  in_heads = set(i for group in heads[:n_batches] for i in group)
  fillers = [i for i in order if i not in in_heads]

  # Slowest fillers first. Each goes to the batch whose time rises least: a
  # size-heavy filler joins a file-heavy batch (or the reverse) at little or
  # no cost, so both throughput limits are used. Ties go to a batch whose
  # largest sites stay similar, then to the batch that would finish soonest.
  for i in sorted(fillers, key=lambda i: -sites.times[i]):
    # n_batches * max_sites >= number of sites, so a batch always has room.
    candidates = [b for b in batches if len(b.site_ids) < max_sites]
    times_with = [b.time_with(i) for b in candidates]
    increases = [round(t - b.time, 6) for b, t in zip(candidates, times_with)]
    least = min(increases)
    tied = [(b, t) for b, t, inc in zip(candidates, times_with, increases) if inc == least]
    best = tied[0][0] if len(tied) == 1 else min(
        tied, key=lambda bt: (bt[0].dissimilar_with(i), bt[1]))[0]
    best.add(i)
  return [b for b in batches if b.site_ids]


def _assign_lanes(batches: List[_Batch], num_lanes: int) -> List[List[_Batch]]:
  """Longest batch first, each to the lane with the least total time."""
  lanes = [[] for _ in range(num_lanes)]
  loads = [0.0] * num_lanes
  for batch in sorted(batches, key=lambda b: -b.time):
    idx = loads.index(min(loads))
    lanes[idx].append(batch)
    loads[idx] += batch.time
  return lanes


def _keeps_similarity(batch: _Batch, add: Optional[int] = None,
                      remove: Optional[int] = None) -> bool:
  """A change may not make a similar batch dissimilar."""
  return batch.dissimilar or not batch.dissimilar_with(add, without=remove)


def _movable(sites: _Sites, batch: _Batch) -> List[int]:
  """Sites that can leave the batch without breaking its similarity."""
  n = len(batch.site_ids)
  if n <= 1:
    return []
  if batch.dissimilar:
    return list(batch.site_ids)
  # Sites below the K-th largest never change the batch's similarity.
  kth = batch.top_counts[sites.min_k - 1] if n - 1 >= sites.min_k else None
  return [i for i in batch.site_ids
          if (kth is not None and sites.files[i] < kth) or _keeps_similarity(batch, remove=i)]


def _improve_lanes(sites: _Sites, lanes: List[List[_Batch]], max_sites: int,
                   max_steps: int) -> None:
  """Moves or swaps sites between lanes while the busiest lane gets shorter.

  A site may move only if neither batch loses its similarity.
  """
  for _ in range(max_steps):
    loads = [sum(b.time for b in lane) for lane in lanes]
    hi = loads.index(max(loads))
    others = sorted((x for x in range(len(lanes)) if x != hi), key=loads.__getitem__)
    if not others or loads[hi] <= _EPSILON:
      return
    gap = loads[hi] - loads[others[0]]

    # Sites in the busiest lane, by how close their saving is to gap / 2.
    # Preselect by site time, then use the exact saving within the batch.
    movable = [(batch, i) for batch in lanes[hi] for i in _movable(sites, batch)]
    movable = heapq.nsmallest(3 * _IMPROVE_CANDIDATES, movable,
                              key=lambda bi: abs(sites.times[bi[1]] - gap / 2))
    candidates = []
    for batch, i in movable:
      saving = batch.time - batch.time_without(i)
      if saving > _EPSILON:
        candidates.append((abs(saving - gap / 2), batch, i))
    candidates = heapq.nsmallest(_IMPROVE_CANDIDATES, candidates, key=lambda c: c[0])

    best = None  # (new busiest of the two lanes, apply)
    for _, src, i in candidates:
      new_src = src.time_without(i)
      for lane_idx in others[:3]:
        for dst in lanes[lane_idx]:
          if len(dst.site_ids) >= max_sites:
            continue
          new_hi = loads[hi] - src.time + new_src
          new_lo = loads[lane_idx] - dst.time + dst.time_with(i)
          worst = max(new_hi, new_lo)
          if (worst < loads[hi] - _EPSILON and (best is None or worst < best[0])
              and _keeps_similarity(dst, i)):
            best = (worst, ("move", src, i, dst))

    if best is None:
      # No single move helps: try swapping a site with a smaller one.
      lo = others[0]
      small = heapq.nsmallest(
          _IMPROVE_CANDIDATES,
          ((sites.times[j], batch, j) for batch in lanes[lo] for j in _movable(sites, batch)),
          key=lambda c: c[0])
      for _, src, i in candidates:
        for _, dst, j in small:
          new_hi = loads[hi] - src.time + src.time_swap(i, j)
          new_lo = loads[lo] - dst.time + dst.time_swap(j, i)
          worst = max(new_hi, new_lo)
          if (worst < loads[hi] - _EPSILON and (best is None or worst < best[0])
              and _keeps_similarity(dst, i, remove=j) and _keeps_similarity(src, j, remove=i)):
            best = (worst, ("swap", src, i, dst, j))
    if best is None:
      return

    action = best[1]
    if action[0] == "move":
      _, src, i, dst = action
      src.remove(i)
      dst.add(i)
    else:
      _, src, i, dst, j = action
      src.remove(i)
      dst.remove(j)
      src.add(j)
      dst.add(i)


def _merge_within_lanes(lanes: List[List[_Batch]], max_sites: int, min_sites: int) -> None:
  """Merges batches in the same lane when that is faster or fills a small batch.

  Batches in a lane run one after another, so one merged batch never takes
  longer than the two separately, and is faster when one is file-heavy and
  the other size-heavy. A merge may not create a new similarity miss or go
  over the site limit.
  """
  for lane in lanes:
    merged_any = True
    while merged_any and len(lane) > 1:
      merged_any = False
      best = None
      for x in range(len(lane)):
        for y in range(x + 1, len(lane)):
          a, b = lane[x], lane[y]
          if len(a.site_ids) + len(b.site_ids) > max_sites:
            continue
          merged = _Batch(a._s, a.site_ids + b.site_ids)
          saving = a.time + b.time - merged.time
          fills_small = len(a.site_ids) < min_sites or len(b.site_ids) < min_sites
          if saving <= _EPSILON and not fills_small:
            continue
          if merged.dissimilar and not (a.dissimilar or b.dissimilar):
            continue
          if best is None or saving > best[0]:
            best = (saving, x, y, merged)
      if best is not None:
        _, x, y, merged = best
        lane[x] = merged
        del lane[y]
        merged_any = True


def _candidate_batch_counts(num_sites: int, max_sites: int, max_batches: int) -> List[int]:
  """Batch counts from the fewest the site limit allows up to max_batches."""
  lowest = max(1, math.ceil(num_sites / max_sites))
  highest = max(lowest, min(max_batches, num_sites))
  if highest == lowest:
    return [lowest]
  span = highest - lowest
  if span < _MAX_CANDIDATE_BATCH_COUNTS:
    return list(range(lowest, highest + 1))
  step = span / (_MAX_CANDIDATE_BATCH_COUNTS - 1)
  return sorted({lowest + round(step * i) for i in range(_MAX_CANDIDATE_BATCH_COUNTS)})


def _lane_evenness(lanes: List[List[_Batch]]) -> float:
  """Mean coefficient of variation of batch times within each lane."""
  values = []
  for lane in lanes:
    times = [b.time for b in lane]
    if len(times) > 1 and statistics.mean(times) > 0:
      values.append(statistics.pstdev(times) / statistics.mean(times))
  return statistics.mean(values) if values else 0.0


def _select_plan(plans: Dict[Tuple[bool, int], Tuple]) -> Tuple[bool, int]:
  """Returns the key of the best plan.

  Fastest plan; ties (within tolerance) go to fewer similarity misses, then
  fewer batches below the minimum sites, then more even batches within lanes,
  then fewer batches.
  """
  fastest = min(p[0] for p in plans.values())
  near_fastest = [key for key, p in plans.items()
                  if p[0] <= fastest * (1 + BATCH_PLAN_ETA_TOLERANCE) + _EPSILON]
  return min(near_fastest, key=lambda key: (
      plans[key][1], plans[key][2], round(plans[key][3], 3), plans[key][4], key))


def _batch_summary(subset: pd.DataFrame) -> Dict[str, Any]:
  def _sum(col, cast):
    if col not in subset.columns:
      return cast(0)
    return cast(pd.to_numeric(subset[col], errors="coerce").fillna(0).sum())

  return {
      "sites": len(subset),
      "dl_count": _sum("DL Count", int),
      "resource_count": _sum("Resource Count", int),
      "folder_count": _sum("Folder Count", int),
      "file_count": _sum("File Count", int),
      "shortcut_count": _sum("Shortcut Count", int),
      "corpus_size": _sum("Corpus Size", float),
  }


def _numeric(df: pd.DataFrame, col: str) -> List[float]:
  """Column as non-negative numbers; missing or invalid values become 0."""
  if col not in df.columns:
    return [0] * len(df)
  return pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).tolist()


def calculate_files_migration_batches(
    df: pd.DataFrame,
    license_metrics: Dict[str, Any],
    min_sites: int,
    max_sites: int,
    num_parallel: int,
    max_batches: int,
    eta_size_col: str = "Corpus Size",
    eta_file_col: str = "File Count",
    log_fn: Optional[Callable[[str], None]] = None,
    format_eta: Optional[Callable[[float], str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], float, List[Dict[str, Any]]]:
  """Groups sites into migration batches and estimates the total time.

  Args:
    df: Site report with a "Site Id" column; numeric columns are coerced in
      place.
    license_metrics: License metrics used to pick the per-batch rate.
    min_sites: Preferred minimum sites per batch. Followed where it doesn't
      make the plan slower.
    max_sites: Maximum sites per batch, never above FILES_MAX_SITES_PER_BATCH.
    num_parallel: Number of batches running in parallel (lanes).
    max_batches: Maximum number of batches in the plan (exceeded only if the
      per-batch site limit requires more).
    eta_size_col: Column holding the size (bytes) used for ETA.
    eta_file_col: Column holding the file count used for ETA and similarity.
    log_fn: Optional callback receiving progress messages.
    format_eta: Optional formatter for hours in progress messages.

  Returns:
    Tuple of (site report with "Suggested Batch", batches, total ETA in hours,
    lanes).
  """
  log_fn = log_fn or (lambda _msg: None)
  format_eta = format_eta or (lambda hours: f"{hours:.1f}h")
  max_sites = max(1, min(int(max_sites or FILES_MAX_SITES_PER_BATCH), FILES_MAX_SITES_PER_BATCH))
  min_sites = max(1, min(int(min_sites or 1), max_sites))
  num_lanes = max(1, int(num_parallel or 1))
  max_batches = max(1, int(max_batches or 1))

  # Ensure numeric columns
  for col in {"Resource Count", "Corpus Size", eta_size_col, eta_file_col}:
    if col not in df.columns:
      df[col] = 0
    else:
      df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
  if "Site Id" not in df.columns:
    df["Site Id"] = [f"site-{i}" for i in range(len(df))]

  df_sorted = df.sort_values(by=eta_file_col, ascending=False, kind="stable")
  df_sorted = df_sorted.reset_index(drop=True)
  if df_sorted.empty:
    df_sorted["Suggested Batch"] = pd.Series(dtype=str)
    return df_sorted, [], 0.0, []

  files = [int(v) for v in _numeric(df_sorted, eta_file_col)]
  sizes = [float(v) for v in _numeric(df_sorted, eta_size_col)]
  sites = _Sites(
      files=files,
      sizes=sizes,
      enc_files=[min(int(v), f) for v, f in zip(_numeric(df_sorted, "Encrypted File Count"), files)],
      enc_sizes=[min(float(v), s) for v, s in zip(_numeric(df_sorted, "Encrypted File Size"), sizes)],
      files_per_sec=files_eta.get_files_batch_rate(license_metrics),
      min_k=max(1, min(SITE_SIMILARITY_MIN_K, max_sites)),
      max_k=max(1, min(SITE_SIMILARITY_MAX_K, max_sites)),
  )
  order = list(range(len(df_sorted)))

  plans = {}

  def _evaluate(use_heads, n_batches):
    key = (not use_heads, n_batches)  # Similar-size plans sort first on ties.
    if key in plans:
      return
    batches = _build_batches(sites, order, n_batches, max_sites, use_heads)
    lanes = _assign_lanes(batches, num_lanes)
    _improve_lanes(sites, lanes, max_sites, _SEARCH_IMPROVE_STEPS)
    plans[key] = _score(lanes)
    total_eta, misses, _, _, num_batches, _ = plans[key]
    log_fn(
        f"Evaluated {n_batches} target batches ({'similar-size' if use_heads else 'balance-only'}): "
        f"{num_batches} batches across {num_lanes} lanes | Total ETA: {format_eta(total_eta)} | "
        f"Batches with dissimilar largest sites: {misses}"
    )

  def _score(lanes):
    batches = [b for lane in lanes for b in lane]
    total_eta = max(sum(b.time for b in lane) for lane in lanes)
    misses = sum(b.dissimilar for b in batches)
    under_min = sum(len(b.site_ids) < min_sites for b in batches)
    return (total_eta, misses, under_min, _lane_evenness(lanes), len(batches), lanes)

  # Coarse pass over the candidate batch counts, then every count around the
  # best. Balance-only plans are the fallback that bounds the total ETA.
  candidates = _candidate_batch_counts(len(order), max_sites, max_batches)
  step = max(1, len(candidates) // _COARSE_SEARCH_POINTS)
  coarse = sorted(set(candidates[::step] + [candidates[-1]]))
  for n_batches in coarse:
    _evaluate(True, n_batches)
    _evaluate(False, n_batches)
  _, best_n = _select_plan(plans)
  pos = candidates.index(best_n)
  for n_batches in candidates[max(0, pos - step):pos + step + 1]:
    _evaluate(True, n_batches)
  # Fully improve the most promising plans, then choose again.
  finalists = {}
  for _ in range(min(_FINALISTS, len(plans))):
    key = _select_plan(plans)
    lanes = plans.pop(key)[5]
    _improve_lanes(sites, lanes, max_sites, _FINAL_IMPROVE_STEPS)
    _merge_within_lanes(lanes, max_sites, min_sites)
    _improve_lanes(sites, lanes, max_sites, _FINAL_IMPROVE_STEPS)
    finalists[key] = _score(lanes)
  total_eta, _, _, _, _, lanes = finalists[_select_plan(finalists)]

  # Name batches by start time across lanes.
  chunks = []
  buckets = []
  for lane in lanes:
    if not lane:
      continue
    lane_chunks = []
    start_time = 0.0
    for batch in sorted(lane, key=lambda b: -b.time):
      positions = sorted(batch.site_ids)
      subset = df_sorted.iloc[positions]
      chunk = _batch_summary(subset)
      chunk.update({
          "eta": batch.time,
          "largest_sites_ratio": sites.similarity_ratio(batch.top_counts, len(batch.site_ids)),
          "largest_sites_dissimilar": batch.dissimilar,
          "df_subset": subset,
          "positions": positions,
          "start_time": start_time,
          "bucket_idx": len(buckets),
      })
      start_time += batch.time
      lane_chunks.append(chunk)
    chunks.extend(lane_chunks)
    buckets.append({
        "id": len(buckets) + 1,
        "total": sum(c["eta"] for c in lane_chunks),
        "batches": lane_chunks,
    })
  chunks.sort(key=lambda c: (c["start_time"], c["bucket_idx"]))
  for i, chunk in enumerate(chunks):
    chunk["name"] = f"Batch {i + 1}"

  dissimilar = [c["name"] for c in chunks if c["largest_sites_dissimilar"]]
  if dissimilar:
    log_fn(
        f"{len(dissimilar)} batch(es) have fewer than {sites.min_k} largest sites within "
        f"{SITE_SIMILARITY_MAX_RATIO}x in file count, as some sites have no similar-size "
        f"peers: {', '.join(dissimilar)}"
    )

  # Assign by position so duplicate Site Ids are handled correctly.
  suggested = [""] * len(df_sorted)
  for chunk in chunks:
    for pos in chunk.pop("positions"):
      suggested[pos] = chunk["name"]
  df_sorted["Suggested Batch"] = suggested
  return df_sorted, chunks, total_eta, buckets

def batch_with_shallow_exclusions(
    df: pd.DataFrame,
    license_metrics: Dict[str, Any],
    batch_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[Any, Any, Any, Any]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], float, List[Dict[str, Any]]]:
  """Batches Shallow Scan sites, excluding sites with >500k-item entities.

  Excluded sites get Suggested Batch = 'Deep Scan Recommended' so they are
  exported to outputs/<ts>/suggested_batches/DeepScanRecommended.csv.

  Args:
    df: Shallow Scan site report.
    license_metrics: License metrics passed through to batch_fn.
    batch_fn: Batching function, called with (df, license_metrics).
    log_fn: Optional callback receiving progress messages.

  Returns:
    Same tuple as calculate_files_migration_batches.
  """
  log_fn = log_fn or (lambda _msg: None)
  if _LARGE_ENTITY_COL not in df.columns:
    numeric_large = pd.Series([0] * len(df), index=df.index)
  else:
    numeric_large = pd.to_numeric(df[_LARGE_ENTITY_COL], errors="coerce").fillna(0)

  df_eligible = df[numeric_large == 0].copy()
  df_excluded = df[numeric_large > 0].copy()

  if not df_excluded.empty:
    df_excluded["Suggested Batch"] = _DEEP_SCAN_RECOMMENDED
    log_fn(
        f"[Shallow Scan] Excluding {len(df_excluded)} site(s) with >500k items "
        f"from ETA calculation (marked as '{_DEEP_SCAN_RECOMMENDED}')."
    )

  if not df_eligible.empty:
    # Convert N/A string columns to 0 temporarily for numeric ETA/batching calculation
    for col in _SHALLOW_NA_COLS:
      if col in df_eligible.columns:
        df_eligible[col] = pd.to_numeric(df_eligible[col], errors="coerce").fillna(0)

    df_eligible_final, batches_list, total_eta, buckets = batch_fn(
        df_eligible, license_metrics
    )

    # Restore N/A display values for shallow scan unsupported metrics
    for col in _SHALLOW_NA_COLS:
      if col in df_eligible_final.columns:
        df_eligible_final[col] = "N/A"
    for b in batches_list:
      b["shortcut_count"] = "N/A"
    for bucket in buckets:
      for b in bucket.get("batches", []):
        b["shortcut_count"] = "N/A"
  else:
    df_eligible_final = df_eligible
    batches_list, total_eta, buckets = [], 0.0, []

  df_combined = pd.concat([df_eligible_final, df_excluded], ignore_index=True)
  return df_combined, batches_list, total_eta, buckets
