#!/usr/bin/env bash
# Weekly analyst-target refresh on Pi 2, committed so Pi 1 can read it.
#
# Pi 2 produces, git transports, Pi 1 consumes — the same shape as
# chitra_holdings.csv. Without the commit the refresh would sit in Pi 2's
# working tree and every Benter_Rating on Pi 1 would stay blank, which is
# exactly the bug this replaces.
#
# ONLY ever touches benter_targets.csv. `git commit <path>` commits that one
# file regardless of what else is staged or dirty, so a cron run can never
# sweep up unrelated work-in-progress.
set -u
cd /home/chakravarti/github/myTrading || exit 1

.venv/bin/python benter_targets.py --refresh || exit 1

if git diff --quiet -- benter_targets.csv; then
    echo "targets unchanged — nothing to commit"
    exit 0
fi
git add benter_targets.csv || exit 1
git commit -q -m "Weekly analyst targets $(date +%F)" -- benter_targets.csv || exit 1
git push -q origin main || { echo "push failed — committed locally"; exit 1; }
echo "committed and pushed $(date +%F)"
