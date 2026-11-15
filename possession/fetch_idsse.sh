#!/bin/bash
# Download the 7 open IDSSE Bundesliga matches (Bassek et al. 2025, Sci Data, CC-BY 4.0) from figshare.
# File ids are the ones kloppy's sportec.load_open_tracking_data uses. ~2.5 GB.
set -euo pipefail
DIR="${MTAH_DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}/external/idsse"
mkdir -p "$DIR"
while read -r m k id; do
  f="$DIR/${m}_${k}.xml"
  [ -s "$f" ] || curl -sSL --retry 3 -o "$f" "https://ndownloader.figshare.com/files/$id"
done <<'IDS'
J03WMX meta 51643475
J03WMX tracking 51643514
J03WN1 meta 51643472
J03WN1 tracking 51643517
J03WPY meta 51643487
J03WPY tracking 51643526
J03WOH meta 51643478
J03WOH tracking 51643520
J03WQQ meta 51643484
J03WQQ tracking 51643529
J03WOY meta 51643481
J03WOY tracking 51643523
J03WR9 meta 51643490
J03WR9 tracking 51643532
IDS
ls -la "$DIR"
