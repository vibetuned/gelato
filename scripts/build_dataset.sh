#!/bin/bash

STAGE=${1:?Usage: $0 <stage>}
BASE="data/dataset-stage-${STAGE}"

case "$STAGE" in
  1)
    GENERATE="easy-writer ${BASE}/mxls 50000"
    ;;
  2)
    GENERATE="inter-writer ${BASE}/mxls 25000"
    ;;
  3)
    GENERATE="synthetic-writer ${BASE}/mxls 40000 \
      --num_voices 1 --no_fingerings --no_ornaments --no_dynamics --no_slurs --no_ties \
      --time_sig_set simple --scale_types major chord_progression \
      --min_measures 4 --max_measures 12"
    ;;
  4)
    GENERATE="synthetic-writer ${BASE}/mxls 50000 \
      --num_voices 1 --no_ornaments --no_dynamics --no_slurs --no_ties \
      --time_sig_set medium --min_measures 6 --max_measures 16"
    ;;
  5)
    GENERATE="synthetic-writer ${BASE}/mxls 80000 \
      --no_ornaments --no_ties \
      --time_sig_set medium --min_measures 8 --max_measures 20"
    ;;
  6)
    GENERATE="synthetic-writer ${BASE}/mxls 100000"
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 1
    ;;
esac

uv run $GENERATE
uv run inject-ids ${BASE}/mxls --out-dir ${BASE}/injected
uv run render-dataset ${BASE}/injected --svg_dir ${BASE}/svgs --img_dir ${BASE}/imgs
uv run paginate-mxl ${BASE}/injected --svg-dir ${BASE}/svgs --out-mxl-dir ${BASE}/paginated
uv run mxl2abc ${BASE}/paginated --output_dir ${BASE}/abcs
uv run clean-dataset --abc_dir ${BASE}/abcs --img_dir ${BASE}/imgs --svg_dir ${BASE}/svgs
uv run strip-abc ${BASE}/abcs --out-dir ${BASE}/abcs-strip
