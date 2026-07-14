#!/bin/bash

# Ensure this script is run from inside the 'netflix_rag' folder
docker run --rm -p 4000:4000 \
  -v "$(pwd)/router.yaml:/dist/config/router.yaml" \
  -v "$(pwd)/supergraph.graphql:/dist/schema/supergraph.graphql" \
  ghcr.io/apollographql/router:v1.30.0 \
  --config /dist/config/router.yaml \
  --supergraph /dist/schema/supergraph.graphql
  