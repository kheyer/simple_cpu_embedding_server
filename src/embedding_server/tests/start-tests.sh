#! /usr/bin/env bash

set -e
set -x
pytest tests --disable-warnings
# pytest tests