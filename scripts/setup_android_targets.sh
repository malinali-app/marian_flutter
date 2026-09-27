#!/usr/bin/env bash
set -euo pipefail

rustup target add \
  aarch64-linux-android \
  armv7-linux-androideabi \
  i686-linux-android \
  x86_64-linux-android

echo "Android Rust targets installed."
echo "Open the example app and run: flutter run"
echo "Cargokit will compile rust/ into jniLibs during the Gradle build."
